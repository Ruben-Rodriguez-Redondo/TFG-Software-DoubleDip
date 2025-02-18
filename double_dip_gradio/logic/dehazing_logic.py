import os

import gradio as gr
import numpy as np
import torch

from double_dip_core.dehazing import Dehaze
from double_dip_core.utils.image_io import np_to_pil, prepare_image
from double_dip_core.utils.imresize import np_imresize
from double_dip_gradio.common.utils import save_image_to_temp, change_scene

stop_flag = False

def stop_program():
    global stop_flag
    stop_flag = True


def image_dehaze(image, num_iter, show_every):
    """
    Function call in the ui, that preprocess params and call the dehaze wrapper class
    Args:
        image (numpy)
        num_iter (int)
        show_every (int)
    """
    global stop_flag

    conf_params = {
        "num_iter": num_iter,
        "show_every": show_every,
        "gt_ambient": None
    }
    dehaze_image = locals().get('dehaze_image', None)
    t_map = locals().get('t_map', None)
    a_map = locals().get('a_map', None)

    if image is not None:
        # Save image temporally
        temp_image_path = save_image_to_temp(image)

        yield dehaze_image, t_map, a_map, gr.update(visible=True),gr.update(visible=True),gr.update(visible=False)
        for dehaze_image, t_map, a_map in main_dehaze_gradio(temp_image_path, conf_params):
            yield dehaze_image, t_map, a_map, gr.update(visible=True),gr.update(visible=True),gr.update(visible=False)

        stop_flag = False
        yield dehaze_image, t_map, a_map, gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
    else:
        return None, None, None, gr.update(visible=False),gr.update(visible=False),gr.update(visible=True)


def main_dehaze_gradio(image_path, conf_params):
    """
    Main that uses wrapper DehazeGradio to perform Dehaze with progress updates
    """

    # Preprocess image
    image = prepare_image(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Obtain the first aproxtimation of the image
    dh = Dehaze(image_name + "_0", image, **conf_params)
    dh_gradio = DehazeGradio(dh)
    yield from dh_gradio.optimize_gradio("First Aproximation")

    # Obtain the optimum ambient lightness value
    if dh_gradio.dh.use_deep_channel_prior:
        assert not dh_gradio.dh.gt_ambient
        conf_params["gt_ambient"] = dh_gradio.dh.best_result.a

    # Upgrade the first obtained image
    assert dh_gradio.dh.post.shape == image.shape, (dh_gradio.dh.post.shape, image.shape)
    dh = Dehaze(image_name + "_{}".format(1), dh_gradio.dh.post, **conf_params)
    dh_gradio = DehazeGradio(dh)
    yield from dh_gradio.optimize_gradio("Upgrading using ambient")


class DehazeGradio:
    """
    Wrapper for Dehaze class (add progress updates)
    """

    def __init__(self, dh: Dehaze):
        self.dh = dh
    def optimize_gradio(self, text, progress=gr.Progress(track_tqdm=False)):
        """Remove haze from image """
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.dh.parameters, lr=self.dh.learning_rate)

        for j in progress.tqdm(range(self.dh.num_iter), desc=text):
            optimizer.zero_grad()
            self.dh._optimization_closure(j)
            self.dh._obtain_current_result(j)
            if self.dh.plot_during_training and j % self.dh.show_every == 0:
                yield from self._get_current_results()
            optimizer.step()
            if self._check_stop() :
                yield from self._get_current_results()
                break
        yield from self._get_current_results()

    def _get_current_results(self):
        """Return current best results."""
        self._update_results()
        yield (
            np_to_pil(self.dh.post),
            np_to_pil(self.dh.t_matting(self.dh.final_t_map)),
            np_to_pil(self.dh.final_a),
        )

    def _update_results(self):
        """Select and process best results"""
        self.dh.final_image = np_imresize(self.dh.best_result.learned, output_shape=self.dh.original_image.shape[1:])
        self.dh.final_t_map = np_imresize(self.dh.best_result.t, output_shape=self.dh.original_image.shape[1:])
        self.dh.final_a = np_imresize(self.dh.best_result.a, output_shape=self.dh.original_image.shape[1:])

        # Refine t-map
        mask_out_np = self.dh.t_matting(self.dh.final_t_map)

        # Dehaze image
        self.dh.post = np.clip((self.dh.original_image - ((1 - mask_out_np) * self.dh.final_a)) / mask_out_np, 0, 1)

    def _check_stop(self):
        return stop_flag