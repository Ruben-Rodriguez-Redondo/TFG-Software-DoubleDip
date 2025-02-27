import os

import gradio as gr
import numpy as np
import torch

from double_dip_core.segmentation import Segmentation
from double_dip_core.utils.image_io import prepare_image, np_to_pil, torch_to_np
from double_dip_gradio.common.utils import save_image_to_temp, set_torch_gpu

stop_flag = False
use_gpu = True


def image_segmentation(image, num_first_step, num_second_step, show_every):
    """
    Function call in the ui, that preprocess params and call the segmentation wrapper class
    Args:
        image (numpy)
        num_first_step (int)
        num_second_step (int)
        show_Every (int)
    """
    global stop_flag
    stop_flag = False
    set_torch_gpu(use_gpu)

    conf_params = {
        "first_step_iter_num": num_first_step,
        "second_step_iter_num": num_second_step,
        "show_every": show_every,
    }
    left = None
    right = None
    learned_mask = None
    learned_image = None

    if image is not None:
        # Save image temporally
        image_path = save_image_to_temp(image)

        yield left, right, learned_mask, learned_image, gr.update(visible=True), gr.update(visible=True), gr.update(
            visible=False)
        for left, right, learned_mask, learned_image in main_segmentation_gradio(image_path, conf_params):
            yield left, right, learned_mask, learned_image, gr.update(visible=True), gr.update(visible=True), gr.update(
                visible=False)
        stop_flag = False
        yield left, right, learned_mask, learned_image, gr.update(visible=True), gr.update(value="Stop",
                                                                                           visible=False), gr.update(
            visible=True)

    else:
        yield left, right, learned_mask, learned_image, gr.update(visible=False), gr.update(value="Stop",
                                                                                            visible=False), gr.update(
            visible=True)


def main_segmentation_gradio(image_path, conf_params):
    """
    Main that uses wrapper SegmentationGradio to perform Segmentation with
    progress updates
    """
    image = prepare_image(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    s = Segmentation(image_name, image, **conf_params)
    s_gradio = SegmentationGradio(s)
    yield from s_gradio.optimize_gradio()


class SegmentationGradio:
    """
    Wrapper for Segmentation class (add progress updates)
    """

    def __init__(self, s: Segmentation):
        self.s = s

    def optimize_gradio(self):
        """Perform image segmentation"""

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        progress = gr.Progress(track_tqdm=False)

        # step 1
        optimizer = torch.optim.Adam(self.s.parameters, lr=self.s.learning_rate)
        for j in progress.tqdm(range(self.s.first_step_iter_num), desc="First step"):
            optimizer.zero_grad()
            self.s._step1_optimization_closure(j)
            self.s._finalize_iteration()
            if self.s.plot_during_training and j % self.s.show_every == 0:
                yield from self._get_current_results()
            optimizer.step()
            if stop_flag:
                yield from self._get_current_results()
                break
        self.s._update_result_closure(1)
        if self.s.plot_during_training:
            yield from self._get_final_results()
        # step 2
        optimizer = torch.optim.Adam(self.s.parameters, lr=self.s.learning_rate)
        for j in progress.tqdm(range(self.s.second_step_iter_num), desc="Second step"):
            optimizer.zero_grad()
            self.s._step2_optimization_closure(j)
            self.s._finalize_iteration()
            if self.s.plot_during_training and j % self.s.show_every == 0:
                yield from self._get_current_results()
            optimizer.step()
            if stop_flag:
                yield from self._get_current_results()
                break
        self.s._update_result_closure(2)
        if self.s.plot_during_training:
            yield from self._get_final_results()

    def _get_current_results(self):
        """Return current best results."""
        for left_out, right_out, mask_out, image in zip(self.s.left_net_outputs,
                                                        self.s.right_net_outputs,
                                                        self.s.mask_net_outputs, self.s.images):
            mask_out_np = torch_to_np(mask_out)
            learned_mask = np_to_pil(np.clip(mask_out_np, 0, 1))
            learned_image = np_to_pil(
                np.clip(mask_out_np * torch_to_np(left_out) + (1 - mask_out_np) * torch_to_np(right_out),
                        0, 1))

            yield (np_to_pil(np.clip(torch_to_np(left_out), 0, 1)), np_to_pil(np.clip(torch_to_np(right_out), 0, 1))
                   , learned_mask, learned_image)

    def _get_final_results(self):
        """Return and process best results"""

        learned_image = np.clip(self.s.best_result.learned_mask * self.s.best_result.left
                                + (1 - self.s.best_result.learned_mask) * self.s.best_result.right,
                                0, 1)
        yield (np_to_pil(self.s.best_result.left), np_to_pil(self.s.best_result.right),
               np_to_pil(self.s.best_result.learned_mask), np_to_pil(learned_image))
