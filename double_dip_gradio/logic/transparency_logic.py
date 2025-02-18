import os

import cv2
import gradio as gr
import numpy as np
import torch

from double_dip_core.transparency_separation import TwoImagesSeparation, Separation
from double_dip_core.utils.image_io import prepare_image, np_to_pil
from double_dip_gradio.common.utils import save_image_to_temp


def show_combined_texture(toggle_view, image_1, image_2):
    """
    Auxiliar function called in the ui, show the combined (image 1 + image 2)/2 if not ambiguous case
    Args:
        toggle_view (gradio Component)
        image_1 (numpy)
        image_2 (numpy)
    """
    if image_1 is not None and image_2 is not None and not toggle_view:
        min_height = min(image_1.shape[0], image_2.shape[0])
        min_width = min(image_1.shape[1], image_2.shape[1])

        resized_1 = cv2.resize(image_1, (min_width, min_height))
        resized_2 = cv2.resize(image_2, (min_width, min_height))

        combined = (resized_1.astype(np.float32) + resized_2.astype(np.float32)) / 2
        combined = np.clip(combined, 0, 255).astype(np.uint8)
        return gr.update(value=combined, visible=True)

    return gr.update(visible=False)


def image_transparency_separation(toggle_view, image_1, image_2, num_iter, show_every):
    """
    Function call in the ui, that preprocess params and call the adequate
    transparency separation wrapper class depending on its ambiguity
    Args:
        toggle_view (gradio Component)
        image_1 (numpy)
        image_2 (numpy)
        num_iter (int)
        show_every (int)
    """
    if toggle_view:
        main_trasnparency = main_two_images_separation_gradio  # Ambiguous case
    else:
        main_trasnparency = main_separation_gradio

    conf_params = {
        "num_iter": num_iter,
        "show_every": show_every,
    }
    if image_1 is not None and image_2 is not None:
        image_1_path = save_image_to_temp(image_1)
        image_2_path = save_image_to_temp(image_2)

        yield None, None, gr.update(visible=True)
        for reflection, transmission in main_trasnparency(image_1_path, image_2_path, conf_params):
            yield reflection, transmission, gr.update(visible=True)
    else:
        return None, None, gr.update(visible=False)


def main_two_images_separation_gradio(path_input1, path_input2, conf_params):
    """
    Main that uses wrapper TwoImagesSeparationGradio to perform transparency
    separation with progress updates
    """
    input1 = prepare_image(path_input1)
    input2 = prepare_image(path_input2)
    input1_name = os.path.splitext(os.path.basename(path_input1))[0]
    input2_name = os.path.splitext(os.path.basename(path_input2))[0]
    t = TwoImagesSeparation(input1_name, input2_name, input1, input2, **conf_params)
    t_gradio = TwoImagesSeparationGradio(t)
    yield from t_gradio.optimize_gradio("Separating two ambiguous images")


def main_separation_gradio(path_input1, path_input2, conf_params):
    t1 = prepare_image(path_input1)
    t2 = prepare_image(path_input2)

    min_height = min(t1.shape[0], t2.shape[0])
    min_width = min(t1.shape[1], t2.shape[1])

    t1 = cv2.resize(t1, (min_width, min_height))
    t2 = cv2.resize(t2, (min_width, min_height))

    s = Separation('textures', (t1 + t2) / 2, **conf_params)
    s_gradio = SeparationGradio(s)
    yield from s_gradio.optimize_gradio("Separating two textures")


class TwoImagesSeparationGradio:
    def __init__(self, t: TwoImagesSeparation):
        self.t = t

    def optimize_gradio(self, text, progress=gr.Progress(track_tqdm=False)):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.t.parameters, lr=self.t.learning_rate)
        for j in progress.tqdm(range(self.t.num_iter), desc=text):
            optimizer.zero_grad()
            self.t._optimization_closure(j)
            self.t._obtain_current_result()
            if self.t.plot_during_training and j % self.t.show_every == 0:
                yield from self._get_current_results()
            optimizer.step()

        yield from self._get_current_results()

    def _get_current_results(self):
        yield np_to_pil(self.t.best_result.reflection), np_to_pil(self.t.best_result.transmission)


class SeparationGradio:
    def __init__(self, s: Separation):
        self.s = s

    def optimize_gradio(self, text, progress=gr.Progress(track_tqdm=False)):
        """Perform image transparency separation"""

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.s.parameters, lr=self.s.learning_rate)
        for j in progress.tqdm(range(self.s.num_iter), desc=text):
            optimizer.zero_grad()
            self.s._optimization_closure(j)
            self.s._obtain_current_result(j)
            if self.s.plot_during_training and j % self.s.show_every == 0:
                yield from self._get_current_results()
            optimizer.step()
        yield from self._get_current_results()

    def _get_current_results(self):
        """Return current best results."""
        yield np_to_pil(2 * self.s.best_result.reflection), np_to_pil(2 * self.s.best_result.transmission)
