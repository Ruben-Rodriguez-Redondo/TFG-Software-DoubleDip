import os

import gradio as gr
import numpy as np
import torch

from double_dip_core.dehazing import DehazeImage, DehazeVideo
from double_dip_core.utils.image_io import np_to_pil, prepare_image, prepare_video
from double_dip_core.utils.imresize import np_imresize
from double_dip_gradio.common.utils import save_image_to_temp, save_video_to_temp, set_torch_gpu

stop_flag = False
use_gpu = True


def image_dehaze(image, num_iter, show_every):
    """
    Function call in the ui, that preprocess params and call the dehaze wrapper class
    Args:
        -image (numpy)
        -num_iter (int)
        -show_every (int)
    """
    global stop_flag
    stop_flag = False
    set_torch_gpu(use_gpu)

    conf_params = {
        "num_iter": num_iter,
        "show_every": show_every,
        "gt_ambient": None
    }
    dehaze_image = None
    t_map = None
    a_map = None

    if image is not None:
        # Save image temporally
        temp_image_path = save_image_to_temp(image)

        yield dehaze_image, t_map, a_map, gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
        for dehaze_image, t_map, a_map in main_dehaze_image_gradio(temp_image_path, conf_params):
            yield dehaze_image, t_map, a_map, gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)

        stop_flag = False
        yield dehaze_image, t_map, a_map, gr.update(visible=True), gr.update(value="Stop", visible=False), gr.update(
            visible=True)
    else:
        yield dehaze_image, t_map, a_map, gr.update(visible=False), gr.update(value="Stop", visible=False), gr.update(
            visible=True)


def video_dehaze(video, num_iter, show_every):
    """
    Function call in the ui, that preprocess params and call the dehaze video wrapper class
    Args:
        -image (numpy)
        -num_iter (int)
        -show_every (int)
    """
    global stop_flag
    stop_flag = False
    set_torch_gpu(use_gpu)
    conf_params = {
        "num_iter": num_iter,
        "show_every": show_every,
        "gt_ambients": None
    }
    dehaze_video = None

    if video is not None:

        yield dehaze_video, None, None, None, gr.update(visible=False), gr.update(visible=True), gr.update(
            visible=True), gr.update(visible=False)
        for dehaze_video, t_map, a_map in main_dehaze_video_gradio(video, conf_params):
            if not isinstance(dehaze_video, str):  # Returns frames
                yield None, dehaze_video, t_map, a_map, gr.update(visible=False), gr.update(visible=True), gr.update(
                    visible=True), gr.update(visible=False)

        stop_flag = False
        yield gr.update(value=dehaze_video), None, None, None, gr.update(visible=True), gr.update(
            visible=False), gr.update(value="Stop", visible=False), gr.update(visible=True)
    else:
        yield dehaze_video, None, None, None, gr.update(visible=False), gr.update(visible=False), gr.update(
            value="Stop", visible=False), gr.update(visible=True)


def main_dehaze_image_gradio(image_path, conf_params):
    """
    Main that uses wrapper DehazeGradio to perform Dehaze with progress updates
    """

    # Preprocess image
    image = prepare_image(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Obtain the first aproxtimation of the image
    dh = DehazeImage(image_name + "_0", image, **conf_params)
    dh_gradio = DehazeImageGradio(dh)
    yield from dh_gradio.optimize_gradio("First Aproximation")

    # Obtain the optimum ambient lightness value
    if dh_gradio.dh.use_deep_channel_prior:
        conf_params["gt_ambient"] = dh_gradio.dh.best_result.a
        conf_params["use_deep_channel_prior"] = False

    # Upgrade the first obtained image
    dh = DehazeImage(image_name + "_{}".format(1), dh_gradio.dh.post, **conf_params)
    dh_gradio = DehazeImageGradio(dh)
    yield from dh_gradio.optimize_gradio("Upgrading using ambient")


def main_dehaze_video_gradio(video_path, conf_params):
    """
    Main that uses wrapper DehazeGradio to perform Dehaze with progress updates
    """
    global stop_flag
    # Prepara image format
    video_frames, fps = prepare_video(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Obtain the first approximation of the image
    ambient_progress = gr.Progress(track_tqdm=False)
    for _ in ambient_progress.tqdm(range(1), desc=" Obtaining image atmosphere \n (may take a while)"):
        dh = DehazeVideo(video_name + "_{}".format(0), video_frames, fps, **conf_params)
    dh_gradio = DehazeVideoGradio(dh)
    yield from dh_gradio.optimize_gradio("First Aproximation")

    # Obtain the optimum ambient lightness value
    if not stop_flag:
        if dh_gradio.dh.use_deep_channel_prior:
            conf_params["gt_ambients"] = []
            conf_params["use_deep_channel_prior"] = False
            for best_result in dh_gradio.dh.best_results:
                conf_params["gt_ambients"].append(best_result.a)

        # Upgrade the first obtained image
        dh = DehazeVideo(video_name + "_{}".format(1), dh_gradio.dh.post, fps, **conf_params)
        dh_gradio = DehazeVideoGradio(dh)
        yield from dh_gradio.optimize_gradio("Upgrading using ambient")

    # Save it as a temp video
    yield from save_video_to_temp(dh_gradio.dh.post, fps)


class DehazeVideoGradio:
    """
    Wrapper for Dehaze class (add progress updates)
    """

    def __init__(self, dh: DehazeVideo):
        self.dh = dh

    def optimize_gradio(self, text):
        """Remove haze from video frames """
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        total_frames = len(self.dh.images)
        optimizer = torch.optim.Adam(self.dh.parameters, lr=self.dh.learning_rate)

        image_progress = gr.Progress(track_tqdm=False)
        for img_index in image_progress.tqdm(range(len(self.dh.images)),
                                             desc="Frames dehazed (preprocessing next frame)",
                                             total=len(self.dh.images)):
            iteration_progress = gr.Progress(track_tqdm=False)
            self.dh._init_image(img_index)
            self.dh.done = False
            self.dh.best_result = None
            for j in iteration_progress.tqdm(range(self.dh.num_iter),
                                             desc=f'\r{text} - Frame {img_index + 1}/{total_frames}'):
                optimizer.zero_grad()
                self.dh._optimization_closure(j, img_index)  # Compute loss for each image
                self.dh._obtain_current_result(j, img_index)  # Evaluate current result for each image

                if self.dh.plot_during_training and j % self.dh.show_every == 0:
                    yield from self._get_current_results(img_index)
                if self.dh.done:
                    break
                optimizer.step()
                if stop_flag:
                    yield from self._get_current_results(img_index)
                    break
            if not self.dh.limit:
                self.dh.limit = self.dh.best_result.ssim
            if stop_flag: break

    def _get_current_results(self, img_index):
        """Return current best results."""
        self._update_results(img_index)
        yield (
            np_to_pil(self.dh.post[img_index]),
            np_to_pil(self.dh.t_matting(self.dh.final_t_map, img_index)),
            np_to_pil(self.dh.final_a),
        )

    def _update_results(self, img_index):
        """Select and process best results"""
        i = img_index
        self.dh.image = self.dh.images[i]
        self.dh.final_image = np_imresize(self.dh.best_results[i].learned, output_shape=self.dh.images[i].shape[1:])
        self.dh.final_t_map = np_imresize(self.dh.best_results[i].t, output_shape=self.dh.images[i].shape[1:])
        self.dh.final_a = np_imresize(self.dh.best_results[i].a, output_shape=self.dh.images[i].shape[1:])

        # Refine transmission map
        mask_out_np = self.dh.t_matting(self.dh.final_t_map, i)
        # Compute final haze-free image
        self.dh.post[i] = np.clip((self.dh.images[i] - ((1 - mask_out_np) * self.dh.final_a)) / mask_out_np, 0, 1)


class DehazeImageGradio:
    """
    Wrapper for Dehaze class (add progress updates)
    """

    def __init__(self, dh: DehazeImage):
        self.dh = dh

    def optimize_gradio(self, text):
        """Remove haze from image """
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        progress = gr.Progress(track_tqdm=False)

        optimizer = torch.optim.Adam(self.dh.parameters, lr=self.dh.learning_rate)
        for j in progress.tqdm(range(self.dh.num_iter), desc=text):
            optimizer.zero_grad()
            self.dh._optimization_closure(j)
            self.dh._obtain_current_result(j)
            if self.dh.plot_during_training and j % self.dh.show_every == 0:
                yield from self._get_current_results()
            optimizer.step()
            if stop_flag:
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
