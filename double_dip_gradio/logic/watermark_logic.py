import os

import cv2
import gradio as gr
import numpy as np
import torch

from double_dip_core.utils.image_io import prepare_image, np_to_pil, torch_to_np
from double_dip_core.watermarks_removal import Watermark, ManyImagesWatermarkNoHint
from double_dip_gradio.common.utils import save_image_to_temp, set_torch_gpu

original_image = None
stop_flag = False
use_gpu = True


def draw_bbox_on_image(image, x1, y1, x2, y2):
    """
    Draw the hint in the ui in real time
    """
    global original_image

    if image is None:
        return image, gr.update(value=None, visible=False)

    if original_image is None:
        original_image = image.copy()

    if None in (x1, y1, x2, y2):
        return original_image, gr.update(value=None, visible=False)

    height, width, _ = image.shape
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
        return original_image, gr.update(value=None, visible=False)

    image_copy = original_image.copy()
    cv2.rectangle(image_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return image_copy, gr.update(value=process_hint(original_image, x1, y1, x2, y2), visible=True)


def process_hint(image, x1, y1, x2, y2):
    """
    Create the hint, i.e a white box over a completely black image
    """
    height, width = image.shape[:2]

    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2 + 1, x1:x2 + 1] = 255

    return mask


def update_label_with_dimensions(image):
    """
    Establish the slider limits using image shape
    """
    global original_image
    if image is None:
        return (gr.update(label="Image 1"), gr.update(value=0, maximum=0),
                gr.update(value=0, maximum=0), gr.update(value=0, maximum=0)
                , gr.update(value=0, maximum=0))


    else:
        if original_image is None:
            height, width = image.shape[:2]

            x1 = width // 4
            y1 = height // 4
            x2 = (3 * width) // 4
            y2 = (3 * height) // 4

            return (gr.update(label=f"Image 1 ({width}x{height})"),
                    gr.update(maximum=width, value=x1),
                    gr.update(maximum=height, value=y1),
                    gr.update(maximum=width, value=x2),
                    gr.update(maximum=height, value=y2))

        return (gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update())


def image_watermark_removal_no_hint(image_1, image_2, image_3, num_iter, show_every):
    """
    Function call in the ui, that preprocess params and call the adequate
    watermark removal wrapper class
    Args:
        image_1 (numpy)
        image_2 (numpy)
        image_3 (numpy)
        num_iter (int)
        show_every (int)
    """
    global stop_flag
    stop_flag = False
    set_torch_gpu(use_gpu)

    conf_params = {
        "num_iter_per_step": num_iter,
        "show_every": show_every,
    }
    clean_imgs = [None, None, None]
    watermark_img = None
    mask = None

    if image_1 is not None and image_2 is not None:
        image_1_path = save_image_to_temp(image_1)
        image_2_path = save_image_to_temp(image_2)
        imgs_paths = [image_1_path, image_2_path]
        if image_3:
            image_3_path = save_image_to_temp(image_3)
            imgs_paths.append(image_3_path)

        imgs_names = [os.path.splitext(os.path.basename(path))[0] for path in imgs_paths]

        yield clean_imgs[0], clean_imgs[1], clean_imgs[2], watermark_img, mask, gr.update(visible=True), gr.update(
            visible=True), gr.update(visible=False)
        for clean_imgs, watermark_img, mask in main_remove_watermark_many_images_gradio(imgs_paths, imgs_names,
                                                                                        conf_params):
            if image_3 is None:
                clean_imgs.append(gr.update(visible=False))
            yield clean_imgs[0], clean_imgs[1], clean_imgs[2], watermark_img, mask, gr.update(visible=True), gr.update(
                visible=True), gr.update(visible=False)
        stop_flag = False
        yield clean_imgs[0], clean_imgs[1], clean_imgs[2], watermark_img, mask, gr.update(visible=True), gr.update(
            value="Stop", visible=False), gr.update(visible=True)

    else:
        yield clean_imgs[0], clean_imgs[1], clean_imgs[2], watermark_img, mask, gr.update(visible=False), gr.update(
            value="Stop", visible=False), gr.update(visible=True)


def image_watermark_removal_hint(x1, y1, x2, y2, num_first_step, num_second_step, show_every):
    """
    Function call in the ui, that preprocess params and call the adequate
    watermark removal wrapper class
    Args:
        x1,y1,x2,y (int): hint coordinates
        num_first_step (int)
        num_second_step (int)
        show_every (int)
    """
    global stop_flag
    stop_flag = False
    set_torch_gpu(use_gpu)

    global original_image

    conf_params = {
        "num_iter_first_step": num_first_step,
        "num_iter_second_step": num_second_step,
        "show_every": show_every,
    }
    clean_img = None
    mask = None
    watermark_img = None

    if original_image is not None:
        image_path = save_image_to_temp(original_image)
        img_hint = process_hint(original_image, x1, y1, x2, y2)
        image_hint_path = save_image_to_temp(img_hint)

        yield clean_img, mask, watermark_img, gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
        for clean_img, mask, watermark_img in main_remove_watermark_hint_gradio(image_path, image_hint_path,
                                                                                conf_params):
            yield clean_img, mask, watermark_img, gr.update(visible=True), gr.update(visible=True), gr.update(
                visible=False)
        stop_flag = False
        yield clean_img, mask, watermark_img, gr.update(visible=True), gr.update(value="Stop",
                                                                                 visible=False), gr.update(visible=True)

    else:
        yield clean_img, mask, watermark_img, gr.update(visible=False), gr.update(value="Stop",
                                                                                  visible=False), gr.update(
            visible=True)


def main_remove_watermark_hint_gradio(img_path, hint_path, conf_params):
    """
    Main that uses wrapper WatermarkGradio to perform
    watermark removal using a hint with updates
    """
    image = prepare_image(img_path)
    hint = prepare_image(hint_path)
    image_name = os.path.splitext(os.path.basename(img_path))[0]
    w = Watermark(image_name, image, watermark_hint=hint, **conf_params)
    w_gradio = WatermarkGradio(w)
    yield from w_gradio.optimize_gradio()


def main_remove_watermark_many_images_gradio(imgs_paths, imgs_names, conf_params):
    """
     Main that uses wrapper ManyImagesWatermarkNoHintGradio to perform
     watermark removal with updates
     """
    imgs = [prepare_image(img_path) for img_path in imgs_paths]
    w = ManyImagesWatermarkNoHint(imgs_names, imgs, plot_during_training=True, **conf_params)
    w_gradio = ManyImagesWatermarkNoHintGradio(w)
    yield from w_gradio.optimize_gradio()


class WatermarkGradio:
    """
    Wrapper for Watermark class (add progress updates)
    """

    def __init__(self, w: Watermark):
        self.w = w

    def optimize_gradio(self):
        """Perform image watermark removal"""
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        progress = gr.Progress(track_tqdm=False)
        # step 1
        optimizer = torch.optim.Adam(self.w.parameters, lr=self.w.learning_rate)
        for j in progress.tqdm(range(self.w.num_iter_first_step), desc="First step"):
            optimizer.zero_grad()
            self.w._step1_optimization_closure(j)
            optimizer.step()
            if self.w.plot_during_training and j % self.w.show_every == 0:
                yield from self._get_current_results(1)
            if stop_flag:
                yield from self._get_current_results(1)
                break
        # step 2
        optimizer = torch.optim.Adam(self.w.parameters, lr=self.w.learning_rate)
        for j in progress.tqdm(range(self.w.num_iter_second_step), desc="Second step"):
            optimizer.zero_grad()
            self.w._step2_optimization_closure(j)
            self.w._step2_finalize_iteration(j)
            if self.w.plot_during_training and j % self.w.show_every == 0:
                yield from self._get_current_results(2)
            optimizer.step()
            if stop_flag:
                yield from self._get_current_results(2)
                break
        self.w._update_result_closure()
        yield from self._get_final_results()

    def _get_current_results(self, step):
        """Return actual results. In step one only uses clean_net"""
        if step == 1:
            yield np_to_pil(torch_to_np(self.w.clean_net_output)), None, None
        else:
            yield np_to_pil(torch_to_np(self.w.clean_net_output)), np_to_pil(
                torch_to_np(self.w.mask_net_output)), np_to_pil(torch_to_np(self.w.watermark_net_output))

    def _get_final_results(self):
        """Return current best results."""
        recovered_mask = self.w.watermark_hint * self.w.best_result.mask
        clear_image_places = np.zeros_like(recovered_mask)
        clear_image_places[recovered_mask < 0.1] = 1
        clean_image = clear_image_places * self.w.image + (1 - clear_image_places) * self.w.best_result.clean
        recovered_watermark = self.w.watermark_hint * self.w.best_result.mask * self.w.best_result.watermark

        yield np_to_pil(clean_image), np_to_pil(self.w.best_result.mask), np_to_pil(recovered_watermark)


class ManyImagesWatermarkNoHintGradio:
    """
    Wrapper for ManyImagesWatermarkNoHint class (add progress updates)
    """

    def __init__(self, w: ManyImagesWatermarkNoHint):
        self.w = w

    def optimize_gradio(self):
        """Perform image watermark removal"""
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        progress = gr.Progress(track_tqdm=False)
        optimizer = torch.optim.Adam(self.w.parameters, lr=self.w.learning_rate)
        for j in progress.tqdm(range(self.w.num_iter_per_step), desc="Removing watermarks"):
            optimizer.zero_grad()
            self.w._optimization_closure(j)
            self.w._finalize_iteration(j)
            if self.w.plot_during_training and j % self.w.show_every == 0:
                yield from self._get_current_results()
            optimizer.step()
            if stop_flag:
                yield from self._get_current_results()
                break
        self.w._update_result_closure()
        yield from self._get_final_results()

    def _get_final_results(self):
        """Return actual results."""
        obtained_watermark = self.w.best_result.mask * self.w.best_result.watermark
        obtained_imgs = self.w.best_result.cleans
        v = np.zeros_like(obtained_watermark)
        v[obtained_watermark < 0.1] = 1
        final_imgs = []
        # Upgrade the clean image
        for im, obt_im in zip(self.w.images, obtained_imgs):
            final_imgs.append(np_to_pil(v * im + (1 - v) * obt_im))
        obtained_watermark[obtained_watermark < 0.1] = 0

        yield final_imgs, np_to_pil(obtained_watermark), np_to_pil(self.w.best_result.mask)

    def _get_current_results(self):
        """Return current best results."""

        clean_out_nps = [np_to_pil(torch_to_np(clean_net_output)) for
                         clean_net_output in self.w.clean_nets_outputs]

        yield (clean_out_nps, np_to_pil(torch_to_np(self.w.watermark_net_output)),
               np_to_pil(torch_to_np(self.w.mask_net_output)))
