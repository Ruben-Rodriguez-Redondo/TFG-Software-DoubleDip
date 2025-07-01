import tempfile

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

from double_dip_core.utils.image_io import np_to_pil


def change_scene():
    """
    Change (show/hide) between functionalities as segmentation, watermark...
    """
    return gr.update(visible=False), gr.update(visible=True)


def save_image_to_temp(image):
    """
    Save image temporally and return its path
    Args:
        image (np.array)
    Return:
        temp_path(str): temporally location file_path of the image
    """
    pil_image = Image.fromarray(image)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        temp_path = temp_file.name
        pil_image.save(temp_path)
    return temp_path


def save_video_to_temp(video_frames, fps):
    """
    Converts a frames secuence into a video and save it into a temp file
    Args:
        video_frames(list):, list of frames (np.array).
        fps(int):videos fps.

    Returns:
        str: Path for temporal video
    """

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            temp_video_path = temp_video.name

        _, height, width = video_frames[0].shape

        # fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(temp_video_path, 6, fps, (width, height))

        for frame in video_frames:
            if frame is None:
                break
            frame = np.array(np_to_pil(frame))
            out.write(frame)

        out.release()
        yield temp_video_path, None, None

    except Exception as e:
        print(f"Error saving temporal video: {e}")
        return None


def get_image_params():
    params = {
        "type": "numpy",
        "sources": ["upload"],
        "show_label": True,
        "show_download_button": False,
        "show_share_button": False,
        "show_fullscreen_button": False,
        "elem_classes": "input-images_remove"

    }
    return params


def get_video_params():
    params = {
        "sources": ["upload"],
        "show_label": True,
        "show_download_button": False,
        "show_share_button": False,
        "include_audio": True,  # Dont change
        "elem_classes": "input-images_remove"

    }

    return params


def stop_process(logic_py):
    setattr(logic_py, "stop_flag", True)
    return gr.update(value="Stopping...")


def set_use_gpu(logic_py, use_gpu):
    setattr(logic_py, "use_gpu", use_gpu)


def set_torch_gpu(use_gpu):
    if use_gpu:
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
        else:
            torch.set_default_device("cpu")
    else:
        torch.set_default_device("cpu")


def get_button_params():
    return {"elem_classes": "gr-button-custom"}


def get_gallery_params():
    return {"interactive": False,
            "container": False,
            "allow_preview": False,
            "show_fullscreen_button": False,
            "object_fit": "fill"}


def get_app_images(images_path):
    imgs = {
        "seg": [(f"{images_path}/seg_image.png", "Input"), (f"{images_path}/seg_learned_mask.png", "Binary Mask"),
                (f"{images_path}/seg_fg.png", "Layer 1 (left)"), (f"{images_path}/seg_bg.png", "Layer 2 (right)")],
        "trans": {"amb": [(f"{images_path}/trans_ambiguous_1.png", "Input 1"),
                          (f"{images_path}/trans_ambiguous_2.png", "Input 2"),
                          (f"{images_path}/trans_ambiguous_reflection.png", "Reflection layer"),
                          (f"{images_path}/trans_ambiguous_transmission.png", "Transmission layer")],
                  },
        "wat": {
            "hint": [(f"{images_path}/wat_image.png", "Input 1"),
                     (f"{images_path}/wat_image_hint.png", "Input 2 (Hint)"),
                     (f"{images_path}/wat_rm_image.png", "Clean input"),
                     (f"{images_path}/wat_mark_hint.png", "Watermark")]
        },
        "deh": [(f"{images_path}/deh_ori.png", "Haze image"), (f"{images_path}/deh_t_map.png", "t-Map"),
                (f"{images_path}/deh_a_map.png", "A-Map"), (f"{images_path}/deh_fin.png", "Dehaze image")]
    }
    return imgs
