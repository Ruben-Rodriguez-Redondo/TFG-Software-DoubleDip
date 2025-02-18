import tempfile

import gradio as gr
import numpy as np
from PIL import Image


def change_scene():
    """
    Change (show/hide) between functionalities as segmentation, watermark...
    """
    return gr.update(visible=False), gr.update(visible=True)


def save_image_to_temp(image: np.ndarray) -> str:
    """
    Save image temporally and return its path
    Args:
        image (np.array)
    Return:
        temp_path(str): temporaly location file_path of the image
    """
    # Convertir el ndarray a una imagen de PIL
    pil_image = Image.fromarray(image)  # Convertir a uint8 para PIL

    # Crear un archivo temporal con un nombre único
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        temp_path = temp_file.name
        pil_image.save(temp_path)  # Guardar la imagen en el archivo temporal
    return temp_path


def get_image_params():
    params = {
        "type": "numpy",
        "sources": ["upload"],
        "show_label": True,
        "show_download_button": False,
        "show_share_button": False,
        "show_fullscreen_button": False,
        "elem_classes": "input-images"

    }
    return params


def get_button_params():
    return {"elem_classes": "gr-button-custom"}


def get_gallery_params():
    return {"interactive": False, "container": False, "object_fit": "fill"}


# Todo remove the extra images once they are used in the results TFG part
def get_app_images(images_path):
    imgs = {
        "seg": [(f"{images_path}/seg_image.png", "Input"), (f"{images_path}/seg_learned_mask.png", "No Binary Mask"),
                (f"{images_path}/seg_bg.png", "Layer 1 (left)"), (f"{images_path}/seg_fg.png", "Layer 2 (right)")],
        "trans": {"amb": [(f"{images_path}/trans_ambiguous_1.png", "Input 1"),
                          (f"{images_path}/trans_ambiguous_2.png", "Input 2"),
                          (f"{images_path}/trans_ambiguous_reflection.png", "Reflection layer"),
                          (f"{images_path}/trans_ambiguous_transmission.png", "Transmission layer")],
                  "no_amb": [f"{images_path}/trans_pre.png", f"{images_path}/trans_reflection.png",
                             f"{images_path}/trans_transmission.png"]
                  },
        "wat": {
            "hint": [(f"{images_path}/wat_image.png", "Input 1"),
                     (f"{images_path}/wat_image_hint.png", "Input 2 (Hint)"),
                     (f"{images_path}/wat_rm_image.png", "Clean input"),
                     (f"{images_path}/wat_mark_hint.png", "Watermark")],
            "no_hint": [f"{images_path}/wat_1.png", f"{images_path}/wat_2.png", f"{images_path}/wat_3.png",
                        f"{images_path}/wat_rm_1.png", f"{images_path}/wat_rm_2.png", f"{images_path}/wat_rm_3.png"]
        },
        "deh": [(f"{images_path}/deh_ori.png", "Haze image"), (f"{images_path}/deh_t_map.png", "A-Map"),
                (f"{images_path}/deh_a_map.png", "Regularized T-MAP"), (f"{images_path}/deh_fin.png", "Dehaze image")]
    }
    return imgs
