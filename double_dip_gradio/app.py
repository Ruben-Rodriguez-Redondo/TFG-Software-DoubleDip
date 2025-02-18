import json
import gradio as gr

from double_dip_gradio.common.utils import change_scene, get_button_params, get_gallery_params, get_app_images
from double_dip_gradio.interfaces.dehazing_ui import get_dehazing_ui
from double_dip_gradio.interfaces.segmentation_ui import get_segmentation_ui
from double_dip_gradio.interfaces.transparency_ui import get_transparency_separation_ui
from double_dip_gradio.interfaces.watermark_ui import get_watermarks_removal_ui


# Todo add stop button
# Todo change iter params names for iteration or itr (a not reserved word), and put in every one
# the same name same as step and the format of the return images
# Todo order the methods in the core mains (step_optimization_clousru, finalize_iteration, it.._plot_clousure,...
# Todo maybe add cpu execution
with open('common/config.json', 'r') as f:
    config = json.load(f)

css = "assets/styles/app.css"
imgs = get_app_images("assets/images")
gallery_params = get_gallery_params()
button_params = get_button_params()

with gr.Blocks(css_paths=css) as app:
    with gr.Column(visible=True, min_width=0, variant="panel", elem_classes="gr-column") as menu_ui:
        gr.Markdown("<h1 style='text-align: center;'>MAIN FUNCTIONALITIES</h1>")

        button_1 = gr.Button("Segmentación", variant="primary", **button_params)
        with gr.Row():
            gr.Gallery(value=imgs["seg"], label="Segmentación", columns=4, **gallery_params)

        button_2 = gr.Button("Transparency Separation", variant="primary", **button_params)
        with gr.Row():
            gr.Gallery(value=imgs["trans"]["amb"], label="Transparency Separation (Ambiguous)", columns=4,
                       **gallery_params)

        button_3 = gr.Button("Watermarks Removal", variant="primary", **button_params)
        with gr.Row():
            gr.Gallery(value=imgs["wat"]["hint"], label="Watermarks Removal (hint)", columns=4, **gallery_params)

        button_4 = gr.Button("Dehazing", variant="primary", **button_params)
        with gr.Row():
            gr.Gallery(value=imgs["deh"], label="Dehazing", columns=4, **gallery_params)

    segmentation_ui = get_segmentation_ui(menu_ui)
    transparency_ui = get_transparency_separation_ui(menu_ui)
    watermark_ui = get_watermarks_removal_ui(menu_ui)
    dehazing_ui = get_dehazing_ui(menu_ui)

    buttons = [
        (button_1, segmentation_ui),
        (button_2, transparency_ui),
        (button_3, watermark_ui),
        (button_4, dehazing_ui)
    ]

    for button, ui in buttons:
        button.click(fn=change_scene, inputs=None, outputs=[menu_ui, ui])


def launch():
    app.launch(**config)


if __name__ == "__main__":
    launch()
