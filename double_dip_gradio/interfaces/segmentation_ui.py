import gradio as gr

import double_dip_gradio.logic.segmentation_logic as segmentation_logic
from double_dip_gradio.common.utils import change_scene, get_image_params, get_button_params, stop_process, set_use_gpu
from double_dip_gradio.logic.segmentation_logic import image_segmentation

image_params = get_image_params()
button_params = get_button_params()


def get_segmentation_ui(menu_ui):
    with gr.Column(visible=False, variant="panel", elem_classes="gr-column") as sementation_separation_ui:
        gr.Markdown("<h1 class='centered-header'>SEGMENTATION</h1>")
        with gr.Row(elem_classes="row-toggles"):
            toggle_gpu = gr.Checkbox(label="Use GPU", value=True, elem_classes="toggles")

        with gr.Row(visible=True):
            image = gr.Image(label="Image", placeholder="#Upload one image \n", **image_params)

        num_first_step = gr.Slider(minimum=1, maximum=10000, step=1, value=2000,
                                   label="Number of iterations first step", visible=True)
        num_second_step = gr.Slider(minimum=1, maximum=10000, step=1, value=4000,
                                    label="Number of iterations second step", visible=True)
        show_every = gr.Slider(minimum=1, maximum=10000, step=1, value=500,
                               label="Show each X iterations ")

        button_execute = gr.Button("Compute Segmentation", variant="primary", **button_params)
        stop_btn = gr.Button("Stop", visible=False, elem_classes="gr-button-custom-stop")

        with gr.Row(visible=False) as output_images_row:
            left = gr.Image(label="Layer 1", interactive=False, **image_params)
            right = gr.Image(label="Layer 2", interactive=False, **image_params)
            learned_mask = gr.Image(label="Learned mask", interactive=False, **image_params)
            learned_image = gr.Image(label="Learned image", interactive=False, **image_params)

        toggle_gpu.change(fn=lambda use_gpu: set_use_gpu(segmentation_logic, use_gpu), inputs=toggle_gpu)

        stop_btn.click(fn=lambda: stop_process(segmentation_logic), outputs=stop_btn)
        button_execute.click(fn=image_segmentation,
                             inputs=[image, num_first_step, num_second_step, show_every],
                             outputs=[left, right, learned_mask, learned_image, output_images_row, stop_btn,
                                      button_execute],
                             show_progress="minimal"
                             )
        button_return = gr.Button("Return to menu", variant="secondary", **button_params)
        button_return.click(fn=change_scene,
                            inputs=None, outputs=[sementation_separation_ui, menu_ui])

    return sementation_separation_ui
