import gradio as gr

import double_dip_gradio.logic.transparency_logic as transparency_logic
from double_dip_gradio.common.utils import get_image_params, change_scene, get_button_params, stop_process, set_use_gpu
from double_dip_gradio.logic.transparency_logic import show_combined_texture, image_transparency_separation

image_params = get_image_params()
button_params = get_button_params()


def get_transparency_separation_ui(menu_ui):
    with gr.Column(visible=False, variant="panel", elem_classes="gr-column") as transparency_separation_ui:
        gr.Markdown("<h1 class='centered-header'>TRANSPARENCY SEPARATION</h1>")

        with gr.Row(elem_classes="row-toggles"):
            toggle_view = gr.Checkbox(label="Ambiguous images", value=True, elem_classes="toggles")
            toggle_gpu = gr.Checkbox(label="Use GPU", value=True, elem_classes="toggles")

        toggle_gpu.change(fn=lambda use_gpu: set_use_gpu(transparency_logic, use_gpu), inputs=toggle_gpu)

        with gr.Row(visible=True):
            image_1 = gr.Image(label="Texture 1", placeholder="#Upload one image \n", **image_params, )
            image_2 = gr.Image(label="Texture 2", placeholder="#Upload another image\n", **image_params)
            combined_image = gr.Image(label="Texture Combine", interactive=False, visible=False, **image_params)

        image_1.change(show_combined_texture, inputs=[toggle_view, image_1, image_2], outputs=[combined_image])
        image_2.change(show_combined_texture, inputs=[toggle_view, image_1, image_2], outputs=[combined_image])
        toggle_view.change(show_combined_texture, inputs=[toggle_view, image_1, image_2], outputs=[combined_image])

        num_iter = gr.Slider(minimum=1, maximum=10000, step=1, value=4000, label="Number of iterations")
        show_every = gr.Slider(minimum=1, maximum=10000, step=1, value=500,
                               label="Show each X iterations ")

        button_execute = gr.Button("Compute Separation", variant="primary", **button_params)
        stop_btn = gr.Button("Stop", visible=False, elem_classes="gr-button-custom-stop")

        with gr.Row(visible=False) as output_images_row:
            reflection_image = gr.Image(label="Reflection image", interactive=False, **image_params)
            transmission_image = gr.Image(label="Transmission image", interactive=False, **image_params)

        stop_btn.click(fn=lambda: stop_process(transparency_logic), outputs=stop_btn)
        button_execute.click(fn=image_transparency_separation,
                             inputs=[toggle_view, image_1, image_2, num_iter, show_every],
                             outputs=[reflection_image, transmission_image, output_images_row, stop_btn,
                                      button_execute],
                             show_progress="minimal"
                             )

        button_return = gr.Button("Return to menu", variant="secondary", **button_params)
        button_return.click(fn=change_scene,
                            inputs=None, outputs=[transparency_separation_ui, menu_ui])

    return transparency_separation_ui
