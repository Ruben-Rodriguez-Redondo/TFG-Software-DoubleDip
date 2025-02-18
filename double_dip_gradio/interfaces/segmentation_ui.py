import gradio as gr

from double_dip_gradio.common.utils import change_scene, get_image_params, get_button_params
from double_dip_gradio.logic.segmentation_logic import image_segmentation

image_params = get_image_params()
button_params = get_button_params()


def get_segmentation_ui(menu_ui):
    with gr.Column(visible=False) as sementation_separation_ui:
        gr.Markdown("<h1 style='text-align: center;'>Segmentation</h1>")

        with gr.Row(visible=True):
            image = gr.Image(label="Image", placeholder="#Upload one image \n", **image_params)

        num_first_step = gr.Slider(minimum=1, maximum=10000, step=1, value=2000,
                                   label="Número de iteraciones primer paso", visible=True)
        num_second_step = gr.Slider(minimum=1, maximum=10000, step=1, value=4000,
                                    label="Número de iteraciones segundo paso", visible=True)
        show_every = gr.Slider(minimum=1, maximum=10000, step=1, value=500,
                               label="Show each X iterations ")

        button_execute = gr.Button("Compute Segmentation", variant="primary", **button_params)
        with gr.Row(visible=False) as output_images_row:
            left = gr.Image(label="Layer 1", interactive=False, **image_params)
            right = gr.Image(label="Layer 2", interactive=False, **image_params)
            learned_mask = gr.Image(label="Learned mask", interactive=False, **image_params)
            learned_image = gr.Image(label="Learned image", interactive=False, **image_params)

        button_execute.click(fn=image_segmentation,
                             inputs=[image, num_first_step, num_second_step, show_every],
                             outputs=[left, right, learned_mask, learned_image, output_images_row],
                             show_progress="minimal"
                             )
        button_return = gr.Button("Return to menu", variant="secondary", **button_params)
        button_return.click(fn=change_scene,
                            inputs=None, outputs=[sementation_separation_ui, menu_ui])

    return sementation_separation_ui
