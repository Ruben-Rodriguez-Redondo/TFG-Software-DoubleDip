import gradio as gr

from double_dip_gradio.common.utils import change_scene, get_image_params, get_button_params
from double_dip_gradio.logic.dehazing_logic import image_dehaze,stop_program

image_params = get_image_params()
button_params = get_button_params()


def get_dehazing_ui(menu_ui):
    with gr.Column(visible=False) as dehazing_ui:
        gr.Markdown("<h1 style='text-align: center;'>DEHAZING</h1>")

        haze_image = gr.Image(label="Haze image", placeholder="#Upload haze image\n",
                              **image_params)

        num_iter = gr.Slider(minimum=1, maximum=10000, step=1, value=4000, label="Número de Iteraciones (num_iter)")
        show_every = gr.Slider(minimum=1, maximum=10000, step=1, value=500,
                               label="Show each X iterations ")

        button_execute = gr.Button("Compute Dehazing", variant="primary", **button_params)
        stop_btn = gr.Button("Stop", visible=False, elem_classes="gr-button-custom-stop")

        with gr.Row(visible=False) as output_images_row:
            dehaze_image = gr.Image(label="Dehaze image", interactive=False, **image_params)
            t_map = gr.Image(label="T-map", interactive=False, **image_params)
            a_map = gr.Image(label="A-map", interactive=False, **image_params)

        stop_btn.click(fn=stop_program)
        button_execute.click(fn=image_dehaze,
                             inputs=[haze_image, num_iter, show_every],
                             outputs=[dehaze_image, t_map, a_map, output_images_row,stop_btn,button_execute],
                             show_progress="minimal"
                             )
        button_return = gr.Button("Return to menu", variant="secondary", **button_params)
        button_return.click(fn=change_scene,
                            inputs=None, outputs=[dehazing_ui, menu_ui])

    return dehazing_ui
