import gradio as gr

import double_dip_gradio.logic.dehazing_logic as dehazing_logic
from double_dip_gradio.common.utils import change_scene, get_image_params, get_button_params, get_video_params, \
    stop_process, set_use_gpu
from double_dip_gradio.logic.dehazing_logic import image_dehaze, video_dehaze

image_params = get_image_params()
button_params = get_button_params()
video_params = get_video_params()


def get_dehazing_ui(menu_ui):
    with gr.Column(visible=False, variant="panel", elem_classes="gr-column") as dehazing_ui:
        gr.Markdown("<h1 class='centered-header'>DEHAZING</h1>")

        with gr.Row(elem_classes="row-toggles"):
            toggle_view = gr.Checkbox(label="Video", value=False, elem_classes="toggles")
            toggle_gpu = gr.Checkbox(label="Use GPU", value=True, elem_classes="toggles")
            set_use_gpu(dehazing_logic, toggle_gpu.value)
        with gr.Row(visible=True) as input_image:
            haze_image = gr.Image(label="Haze image", placeholder="#Upload haze image\n",
                                  **image_params)

        with gr.Row(visible=False) as input_video:
            haze_video = gr.PlayableVideo(label="Haze video", **video_params)

        num_iter = gr.Slider(minimum=1, maximum=10000, step=1, value=4000, label="Number of Iterations")
        show_every = gr.Slider(minimum=1, maximum=10000, step=1, value=500,
                               label="Show each X iterations ")

        button_image = gr.Button("Compute Dehazing", visible=True, variant="primary", **button_params)
        button_video = gr.Button("Compute Dehazing", visible=False, variant="primary", **button_params)

        stop_btn = gr.Button("Stop", visible=False, elem_classes="gr-button-custom-stop")

        with gr.Row(visible=False) as output_images_row:
            dehaze_image = gr.Image(label="Dehaze image", interactive=False, **image_params)
            t_map = gr.Image(label="T-map", interactive=False, **image_params)
            a_map = gr.Image(label="A-map", interactive=False, **image_params)

        with gr.Row(visible=False) as output_video_row:
            dehaze_video = gr.PlayableVideo(label="Dehaze video", autoplay=True, elem_id="input-images_remove")

        stop_btn.click(fn=lambda: stop_process(dehazing_logic), outputs=stop_btn)

        button_image.click(fn=image_dehaze,
                           inputs=[haze_image, num_iter, show_every],
                           outputs=[dehaze_image, t_map, a_map, output_images_row, stop_btn, button_image],
                           show_progress="minimal"
                           )
        button_video.click(fn=video_dehaze,
                           inputs=[haze_video, num_iter, show_every],
                           outputs=[dehaze_video, dehaze_image, t_map, a_map, output_video_row, output_images_row,
                                    stop_btn, button_video],
                           show_progress="minimal"
                           )
        toggle_view.change(
            lambda x: (
                *(gr.update(visible=not x),) * 2,
                *(gr.update(visible=x),) * 2,
                *(gr.update(visible=False),) * 2
            ),
            inputs=[toggle_view],
            outputs=[input_image, button_image, input_video, button_video,
                     output_images_row, output_video_row]
        )
        toggle_gpu.change(fn=lambda use_gpu: set_use_gpu(dehazing_logic, use_gpu), inputs=toggle_gpu)

        button_return = gr.Button("Return to menu", variant="secondary", **button_params)
        button_return.click(fn=change_scene,
                            inputs=None, outputs=[dehazing_ui, menu_ui])

    return dehazing_ui
