import gradio as gr

from double_dip_gradio.common.utils import get_image_params, change_scene, get_button_params
from double_dip_gradio.logic.watermark_logic import update_label_with_dimensions, draw_bbox_on_image, \
    image_watermark_removal_hint, image_watermark_removal_no_hint

image_params = get_image_params()
button_params = get_button_params()


def get_watermarks_removal_ui(menu_ui):
    with gr.Column(visible=False) as transparency_separation_ui:
        gr.Markdown("#Watermak removal")

        toggle_view = gr.Checkbox(label="No hint (boundind box)", value=False)

        with gr.Row(visible=True) as input_image_hint:
            image = gr.Image(label="Image", placeholder="#Upload one image \n", **image_params)
            image_hint = gr.Image(label="Image hint", interactive=False, visible=False, **image_params)
        with gr.Row(visible=True) as input_bbox_hint:
            x1 = gr.Slider(label="Coordenada X (esquina superior izq)", minimum=0, maximum=0, step=1, value=151)
            y1 = gr.Slider(label="Coordenada Y (esquina superior izq)", minimum=0, maximum=0, step=1, value=127)
            x2 = gr.Slider(label="Coordenada X (esquina inferior der)", minimum=0, maximum=0, step=1, value=334)
            y2 = gr.Slider(label="Coordenada Y (esquina inferior der)", minimum=0, maximum=0, step=1, value=194)

        image.change(update_label_with_dimensions, inputs=[image], outputs=[image, x1, y1, x2, y2])

        x1.change(fn=draw_bbox_on_image, inputs=[image, x1, y1, x2, y2], outputs=[image, image_hint])
        y1.change(fn=draw_bbox_on_image, inputs=[image, x1, y1, x2, y2], outputs=[image, image_hint])
        x2.change(fn=draw_bbox_on_image, inputs=[image, x1, y1, x2, y2], outputs=[image, image_hint])
        y2.change(fn=draw_bbox_on_image, inputs=[image, x1, y1, x2, y2], outputs=[image, image_hint])

        with gr.Row(visible=False) as inputs_no_hint:
            image_1 = gr.Image(label="Image 1", placeholder="#Upload one image \n", **image_params)
            image_2 = gr.Image(label="Image 2", placeholder="#Upload another image\n", **image_params)
            image_3 = gr.Image(label="Image 3", placeholder="#Upload another image \n (opcional)", **image_params)

        num_first_step = gr.Slider(minimum=1, maximum=10000, step=1, value=4000,
                                   label="Número de iteraciones primer paso", visible=True)
        num_second_step = gr.Slider(minimum=1, maximum=10000, step=1, value=7000,
                                    label="Número de iteraciones segundo paso", visible=True)

        num_iter = gr.Slider(minimum=1, maximum=10000, step=1, value=4000,
                             label="Número de Iteraciones (num_iter)", visible=False)

        # Always visible
        show_every = gr.Slider(minimum=1, maximum=10000, step=1, value=500,
                               label="Show each X iterations ")

        button_hint = gr.Button("Compute Watermark Removal (hint)", variant="primary", visible=True, **button_params)

        button_no_hint = gr.Button("Compute Watermark Removal (no hint)", variant="primary", visible=False,
                                   **button_params)

        with gr.Row(visible=False) as output_hint_row:
            clean_hint = gr.Image(label="Clean image", interactive=False, **image_params)
            watermark_hint = gr.Image(label="Watermark", interactive=False, **image_params)
            mask = gr.Image(label="Mask image", interactive=False, **image_params)

        with gr.Row(visible=False) as output_no_hint_row:
            clean_1 = gr.Image(label="Clean image 1", interactive=False, **image_params)
            clean_2 = gr.Image(label="Clean image 2", interactive=False, **image_params)
            clean_3 = gr.Image(label="Clean image 3", interactive=False, **image_params)
            watermark_no_hint = gr.Image(label="Watermark", interactive=False, **image_params)
            mask_no_hint = gr.Image(label="Mask image", interactive=False, **image_params)

        button_hint.click(fn=image_watermark_removal_hint,
                          inputs=[x1, y1, x2, y2, num_first_step, num_second_step, show_every],
                          outputs=[clean_hint, mask, watermark_hint, output_hint_row],
                          show_progress="minimal"
                          )

        button_no_hint.click(fn=image_watermark_removal_no_hint,
                             inputs=[image_1, image_2, image_3, num_iter, show_every],
                             outputs=[clean_1, clean_2, clean_3, watermark_no_hint, mask_no_hint, output_no_hint_row],
                             show_progress="minimal"
                             )

        toggle_view.change(
            lambda x: (
                *(gr.update(visible=x),) * 3,
                *(gr.update(visible=not x),) * 5,
                *(gr.update(visible=False),) * 2  # Hidde outputs rows when change
            ),
            inputs=[toggle_view],
            outputs=[inputs_no_hint, num_iter, button_no_hint,
                     input_image_hint, input_bbox_hint, num_first_step
                , num_second_step, button_hint, output_hint_row, output_no_hint_row]
        )

        button_return = gr.Button("Return to menu", variant="secondary", **button_params)
        button_return.click(fn=change_scene,
                            inputs=None, outputs=[transparency_separation_ui, menu_ui])

    return transparency_separation_ui
