import subprocess

import gradio as gr
import os
import pathlib


title = "# Thin-Plate Spline Motion Model for Image Animation"
DESCRIPTION = '''### Gradio demo for <b>Thin-Plate Spline Motion Model for Image Animation</b>, CVPR 2022. <a href='https://arxiv.org/abs/2203.14367'>[Paper]</a><a href='https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model'>[Github Code]</a>
<img id="overview" alt="overview" src="https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model/raw/main/assets/vox.gif" />
'''
FOOTER = '<img id="visitor-badge" alt="visitor badge" src="https://visitor-badge.glitch.me/badge?page_id=gradio-blocks.Image-Animation-using-Thin-Plate-Spline-Motion-Model" />'


def get_style_image_path(style_name: str) -> str:
    base_path = 'assets'
    filenames = {
        'source': 'source.png',
        'driving': 'driving.mp4',
    }
    return f'{base_path}/{filenames[style_name]}'


def get_style_image_markdown_text(style_name: str) -> str:
    url = get_style_image_path(style_name)
    return f'<img id="style-image" src="{url}" alt="style image">'


def update_style_image(style_name: str) -> dict:
    text = get_style_image_markdown_text(style_name)
    return gr.Markdown.update(value=text)


def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])


def set_example_video(example: list) -> dict:
    return gr.Video.update(value=example[0])


def inference(img, video):
    if not os.path.exists('temp'):
        os.system('mkdir temp')

    img.save("image.png", "PNG")
    #video.save("driving.mp4")

    env = os.environ.copy()
    env["FAIR_COMPUTE_SERVER_ADDRESS"] = "http://localhost:8000/api/v1"
    video_name = str(video).replace(':', '/:')
    subprocess.check_call(
        f"c:\\Users\\koshm\\Projects\\faircompute\\target\\debug\\fair.exe task -l docker -r nvidia -i gradio_demo --input-file image.png:/workspace/image.png --input-file {video_name}:/workspace/smile.pm4 -o /workspace/result.mp4:result.mp4", shell=True, env=env)
    return 'result.mp4'


def main():
    with gr.Blocks(theme="huggingface", css='style.css') as demo:
        gr.Markdown(title)
        gr.Markdown(DESCRIPTION)

        with gr.Box():
            gr.Markdown('''## Step 1 (Provide Input Face Image)
- Drop an image containing a face to the **Input Image**.
    - If there are multiple faces in the image, use Edit button in the upper right corner and crop the input image beforehand.
''')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        input_image = gr.Image(label='Input Image',
                                               type="pil")

            with gr.Row():
                paths = sorted(pathlib.Path('assets').glob('*.png'))
                example_images = gr.Dataset(components=[input_image],
                                            samples=[[path.as_posix()]
                                                     for path in paths])

        with gr.Box():
            gr.Markdown('''## Step 2 (Select Driving Video)
- Select **Style Driving Video for the face image animation**.
''')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        driving_video = gr.Video(label='Driving Video',
                                                 format="mp4")

            with gr.Row():
                paths = sorted(pathlib.Path('assets').glob('*.mp4'))
                example_video = gr.Dataset(components=[driving_video],
                                           samples=[[path.as_posix()]
                                                    for path in paths])

        with gr.Box():
            gr.Markdown('''## Step 3 (Generate Animated Image based on the Video)
- Hit the **Generate** button. (Note: As it runs on the CPU, it takes ~ 3 minutes to generate final results.)
''')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        generate_button = gr.Button('Generate')

                with gr.Column():
                    result = gr.Video(type="file", label="Output")
        gr.Markdown(FOOTER)
        generate_button.click(fn=inference,
                              inputs=[
                                  input_image,
                                  driving_video
                              ],
                              outputs=result)
        example_images.click(fn=set_example_image,
                             inputs=example_images,
                             outputs=example_images.components)
        example_video.click(fn=set_example_video,
                            inputs=example_video,
                            outputs=example_video.components)

    demo.launch(
        enable_queue=True,
        debug=True
    )


if __name__ == '__main__':
    main()