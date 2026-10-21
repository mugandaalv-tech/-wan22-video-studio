
cat > app.py <<'PY'
import os
import tempfile
import traceback

import gradio as gr

MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
pipe = None


def load_model():
    global pipe

        if pipe is not None:
                return pipe

                    import torch
                        from diffusers import DiffusionPipeline

                            if not torch.cuda.is_available():
                                    raise RuntimeError(
                                                    "No CUDA GPU detected. Deploy this app to a compatible "
                                                                "GPU-backed Hugging Face Space to generate videos."
                                    )

                                        pipe = DiffusionPipeline.from_pretrained(
                                                    MODEL_ID,
                                                            torch_dtype=torch.bfloat16,
                                                                    device_map="cuda",
                                        )
                                            return pipe


                                            def generate_video(prompt, resolution, steps, seed):
                                                if not prompt or not prompt.strip():
                                                        raise gr.Error("Please enter a description for your video.")

                                                            try:
                                                                    import torch
                                                                            from diffusers.utils import export_to_video

                                                                                    model = load_model()

                                                                                            width, height = (
                                                                                                            (832, 480) if resolution == "480p"
                                                                                                                        else (1280, 720)
                                                                                            )

                                                                                                    generator = torch.Generator(device="cuda").manual_seed(
                                                                                                                    int(seed)
                                                                                                    )

                                                                                                            result = model(
                                                                                                                            prompt=prompt.strip(),
                                                                                                                                        width=width,
                                                                                                                                                    height=height,
                                                                                                                                                                num_frames=81,
                                                                                                                                                                            num_inference_steps=int(steps),
                                                                                                                                                                                        generator=generator,
                                                                                                            )

                                                                                                                    frames = result.frames[0]

                                                                                                                            output = tempfile.NamedTemporaryFile(
                                                                                                                                            suffix=".mp4", delete=False
                                                                                                                            )
                                                                                                                                    output.close()

                                                                                                                                            export_to_video(
                                                                                                                                                            frames, output.name, fps=16
                                                                                                                                            )

                                                                                                                                                    return output.name, (
                                                                                                                                                                    f"Video generated successfully at {resolution}."
                                                                                                                                                    )

                                                                                                                                                        except Exception as exc:
                                                                                                                                                                traceback.print_exc()
                                                                                                                                                                        raise gr.Error(
                                                                                                                                                                                        f"Generation failed: {exc}. Check the GPU, "
                                                                                                                                                                                                    "model access, dependencies, and server logs."
                                                                                                                                                                        )


                                                                                                                                                                        with gr.Blocks(
                                                                                                                                                                                theme=gr.themes.Soft(),
                                                                                                                                                                                    title="Wan AI Video Studio",
                                                                                                                                                                        ) as demo:
                                                                                                                                                                            gr.Markdown(
                                                                                                                                                                                        """
                                                                                                                                                                                                # 🎬 Wan AI Video Studio
                                                                                                                                                                                                        ### Cinematic AI video generation powered by Wan2.2 A14B

                                                                                                                                                                                                                Describe your scene and create a video from text.
                                                                                                                                                                                                                        Video generation requires a compatible GPU.
                                                                                                                                                                                                                                """
                                                                                                                                                                            )

                                                                                                                                                                                with gr.Row():
                                                                                                                                                                                        with gr.Column():
                                                                                                                                                                                                    prompt = gr.Textbox(
                                                                                                                                                                                                                        label="Describe your video",
                                                                                                                                                                                                                                        placeholder=(
                                                                                                                                                                                                                                                                "A cinematic aerial shot of a futuristic "
                                                                                                                                                                                                                                                                                    "city at night, rain reflecting neon lights..."
                                                                                                                                                                                                                                        ),
                                                                                                                                                                                                                                                        lines=5,
                                                                                                                                                                                                    )

                                                                                                                                                                                                                resolution = gr.Radio(
                                                                                                                                                                                                                                    choices=["480p", "720p"],
                                                                                                                                                                                                                                                    value="480p",
                                                                                                                                                                                                                                                                    label="Resolution",
                                                                                                                                                                                                                )

                                                                                                                                                                                                                            steps = gr.Slider(
                                                                                                                                                                                                                                                minimum=10,
                                                                                                                                                                                                                                                                maximum=50,
                                                                                                                                                                                                                                                                                value=30,
                                                                                                                                                                                                                                                                                                step=1,
                                                                                                                                                                                                                                                                                                                label="Inference steps",
                                                                                                                                                                                                                            )

                                                                                                                                                                                                                                        seed = gr.Number(
                                                                                                                                                                                                                                                            value=42,
                                                                                                                                                                                                                                                                            precision=0,
                                                                                                                                                                                                                                                                                            label="Random seed",
                                                                                                                                                                                                                                        )

                                                                                                                                                                                                                                                    generate = gr.Button(
                                                                                                                                                                                                                                                                        "✨ Generate video",
                                                                                                                                                                                                                                                                                        variant="primary",
                                                                                                                                                                                                                                                    )

                                                                                                                                                                                                                                                            with gr.Column():
                                                                                                                                                                                                                                                                        video = gr.Video(label="Generated video")
                                                                                                                                                                                                                                                                                    status = gr.Textbox(label="Generation status")

                                                                                                                                                                                                                                                                                        generate.click(
                                                                                                                                                                                                                                                                                                    fn=generate_video,
                                                                                                                                                                                                                                                                                                            inputs=[prompt, resolution, steps, seed],
                                                                                                                                                                                                                                                                                                                    outputs=[video, status],
                                                                                                                                                                                                                                                                                        )

                                                                                                                                                                                                                                                                                            gr.Markdown(
                                                                                                                                                                                                                                                                                                        """
                                                                                                                                                                                                                                                                                                                ---
                                                                                                                                                                                                                                                                                                                        **Model:** [Wan2.2 T2V A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)

                                                                                                                                                                                                                                                                                                                                **Note:** The model is large and requires substantial GPU
                                                                                                                                                                                                                                                                                                                                        memory. Actual video generation is not available in a
                                                                                                                                                                                                                                                                                                                                                CPU-only Codespace.
                                                                                                                                                                                                                                                                                                                                                        """
                                                                                                                                                                                                                                                                                            )


                                                                                                                                                                                                                                                                                            if __name__ == "__main__":
                                                                                                                                                                                                                                                                                                demo.queue(max_size=2).launch(
                                                                                                                                                                                                                                                                                                            server_name="0.0.0.0",
                                                                                                                                                                                                                                                                                                                    server_port=int(os.environ.get("PORT", 7860)),
                                                                                                                                                                                                                                                                                                )
                                                                                                                                                                                                                                                                                                PY

                                                                                                                                                                                                                                                                                                cat > requirements.txt <<'REQ'
                                                                                                                                                                                                                                                                                                gradio>=5.0
                                                                                                                                                                                                                                                                                                torch>=2.4
                                                                                                                                                                                                                                                                                                diffusers>=0.35.0
                                                                                                                                                                                                                                                                                                transformers
                                                                                                                                                                                                                                                                                                accelerate
                                                                                                                                                                                                                                                                                                safetensors
                                                                                                                                                                                                                                                                                                imageio
                                                                                                                                                                                                                                                                                                imageio-ffmpeg
                                                                                                                                                                                                                                                                                                REQ

                                                                                                                                                                                                                                                                                                cat > README.md <<'MD'
                                                                                                                                                                                                                                                                                                ---
                                                                                                                                                                                                                                                                                                title: Wan AI Video Studio
                                                                                                                                                                                                                                                                                                emoji: 🎬
                                                                                                                                                                                                                                                                                                colorFrom: purple
                                                                                                                                                                                                                                                                                                colorTo: blue
                                                                                                                                                                                                                                                                                                sdk: gradio
                                                                                                                                                                                                                                                                                                app_file: app.py
                                                                                                                                                                                                                                                                                                pinned: false
                                                                                                                                                                                                                                                                                                ---

                                                                                                                                                                                                                                                                                                # Wan AI Video Studio

                                                                                                                                                                                                                                                                                                A text-to-video interface for the Wan2.2 T2V A14B Diffusers model.

                                                                                                                                                                                                                                                                                                ## Hardware

                                                                                                                                                                                                                                                                                                Use a compatible GPU environment with sufficient VRAM and storage.
                                                                                                                                                                                                                                                                                                The full model is large and is not intended for CPU-only execution.

                                                                                                                                                                                                                                                                                                ## Run locally

                                                                                                                                                                                                                                                                                                ```bash
                                                                                                                                                                                                                                                                                                pip install -r requirements.txt
                                                                                                                                                                                                                                                                                                python app.py
                                                                                                                                                                                                                                                                                                ```

                                                                                                                                                                                                                                                                                                Model:
                                                                                                                                                                                                                                                                                                https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers
                                                                                                                                                                                                                                                                                                MD

                                                                                                                                                                                                                                                                                                git add app.py requirements.txt README.md
                                                                                                                                                                                                                                                                                                git commit -m "Create Wan2.2 video generation demo"
                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                        )
                                    )