import gradio as gr
import torch

from audio_preprocessing import process_and_classify
from html import escape

# text_parts = [
#     "Sometimes the right thing to do is to throw out the old schools of thought in the name of progress and reform.",
#     "Sometimes the right thing to do is to sit and listen to the wisdom of those who have come before us.",
#     "How will you know what the right choice is in these crucial moments?",
#     "You won't.",
#     "How do I give advice to this many people about their life choices?",
#     "I won't.",
#     "The scary news is you're on your own now.",
#     "But the cool news is you're on your own now."
# ]
#
# emotions_1 = ["neutral", "neutral", "neutral", "neutral", "neutral", "neutral", "neutral", "neutral"]
# emotions_2 = ["neutral", "neutral", "neutral", "neutral", "neutral", "neutral", "fear", "happy"]


def visualize_emotions(audio_path):
    texts, audio_preds, text_preds = process_and_classify(audio_path)
    audio_labels = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprised"]
    # audio_labels = ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "surprised"]
    text_labels = ["neutral", "happy", "anger", "sad", "surprise", "fear", "disgust"]

    html = ""
    for txt, a, t in zip(texts, audio_preds, text_preds):
        if isinstance(a, torch.Tensor):
            a = a.item()
        if isinstance(t, torch.Tensor):
            t = t.item()

        a_emo = audio_labels[a]
        t_emo = text_labels[t]

        if a_emo == t_emo:
            html += f"<p><b>{escape(a_emo)}</b><br>{escape(txt)}</p>"
        else:
            html += f"<p><b><s>{escape(a_emo)}</s> → {escape(t_emo)}</b><br>{escape(txt)}</p>"
    return html


def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 🎧 Speech Emotion Tune")

        audio_input = gr.Audio(type="filepath", label="Upload Audio")
        output = gr.HTML(label="Emotion-labeled Transcript")

        with gr.Row():
            submit_btn = gr.Button("Process and Show Results")
            clear_btn = gr.Button("Clear")

        submit_btn.click(fn=visualize_emotions, inputs=audio_input, outputs=output)

        clear_btn.click(
            fn=lambda: (None, ""),
            inputs=[],
            outputs=[audio_input, output]
        )

    return demo
