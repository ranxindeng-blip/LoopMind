"""
Gradio demo — MIDI melody → compatible stems per category.

Run location : browser Colab (share=True for public URL)
"""
import os
import tempfile
import traceback
import numpy as np
import torch
import pretty_midi
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from pydub import AudioSegment

from data.features import midi_to_chroma
from models.dual_encoder import DualEncoder, CATEGORIES

CAT_EMOJI = {"drums": "🥁", "bass": "🎸", "piano": "🎹", "guitar": "🎛"}
CAT_LABEL = {"drums": "Drums", "bass": "Bass",
             "piano": "Piano / Chords", "guitar": "Guitar / Texture"}

PRESET_DIR = os.path.join(os.path.dirname(__file__), "presets")
PRESETS = {
    "🎵 Pop Major":        "pop_major.mid",
    "😢 Sad Minor":        "sad_minor.mid",
    "🕺 Funky Syncopated": "funky_syncopated.mid",
    "🌊 Pad / Long Note":  "pad_longnote.mid",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def piano_roll_image(midi_path: str) -> str:
    try:
        pm   = pretty_midi.PrettyMIDI(midi_path)
        roll = pm.get_piano_roll(fs=50)
        active = np.where(roll.any(axis=1))[0]
        if len(active):
            lo = max(0, active[0] - 3)
            hi = min(127, active[-1] + 3)
            roll = roll[lo:hi+1, :]
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.imshow(roll, aspect="auto", origin="lower",
                  cmap="magma", interpolation="nearest")
        ax.set_xlabel("Time (frames @ 50fps)")
        ax.set_ylabel("MIDI Pitch")
        ax.set_title("Melody — Piano Roll", fontsize=11, fontweight="bold")
        plt.tight_layout()
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return tmp.name
    except Exception:
        traceback.print_exc()
        return None


def mix_stems(paths: list) -> str:
    try:
        segments = [AudioSegment.from_file(p) for p in paths if p]
        if not segments:
            return None
        min_len = min(len(s) for s in segments)
        mixed   = segments[0][:min_len]
        for s in segments[1:]:
            mixed = mixed.overlay(s[:min_len])
        mixed = mixed.normalize()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        mixed.export(tmp.name, format="wav")
        return tmp.name
    except Exception:
        traceback.print_exc()
        return None


def retrieve_all(midi_path: str, model, library: dict, device) -> dict:
    try:
        feat = midi_to_chroma(midi_path)
        x    = torch.from_numpy(feat).unsqueeze(0).to(device)
        with torch.no_grad():
            embeddings = model.encode_query(x)

        results = {}
        for cat in CATEGORIES:
            emb   = embeddings[cat].squeeze(0).cpu().numpy()
            embs  = library[cat]["embeddings"]
            paths = library[cat]["paths"]
            tracks= library[cat]["tracks"]
            q_norm = emb / (np.linalg.norm(emb) + 1e-8)
            e_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
            scores = e_norm @ q_norm
            top3   = np.argsort(scores)[::-1][:3]
            results[cat] = [(paths[i], float(scores[i]), tracks[i]) for i in top3]
        return results
    except Exception:
        traceback.print_exc()
        return {}


# ── Gradio app ────────────────────────────────────────────────────────────────

def launch(model: DualEncoder, library: dict, device=None, share: bool = True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    def process(midi_path: str):
        if not midi_path or not os.path.exists(midi_path):
            return [None] * (1 + 4*6 + 1)

        roll  = piano_roll_image(midi_path)
        res   = retrieve_all(midi_path, model, library, device)
        outs  = [roll]
        top1_paths = []

        for cat in CATEGORIES:
            hits = res.get(cat, [])
            for i in range(3):
                if i < len(hits):
                    path, score, track = hits[i]
                    outs += [path, f"{CAT_EMOJI[cat]} {track} | score: {score:.3f}"]
                    if i == 0:
                        top1_paths.append(path)
                else:
                    outs += [None, ""]

        outs.append(mix_stems(top1_paths))
        return outs

    # All outputs: roll_img + (audio + label) × 3 × 4 cats + mix
    # Total = 1 + 24 + 1 = 26
    with gr.Blocks(title="LoopMind") as demo:
        gr.Markdown("""
# 🎵 LoopMind — MIDI Melody to Arrangement
**Select a preset melody or upload your own MIDI.
LoopMind retrieves compatible drums, bass, piano, and guitar stems.**
---""")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎹 Input Melody")
                gr.Markdown("**Preset Melodies:**")
                preset_btns = [gr.Button(name, size="sm") for name in PRESETS]
                upload = gr.File(label="Upload your own MIDI",
                                 file_types=[".mid", ".midi"])
                roll_img = gr.Image(label="Piano Roll", height=200)

            with gr.Column(scale=2):
                gr.Markdown("### 🎼 Retrieved Stems")
                audio_cols, label_cols = {}, {}
                for cat in CATEGORIES:
                    gr.Markdown(f"**{CAT_EMOJI[cat]} {CAT_LABEL[cat]}**")
                    audio_cols[cat], label_cols[cat] = [], []
                    with gr.Row():
                        for i in range(3):
                            with gr.Column(min_width=160):
                                lbl = gr.Textbox(label=f"#{i+1}",
                                                 interactive=False, lines=1)
                                aud = gr.Audio(type="filepath",
                                               show_label=False)
                                label_cols[cat].append(lbl)
                                audio_cols[cat].append(aud)

        gr.Markdown("---")
        gr.Markdown("### 🎧 Auto Mix  *(top-1 per category)*")
        mix_out = gr.Audio(label="Mixed Arrangement", type="filepath")

        all_outputs = [roll_img]
        for cat in CATEGORIES:
            for i in range(3):
                all_outputs += [audio_cols[cat][i], label_cols[cat][i]]
        all_outputs.append(mix_out)

        # Wire preset buttons
        for btn, preset_name in zip(preset_btns, PRESETS.keys()):
            midi_path = os.path.join(PRESET_DIR, PRESETS[preset_name])
            btn.click(fn=lambda p=midi_path: process(p), outputs=all_outputs)

        # Wire upload
        upload.change(fn=lambda f: process(f.name if f else None),
                      inputs=upload, outputs=all_outputs)

    demo.launch(share=share)
