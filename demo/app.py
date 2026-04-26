"""
Gradio demo — MIDI melody → compatible stems per category.

Run location : browser Colab (for public share=True URL)
               OR local Mac (for local testing)
"""
import os
import tempfile
import numpy as np
import torch
import pretty_midi
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from pydub import AudioSegment

from data.features       import midi_to_chroma
from models.dual_encoder import DualEncoder, CATEGORIES
from build_library       import retrieve

CAT_EMOJI = {"drums": "🥁", "bass": "🎸", "piano": "🎹", "guitar": "🎛"}
CAT_LABEL = {"drums": "Drums", "bass": "Bass", "piano": "Piano / Chords", "guitar": "Guitar / Texture"}

PRESET_DIR = os.path.join(os.path.dirname(__file__), "presets")
PRESETS = {
    "🎵 Pop Major":        os.path.join(PRESET_DIR, "pop_major.mid"),
    "😢 Sad Minor":        os.path.join(PRESET_DIR, "sad_minor.mid"),
    "🕺 Funky Syncopated": os.path.join(PRESET_DIR, "funky_syncopated.mid"),
    "🌊 Pad / Long Note":  os.path.join(PRESET_DIR, "pad_longnote.mid"),
}


# ── Audio helpers ─────────────────────────────────────────────────────────────

def piano_roll_image(midi_path: str, title: str = "") -> str:
    pm = pretty_midi.PrettyMIDI(midi_path)
    roll = pm.get_piano_roll(fs=50)   # [128, T]
    # Crop to active pitch range
    active = np.where(roll.any(axis=1))[0]
    if len(active):
        lo, hi = max(0, active[0]-2), min(127, active[-1]+2)
        roll = roll[lo:hi+1, :]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(roll, aspect="auto", origin="lower",
              cmap="magma", interpolation="nearest")
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("MIDI Pitch")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


def mix_stems(paths: list) -> str:
    segments = []
    for p in paths:
        try:
            segments.append(AudioSegment.from_file(p))
        except Exception:
            continue
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


# ── Main demo builder ─────────────────────────────────────────────────────────

def build_demo(model: DualEncoder, library: dict, device):
    model.eval()

    def run_retrieval(midi_path: str) -> tuple:
        """Encode query MIDI and retrieve top-3 per category."""
        import traceback
        try:
            feat = midi_to_chroma(midi_path)
            x    = torch.from_numpy(feat).unsqueeze(0).to(device)
            with torch.no_grad():
                embeddings = model.encode_query(x)   # {cat: [1, D]}

            results = {}
            for cat in CATEGORIES:
                emb  = embeddings[cat].squeeze(0).cpu().numpy()
                hits = retrieve(emb, library, cat, top_k=3)
                results[cat] = hits   # [(path, score, track), ...]
            return results
        except Exception:
            traceback.print_exc()
            return {}

    def on_preset(preset_name: str):
        midi_path = PRESETS.get(preset_name)
        if not midi_path or not os.path.exists(midi_path):
            return [None] * 16
        return _build_outputs(midi_path)

    def on_upload(midi_file):
        if midi_file is None:
            return [None] * 16
        return _build_outputs(midi_file)

    def _build_outputs(midi_path: str):
        roll_img = piano_roll_image(midi_path, "Melody — Piano Roll")
        results  = run_retrieval(midi_path)

        outputs = [roll_img]
        all_audio_paths = []

        for cat in CATEGORIES:
            hits = results.get(cat, [])
            for i in range(3):
                if i < len(hits):
                    path, score, track = hits[i]
                    label = f"{CAT_EMOJI[cat]} {track}  •  score {score:.3f}"
                    outputs.append(path)
                    outputs.append(label)
                    all_audio_paths.append(path)
                else:
                    outputs.append(None)
                    outputs.append("")

        # Mix: one stem per category (top-1)
        mix_paths = []
        for cat in CATEGORIES:
            hits = results.get(cat, [])
            if hits:
                mix_paths.append(hits[0][0])
        outputs.append(mix_stems(mix_paths))
        return outputs

    # ── Gradio layout ──────────────────────────────────────────────────────────
    with gr.Blocks(title="LoopMind") as demo:
        gr.Markdown("""
# 🎵 LoopMind — MIDI Melody to Arrangement
### Select a preset melody or upload your own MIDI. Retrieve compatible stems per instrument.
---""")

        with gr.Row():
            # Left: input
            with gr.Column(scale=1):
                gr.Markdown("### Input Melody")
                with gr.Row():
                    for preset_name in PRESETS:
                        btn = gr.Button(preset_name, size="sm")

                upload = gr.File(label="Or upload your MIDI", file_types=[".mid", ".midi"])
                roll_img = gr.Image(label="Piano Roll", height=220)

            # Right: results
            with gr.Column(scale=2):
                gr.Markdown("### Retrieved Stems")
                audio_components = {}
                label_components = {}

                for cat in CATEGORIES:
                    gr.Markdown(f"**{CAT_EMOJI[cat]} {CAT_LABEL[cat]}**")
                    with gr.Row():
                        audio_components[cat] = []
                        label_components[cat] = []
                        for i in range(3):
                            with gr.Column(min_width=180):
                                lbl   = gr.Textbox(label=f"#{i+1}", interactive=False, lines=1)
                                audio = gr.Audio(type="filepath", show_label=False)
                                label_components[cat].append(lbl)
                                audio_components[cat].append(audio)

        gr.Markdown("---")
        gr.Markdown("### 🎧 Auto Mix (top-1 per category)")
        mix_out = gr.Audio(label="Mixed Arrangement", type="filepath")

        # Build flat output list:
        # [roll_img, drums_a0, drums_l0, drums_a1, drums_l1, drums_a2, drums_l2,
        #            bass_..., piano_..., guitar_..., mix]
        all_outputs = [roll_img]
        for cat in CATEGORIES:
            for i in range(3):
                all_outputs.append(audio_components[cat][i])
                all_outputs.append(label_components[cat][i])
        all_outputs.append(mix_out)

        # Wire preset buttons
        for preset_name in PRESETS:
            # Need closure to capture preset_name
            def make_handler(name):
                def handler():
                    return on_preset(name)
                return handler
            # find the button — re-create in order
        # Wire via separate loop with captured names
        with gr.Row(visible=False):
            preset_selector = gr.Dropdown(choices=list(PRESETS.keys()), label="preset")

        preset_selector.change(fn=on_preset, inputs=preset_selector, outputs=all_outputs)
        upload.change(fn=on_upload, inputs=upload, outputs=all_outputs)

        gr.Markdown("""
---
*Demo version: chroma+mel features, lightweight MLP dual encoder trained on BabySlakh.*
*Final version: CNN+Transformer on full Slakh2100.*""")

    return demo


def launch(model, library, device=None, share: bool = True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demo = build_demo(model, library, device)
    demo.launch(share=share)
