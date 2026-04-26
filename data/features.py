"""
Feature extraction for LoopMind full model.

Query side  (MIDI melody) → piano roll sequence  → [T_SEQ, 128]
Audio side  (all stems)   → log-mel spectrogram  → [T_SEQ, N_MELS]

Both outputs share the same time dimension T_SEQ so the CNN+Transformer
encoder can treat them uniformly.

Legacy mean+std functions (midi_to_chroma, audio_to_chroma, audio_to_mel)
are kept for the Gradio demo, which uses the older MLP model.

Run location : browser Colab
Output stored : Google Drive (cache_dir)
"""
import os
import hashlib
import numpy as np
import librosa
import pretty_midi

# ── Shared constants ──────────────────────────────────────────────────────────

SR             = 22050
HOP_LENGTH     = 512       # audio frames per mel hop
N_MELS         = 128       # mel bins == piano roll pitch bins → uniform [T, 128]
T_SEQ          = 256       # fixed time steps for both query and audio
PIANO_ROLL_FPS = 50        # frames/sec for piano roll (256 frames ≈ 5.1 s)


# ── Full-model features (time-series) ─────────────────────────────────────────

def midi_to_pianoroll(midi_path: str) -> np.ndarray:
    """
    MIDI file → piano roll sequence [T_SEQ, 128], float32 in [0, 1].

    Uses pretty_midi get_piano_roll at PIANO_ROLL_FPS, then crops/pads to T_SEQ.
    Velocity values are normalised to [0, 1].
    """
    pm   = pretty_midi.PrettyMIDI(midi_path)
    roll = pm.get_piano_roll(fs=PIANO_ROLL_FPS)   # [128, T_raw]
    roll = roll.T                                  # [T_raw, 128]
    roll = _trim_or_pad(roll, T_SEQ)               # [T_SEQ, 128]
    roll = roll / (roll.max() + 1e-8)             # normalise to [0, 1]
    return roll.astype(np.float32)


def audio_to_mel_seq(audio_path: str) -> np.ndarray:
    """
    Audio file → log-mel spectrogram sequence [T_SEQ, N_MELS], float32.

    Mel bins (N_MELS=128) match the 128 piano roll pitches so both modalities
    share the same feature width going into the encoder.
    """
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    mel  = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH
    )                                              # [N_MELS, T_raw]
    log_mel = librosa.power_to_db(mel, ref=np.max) # dB scale
    log_mel = log_mel.T                            # [T_raw, N_MELS]
    log_mel = _trim_or_pad(log_mel, T_SEQ)         # [T_SEQ, N_MELS]
    # Standardise to zero mean / unit std per sample
    mu  = log_mel.mean()
    std = log_mel.std() + 1e-8
    log_mel = (log_mel - mu) / std
    return log_mel.astype(np.float32)


def _trim_or_pad(arr: np.ndarray, length: int) -> np.ndarray:
    """Trim or zero-pad a [T, F] array along the time axis to exactly `length`."""
    T = arr.shape[0]
    if T >= length:
        return arr[:length]
    pad = np.zeros((length - T, arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


# ── Batch extraction with caching ─────────────────────────────────────────────

def extract_and_cache(records: list, cache_dir: str) -> dict:
    """
    Pre-compute and disk-cache all time-series features for all records.

    Returns a nested lookup dict:
        features[track]["melody"]        = np.ndarray [T_SEQ, 128]
        features[track][cat][audio_path] = np.ndarray [T_SEQ, N_MELS]
    """
    os.makedirs(cache_dir, exist_ok=True)
    features = {}
    total = len(records)

    for idx, rec in enumerate(records):
        track = rec["track"]
        features[track] = {
            "melody": None,
            "drums": {}, "bass": {}, "piano": {}, "guitar": {},
        }

        # Melody: MIDI → piano roll
        midi_path = rec["melody_midi"]
        ck = _cache_path(cache_dir, midi_path, "pianoroll")
        if os.path.exists(ck):
            feat = np.load(ck)
        else:
            feat = midi_to_pianoroll(midi_path)
            np.save(ck, feat)
        features[track]["melody"] = feat

        # Accompaniment: audio → mel seq
        for cat in ["drums", "bass", "piano", "guitar"]:
            for audio_path in rec[cat]:
                ck = _cache_path(cache_dir, audio_path, cat)
                if os.path.exists(ck):
                    feat = np.load(ck)
                else:
                    try:
                        feat = audio_to_mel_seq(audio_path)
                    except Exception as e:
                        print(f"  [WARN] {audio_path}: {e}")
                        continue
                    np.save(ck, feat)
                features[track][cat][audio_path] = feat

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"  [{idx+1}/{total}] {track} done")

    return features


def _cache_path(cache_dir: str, source_path: str, tag: str) -> str:
    """Deterministic .npy path: hash the source path to avoid filesystem length limits."""
    key = hashlib.md5(source_path.encode()).hexdigest()
    return os.path.join(cache_dir, f"{tag}_{key}.npy")


# ── Legacy mean+std features (Gradio demo only) ───────────────────────────────

_LEGACY_SR         = 16000
_LEGACY_HOP        = 256
_LEGACY_DURATION   = 4.0


def midi_to_chroma(midi_path: str) -> np.ndarray:
    """Legacy: MIDI → chroma mean+std [24-d]. Used by demo/app.py."""
    try:
        pm    = pretty_midi.PrettyMIDI(midi_path)
        audio = pm.fluidsynth(fs=_LEGACY_SR)
        return _legacy_audio_to_chroma(audio)
    except Exception:
        return _legacy_midi_pianoroll_chroma(midi_path)


def _legacy_audio_to_chroma(y: np.ndarray) -> np.ndarray:
    n = int(_LEGACY_SR * _LEGACY_DURATION)
    y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    chroma = librosa.feature.chroma_cqt(y=y, sr=_LEGACY_SR, hop_length=_LEGACY_HOP)
    return np.concatenate([chroma.mean(axis=1), chroma.std(axis=1)]).astype(np.float32)


def _legacy_midi_pianoroll_chroma(midi_path: str) -> np.ndarray:
    pm   = pretty_midi.PrettyMIDI(midi_path)
    roll = pm.get_piano_roll(fs=50)
    n    = int(_LEGACY_DURATION * 50)
    roll = roll[:, :n] if roll.shape[1] >= n else np.pad(roll, ((0,0),(0, n-roll.shape[1])))
    chroma = np.zeros((12, roll.shape[1]))
    for p in range(128):
        chroma[p % 12] += roll[p]
    chroma /= (chroma.max() + 1e-8)
    return np.concatenate([chroma.mean(axis=1), chroma.std(axis=1)]).astype(np.float32)
