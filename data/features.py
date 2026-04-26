"""
Feature extraction for demo version.

Query side  (MIDI melody)  → chroma mean+std → 24-d vector
Audio side  (drums)        → mel   mean+std  → 256-d vector
Audio side  (bass/piano/guitar) → chroma mean+std → 24-d vector

Run location : local Mac
Output stored : local (data/cache/*.npy)
"""
import os
import numpy as np
import librosa
import pretty_midi

SR         = 16000
DURATION   = 4.0        # seconds per window
HOP_LENGTH = 256
N_MELS     = 128
N_CHROMA   = 12


# ── MIDI → chroma ─────────────────────────────────────────────────────────────

def midi_to_chroma(midi_path: str, duration: float = DURATION) -> np.ndarray:
    """
    MIDI file → chroma mean+std [24-d].
    Synthesize the MIDI to audio first, then compute chroma.
    Uses pretty_midi's built-in fluidsynth synthesis.
    Falls back to piano-roll chroma if synthesis fails.
    """
    try:
        pm    = pretty_midi.PrettyMIDI(midi_path)
        audio = pm.fluidsynth(fs=SR)
        # Trim or pad to fixed duration
        n = int(SR * duration)
        if len(audio) > n:
            audio = audio[:n]
        elif len(audio) < n:
            audio = np.pad(audio, (0, n - len(audio)))
        return _audio_to_chroma(audio)
    except Exception:
        # Fallback: piano-roll based chroma
        return _midi_pianoroll_chroma(midi_path, duration)


def _midi_pianoroll_chroma(midi_path: str, duration: float) -> np.ndarray:
    """Piano-roll folded to 12 chroma bins, mean+std."""
    pm      = pretty_midi.PrettyMIDI(midi_path)
    fs      = 50  # frames per second
    n_frames = int(duration * fs)
    roll    = pm.get_piano_roll(fs=fs)            # [128, T]
    roll    = roll[:, :n_frames]
    if roll.shape[1] < n_frames:
        roll = np.pad(roll, ((0,0),(0, n_frames - roll.shape[1])))
    # Fold to 12 chroma bins
    chroma = np.zeros((12, roll.shape[1]))
    for pitch in range(128):
        chroma[pitch % 12] += roll[pitch]
    chroma = chroma / (chroma.max() + 1e-8)
    mean = chroma.mean(axis=1)                    # [12]
    std  = chroma.std(axis=1)                     # [12]
    return np.concatenate([mean, std]).astype(np.float32)   # [24]


# ── Audio → chroma ────────────────────────────────────────────────────────────

def audio_to_chroma(audio_path: str, duration: float = DURATION) -> np.ndarray:
    """Audio file → chroma mean+std [24-d]. For bass, piano, guitar."""
    y = _load_audio(audio_path, duration)
    return _audio_to_chroma(y)


def _audio_to_chroma(y: np.ndarray) -> np.ndarray:
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP_LENGTH)
    mean   = chroma.mean(axis=1)    # [12]
    std    = chroma.std(axis=1)     # [12]
    return np.concatenate([mean, std]).astype(np.float32)   # [24]


# ── Audio → mel (drums) ───────────────────────────────────────────────────────

def audio_to_mel(audio_path: str, duration: float = DURATION) -> np.ndarray:
    """Audio file → mel mean+std [256-d]. For drums."""
    y   = _load_audio(audio_path, duration)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    mean    = log_mel.mean(axis=1)   # [128]
    std     = log_mel.std(axis=1)    # [128]
    return np.concatenate([mean, std]).astype(np.float32)   # [256]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_audio(path: str, duration: float) -> np.ndarray:
    n = int(SR * duration)
    y, _ = librosa.load(path, sr=SR, mono=True, duration=duration)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y


def get_feature_dim(category: str) -> int:
    """Return feature vector dimension for a given category."""
    return 256 if category == "drums" else 24


# ── Batch extraction with caching ─────────────────────────────────────────────

def extract_and_cache(records: list, cache_dir: str) -> dict:
    """
    Pre-compute and cache all features for all records.
    Returns a lookup dict:
        features[track][category][path] = np.ndarray
    """
    os.makedirs(cache_dir, exist_ok=True)
    features = {}

    for rec in records:
        track = rec["track"]
        features[track] = {"melody": None, "drums": {}, "bass": {}, "piano": {}, "guitar": {}}

        # Melody MIDI
        midi_path = rec["melody_midi"]
        cache_key = _cache_key(cache_dir, midi_path, "midi_chroma")
        if os.path.exists(cache_key):
            feat = np.load(cache_key)
        else:
            feat = midi_to_chroma(midi_path)
            np.save(cache_key, feat)
        features[track]["melody"] = feat

        # Accompaniment audio
        for cat in ["drums", "bass", "piano", "guitar"]:
            extractor = audio_to_mel if cat == "drums" else audio_to_chroma
            for audio_path in rec[cat]:
                ck = _cache_key(cache_dir, audio_path, cat)
                if os.path.exists(ck):
                    feat = np.load(ck)
                else:
                    feat = extractor(audio_path)
                    np.save(ck, feat)
                features[track][cat][audio_path] = feat

        print(f"  [{track}] features extracted")

    return features


def _cache_key(cache_dir: str, path: str, suffix: str) -> str:
    name = path.replace("/", "_").replace(".", "_")
    return os.path.join(cache_dir, f"{name}_{suffix}.npy")
