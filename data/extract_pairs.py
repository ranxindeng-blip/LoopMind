"""
Scan Slakh2100 (train + validation splits), identify the melody MIDI stem and
accompaniment audio stems per track, and write pairs.json to the cache dir.

Run location : browser Colab
Output stored : Google Drive (cache_dir)
"""
import os
import json
import yaml
import pretty_midi

# ── Constants ─────────────────────────────────────────────────────────────────

# Melody candidates must have avg pitch >= C3 to exclude bass / low string pads
MIN_MELODY_PITCH = 48   # MIDI note 48 = C3

BASS_KEYWORDS   = ["bass"]
PIANO_KEYWORDS  = ["piano", "keyboard", "organ", "harpsichord", "clavinet"]
GUITAR_KEYWORDS = ["guitar", "banjo", "mandolin", "ukulele"]
MELODY_KEYWORDS = [
    "piano", "guitar", "violin", "viola", "cello", "flute", "trumpet",
    "saxophone", "clarinet", "oboe", "lead", "synth", "organ",
    "strings", "brass", "mallet", "xylophone", "marimba",
]

# Slakh2100 dataset splits to scan (test split is held out for evaluation only)
SLAKH_SPLITS = ["train", "validation"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def classify_stem(inst_class: str, is_drum: bool) -> str:
    """Return accompaniment category string, or None if stem is unclassified."""
    inst_lower = inst_class.lower()
    if is_drum:
        return "drums"
    if any(k in inst_lower for k in BASS_KEYWORDS):
        return "bass"
    if any(k in inst_lower for k in PIANO_KEYWORDS):
        return "piano"
    if any(k in inst_lower for k in GUITAR_KEYWORDS):
        return "guitar"
    return None


def is_melody_candidate(inst_class: str, is_drum: bool) -> bool:
    """Return True if this stem is eligible to be the melody track."""
    if is_drum:
        return False
    inst_lower = inst_class.lower()
    if any(k in inst_lower for k in BASS_KEYWORDS):
        return False
    return any(k in inst_lower for k in MELODY_KEYWORDS)


def avg_pitch(midi_path: str) -> float:
    """Mean MIDI pitch of all notes in a file. Returns 0.0 on failure."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        pitches = [n.pitch for inst in pm.instruments for n in inst.notes]
        return float(sum(pitches) / len(pitches)) if pitches else 0.0
    except Exception:
        return 0.0


def get_bpm(midi_path: str) -> float:
    """Return the first tempo (BPM) found in a MIDI file, default 120."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        _, tempi = pm.get_tempo_changes()   # returns (times, tempi_in_bpm)
        return float(tempi[0]) if len(tempi) > 0 else 120.0
    except Exception:
        return 120.0


def _find_audio(stems_dir: str, stem_id: str) -> str:
    """Return the audio path for a stem, checking .wav then .flac (Slakh2100 redux)."""
    for ext in (".wav", ".flac"):
        p = os.path.join(stems_dir, f"{stem_id}{ext}")
        if os.path.exists(p):
            return p
    return os.path.join(stems_dir, f"{stem_id}.wav")   # fallback (will fail exists check)


def _collect_tracks(data_root: str) -> list[tuple[str, str]]:
    """
    Return a sorted list of (track_name, track_dir) pairs from all SLAKH_SPLITS.
    Tracks from 'train' come before 'validation'.
    """
    result = []
    for split in SLAKH_SPLITS:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        for name in sorted(os.listdir(split_dir)):
            full = os.path.join(split_dir, name)
            if os.path.isdir(full):
                result.append((name, full))
    return result


# ── Main entry point ──────────────────────────────────────────────────────────

def extract_pairs(data_root: str, cache_dir: str,
                  max_tracks: int = 400) -> list:
    """
    For each track (up to max_tracks):
      - Pick the melody MIDI stem:
          * must be a melody-keyword instrument
          * avg pitch >= MIN_MELODY_PITCH (filters out low string pads / bass doubles)
          * highest avg pitch among remaining candidates
      - Collect accompaniment audio stems per category (drums/bass/piano/guitar)
      - Extract track BPM from the melody MIDI

    Each record:
    {
        "track":        "Track00001",
        "split":        "train",
        "melody_midi":  "/path/to/Sxx.mid",
        "bpm":          120.0,
        "drums":        ["/path/to/Syy.wav", ...],
        "bass":         [...],
        "piano":        [...],
        "guitar":       [...]
    }
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "pairs.json")

    if os.path.exists(cache_path):
        print(f"Loading cached pairs from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    all_tracks = _collect_tracks(data_root)
    if max_tracks and max_tracks < len(all_tracks):
        all_tracks = all_tracks[:max_tracks]
        print(f"Using subset: {max_tracks} of {len(all_tracks) + max_tracks} total tracks")

    records = []
    skipped = 0

    for track_name, track_dir in all_tracks:
        stems_dir = os.path.join(track_dir, "stems")
        midi_dir  = os.path.join(track_dir, "MIDI")
        meta_path = os.path.join(track_dir, "metadata.yaml")
        split     = os.path.basename(os.path.dirname(track_dir))

        if not all(os.path.exists(p) for p in [stems_dir, midi_dir, meta_path]):
            skipped += 1
            continue

        with open(meta_path) as f:
            meta = yaml.safe_load(f)

        melody_candidates = []   # (avg_pitch, stem_id, midi_path)
        accompaniment = {"drums": [], "bass": [], "piano": [], "guitar": []}

        for stem_id, info in meta.get("stems", {}).items():
            inst_class = info.get("inst_class", "")
            is_drum    = info.get("is_drum", False)

            audio_path = _find_audio(stems_dir, stem_id)
            midi_path  = os.path.join(midi_dir,  f"{stem_id}.mid")

            # Accompaniment side
            cat = classify_stem(inst_class, is_drum)
            if cat and os.path.exists(audio_path):
                accompaniment[cat].append(audio_path)

            # Melody candidate side
            if is_melody_candidate(inst_class, is_drum) and os.path.exists(midi_path):
                ap = avg_pitch(midi_path)
                if ap >= MIN_MELODY_PITCH:   # exclude low-register pads
                    melody_candidates.append((ap, stem_id, midi_path))

        if not melody_candidates:
            skipped += 1
            continue

        # Highest average pitch among eligible candidates
        melody_candidates.sort(key=lambda x: x[0], reverse=True)
        _, _, melody_midi = melody_candidates[0]
        bpm = get_bpm(melody_midi)

        record = {
            "track":       track_name,
            "split":       split,
            "melody_midi": melody_midi,
            "bpm":         bpm,
            "drums":       accompaniment["drums"],
            "bass":        accompaniment["bass"],
            "piano":       accompaniment["piano"],
            "guitar":      accompaniment["guitar"],
        }
        records.append(record)
        cats_found = [c for c in ["drums", "bass", "piano", "guitar"] if accompaniment[c]]
        print(f"  [{track_name}] bpm={bpm:.0f} melody={os.path.basename(melody_midi)} | {cats_found}")

    print(f"\nTotal records: {len(records)}  |  skipped: {skipped}")
    with open(cache_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved to {cache_path}")
    return records
