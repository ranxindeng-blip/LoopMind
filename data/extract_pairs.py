"""
Scan BabySlakh, identify melody MIDI stem + accompaniment audio stems per track.
Outputs a list of pair records saved as pairs.json.

Run location : local Mac
Output stored : local (data/cache/pairs.json)
"""
import os
import json
import yaml
import pretty_midi

# ── Category mapping ──────────────────────────────────────────────────────────

BASS_KEYWORDS    = ["bass"]
DRUM_FLAG        = "is_drum"
PIANO_KEYWORDS   = ["piano", "keyboard", "organ", "harpsichord", "clavinet"]
GUITAR_KEYWORDS  = ["guitar", "banjo", "mandolin", "ukulele"]

# Instruments eligible to be the melody stem
MELODY_KEYWORDS  = [
    "piano", "guitar", "violin", "viola", "cello", "flute", "trumpet",
    "saxophone", "clarinet", "oboe", "lead", "synth", "organ",
    "strings", "brass", "mallet", "xylophone", "marimba"
]


def classify_stem(inst_class: str, is_drum: bool) -> str:
    """Return category string or None if stem should be skipped."""
    inst_lower = inst_class.lower()
    if is_drum:
        return "drums"
    if any(k in inst_lower for k in BASS_KEYWORDS):
        return "bass"
    if any(k in inst_lower for k in PIANO_KEYWORDS):
        return "piano"
    if any(k in inst_lower for k in GUITAR_KEYWORDS):
        return "guitar"
    return None  # not assigned to any accompaniment category


def is_melody_candidate(inst_class: str, is_drum: bool) -> bool:
    if is_drum:
        return False
    inst_lower = inst_class.lower()
    if any(k in inst_lower for k in BASS_KEYWORDS):
        return False
    return any(k in inst_lower for k in MELODY_KEYWORDS)


def avg_pitch(midi_path: str) -> float:
    """Return mean pitch of all notes in a MIDI file. Used to select melody."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        pitches = [n.pitch for inst in pm.instruments for n in inst.notes]
        return sum(pitches) / len(pitches) if pitches else 0.0
    except Exception:
        return 0.0


def extract_pairs(babyslakh_root: str, cache_dir: str) -> list:
    """
    For each track:
      - Pick the melody MIDI stem (highest avg pitch among melody candidates)
      - Collect accompaniment audio stems per category
      - Return list of pair records

    Each record:
    {
        "track":        "Track00001",
        "melody_midi":  "/path/to/Sxx.mid",
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

    tracks = sorted([
        d for d in os.listdir(babyslakh_root)
        if os.path.isdir(os.path.join(babyslakh_root, d))
    ])

    records = []
    for track in tracks:
        track_dir   = os.path.join(babyslakh_root, track)
        stems_dir   = os.path.join(track_dir, "stems")
        midi_dir    = os.path.join(track_dir, "MIDI")
        meta_path   = os.path.join(track_dir, "metadata.yaml")

        if not all(os.path.exists(p) for p in [stems_dir, midi_dir, meta_path]):
            continue

        with open(meta_path) as f:
            meta = yaml.safe_load(f)

        melody_candidates = []   # (avg_pitch, stem_id, midi_path)
        accompaniment     = {"drums": [], "bass": [], "piano": [], "guitar": []}

        for stem_id, info in meta.get("stems", {}).items():
            inst_class = info.get("inst_class", "")
            is_drum    = info.get("is_drum", False)

            audio_path = os.path.join(stems_dir, f"{stem_id}.wav")
            midi_path  = os.path.join(midi_dir,  f"{stem_id}.mid")

            # Accompaniment side
            cat = classify_stem(inst_class, is_drum)
            if cat and os.path.exists(audio_path):
                accompaniment[cat].append(audio_path)

            # Melody candidate side
            if is_melody_candidate(inst_class, is_drum) and os.path.exists(midi_path):
                ap = avg_pitch(midi_path)
                melody_candidates.append((ap, stem_id, midi_path))

        if not melody_candidates:
            print(f"  [{track}] No melody candidate found, skipping")
            continue

        # Pick melody stem with highest average pitch
        melody_candidates.sort(key=lambda x: x[0], reverse=True)
        _, _, melody_midi = melody_candidates[0]

        record = {
            "track":       track,
            "melody_midi": melody_midi,
            "drums":       accompaniment["drums"],
            "bass":        accompaniment["bass"],
            "piano":       accompaniment["piano"],
            "guitar":      accompaniment["guitar"],
        }
        records.append(record)
        cats_found = [c for c in ["drums","bass","piano","guitar"] if accompaniment[c]]
        print(f"  [{track}] melody={os.path.basename(melody_midi)} | {cats_found}")

    print(f"\nTotal tracks with melody: {len(records)}")
    with open(cache_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved to {cache_path}")
    return records
