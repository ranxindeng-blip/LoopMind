"""Run once locally to generate 4 preset MIDI melody files."""
import os
import pretty_midi

OUT = os.path.join(os.path.dirname(__file__), "demo", "presets")
os.makedirs(OUT, exist_ok=True)

def make_midi(notes, bpm=100, program=0) -> pretty_midi.PrettyMIDI:
    pm   = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=program)
    for pitch, start, end in notes:
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=pitch,
                                           start=start, end=end))
    pm.instruments.append(inst)
    return pm

beat = 60 / 100  # one beat at 100 BPM

# 1. Pop Major — C major, simple, upbeat (C D E G A)
pop = [
    (60, 0*beat, 0.8*beat), (62, 1*beat, 1.8*beat),
    (64, 2*beat, 2.8*beat), (67, 3*beat, 3.8*beat),
    (69, 4*beat, 4.8*beat), (67, 5*beat, 5.8*beat),
    (64, 6*beat, 6.8*beat), (60, 7*beat, 8.0*beat),
]
make_midi(pop, bpm=100).write(os.path.join(OUT, "pop_major.mid"))

# 2. Sad Minor — A natural minor, slow, descending
beat2 = 60 / 72
sad = [
    (69, 0*beat2, 1.8*beat2), (67, 2*beat2, 3.8*beat2),
    (65, 4*beat2, 5.8*beat2), (64, 6*beat2, 7.8*beat2),
    (62, 8*beat2, 9.8*beat2), (60, 10*beat2, 12.0*beat2),
]
make_midi(sad, bpm=72).write(os.path.join(OUT, "sad_minor.mid"))

# 3. Funky Syncopated — syncopated 16th-note feel, D minor pentatonic
beat3 = 60 / 105
funky = [
    (62, 0.0*beat3, 0.4*beat3), (65, 0.5*beat3, 0.9*beat3),
    (67, 0.75*beat3,1.1*beat3), (69, 1.0*beat3, 1.4*beat3),
    (67, 1.5*beat3, 1.9*beat3), (65, 2.0*beat3, 2.4*beat3),
    (62, 2.5*beat3, 2.9*beat3), (60, 3.0*beat3, 3.4*beat3),
    (62, 3.5*beat3, 3.9*beat3), (65, 4.0*beat3, 4.4*beat3),
    (67, 4.5*beat3, 5.9*beat3), (65, 6.0*beat3, 6.4*beat3),
    (62, 6.5*beat3, 7.9*beat3),
]
make_midi(funky, bpm=105).write(os.path.join(OUT, "funky_syncopated.mid"))

# 4. Pad / Long Note — slow, whole notes, C major (C E G B)
beat4 = 60 / 60
pad = [
    (60, 0*beat4, 1.9*beat4), (64, 2*beat4, 3.9*beat4),
    (67, 4*beat4, 5.9*beat4), (71, 6*beat4, 8.0*beat4),
]
make_midi(pad, bpm=60).write(os.path.join(OUT, "pad_longnote.mid"))

print("✅ 4 preset MIDI files created in demo/presets/:")
for f in os.listdir(OUT):
    print(f"   {f}")
