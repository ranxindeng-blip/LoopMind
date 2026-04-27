# LoopMind — Project Presentation Script

---

## 1. What Is This Project?

LoopMind solves a practical problem in music production:

> I have a melody. How do I quickly find compatible drums, bass, piano, and guitar to go with it?

The traditional approach is to manually browse a sample library and audition tracks one by one — time-consuming and driven by guesswork. We use deep learning to automate this: **given a MIDI melody, the system retrieves audio stems that are musically compatible**.

The technical name for this task is **cross-modal retrieval**: the input is MIDI (symbolic music notation) and the output is audio (actual sound waveforms). These are two completely different signal types, and we train a model to understand the relationship between them.

---

## 2. Why Does This Matter?

Music production has a well-known bottleneck: melodic ideas come easily, but turning them into a full arrangement is hard. Existing tools either require manual searching or produce random accompaniments with no understanding of musical style.

Our system performs **content-based matching** — not keyword search, but genuine understanding of tonality, rhythm, and mood, followed by retrieval of accompaniment that fits. This has real value for independent musicians, game audio designers, and film composers.

---

## 3. Dataset: Slakh2100

We use **Slakh2100**, a dataset specifically designed for music machine learning research, containing approximately 2,100 songs.

Each track in the dataset is structured as follows:

```
Track00001/
├── MIDI/          ← Per-instrument MIDI files (piano, guitar, drums, ...)
├── stems/         ← Corresponding audio files (.flac), one per instrument
└── metadata.yaml  ← Instrument label for each stem
```

The key property: **all stems within a track were recorded together as real multi-track music**, so they are naturally style-consistent and rhythmically aligned. This is exactly what we need for training.

---

## 4. How Are Labels Created?

You might wonder: does this require manual annotation?

**No. The labels are implicit in the dataset structure.**

This approach is called **self-supervised contrastive learning**. The logic is simple:

- Melody from Track A + drums from Track A → **positive pair** (they naturally belong together)
- Melody from Track A + drums from Track B → **negative pair** (they are from different songs)

The model's training objective is to pull positive pairs close together in vector space and push negative pairs apart. No human annotation is needed — the dataset structure itself provides the supervision signal.

The loss function used is **InfoNCE (Noise-Contrastive Estimation)**, the standard loss for contrastive learning.

---

## 5. Features

We have two input modalities, each represented differently.

### Query Side: MIDI → Piano Roll

A MIDI file records "which key was pressed at what time and for how long." We convert this into a 2D matrix called a **Piano Roll**:
- **X-axis**: time (one frame per 1/50 second)
- **Y-axis**: 128 MIDI pitch values
- **Value**: whether that pitch is active at that frame (0 or 1)
- **Output shape**: `[256 frames, 128 pitches]`

### Audio Side: Stem → Log-Mel Spectrogram

An audio file is a continuous sound wave. We convert it into a **Mel Spectrogram**, a 2D time-frequency representation:
- **X-axis**: time frames
- **Y-axis**: 128 Mel frequency bins (approximating human auditory perception)
- **Value**: energy intensity at each time-frequency cell (in dB)
- **Output shape**: `[256 frames, 128 frequency bins]`

**Why this design?** Both modalities produce the same shape `[256, 128]`, so a single network architecture can process both.

---

## 6. Model: Dual Encoder with CNN + Transformer

The model is called a **Dual Encoder**. Two encoders, one per modality:

```
MIDI Piano Roll [256, 128]  →  QueryEncoder  →  embedding [128-d]
Audio Mel Spec  [256, 128]  →  AudioEncoder  →  embedding [128-d]
```

Each encoder uses the same architectural pattern: **CNN → Transformer → Global Average Pooling → Linear Head**.

### CNN: Capturing Local Patterns

We treat the input like an image and apply 1D convolutions along the time axis. This captures local patterns: a syncopated rhythm here, an ascending arpeggio there, a drum fill at a particular moment.

### Transformer: Understanding Global Context

After CNN processing, the Transformer attends across all time steps simultaneously, building a global understanding of the sequence: overall key, emotional arc, rhythmic density. This is where the architecture benefits from sequence-level context that CNNs alone cannot capture.

### Global Average Pooling: Compressing to a Vector

The Transformer outputs a sequence of vectors, one per time frame. We average across all frames to produce a single fixed-length vector — a semantic "fingerprint" of the audio or MIDI clip.

### 4 Category-Specific Projection Heads

The final layer consists of 4 separate linear projections, one per accompaniment category (drums / bass / piano / guitar). This maps the shared representation into category-specific embedding spaces, allowing the model to specialize its understanding per instrument role.

**QueryEncoder**: one shared CNN+Transformer backbone, four projection heads.  
**AudioEncoder**: four independent CNN+Transformer backbones, one per category (drums, bass, piano, guitar have very different audio characteristics).

---

## 7. Training Pipeline

```
Each batch contains (melody, stem, category) triples

For each active category in the batch:
    zq = QueryEncoder(melody_piano_roll)    # [N, 128]
    za = AudioEncoder(stem_mel_spec)        # [N, 128]

InfoNCE Loss per category:
    Diagonal of similarity matrix = positive pairs (same track)
    Off-diagonal = negative pairs (different tracks)

Total loss = sum of losses across all 4 categories
One backward pass, one optimizer step
```

After training, the model has learned a shared embedding space where musically compatible melody–stem pairs cluster together.

---

## 8. Inference Pipeline (Retrieval)

```
1. User inputs a MIDI melody file

2. Feature extraction:
   Piano Roll → [256, 128]

3. Query encoding:
   QueryEncoder → melody embedding [128-d]

4. Search the pre-built stem library:
   (Library = all audio stems pre-encoded and stored as embeddings)
   Cosine similarity between melody embedding and all stem embeddings
   → Return Top-3 results per category (drums / bass / piano / guitar)

5. User selects stems → system mixes them:
   BPM alignment (time-stretch to match query tempo)
   Onset alignment (trim to first detected beat)
   → Final mixed arrangement output
```

---

## 9. Evaluation Metrics

We evaluate using **Recall@K (R@K)**:

| Metric | Definition |
|--------|-----------|
| R@1 | Probability that the #1 retrieved result is the correct match |
| R@5 | Probability that the correct match appears in the top 5 results |
| R@10 | Probability that the correct match appears in the top 10 results |

**Baseline**: random retrieval = 1/N (where N is the library size).  
**Goal**: significantly outperform random on all categories.

---

## 10. Summary

> LoopMind trains a dual encoder with contrastive learning to project MIDI melodies and audio stems into a shared semantic embedding space, enabling cross-modal style-compatible retrieval — give it a melody, and it finds matching drums, bass, piano, and guitar.

---

## Key Design Decisions (for Q&A)

| Decision | Reason |
|----------|--------|
| InfoNCE loss | Standard for cross-modal contrastive learning; efficient with in-batch negatives |
| Shared input shape `[256, 128]` | Enables the same architecture for both modalities |
| CNN before Transformer | CNN captures local temporal patterns efficiently; Transformer then models global dependencies |
| 4 independent AudioEncoders | Drums, bass, piano, guitar have fundamentally different spectral characteristics |
| Self-supervised labels | No annotation cost; dataset structure provides natural positive pairs |
| Subset of 400 tracks | Balances training time and Drive storage constraints for this prototype |
