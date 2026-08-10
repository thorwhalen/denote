# Audio-to-Symbol: A Comprehensive Survey of Automatic Music Transcription and MIR Tools (2020–2025)

**The field of automatic music transcription has entered a transformer-dominated era where piano transcription exceeds 97% onset F1, yet multi-instrument and real-time transcription remain unsolved.** This survey catalogs every significant tool, library, API, and academic method that converts raw audio into symbolic musical representations — MIDI, chords, pitch contours, beats, keys, lyrics, structural labels, and more. The landscape reveals a fragmented ecosystem of 60+ tools across permissive and restrictive licenses, with no unified Python façade yet existing. For the architect building such a façade, the critical finding is that a permissively licensed core stack is achievable using librosa, Basic Pitch, Beat This!, torchcrepe, and Demucs, with GPL/AGPL tools relegated to optional backends. The following sections organize every tool by representation type, provide structured comparison tables, answer ten targeted research questions, and conclude with a recommended architecture for a Python wrapper library.

---

## 1. Polyphonic AMT and audio-to-MIDI transcription

### Background (pre-2022)

Modern neural AMT traces to Google Magenta's **Onsets and Frames** [1], a CNN+BiLSTM architecture that decoupled onset detection from frame-level pitch estimation, achieving **94.8% note onset F1** on MAESTRO. This dual-head paradigm became the foundation for all subsequent work. ByteDance's **High-Resolution Piano Transcription** by Kong et al. [2] advanced the state of the art to **96.72% onset F1** on MAESTRO by regressing continuous onset/offset times rather than frame-level binary predictions, and uniquely included pedal transcription (onset F1 = 91.86%). Google's **MT3** [3] extended seq2seq transcription to multiple instruments simultaneously using a T5-based encoder-decoder transformer, demonstrating strong results on Slakh2100, MusicNet, GuitarSet, and URMP — though it struggles with instrument switching mid-phrase and does not transcribe vocals.

### Recent advances (2022–2025)

The period from 2022 onward brought four critical developments. First, Sony's **hFT-Transformer** [4] (ISMIR 2023) introduced a hierarchical frequency-time transformer that achieved state-of-the-art results across all four standard piano metrics on both MAESTRO and MAPS, surpassing Kong et al. Second, **YourMT3+** [5] (2024) combined MT3 with a Perceiver-TF encoder, Mixture of Experts, and a multi-channel decoder, significantly outperforming MT3 on ten public datasets with under 2.5% parameter increase. Third, **DiffRoll** [6] from Sony AI introduced the first diffusion-based approach to AMT, generating piano rolls from Gaussian noise conditioned on spectrograms — outperforming its discriminative counterpart by 19 percentage points and enabling transcription, generation, and inpainting in a single framework. Fourth, **Mobile-AMT** [7] (EUSIPCO 2024) demonstrated real-time polyphonic piano transcription on mobile devices with **82.9% computational reduction** while maintaining competitive accuracy.

The most recent piano checkpoint from the **Aria-MIDI** project [8] (ICLR 2025) retrained Kong et al.'s architecture with updated data augmentation, reaching **97.63% note-level F1** on MAESTRO and **90.58% on MAPS** — the highest published numbers for piano transcription as of early 2026.

Spotify's **Basic Pitch** [9] occupies a unique niche: a lightweight, instrument-agnostic CNN (~17K parameters) that runs faster than real-time on CPU, supports pitch bends, and works in the browser via WASM. It sacrifices leaderboard accuracy for generalization and accessibility, making it the most practical tool for quick audio-to-MIDI conversion across instruments.

### Tool comparison table: AMT / MIDI

| Tool | Scope | Architecture | Onset F1 (MAESTRO) | License | Install | Real-time | Last Active |
|---|---|---|---|---|---|---|---|
| **Basic Pitch** | Instrument-agnostic | Lightweight CNN | Competitive | Apache 2.0 | `pip install basic-pitch` | Yes (CPU) | 2024 |
| **Kong et al.** | Piano + pedals | CRNN | 96.72% | MIT | `pip install piano_transcription_inference` | No | 2025 |
| **hFT-Transformer** | Piano | Hierarchical Transformer | SOTA (all 4 metrics) | MIT | git clone | No | 2023 |
| **MT3** | Multi-instrument | T5 seq2seq (JAX) | ~96% (piano) | Apache 2.0 | git clone + Colab | No | 2022 |
| **YourMT3+** | Multi-instrument | PerceiverTF + MoE | Exceeds MT3 | — | Research code | No | 2024 |
| **Omnizart** | Piano/drums/vocal/chord/beat | Multiple CNNs | — | MIT | `pip install omnizart` | No | 2022 |
| **Onsets and Frames** | Piano | CNN + BiLSTM | 94.80% | Apache 2.0 | `pip install magenta` | TFLite | 2021 |
| **DiffRoll** | Piano | Conditional diffusion | +19pp over baseline | — | Research code | No | 2024 |
| **Mobile-AMT** | Piano (mobile) | Efficient recurrent+conv | Competitive | — | Research code | **Yes** | 2024 |
| **Pop2Piano** | Piano cover generation | T5 (HuggingFace) | N/A (creative) | MIT | HuggingFace | No | 2023 |

**Current SOTA for polyphonic multi-instrument AMT** is YourMT3+ [5], which combines PerceiverTF's cross-attention encoder with MoE routing. Its key failure modes include instrument leakage (confusing violin with cello), inability to transcribe vocals, poor generalization to out-of-distribution recording conditions, and false-positive notes in dense polyphonic passages. For piano-only, the Aria-MIDI checkpoint [8] achieves 97.63% onset F1 on MAESTRO, though hFT-Transformer [4] leads on the full four-metric suite including offset and velocity.

**No open-source model produces MusicXML or quantized notation directly from audio.** The standard pipeline is Audio → MIDI (via any AMT model) → MusicXML (via music21 or MuseScore import). Commercial tools that perform this end-to-end include **Klangio** (REST API, exports MusicXML/PDF/Guitar Pro), **AnthemScore** ($39, desktop-only), and **Melody Scanner** (subscription SaaS).

---

## 2. Pitch estimation and melody extraction

### Background and SOTA landscape

Monophonic pitch estimation was transformed by **CREPE** [10] (ICASSP 2018), a 6-layer CNN with 22.2M parameters operating directly on time-domain waveforms. CREPE achieved **96.7% Raw Pitch Accuracy** (RPA, 50-cent threshold) on MDB-stem-synth and remains the most widely cited baseline. Its PyTorch port **torchcrepe** [11] adds Viterbi decoding (reducing octave errors), GPU-accelerated batch inference, and pitch embedding extraction — making it the preferred implementation for modern pipelines.

Two 2023 papers dramatically improved the speed-accuracy tradeoff. **PESTO** [12] (ISMIR 2023 Best Paper) demonstrated that self-supervised training with a transposition-equivariant objective achieves **96.1% RPA on MIR-1K** with fewer than **30,000 parameters** — 800× smaller than CREPE — running **12× faster than real-time on CPU**. **PENN** [13] (FCNF0++) improved CREPE's architecture with finer pitch quantization (3.125 vs. 20 cents) and cross-domain training on speech and music simultaneously, achieving **408× real-time on an RTX 3090 GPU** and **11.2× on CPU**.

The most recent entrant is **SwiftF0** [14] (2025), which achieves 90.2% harmonic-mean accuracy across eight diverse datasets with only 95.8K parameters, running 42× faster than CREPE on CPU and outperforming it by over 12 percentage points in noisy conditions. For singing voice specifically, **RMVPE** [15] (Interspeech 2023) uses a deep U-Net to estimate vocal pitch directly from polyphonic audio without source separation, scoring 87.2% on the pitch-benchmark and becoming the default in singing voice conversion pipelines.

**Melody extraction from polyphonic audio** (identifying the predominant melody line in a mix) remains served primarily by **Melodia** [16], a signal-processing pipeline available as a VAMP plugin. VAMP plugins are accessed from Python via `pip install vamp`, then `vamp.collect(audio_data, sample_rate, "mtg-melodia:melodia")`. Similarly, **pYIN** [17] (probabilistic YIN) is available both as a VAMP plugin and natively in librosa via `librosa.pyin()`.

### Tool comparison table: Pitch / F0

| Tool | Params | RPA (MDB, 50¢) | Speed | License | Install | Real-time |
|---|---|---|---|---|---|---|
| **CREPE** | 22.2M | 96.7% | ~1× RT (CPU) | MIT | `pip install crepe` | No |
| **torchcrepe** | 22.2M | 96.7% | GPU-accelerated | MIT | `pip install torchcrepe` | Near |
| **PESTO** | <30K | ~96.1% (MIR-1K) | 12× RT (CPU) | MIT | `pip install pesto-pitch` | Yes |
| **PENN** | ~CREPE | SOTA cross-domain | 408× RT (GPU) | MIT | `pip install penn` | Yes |
| **SwiftF0** | 95.8K | 90.2% (8-dataset avg) | 42× CREPE (CPU) | — | Research | Yes |
| **RMVPE** | Large | 87.2% (vocal) | Moderate | — | GitHub | No |
| **SPICE** | Medium | ~90.7% (MIR-1K) | Mobile (TFLite) | Apache 2.0 | TF Hub | Yes |
| **pYIN** | N/A (DSP) | Pre-CREPE SOTA | Very fast | ISC (librosa) | `pip install librosa` | Yes |
| **aubio** | N/A (DSP) | Good (not benchmarked) | Real-time native | GPL-3.0 | `pip install aubio` | Yes |

Standard evaluation metrics are **RPA** (Raw Pitch Accuracy at 50-cent threshold), **RCA** (Raw Chroma Accuracy, octave-invariant), and **Voicing Recall/False Alarm**. The primary datasets are **MDB-stem-synth** (230 tracks, 25 instruments, 15.5 hours) and **MIR-1K** (1000 singing clips).

---

## 3. Chord recognition and harmonic analysis

### The state of chord recognition in 2025

Automatic chord recognition has seen steady but modest progress since the transformer era began. **BTC** [18] (Bi-directional Transformer for Chords, ISMIR 2019) processes 10-second CQT windows through multi-head self-attention and supports both 25-class (major/minor) and ~170-class (large vocabulary) chord recognition, achieving approximately **80–87% Weighted Chord Symbol Recall** (WCSR) on standard datasets. It remains the most widely used open-source baseline.

**madmom**'s CNN+CRF chord recognition pipeline [19] uses a fully convolutional deep auditory model followed by a linear-chain CRF for temporal smoothing. It outputs only 25 chord classes (12 major + 12 minor + no-chord) but was the top MIREX performer in 2016 and remains reliable for major/minor-only applications. **Chordino** [20], a VAMP plugin based on NNLS approximate transcription and template matching, supports customizable vocabularies including sevenths and inversions via a `chord.dict` file, and is accessible from Python via `vamp.collect(data, rate, "nnls-chroma:chordino")` or the higher-level `pip install chord-extractor`.

The newest open-source model is **ChordFormer** [21] (arXiv, February 2025), a Conformer-based architecture that applies CNN+self-attention blocks with reweighted loss for class imbalance and CRF decoding. It achieves **+2% frame-wise and +6% class-wise accuracy** over prior state-of-the-art on a combined 1,217-song dataset — the current best for large-vocabulary chord recognition. A separate LLM-based approach [22] (arXiv, 2025) uses GPT-4o in a five-stage Chain-of-Thought pipeline (source separation → key detection → initial chord recognition → beat tracking → LLM refinement), gaining 1–2.77% MIREX metric improvement but requiring costly API calls.

### Roman numeral analysis and figured bass

**No mature end-to-end tool produces Roman numeral analysis from audio.** The most promising work is by Fricke et al. [23] (EvoMUSART 2024), who adapted **AugmentedNet** [24] (originally a symbolic-domain CRNN for Roman numeral analysis, ISMIR 2021) to accept audio input via chromagrams and semitone spectra, finding accuracy comparable to symbolic input. **ChordGNN** [25] (ISMIR 2023) uses graph neural networks to outperform AugmentedNet on symbolic data but remains symbolic-only.

The practical pipeline for Roman numeral analysis from audio is: extract chords (BTC/madmom) → detect key (Essentia/madmom) → convert to Roman numerals using **music21** [26], which provides `roman.romanNumeralFromChord(chord_obj, key_obj)` with full support for inversions, secondary dominants, augmented sixths, and Neapolitan chords. For **figured bass**, no audio-based tool exists whatsoever; music21's `figuredBass.notation.Notation` objects operate exclusively on symbolic input.

### Tool comparison table: Chords / Harmony / Key

| Tool | Vocabulary | WCSR (MajMin) | Method | License | Install |
|---|---|---|---|---|---|
| **ChordFormer** | Large (~170+) | SOTA (+2% frame) | Conformer + CRF | — | Research code |
| **BTC** | 25 or ~170 | ~80–87% | Bi-dir Transformer | Academic | git clone |
| **madmom CRF** | 25 (maj/min) | MIREX 2016 winner | CNN + CRF | BSD / CC-NC | `pip install madmom` |
| **Chordino** | Configurable | ~79–81% | NNLS + templates | GPL | VAMP plugin |
| **Omnizart chord** | Varies | — | Harmony Transformer | MIT | `pip install omnizart` |
| **Essentia** | 24 (triads only) | ~61% | HPCP + templates | AGPL-3.0 | `pip install essentia` |
| **autochord** | 25 | — | BiLSTM-CRF | Open source | `pip install autochord` |
| **Chord.ai** | Broad (7ths, aug, dim) | "Outstanding" (user reports) | Proprietary DL | Commercial | App only, no API |

For **key detection**, the best options are **madmom CNNKeyRecognitionProcessor** [27] (CNN end-to-end, genre-adaptable), **Essentia KeyExtractor** [28] (with profile selection: 'edma' for EDM, 'temperley' for pop/classical), and **libKeyFinder** [29] (GPL-3.0, integrated in Mixxx DJ). State-of-the-art accuracy ranges from 75–90% depending on genre, with **MARBLE SSL-based models** [30] representing the cutting edge on the GiantSteps Key benchmark.

A critical finding on **chord annotation subjectivity**: Koops et al. [31] (2019) found only **73% agreement** among four expert annotators on chord labels for the same recordings, establishing a practical ceiling for automated systems.

---

## 4. Beat tracking, tempo estimation, and meter

### The post-DBN era

Beat tracking has been one of MIR's most mature tasks, with madmom's DBN-based systems [19] dominating from 2016 through 2023. The standard approach combined RNN-based beat activation functions with a Dynamic Bayesian Network for metrically consistent output, typically achieving **F1 > 0.90** on Ballroom and **~0.85** on GTZAN.

**Beat This!** [32] (ISMIR 2024) is the new offline SOTA. It uses a transformer encoder with Rotary Position Embeddings (RoPE), trained on **16 diverse datasets** with shift-tolerant binary cross-entropy loss that eliminates the need for DBN post-processing. It significantly outperforms all prior systems including madmom and the Beat Transformer [33] across multiple benchmarks. The MIT license and clean Python API (`pip install` from GitHub, ensemble of pretrained models) make it the recommended offline beat tracker.

For **real-time** applications, **BeatNet** [34] (ISMIR 2021) remains the most feature-complete system, jointly tracking beat, downbeat, tempo, and meter using a CRNN activation function fed into a Monte Carlo particle filter. It is the only tool that estimates **time signature in real-time**. **BEAST** [35] (ICASSP 2024) introduced a streaming transformer achieving **~80% beat F1** at under 50ms latency — approximately **10–15 percentage points below offline SOTA**, with the gap widening to ~30 percentage points for downbeat tracking.

### Tool comparison table: Beat / Tempo / Meter

| Tool | Beat | Downbeat | Meter | Method | Real-time | License | Last Active |
|---|---|---|---|---|---|---|---|
| **Beat This!** | SOTA | SOTA | No | Transformer + RoPE | No | MIT | 2024 |
| **BeatNet** | Good | Good | **Yes** | CRNN + particle filter | **Yes** | GPL-3.0 | 2023 |
| **BEAST** | ~80% F1 | ~53% F1 | No | Streaming Transformer | **Yes** | — | 2024 |
| **madmom DBN** | Excellent | Excellent | Partial | RNN + DBN/HMM | Partial | BSD/CC-NC | 2020 |
| **librosa** | Adequate | No | No | Onset DP | No | ISC | 2025 |
| **aubio** | Adequate | No | No | Signal processing | **Yes** | GPL-3.0 | 2021 |
| **Essentia** | Good | No | No | Multi-feature + CNN | Partial | AGPL-3.0 | 2024 |

---

## 5. Structural segmentation reaches functional labeling

Structural segmentation — identifying section boundaries and labeling them as intro, verse, chorus, bridge, or outro — has advanced significantly with deep learning. **ALL-IN-ONE** [36] (Kim and Nam, 2023) is the only publicly available model that jointly performs beat tracking, downbeat tracking, structural segmentation, and **functional structure labeling** in a single pass. It uses source-separated spectrograms (via Demucs) processed through 1D Dilated Neighborhood Attention and achieves SOTA on the Harmonix Set. The clean pip-installable API (`pip install allin1; allin1.analyze('song.wav')`) returns BPM, beats, downbeats, and segment labels directly.

**SongFormer** [37] (2025) sets a new SOTA for structural segmentation using a 4-layer transformer with RoPE that fuses multi-resolution self-supervised representations. Trained on **SongFormDB** (10,000+ tracks across 47 languages), it outperforms even Gemini 2.5 Pro on the SongFormBench evaluation. It is available on HuggingFace but focuses exclusively on structure — no beat or tempo.

**MSAF** [38] (Music Structure Analysis Framework) provides classical unsupervised algorithms (checkerboard kernel, spectral clustering, CNMF) that output similarity-based labels (A/B/C) rather than functional labels. It remains useful for research and comparison but is outperformed by deep learning methods. librosa provides basic building blocks via `librosa.segment.recurrence_matrix()` and agglomerative clustering.

---

## 6. Lyrics transcription and phoneme alignment

**Automatic lyrics transcription** from singing audio has been transformed by large-scale ASR models. OpenAI's **Whisper** [39] achieves ~3% WER on clean, close-mic vocal recordings but degrades to 50%+ on accompanied singing, with hallucination during instrumental passages. **WhisperX** [40] (Interspeech 2023) adds voice activity detection (pyannote-audio) and word-level forced alignment (wav2vec2.0), running **70× real-time** with GPU batching and dramatically reducing hallucination.

The current SOTA is **LyricWhiz** [41] (ISMIR 2023), a zero-shot pipeline combining PANNs for vocal detection, Whisper for multiple transcription passes, and GPT-4 for Chain-of-Thought post-processing. It significantly reduces WER across genres including challenging rock and metal. **SongTrans** [42] (2024) unifies lyrics transcription, note transcription, and alignment in a single autoregressive model — the first to handle all three without source separation preprocessing.

For **phoneme-level forced alignment**, the **Montreal Forced Aligner** (MFA) [43] is the standard tool: Kaldi-based, MIT-licensed, supporting many languages, with ~17ms median boundary error on speech. When retrained on singing data, it achieves good performance for singing voice synthesis data preparation. WhisperX also provides word-level alignment via wav2vec2.

---

## 7. Instrument recognition and source separation

**PANNs** [44] (Pretrained Audio Neural Networks) provide 527-class audio tagging including many instrument categories, achieving **mAP 0.439** on AudioSet with the CNN14 architecture. They serve as the workhorse for instrument classification in production pipelines. **CLAP** [45] (Contrastive Language-Audio Pretraining) enables **zero-shot instrument classification** via text queries like "acoustic guitar playing" — trained on 4.6M audio-text pairs, it achieves SOTA across 26 downstream tasks without task-specific fine-tuning. **MERT** [46] (Music Understanding Model) provides general-purpose music embeddings via a BERT-style transformer pre-trained with dual acoustic and musical teachers, achieving SOTA on 14 music understanding tasks including instrument classification. Available on HuggingFace (`m-a-p/MERT-v1-330M`), it represents the most versatile foundation model for MIR.

**Source separation** is a critical preprocessing step for most MIR tasks. **Demucs v4/HTDemucs** [47] (Meta, MIT license) achieves **9.20 dB SDR** on MUSDB-HQ using a hybrid time-frequency U-Net with cross-domain transformer attention, and supports 4-stem (vocals/drums/bass/other) or 6-stem separation. **Spleeter** [48] (Deezer, MIT) runs **100× real-time** on GPU but with lower quality (bandwidth limited to 11–16 kHz). **BSRNN** [49] and its transformer variant BS-RoFormer won the SDX23 Challenge but lack official open-source implementations.

---

## 8. Other symbolic representations: tablature, lead sheets, and alignment

**Guitar tablature from audio** remains an open problem. **TabCNN** [50] (ISMIR 2019) estimates tablature from solo acoustic guitar using GuitarSet, and **TapToTab** [51] (2024) combines video and audio for tab generation. Commercially, **Klangio** exports Guitar Pro tablature via its REST API. No robust open-source tool handles polyphonic or mixed-audio tablature.

**Lead sheet generation** (melody + chords + lyrics) has no end-to-end open-source solution. The practical approach combines Demucs (separation) → Basic Pitch or CREPE (melody) → BTC or madmom (chords) → Whisper or LyricWhiz (lyrics) → assembly via music21. Commercially, **ScoreCloud** and **Melody Scanner** (Klangio) offer this workflow.

**Audio-to-score alignment** (synchronizing a known score to a recording) is well-served by **Synctoolbox** [52] (Müller et al.), librosa's `librosa.sequence.dtw()`, and **pyAMPACT** [53] (2024). **ChordSync** [54] (2024) uses a Conformer for chord-level alignment without weak supervision.

---

## 9. Framework libraries and evaluation infrastructure

The MIR ecosystem rests on four core framework libraries, each with distinct API philosophies:

**librosa** [55] (ISC license, v0.11.0, March 2025) provides a purely functional API with NumPy arrays as the universal data type. It covers spectral features (STFT, CQT, chroma, MFCC), onset detection, beat tracking, pitch estimation (pYIN), harmonic-percussive separation, DTW, and display utilities. It is the lowest-friction entry point for any audio analysis task.

**Essentia** [28] (AGPL-3.0, 250+ algorithms) provides both imperative (`essentia.standard`) and streaming (`essentia.streaming`) modes via C++ with Python bindings. Its strength lies in integrated TensorFlow model inference for genre, mood, instrument, and tempo classification via pre-trained models. The AGPL license and CC BY-NC-ND 4.0 model weights are significant barriers for commercial use.

**madmom** [19] (BSD code, CC-BY-NC-SA models) uses a processor-based pipeline pattern where all operations are encapsulated in callable `Processor` objects composable via `SequentialProcessor`. Its beat tracking, onset detection, and chord recognition processors were MIREX top-ranked for years. The codebase is aging (last PyPI release: 2018) with Python 3.10+ compatibility issues, but it remains widely used as a component in newer systems.

**mir_eval** [56] (MIT, v0.8.2) provides standardized evaluation metrics for every MIR task: beat (F-measure, CMLc/t, AMLc/t), chord (root/thirds/triads/tetrads/sevenths with and without inversions, WCSR), melody (RPA, RCA, OA, voicing metrics), onset (F/P/R), transcription (onset and onset-offset F-measure), separation (SDR, SIR, SAR, SI-SDR), key (weighted score), and structural segmentation (pairwise F, Rand index, NCE, boundary F-measure). **mirdata** [57] (BSD-3) standardizes dataset access for 50+ MIR datasets with `mirdata.initialize('maestro').download()`. **JAMS** [58] (JSON Annotated Music Specification) provides a hierarchical JSON format for multi-annotator, multi-task music annotations with built-in evaluation wrapping mir_eval. **pretty_midi** [59] (MIT) provides object-oriented MIDI manipulation with times in seconds (not ticks), making it ideal for audio-aligned work.

---

## 10. Commercial and cloud-based services

| Service | Capabilities | API | Pricing | Notes |
|---|---|---|---|---|
| **Klangio** | Multi-instrument transcription, MIDI, MusicXML, PDF, tabs, chords, BPM | **Full REST API** | Subscription + tickets | Most comprehensive transcription API |
| **Moises / music.ai** | Stem separation, chords, beat, key, lyrics | **REST API** (async jobs) | Per-job billing | WASPAA 2025 paper on efficiency |
| **AudioShake** | Stem separation, lyrics + word-level alignment | **REST API + SDK** (iOS/Win/Android/Linux) | From $19.99/mo | Won Sony Demixing Challenge |
| **LALAL.AI** | Source separation (10+ stems) | **REST API** (v1, Feb 2026) | Minute-based credits | Strong Pro Instrument benchmark |
| **Chord.ai** | Chords, beat, key, stems, MIDI, lyrics | **No API** (app only) | ~$9/mo | Uses Basic Pitch internally |
| **AnthemScore** | Audio → sheet music (PDF, MusicXML, MIDI) | **No API** (desktop) | $39 one-time | Good for solo piano |
| **Melodyne** | Note-level audio editing, polyphonic DNA | **ARA SDK** (C++ only) | $99–$699 | ARA2 is Apache 2.0 |
| **Spotify Audio Features** | Key, tempo, time signature, sections, segments | **Deprecated** (Nov 2024) | N/A | New apps get 403 errors |

---

## 11. Unified multi-task models producing multiple representations

**No single model produces pitch + chords + beat + key in one forward pass.** The closest approaches are:

**ALL-IN-ONE** [36] jointly produces beat + downbeat + BPM + structural segmentation + functional labels in one pass. It is the most practically useful unified model, installable via `pip install allin1`. **MT3/YourMT3+** [3][5] produces multi-instrument MIDI transcription in one pass (pitch + onset/offset + instrument labels). **Omnizart** [60] wraps six separate models (piano, drums, vocals, vocal contour, chords, beat) under a single CLI/Python API but runs them independently. **MERT** [46] produces general-purpose embeddings that achieve SOTA on 14 tasks when fine-tuned with task-specific heads, but does not directly output symbolic representations.

The fundamental barrier is that different representations operate at different temporal granularities (frame-level pitch vs. segment-level structure vs. note-level MIDI) and require different loss functions, making true end-to-end multi-task learning architecturally challenging.

---

## 12. Answers to the ten research questions

**Q1. Current SOTA for polyphonic AMT (all instruments)?** YourMT3+ [5] with PerceiverTF + MoE encoder, outperforming MT3 across ten datasets. Failure modes: instrument leakage (violin↔cello), no vocal support, poor generalization to out-of-distribution recordings, false positives in dense textures.

**Q2. Real-time transcription tools (<500ms latency)?** Basic Pitch (CPU, browser WASM), Mobile-AMT (mobile devices, 82.9% less compute), Onsets and Frames TFLite (mobile). Accuracy trade-off: approximately 5–15% lower F1 than offline SOTA. For beat tracking: BeatNet (real-time with meter) and BEAST (~80% F1 at <50ms latency) vs. Beat This! (90%+ offline).

**Q3. Best open-source chord recognition tool (2024–2025)?** **ChordFormer** [21] for large-vocabulary accuracy (+2% frame, +6% class over prior SOTA). **BTC** [18] for the best balance of accuracy, vocabulary options, and ease of use. **madmom** [19] for major/minor-only with the simplest pip install.

**Q4. Roman numeral analysis or figured bass from audio?** Roman numerals: Fricke et al. [23] adapted AugmentedNet for audio input (EvoMUSART 2024) — the only published work. Practical pipeline: BTC → Essentia key → music21 `romanNumeralFromChord()`. Figured bass: **no tool exists**; only music21's symbolic `figuredBass` module could be used downstream.

**Q5. MusicXML or score-level output from audio?** No open-source neural model does this end-to-end. Pipeline: AMT model → MIDI → music21/MuseScore for quantization → MusicXML export. Commercial: Klangio API (REST, exports MusicXML/PDF), AnthemScore, Melody Scanner.

**Q6. Structural segmentation tools?** ALL-IN-ONE [36] (functional labels: verse/chorus/bridge, MIT, `pip install allin1`). SongFormer [37] (2025 SOTA, HuggingFace). MSAF [38] (classical unsupervised, MIT, `pip install msaf`).

**Q7. Lyrics transcription and phoneme alignment?** Transcription: LyricWhiz [41] (SOTA, Whisper + GPT-4), WhisperX [40] (word-level, 70× RT), SongTrans [42] (unified lyrics+notes). Alignment: Montreal Forced Aligner [43] (MIT, trainable), WhisperX forced alignment.

**Q8. Unified multi-task models?** ALL-IN-ONE (beat+downbeat+structure+labels), MT3/YourMT3+ (multi-instrument MIDI), Omnizart (6 tasks, separate models). No model does pitch+chords+beat+key in one pass.

**Q9. Existing Python façade patterns?** No comprehensive unified AMT library exists. Omnizart is closest (multi-task CLI/API, separate models). torchaudio.pipelines provides the best design precedent (Bundle pattern with lazy model loading and metadata). mirdata and mir_eval provide unified interfaces for data and evaluation respectively.

**Q10. Licensing landmines?** Critical: **aubio** (GPL-3.0, importing makes your package GPL), **Essentia** (AGPL-3.0, even SaaS triggers copyleft), **madmom model weights** (CC-BY-NC-SA, non-commercial only), **Essentia model weights** (CC BY-NC-ND 4.0, non-commercial, no derivatives). Safe core: librosa (ISC), Basic Pitch (Apache 2.0), Demucs (MIT), pretty_midi (MIT), mir_eval (MIT), torchaudio (BSD), Beat This! (MIT), torchcrepe (MIT), PESTO (MIT), PENN (MIT).

---

## 13. Gaps and custom implementation needs

Several areas lack good open-source solutions and would require custom implementation in a Python façade:

**Roman numeral analysis from audio** has only one experimental paper (Fricke et al., 2024). A production pipeline needs a chord recognizer, key detector, and music21 conversion layer — all glued together with temporal alignment logic and handling of modulations, secondary dominants, and chromatic chords. **Figured bass from audio** is entirely unaddressed.

**MusicXML/score quantization from MIDI** requires rhythmic quantization (snapping note onsets/offsets to a metric grid), voice separation (assigning notes to staves), beam grouping, and enharmonic spelling. No open-source library does this well; music21 can write MusicXML but does not perform intelligent quantization from raw MIDI timing.

**Lead sheet generation** (melody + chords + lyrics assembled into a single document) requires orchestrating three separate extraction pipelines plus layout — a pure integration challenge with no existing solution.

**Large-vocabulary chord recognition with inversions** is addressed by ChordFormer but without pip-installable code. A custom wrapper around BTC or a reimplementation of the Conformer architecture would be needed.

**Singing-specific lyrics transcription** without LLM API dependency (avoiding GPT-4 costs in LyricWhiz) requires fine-tuning Whisper on singing data or training a dedicated model.

**Real-time multi-instrument transcription** does not exist — Mobile-AMT handles piano only, and no model transcribes guitar, bass, drums, and vocals in real-time.

---

## 14. Recommended stack for a Python façade library

The following stack uses exclusively **permissive licenses** (MIT, Apache 2.0, ISC, BSD) for the core, with GPL/AGPL tools as optional extras:

### Core dependencies (always installed)

```bash
pip install librosa pretty-midi mir_eval jams mirdata music21
```

These provide audio I/O, MIDI manipulation, evaluation metrics, annotation formats, dataset access, and symbolic music theory — all under permissive licenses.

### Task-specific backends (optional extras)

```bash
# AMT / MIDI transcription
pip install basic-pitch                    # Apache 2.0, instrument-agnostic
pip install piano_transcription_inference  # MIT, piano SOTA

# Pitch / F0
pip install torchcrepe    # MIT, GPU-accelerated CREPE
pip install pesto-pitch   # MIT, lightweight, real-time
pip install penn          # MIT, fastest on GPU

# Beat / tempo / structure
pip install beat_this     # MIT, offline SOTA (from GitHub)
pip install allin1        # MIT, beat+downbeat+structure+labels

# Source separation (preprocessing)
pip install demucs        # MIT, SOTA quality

# Lyrics
pip install whisperx      # BSD-4, word-level alignment

# Foundation model
pip install transformers  # Apache 2.0 (for MERT embeddings)
```

### Optional GPL/AGPL extras (isolated via lazy imports)

```bash
pip install myfacade[essentia]  # AGPL — key detection, tempo, instrument models
pip install myfacade[madmom]    # BSD code, CC-NC models — beat/chord baselines
pip install myfacade[aubio]     # GPL — real-time pitch/beat
pip install myfacade[beatnet]   # GPL — real-time beat+meter
```

### Façade API design pattern

Following torchaudio's Bundle pattern and madmom's Processor pattern:

```python
from myfacade import transcribe, analyze

# High-level unified API
result = analyze("song.wav", tasks=["midi", "chords", "beats", "key", "structure"])

# result.midi         → pretty_midi.PrettyMIDI object
# result.chords       → List[Interval(start, end, label)]
# result.beats        → List[float] (seconds)
# result.downbeats    → List[float]
# result.key          → Key(tonic="C", mode="major", confidence=0.92)
# result.structure    → List[Segment(start, end, label="chorus")]
# result.bpm          → float

# Task-specific with backend selection
from myfacade.pitch import PitchTracker

tracker = PitchTracker(backend="pesto")  # or "crepe", "penn", "pyin"
f0, confidence = tracker.predict("vocal.wav")

from myfacade.chords import ChordRecognizer

recognizer = ChordRecognizer(backend="btc", vocabulary="large")
chords = recognizer.predict("song.wav")

# Evaluation integration
from myfacade.evaluate import evaluate_beats

scores = evaluate_beats(predicted=result.beats, reference="ref.beats")
```

Key architectural decisions: (1) abstract base classes per task (BeatTracker, ChordRecognizer, PitchEstimator, Transcriber, StructureAnalyzer, LyricsTranscriber); (2) lazy imports to avoid loading heavy backends until needed; (3) standardized output dataclasses compatible with mir_eval and JAMS; (4) factory functions that auto-select the best available backend; (5) model weights downloaded on demand at runtime, never bundled (avoiding CC-NC redistribution issues).

---

## 15. Evaluation datasets at a glance

| Dataset | Size | Tasks | License |
|---|---|---|---|
| **MAESTRO** | 200 hrs, 1,276 performances | Piano transcription | CC BY-NC-SA 4.0 |
| **MAPS** | ~60 hrs piano | Piano transcription | Research only |
| **Slakh2100** | 145 hrs, 2,100 tracks, 34 instruments | Multi-instrument transcription, separation | CC BY 4.0 |
| **MusicNet** | 34 hrs, 330 classical recordings | Transcription, instrument recognition | CC BY 4.0 |
| **GuitarSet** | 3 hrs, 360 excerpts | Guitar transcription, chords | CC BY 4.0 |
| **MUSDB18-HQ** | 10 hrs, 150 tracks | Source separation | Mixed |
| **Isophonics** | ~200 songs (annotations only) | Chords, beat, key, structure | Free |
| **Billboard** | ~700 songs (annotations) | Chord recognition | Research use |
| **SALAMI** | 1,500+ tracks (annotations) | Structural segmentation | CC0 |
| **Harmonix Set** | ~900 tracks | Beat, downbeat, structure | Research |
| **MDB-stem-synth** | 15.5 hrs, 230 tracks | Pitch estimation | Research |
| **DALI** | 5,358 songs | Lyrics alignment, vocal melody | Research |
| **Schubert Winterreise** | 24 songs, 9 performances | Cross-modal alignment, chords, key | CC BY-SA 4.0 |
| **GiantSteps Key** | 604 EDM tracks | Key detection | Research |

---

## Conclusion

The AMT/MIR field in 2025 is simultaneously mature and fragmented. **Piano transcription is effectively solved** for in-distribution data (97.6% onset F1), yet multi-instrument transcription, real-time operation, and out-of-distribution robustness remain active research frontiers. The transformer architecture now dominates every task from beat tracking (Beat This!) to chord recognition (ChordFormer) to structural segmentation (SongFormer), with self-supervised foundation models (MERT, CLAP) emerging as versatile feature extractors.

Three critical gaps define the opportunity for a Python façade library. First, **no unified tool chain exists** — building a complete music analysis pipeline requires integrating 5–10 tools with incompatible APIs, conflicting dependencies, and mixed licenses. Second, **higher-level music theory outputs** (Roman numeral analysis, figured bass, lead sheets) require custom pipelines combining audio extraction with symbolic reasoning via music21 — this integration layer is entirely missing. Third, the **licensing landscape** creates a minefield: the most established tools (Essentia, madmom models, aubio) carry copyleft or non-commercial restrictions that prevent their inclusion in a permissively licensed package.

The viable path forward is a façade built on the permissive core of librosa + Basic Pitch + Beat This! + torchcrepe/PESTO + Demucs + music21, with GPL/AGPL backends as optional extras loaded lazily. The torchaudio Bundle pattern provides the right design blueprint: encapsulate each backend behind abstract task interfaces with standardized output dataclasses compatible with mir_eval and JAMS. The most impactful novel contribution such a library could make would be the Roman-numeral-from-audio pipeline and MusicXML quantization layer — two areas where no existing tool provides a clean programmatic solution.

---

**References**

Here are all 60 references, one per line, each with a hyperlink:

[1] Hawthorne, C. et al. "Onsets and Frames: Dual-Objective Piano Transcription." ISMIR 2018. [Magenta](https://magenta.tensorflow.org/onsets-frames)

[2] Kong, Q. et al. "High-Resolution Piano Transcription with Pedals by Regressing Onset and Offset Times." IEEE/ACM TASLP, 2021. [arXiv:2010.01815](https://arxiv.org/abs/2010.01815)

[3] Gardner, J. et al. "MT3: Multi-Task Multitrack Music Transcription." ICLR 2022. [arXiv:2111.03017](https://arxiv.org/abs/2111.03017)

[4] Toyama, K. et al. "Automatic Piano Transcription with Hierarchical Frequency-Time Transformer." ISMIR 2023. [arXiv:2307.04305](https://arxiv.org/abs/2307.04305)

[5] Chang, S. et al. "YourMT3+: Multi-instrument Music Transcription with Enhanced Transformer Architectures and Cross-dataset Stem Augmentation." 2024. [arXiv:2407.04822](https://arxiv.org/abs/2407.04822)

[6] Sony AI. "DiffRoll: Diffusion-based Generative Music Transcription with Unsupervised Pretraining Capability." 2023. [Sony AI](https://ai.sony/publications/DiffRoll-Diffusion-based-Generative-Music-Transcription-with-Unsupervised-Pretraining-Capability/)

[7] "Mobile-AMT: Real-Time Polyphonic Piano Transcription for In-the-Wild Recordings." EUSIPCO 2024. [OpenReview](https://openreview.net/forum?id=1QTsNlmlDk)

[8] "Aria-MIDI: A Dataset of Piano MIDI Files for Symbolic Music Modeling." ICLR 2025. [arXiv:2504.15071](https://arxiv.org/abs/2504.15071)

[9] Bittner, R. M. et al. "A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation." ICASSP 2022. [GitHub: spotify/basic-pitch](https://github.com/spotify/basic-pitch)

[10] Kim, J. W. et al. "CREPE: A Convolutional Representation for Pitch Estimation." ICASSP 2018. [arXiv:1802.06182](https://arxiv.org/abs/1802.06182)

[11] Morrison, M. "torchcrepe." [GitHub: maxrmorrison/torchcrepe](https://github.com/maxrmorrison/torchcrepe)

[12] Riou, A. et al. "PESTO: Pitch Estimation with Self-supervised Transposition-equivariant Objective." ISMIR 2023 Best Paper. [arXiv:2309.02265](https://arxiv.org/abs/2309.02265)

[13] Morrison, M. et al. "Cross-domain Neural Pitch and Periodicity Estimation." 2023. [arXiv:2301.12258](https://arxiv.org/abs/2301.12258)

[14] "SwiftF0: Fast and Accurate Monophonic Pitch Detection." 2025. [arXiv:2508.18440](https://arxiv.org/abs/2508.18440)

[15] "RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music." Interspeech 2023. [arXiv:2306.15412](https://arxiv.org/abs/2306.15412)

[16] Salamon, J. and Gómez, E. "Melody Extraction from Polyphonic Music Signals using Pitch Contour Characteristics." IEEE TASLP, 20(6), 2012. [MTG-UPF Melodia](https://www.upf.edu/web/mtg/melodia)

[17] Mauch, M. and Dixon, S. "pYIN: A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions." ICASSP 2014. [Sound Software](https://code.soundsoftware.ac.uk/projects/pyin)

[18] Park, J. et al. "A Bi-Directional Transformer for Musical Chord Recognition." ISMIR 2019. [arXiv:1907.02698](https://arxiv.org/abs/1907.02698)

[19] Böck, S. et al. "madmom: a new Python Audio and Music Signal Processing Library." ACM Multimedia 2016. [ACM DL](https://dl.acm.org/doi/10.1145/2964284.2973795)

[20] Mauch, M. and Dixon, S. "Approximate Note Transcription for the Automatic Identification of Difficult Chords." ISMIR 2010. [Isophonics](http://www.isophonics.net/nnls-chroma)

[21] Akram et al. "ChordFormer: A Conformer-Based Architecture for Large-Vocabulary Audio Chord Recognition." 2025. [arXiv:2502.11840](https://arxiv.org/abs/2502.11840)

[22] "Enhancing Automatic Chord Recognition through LLM Chain-of-Thought Reasoning." 2025. [arXiv:2509.18700](https://arxiv.org/abs/2509.18700)

[23] Fricke et al. "Adaptation and Optimization of AugmentedNet for Roman Numeral Analysis Applied to Audio Signals." EvoMUSART 2024. [Springer](https://link.springer.com/chapter/10.1007/978-3-031-56992-0_10)

[24] Nápoles López et al. "AugmentedNet: A Roman Numeral Analysis Network with Synthetic Training Examples." ISMIR 2021. [arXiv:2106.02919](https://arxiv.org/abs/2106.02919)

[25] Karystinaios, E. and Widmer, G. "Roman Numeral Analysis with Graph Neural Networks." ISMIR 2023. [arXiv:2307.03544](https://arxiv.org/abs/2307.03544)

[26] Cuthbert, M. S. and Ariza, C. "music21: A Toolkit for Computer-Aided Musicology." ISMIR 2010. [music21.org](https://music21.org/)

[27] Korzeniowski, F. and Widmer, G. "End-to-End Musical Key Estimation Using a Convolutional Neural Network." 2017. [arXiv:1706.02921](https://arxiv.org/abs/1706.02921)

[28] Bogdanov, D. et al. "Essentia: an Audio Analysis Library for Music Information Retrieval." ISMIR 2013. [essentia.upf.edu](https://essentia.upf.edu/)

[29] Sha'ath, I. "Estimation of Key in Digital Music Recordings." M.Sc. thesis, 2011. [GitHub: mixxxdj/libkeyfinder](https://github.com/mixxxdj/libkeyfinder)

[30] "MARBLE Benchmark." [GitHub: a43992899/MARBLE-Benchmark](https://github.com/a43992899/MARBLE-Benchmark)

[31] Koops, H. V. et al. "Improving Audio Chord Estimation by Alignment and Integration of Crowd-Sourced Symbolic Music." TISMIR, 2019. [TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.81)

[32] Foscarin, F., Schlüter, J., Widmer, G. "Beat This! Accurate Beat Tracking Without DBN Postprocessing." ISMIR 2024. [GitHub: CPJKU/beat_this](https://github.com/CPJKU/beat_this)

[33] Zhao, H., Xia, G., Wang, Y. "Beat Transformer: Demixed Beat and Downbeat Tracking with Dilated Self-Attention." ISMIR 2022. [arXiv:2209.07140](https://arxiv.org/abs/2209.07140)

[34] Heydari, M., Cwitkowitz, F., Duan, Z. "BeatNet: CRNN and Particle Filtering for Online Joint Beat Downbeat and Meter Tracking." ISMIR 2021. [GitHub: mjhydri/BeatNet](https://github.com/mjhydri/BeatNet)

[35] Chang, C. and Su, L. "BEAST: Online Joint Beat and Downbeat Tracking Based on Streaming Transformer." ICASSP 2024. [arXiv:2312.17156](https://arxiv.org/abs/2312.17156)

[36] Kim, T. and Nam, J. "All-In-One Metrical And Functional Structure Analysis With Neighborhood Attentions on Demixed Audio." 2023. [arXiv:2307.16425](https://arxiv.org/abs/2307.16425)

[37] Hao et al. "SongFormer: Scaling Music Structure Analysis with Heterogeneous Supervision." 2025. [arXiv:2510.02797](https://arxiv.org/abs/2510.02797)

[38] Nieto, O. and Bello, J. P. "Systematic Exploration of Computational Music Structure Research." ISMIR 2016. [MSAF docs](https://pythonhosted.org/msaf/)

[39] Radford, A. et al. "Robust Speech Recognition via Large-Scale Weak Supervision." ICML 2023. [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)

[40] Bain, M. et al. "WhisperX: Time-Accurate Speech Transcription of Long-Form Audio." Interspeech 2023. [arXiv:2303.00747](https://arxiv.org/abs/2303.00747)

[41] Zhuo et al. "LyricWhiz: Robust Multilingual Zero-shot Lyrics Transcription by Whispering to ChatGPT." ISMIR 2023. [arXiv:2306.17103](https://arxiv.org/abs/2306.17103)

[42] "SongTrans: An Unified Song Transcription and Alignment Method for Lyrics and Notes." 2024. [arXiv:2409.14619](https://arxiv.org/abs/2409.14619)

[43] McAuliffe, M. et al. "Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi." Interspeech 2017. [MFA docs](https://montrealcorpustools.github.io/Montreal-Forced-Aligner/)

[44] Kong, Q. et al. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." IEEE/ACM TASLP, 2020. [arXiv:1912.10211](https://arxiv.org/abs/1912.10211)

[45] Elizalde, B. et al. "CLAP: Learning Audio Concepts from Natural Language Supervision." ICASSP 2023. [arXiv:2206.04769](https://arxiv.org/abs/2206.04769)

[46] Li, Y. et al. "MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training." ICLR 2024. [arXiv:2306.00107](https://arxiv.org/abs/2306.00107)

[47] Rouard, S. et al. "Hybrid Transformers for Music Source Separation." ICASSP 2023. [arXiv:2211.08553](https://arxiv.org/abs/2211.08553)

[48] Hennequin, R. et al. "Spleeter: A Fast and Efficient Music Source Separation Tool with Pre-Trained Models." JOSS, 2020. [JOSS](https://www.theoj.org/joss-papers/joss.02154/10.21105.joss.02154.pdf)

[49] Luo, Y. and Yu, D. "Music Source Separation with Band-split RNN." IEEE/ACM TASLP, 2023. [ResearchGate](https://www.researchgate.net/publication/370621071_Music_Source_Separation_With_Band-Split_RNN)

[50] Wiggins, A. and Kim, Y. "Guitar Tablature Estimation with a Convolutional Neural Network." ISMIR 2019. [Semantic Scholar](https://www.semanticscholar.org/paper/Generating-Guitar-Tablatures-with-Neural-Networks-Mistler/7065897be10f94edb533ad8b98364372a85106da)

[51] "TapToTab." 2024. [arXiv:2409.08618](https://arxiv.org/abs/2409.08618)

[52] Müller, M. et al. "Synctoolbox: A Python Package for Efficient, Robust, and Accurate Music Synchronization." JOSS, 2021. [AudioLabs](https://www.audiolabs-erlangen.de/resources/MIR/FMP/B/B_ResourcesMIR.html)

[53] Devaney et al. "pyAMPACT: A Score-Audio Alignment Toolkit for Performance Data Estimation and Multi-modal Processing." 2024. [arXiv:2412.05436](https://arxiv.org/abs/2412.05436)

[54] Poltronieri et al. "ChordSync: Conformer-Based Alignment of Chord Annotations to Music Audio." 2024. [arXiv:2408.00674](https://arxiv.org/abs/2408.00674)

[55] McFee, B. et al. "librosa: Audio and Music Signal Analysis in Python." SciPy 2015. [librosa.org](https://librosa.org/)

[56] Raffel, C. et al. "mir_eval: A Transparent Implementation of Common MIR Metrics." ISMIR 2014. [ISMIR proceedings](https://archives.ismir.net/ismir2014/poster/000039.pdf)

[57] Bittner, R. M. et al. "mirdata: Software for Reproducible Usage of Datasets in MIR Research." ISMIR 2019. [GitHub: mir-dataset-loaders/mirdata](https://github.com/mir-dataset-loaders/mirdata)

[58] Humphrey, E. et al. "JAMS: A JSON Annotated Music Specification for Reproducible MIR Research." ISMIR 2014. [ResearchGate](https://www.researchgate.net/publication/265508524_JAMS_A_JSON_Annotated_Music_Specification_for_Reproducible_MIR_Research)

[59] Raffel, C. and Ellis, D. P. W. "Intuitive Analysis, Creation and Manipulation of MIDI Data with pretty_midi." ISMIR Late-Breaking Demo, 2014. [PDF](https://colinraffel.com/publications/ismir2014intuitive.pdf)

[60] Wu, Y.-T. et al. "Omnizart: A General Toolbox for Automatic Music Transcription." JOSS, 6(68), 2021. [arXiv:2106.00497](https://arxiv.org/abs/2106.00497)