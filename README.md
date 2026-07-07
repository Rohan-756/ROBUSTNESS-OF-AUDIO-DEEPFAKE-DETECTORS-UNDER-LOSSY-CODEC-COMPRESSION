# Robustness of Audio Deepfake Detectors Under Lossy Codec Compression

This repository contains the source code and experimental results for investigating the robustness of audio deepfake detectors under various lossy consumer codec compression scenarios (such as MP3, AAC, and Opus at social media-typical bitrates). 

Our research builds directly on the baseline paper:
> **Benchmarking Audio Deepfake Detection Robustness in Real-world Communication Scenarios** (EUSIPCO 2025) by Shi et al.

---

## Key Research Contributions

1. **Consumer Codec Coverage**: Unlike studies focusing solely on legacy telephony codecs (e.g., AMR-WB, SILK), we evaluate detectors against consumer internet codecs (**MP3**, **AAC**, and **Opus**) at bitrates common in messaging apps like WhatsApp and Telegram.
2. **Augmentation Ablation**: Systematic analysis of which specific codec subset (or combination) chosen during data augmentation generalizes best to unseen compression formats.
3. **Per-TTS-System Breakdown**: Instead of collapsing all results into a single EER value, we evaluate performance across individual text-to-speech (TTS) and voice conversion (VC) architectures (19 systems in ASVspoof 2019).
4. **Cross-Dataset Generalization**: We analyze model transferability by training on ASVspoof 2019 LA and evaluating on WaveFake.
5. **Vocoder-Codec Interactions**: We identify how specific vocoder families interact with lossy compression, determining why certain neural vocoders display higher fragility.

---

## 📁 Repository Structure

```directory
.
├── code_files/
│   ├── capstone-step-1.ipynb          # Environment verification & pilot phase experiments
│   ├── capstone-step-2.ipynb          # RawNet2 baseline training & ASVspoof dataset loading
│   ├── capstone-step-3.ipynb          # Initial WaveFake dataset integration
│   ├── capstone-step-3-extended.ipynb # Extended WaveFake evaluation & cross-dataset pipelines
│   ├── capstone-step-45-universal-fixed.py # Unified codec augmentation, AASIST/RawNet2 training, & ablation loop
│   └── capstone-step-6.ipynb          # Result analysis & final metric compilation
└── outputs/
    ├── run2_aasist_mp3_only_results.json
    ├── run3_aasist_opus_only_results.json
    ├── run4_aasist_all_codecs_results.json
    ├── run5_aasist_mp3_opus_results.pdf
    ├── run6_aasist_aac_only_results(1).json
    ├── run7_rawnet2_mp3_only_results.json
    ├── run8_rawnet2_all_codecs_results.json
    └── step_45_final_outputs.zip      # Archived raw ablation run results
```

---

## Execution & Pipeline Steps

### Step 1: Initialization and Pilot
- Verification of the ASVspoof 2019 dataset structure.
- FFmpeg-based simulation of consumer lossy codecs.
- CQT feature extraction and disproving the phase-coherence hypothesis.

### Step 2: Model Baseline Training
- Baseline training of RawNet2 and AASIST on clean ASVspoof 2019 data.
- Establishing baseline Equal Error Rates (EER) under zero-compression.

### Step 3: Cross-Dataset Integration
- Incorporating the WaveFake dataset to evaluate generalization to modern vocoders.
- Establishing base cross-dataset evaluation pipelines.

### Steps 4 & 5: Codec Augmentation & Ablation (Universal Fixed Pipeline)
- Utilizes `capstone-step-45-universal-fixed.py` to run systematic training configurations:
  - **Run 2**: AASIST trained with MP3-only codec augmentation.
  - **Run 3**: AASIST trained with Opus-only codec augmentation.
  - **Run 4**: AASIST trained with all codecs (MP3 + AAC + Opus) codec augmentation.
  - **Run 6**: AASIST trained with AAC-only codec augmentation.
  - **Run 7**: RawNet2 trained with MP3-only codec augmentation.
  - **Run 8**: RawNet2 trained with all codecs (MP3 + AAC + Opus) codec augmentation.

### Step 6: Summary & Compilation
- Processing evaluation outputs to generate comparative tables, EER summaries, and plot results.

---

## Summary of Experiments

The JSON files located in the `outputs/` directory contain detailed test results (EERs) for both models evaluated across:
- **Clean Dev/Eval sets**
- **Compressed Dev/Eval sets** (different codec-bitrate pairings)
- **Unseen vocoders** from WaveFake

For an overview of the ablation findings and finalized project findings, refer to the compiled outputs in `outputs/`.
