# Solfeggio OCR Studio (`solfeggio2md`)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-CUDA%2012.8-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-117%2F117%20Passed-success.svg)](tests/run_tests.py)

Convert scanned music theory textbooks, solfeggio exercises, and polyphonic sheet music PDFs into clean Markdown with embedded ABC notation and vector SVG scores.

Designed to run locally on consumer hardware with **8 GB VRAM** (e.g., RTX 3060, 4060, 4070, 5070 Mobile).

---

## Why this exists

Standard OCR engines (Tesseract, PaddleOCR) either completely ignore musical staves or turn them into gibberish characters across your text. Modern Vision LLMs (like Qwen-VL or GPT-4o) are great with prose, but they struggle with precise note pitches, hallucinate melodies, and run out of VRAM if you try to load specialized music models at the same time.

**Solfeggio OCR Studio** uses a 4-step sequential pipeline:
1. **Find and mask the music**: It detects staves, straightens them, and replaces them on the page with placeholder tags (`<!-- MUSIC_STUB_ID:... -->`).
2. **Read the music with specialized models**: It cuts out each stave, enhances the lines, and transcribes notes into Humdrum `**kern` using Transcoda and SMT.
3. **Validate the notes**: It checks that measure durations match time signatures, runs the C++ Verovio engine to ensure the notation is valid, and converts it to multi-voice ABC (`V:1 treble`, `V:2 bass`).
4. **Read the text and assemble**: It feeds the masked page to a local Vision LLM to transcribe the text, headers, and exercise numbers. Because the notes were wiped out, the model doesn't hallucinate music. Finally, it swaps the placeholders for playable ABC code blocks and vector SVGs.

Everything runs sequentially. The music model unloads before the text model loads, keeping peak VRAM under **~6.2 GB**.

---

## Quick Start (Windows)

You don't need to configure environment variables, edit JSON files, or compile C++ libraries.

1. **Download the project**:
   - Clone via git: `git clone https://github.com/greenyamao/solfeggio2md.git`
   - *Or click **Code ➔ Download ZIP** on GitHub and extract it.*
2. **Install**:
   - Double-click **`install.bat`** (it creates `.venv`, installs PyTorch with CUDA, and pulls dependencies).
   - *(If you forget, just double-click **`start.bat`** — it will offer to run the installer for you).*
3. **Launch**:
   - Double-click **`start.bat`**.

### Setting up Text OCR

The app needs a Vision model to read the non-music text on the page:

- **Option A (Easiest — via LM Studio)**:
  If you already use [LM Studio](https://lmstudio.ai/):
  1. Load a vision model (e.g. `Qwen2.5-VL-7B-Instruct` or `Qwen3.5-9B`).
  2. Click **Start Server** on port `1234`.
  3. Solfeggio OCR Studio connects to it automatically.

- **Option B (Standalone / Offline)**:
  Open the app, go to the **Settings** tab in the sidebar, and click **Download Model**. It will fetch the GGUF model and vision projector from Hugging Face into `models/`.

> [!NOTE]
> All other model weights (YOLO layout detector 38 MB, Transcoda OMR, SMT) download automatically on their first run. You don't need to hunt down files manually.

---

## Linux & Advanced Setup

For Linux workstations or headless servers:

```bash
# 1. Clone & enter
git clone https://github.com/greenyamao/solfeggio2md.git
cd solfeggio2md

# 2. Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. PyTorch with CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Dependencies
pip install -r requirements.txt

# 5. Launch GUI or CLI
python run_native_app.py
```

---

## How It Works Under the Hood

### 1. Ingestion, 2D-FFT Deskew & Layout
- **Spread splitting**: Detects 2-page spreads (aspect ratio between 1.22 and 2.2) and finds the spine fold by looking for the minimum ink column in the gutter, cutting it into left and right pages.
- **Deskew via 2D-FFT**: Staff lines create strong horizontal frequency spikes. A 2D-RFFT on the GPU calculates the rotation angle in ~1.3 ms, straightening the image so staff lines are level.
- **YOLO OLA v2.0**: Detects single staves, piano accolades (`grand_staff`), and titles.
- **Collinear Healing**: Reconnects stave segments that were split across columns if they share the same horizontal baseline.
- **Masking**: Erases staves on the full-page image with white rectangles containing `<!-- MUSIC_STUB_ID:P0001_S01 -->`.

### 2. Optical Music Recognition (OMR)
- **Score Enhancer**: Runs a lightweight Real-CUGAN 2x Cascaded U-Net and GPU background division to clean up scan yellowing and reconnect broken 1-pixel staff lines without hallucinations.
- **Model routing**: Single staves go to `Transcoda-59M` (`btrkeks/transcoda-59M-zeroshot-v1`); piano accolades go to `SMT-GrandStaff` (`antoniorv6/smt-grandstaff`).
- Output is generated in Humdrum `**kern` format.

### 3. Validation & ABC Conversion
Raw neural OMR output can make rhythmic mistakes or loop. Before converting, the engine validates:
- **Measure math**: Sums note and rest durations per measure against the time signature ($4/4$, $3/4$, $6/8$).
- **Spine balancing**: Ensures split (`*^`) and merge (`*v`) columns stay aligned so the C++ parser doesn't crash.
- **Loop guards**: Detects repeating token loops and unrealistic note densities.
- **C++ Verovio Core**: Validates the Humdrum syntax and compiles it to MusicXML in ~25 ms.
- **Willem Vree `xml2abc`**: Converts MusicXML into polyphonic ABC notation with separated voices (`V:1 treble`, `V:2 bass`) without dropping notes.

### 4. Text OCR & Markdown Assembly
- **VLM pass**: Qwen Vision receives the masked page image. Because the notes are gone, it focuses 100% on textbook text, exercise numbers, and headers, copying the `MUSIC_STUB_ID` tags into the text verbatim.
- **Assembly**: Replaces each stub tag with the validated ABC notation block and vector SVG.
- **Front/Back matter**: Automatically detects Roman numeral pages (prefaces, indices) and skips the music stage when no staves exist.

### 5. Memory Management (The 8 GB VRAM Rule)
Running OMR and a Vision LLM at the same time on an 8 GB card causes immediate CUDA out-of-memory errors. The pipeline separates them into sequential batch passes:

| Phase | What runs | Typical VRAM |
| :--- | :--- | :---: |
| **Phase 1: Layout & Deskew** | YOLO OLA + PyTorch FFT | ~1.4 GB |
| *Purge* | Model unloaded, `torch.cuda.empty_cache()` | 0 GB |
| **Phase 2: Music OMR** | Transcoda-59M / SMT + Real-CUGAN | ~3.8 GB |
| *Purge* | Model unloaded, `gc.collect()` | 0 GB |
| **Phase 3: Validation** | C++ Verovio + xml2abc | 0 GB (CPU only) |
| **Phase 4: Text OCR** | Qwen-VL via llama-server / LM Studio | ~6.2 GB |

---

## Desktop GUI Overview

Built with **PySide6** and **QFluentWidgets**:
- **Zero-scroll dashboard**: Everything fits cleanly in a single screen without nested scrolling.
- **Zero-copy shared memory**: Uses Windows named shared memory blocks (`SharedMemoryFrameReader`) to stream 60 FPS page renders without freezing the UI thread.
- **Telemetry Ribbon**: Shows real-time CPU, RAM, GPU Compute %, and aggregate VRAM (combining the Python worker and background llama-server).
- **Page Inspector**:
  - *Full Page Layout*: YOLO bounding boxes and labels.
  - *Deskew Preview*: Scan crop alongside straightened staff lines.
  - *Vector Score & ABC*: Original crop side-by-side with Verovio vector SVG, ABC code, and quality validation cards.
- **SSD wear protection**: When `keep_intermediate_files: false` (default), temporary crop images are cleaned up after assembly, saving gigabytes of disk writes.

---

## CLI & Diagnostic Commands

You can run the pipeline or inspect individual pages from the command line:

```bash
# Run batch conversion on PDFs in the in/ directory
python -m core.pipeline_batch_runner

# Quick 1-line status audit of a book
python -m tools.debug_toolkit audit --book "in/Counterpoint.pdf"

# Inspect a single page and save debug images to scratch/
python -m tools.debug_toolkit page --book "in/Counterpoint.pdf" --page 25 --render

# Scan a page range for split staves or layout issues
python -m tools.debug_toolkit scan --book "in/Harmony.pdf" --range 1-50

# Generate an interactive HTML report comparing crops with vector SVG renders
python -m tools.debug_toolkit report --book "in/Counterpoint.pdf" --range 1-10 --open

# Run the 117-test regression suite
python tests/run_tests.py
```

---

## Configuration (`config.json`)

The default configuration is already tuned for 8 GB GPUs. If you want to tweak settings, you can do so in the GUI **Settings** tab or directly in `config.json`:

```json
{
  "vlm_backend": "embedded",
  "vlm_model_file": "Qwen3.5-9B-Q4_K_M.gguf",
  "vlm_mmproj_file": "mmproj-Qwen3.5-9B-BF16.gguf",
  "vlm_embedded_port": 1234,
  "dpi": 200,
  "enable_score_enhancer": true,
  "enable_cugan_sr": true,
  "keep_intermediate_files": false,
  "overwrite": false
}
```

---

## Acknowledgments & Third-Party Code

This project relies on several open-source models and libraries:

- **[Sheet Music Transformer (SMT)](https://github.com/antoniorv6/SMT)**: Dual-staff piano accolade architecture in `core/smt_model/` by Antonio Ríos-Vila (MIT License, see [core/smt_model/LICENSE](core/smt_model/LICENSE)).
- **[OMR Layout Analysis (OLA v2.0)](https://github.com/v-dvorak/omr-layout-analysis)**: YOLO layout model by Charles University.
- **[Transcoda-59M](https://huggingface.co/btrkeks/transcoda-59M-zeroshot-v1)**: Zero-shot single-staff encoder-decoder model by btrkeks.
- **[Verovio](https://www.verovio.org/)**: C++ music engraving and validation engine (LGPL).
- **[xml2abc](https://github.com/sebastian-eck/abc_xml_converter)**: MusicXML to ABC conversion by Willem Vree.
- **Real-CUGAN**: Line-art super-resolution architecture adapted for clean staff line restoration.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
Vendored subcomponents retain their respective licenses (e.g. SMT under MIT).