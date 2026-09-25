# Solfeggio OCR Studio

Production-grade pipeline for converting music theory textbooks, solfeggio exercises, and polyphonic sheet music into hybrid Markdown with embedded polyphonic ABC notation.

---

## Quick Start

Launch via the batch runner:
```bat
start.bat
```
Or execute directly using the virtual environment:
```powershell
.\.venv\Scripts\python.exe run_native_app.py
```

---

## Architecture & Technology Stack

- **Desktop GUI**: Native Windows 11 Fluent interface built with PySide6 and QFluentWidgets, featuring zero-copy shared memory frame streaming and sub-millisecond IPC.
- **Layout Analysis & Slicing**: YOLO OLA v2.0 (`weights/ola-layout-analysis-2.0-2025-03-09.pt`) with dynamic collinear segment healing (`heal_collinear_segments`) and automatic two-page spread bisection.
- **Optical Music Recognition (OMR)**: Routed neural engine using `Transcoda-59M` (`btrkeks/transcoda-59M-zeroshot-v1`) for single staves and `SMT-GrandStaff` for piano systems.
- **Notation Validation & Export**: C++ `verovio` core coupled with `xml2abc.py` for pristine polyphonic ABC notation export (`V:1 treble`, `V:2 bass`).
- **Vision-Language Model (VLM)**: Embedded `llama-server` or local LM Studio server running Qwen vision models via REST API (`POST /v1/chat/completions`) with strict masking.
- **Fault-Tolerant Checkpoints**: Atomic `checkpoint.json` per document directory ensures zero-overhead idempotency and crash resilience across all 4 pipeline phases.

---

## Requirements & Environment

- Python 3.12 (inside `.venv` virtual environment).
- `opencv-python-headless` (never install standard `opencv-python`).
- GPU: CUDA-compatible NVIDIA GPU (8 GB+ VRAM recommended). Strictly enforces Sequential GPU Ownership with memory purge barriers between batch passes.