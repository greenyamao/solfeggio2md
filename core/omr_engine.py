"""
Routed Multi-Model OMR Engine:
- Transcoda-59M: SOTA vision-encoder-decoder for single-staff melodies (solfeggio, vocal lines) and general zero-shot OMR.
- SMT-GrandStaff: Sheet Music Transformer for dual-staff piano accolades.
- ABCBridge: Verovio C++ validation + Willem Vree xml2abc translation.
- Full GPU memory purge protocol before VLM handoff.
"""

import gc
from pathlib import Path
import sys
from typing import Dict, Any, Optional, Tuple, List

import os
import re
import logging
import warnings
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from transformers import logging as transformers_logging

# Silence noisy third-party logging
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()
logging.getLogger("torch").setLevel(logging.ERROR)
try:
    from loguru import logger
    logger.disable("transformers_modules")
except ImportError:
    pass

ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.abc_bridge import ABCBridge
from core.score_enhancer import ScoreEnhancer


class OMREngine:
    def __init__(
        self,
        device: str = "cpu",
        transcoda_repo: str = "btrkeks/transcoda-59M-zeroshot-v1",
        smt_repo: str = "antoniorv6/smt-grandstaff",
        enable_score_enhancer: bool = True,
        enable_cugan: bool = True,
    ):
        self.device = device
        self.transcoda_repo = transcoda_repo
        self.smt_repo = smt_repo
        self.enable_score_enhancer = enable_score_enhancer
        self.enable_cugan = enable_cugan
        
        self.transcoda_model = None
        self.transcoda_tokenizer = None
        self.smt_model = None
        
        self.bridge = ABCBridge()
        self.enhancer = ScoreEnhancer(device=self.device, enable_cugan=self.enable_cugan)

    def _ensure_transcoda_loaded(self):
        if self.transcoda_model is None:
            is_cuda = self.device == "cuda" or "cuda" in str(self.device).lower()
            load_dtype = torch.float16 if (is_cuda and torch.cuda.is_available()) else torch.float32
            self.transcoda_model = AutoModelForCausalLM.from_pretrained(
                self.transcoda_repo,
                dtype=load_dtype,
                trust_remote_code=True
            ).to(self.device).eval()
            self.transcoda_tokenizer = PreTrainedTokenizerFast.from_pretrained(self.transcoda_repo)
            eos_ids = [self.transcoda_tokenizer.eos_token_id or 2]
            for bar_tok in ["==", "=||", "=:|", "=:|!"]:
                tok_id = self.transcoda_tokenizer.convert_tokens_to_ids(bar_tok)
                if tok_id is not None and tok_id != self.transcoda_tokenizer.unk_token_id:
                    eos_ids.append(tok_id)
            self.transcoda_eos_token_ids = list(dict.fromkeys(eos_ids))
            # Permanently suppress lute/guitar tablature tokens on standard notation
            self.transcoda_bad_words_ids = [[1694], [2093], [120, 40], [1693]]

    def _ensure_smt_loaded(self):
        if self.smt_model is None:
            from SMT.smt_model import SMTModelForCausalLM
            self.smt_model = SMTModelForCausalLM.from_pretrained(self.smt_repo).to(self.device).eval()

    def _preprocess_transcoda(self, img_bgr: np.ndarray, target_w: int = 1050, target_h: int = 1485) -> torch.Tensor:
        """
        Transcoda expects normalized RGB float32 in [-1, 1] of shape (1, 3, target_h, target_w).
        Uses pure PyTorch GPU bilinear interpolation to eliminate CPU bottlenecks.
        """
        h, w = img_bgr.shape[:2]
        new_h = max(1, int(round(h * (target_w / float(max(1, w))))))
        clip_h = min(new_h, target_h)

        is_cuda = self.device == "cuda" or "cuda" in str(self.device).lower()
        dev = self.device if (is_cuda and torch.cuda.is_available()) else "cpu"
        dtype = torch.float32

        pixel_values = torch.ones((1, 3, target_h, target_w), dtype=dtype, device=dev)
        img_rgb = img_bgr[:, :, ::-1].copy()
        t = torch.from_numpy(img_rgb).to(device=dev, dtype=dtype).permute(2, 0, 1).unsqueeze(0)
        t = ((t / 255.0) - 0.5) / 0.5
        resized = F.interpolate(t, size=(new_h, target_w), mode="bilinear", align_corners=False)
        pixel_values[0, :, :clip_h, :] = resized[0, :, :clip_h, :]
        return pixel_values

    def _collate_crops_batch(
        self,
        crops_bgr: List[np.ndarray],
        target_w: int = 1050,
        notation_classes: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dynamically resizes and collates a mini-batch of crops preserving natural aspect ratio
        and staff line height, rounded up to multiples of 32 for ConvNeXt, capped at 1485.
        Uses pure PyTorch GPU bilinear interpolation to eliminate CPU bottlenecks.
        Masks out white padding via image_sizes (valid_h, valid_w).
        """
        batch_size = len(crops_bgr)
        item_dims = []
        classes = notation_classes or (["staff"] * batch_size)

        for crop, cls_name in zip(crops_bgr, classes):
            h, w = crop.shape[:2]
            if w >= 800:
                scale = min(1.0, target_w / float(max(1, w)))
            else:
                # For narrower snippets or single staves, preserve natural staff line spacing:
                is_grand = "grand" in str(cls_name).lower() or h > 150
                target_staff_h = 80.0 if is_grand else 40.0
                scale = min(1.0, target_staff_h / float(max(1, h))) if h > 50 else 1.0
                if w * scale > target_w:
                    scale = target_w / float(max(1, w))
            new_h = max(1, int(round(h * scale)))
            new_w = min(target_w, max(1, int(round(w * scale))))
            item_dims.append((new_h, new_w))

        max_h = max(d[0] for d in item_dims) if item_dims else 32
        batch_h = min(1485, max(32, int(np.ceil(max_h / 32.0) * 32)))

        is_cuda = self.device == "cuda" or "cuda" in str(self.device).lower()
        dev = self.device if (is_cuda and torch.cuda.is_available()) else "cpu"
        dtype = torch.float16 if (is_cuda and torch.cuda.is_available()) else torch.float32

        # Pure white background (255) normalized into Transcoda space: ((255 / 255.0) - 0.5) / 0.5 = 1.0
        pixel_values = torch.ones((batch_size, 3, batch_h, target_w), dtype=dtype, device=dev)

        for i, crop in enumerate(crops_bgr):
            new_h, new_w = item_dims[i]
            crop_rgb = crop[:, :, ::-1].copy()
            t = torch.from_numpy(crop_rgb).to(device=dev, dtype=dtype).permute(2, 0, 1).unsqueeze(0)
            t = ((t / 255.0) - 0.5) / 0.5
            resized = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
            clip_h = min(new_h, batch_h)
            pixel_values[i, :, :clip_h, :new_w] = resized[0, :, :clip_h, :]

        image_sizes = torch.tensor([[d[0], d[1]] for d in item_dims], device=dev)
        return pixel_values, image_sizes

    def transcribe_crops_batch(
        self,
        crops_bgr: List[np.ndarray],
        notation_classes: Optional[List[str]] = None,
        titles: Optional[List[str]] = None,
        max_tokens: int = 512
    ) -> List[Dict[str, Any]]:
        """
        Batched inference for mini-batches of crops (typically 4-8 crops) via Transcoda-59M.
        """
        if not crops_bgr:
            return []

        count = len(crops_bgr)
        if notation_classes is None:
            notation_classes = ["staff"] * count
        if titles is None:
            titles = [None] * count

        # Height-aware partitioning: avoid mixing single staves (h < 90) with multi-staff systems (h >= 90)
        # in the same mini-batch to eliminate massive blank padding degradation.
        if count > 1 and any(c.shape[0] < 90 for c in crops_bgr) and any(c.shape[0] >= 90 for c in crops_bgr):
            idx_small = [i for i, c in enumerate(crops_bgr) if c.shape[0] < 90]
            idx_tall = [i for i, c in enumerate(crops_bgr) if c.shape[0] >= 90]

            res_small = self.transcribe_crops_batch(
                [crops_bgr[i] for i in idx_small],
                [notation_classes[i] for i in idx_small],
                [titles[i] for i in idx_small],
                max_tokens=max_tokens
            )
            res_tall = self.transcribe_crops_batch(
                [crops_bgr[i] for i in idx_tall],
                [notation_classes[i] for i in idx_tall],
                [titles[i] for i in idx_tall],
                max_tokens=max_tokens
            )
            combined = [None] * count
            for idx, r in zip(idx_small, res_small):
                combined[idx] = r
            for idx, r in zip(idx_tall, res_tall):
                combined[idx] = r
            return combined

        try:
            self._ensure_transcoda_loaded()

            # Record original widths for music density token bounding without artificial early cutoffs
            widest_crop_px = max(c.shape[1] for c in crops_bgr) if crops_bgr else 1000
            effective_max_tokens = min(int(max_tokens), max(160, int(widest_crop_px * 0.35)))

            # Neural stroke restoration & GPU background division (avoiding double-SR if already 2x)
            if self.enable_score_enhancer and self.enhancer is not None:
                run_sr = self.enable_cugan and any(c.shape[1] < 1600 for c in crops_bgr)
                if hasattr(self.enhancer, "enhance_crops_batch"):
                    processed_crops = self.enhancer.enhance_crops_batch(crops_bgr, run_sr=run_sr)
                else:
                    processed_crops = [
                        self.enhancer.enhance_crop(c, run_sr=(self.enable_cugan and c.shape[1] < 1600))
                        for c in crops_bgr
                    ]
            else:
                processed_crops = crops_bgr

            pixel_values, image_sizes = self._collate_crops_batch(processed_crops, notation_classes=notation_classes)

            with torch.inference_mode():
                eos_ids = getattr(self, "transcoda_eos_token_ids", [2, 212, 236, 155, 156])
                bad_words_ids = getattr(self, "transcoda_bad_words_ids", None)
                out = self.transcoda_model.generate(
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    max_length=effective_max_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.15,
                    eos_token_id=eos_ids,
                    bad_words_ids=bad_words_ids
                )

            decoded_kerns = self.transcoda_tokenizer.batch_decode(out, skip_special_tokens=True)
            results = []
            for raw_k, tit in zip(decoded_kerns, titles):
                clean_k = self._sanitize_runaway_kern(raw_k)
                try:
                    norm_k = self.bridge.normalize_humdrum(clean_k)
                    abc_str = self.bridge.kern_to_abc(norm_k, title=tit)
                    results.append({
                        "abc": abc_str,
                        "raw_kern": norm_k,
                        "model_used": "transcoda-59M",
                        "status": "success"
                    })
                except Exception as conv_err:
                    fallback_abc = f"X:1\n{('T:' + tit + chr(10)) if tit else ''}L:1/4\nM:none\nI:linebreak $\nK:C\nz4 |"
                    results.append({
                        "abc": fallback_abc,
                        "raw_kern": clean_k,
                        "model_used": "transcoda-59M",
                        "status": "warning",
                        "error": str(conv_err)
                    })
            return results
        except Exception as e:
            return [
                {
                    "abc": f"% [OMR Batch Error: {e}]",
                    "raw_kern": "",
                    "model_used": "error",
                    "status": "error",
                    "error": str(e)
                }
                for _ in crops_bgr
            ]

    def transcribe_crop(
        self,
        crop_bgr: np.ndarray,
        notation_class: str = "staff",
        title: Optional[str] = None,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Single-crop wrapper routing to transcribe_crops_batch.
        Returns dictionary with:
            { 'abc': str, 'raw_kern': str, 'model_used': str, 'status': 'success' | 'error' }
        """
        results = self.transcribe_crops_batch(
            crops_bgr=[crop_bgr],
            notation_classes=[notation_class],
            titles=[title],
            max_tokens=max_tokens
        )
        return results[0] if results else {
            "abc": "% [OMR Error: Empty batch]",
            "raw_kern": "",
            "model_used": "error",
            "status": "error",
            "error": "Empty batch"
        }

    @staticmethod
    def _sanitize_runaway_kern(raw_kern: str, max_repeats: int = 2) -> str:
        """
        Detects and truncates degenerative runaway loops and attention wrap-arounds in Humdrum **kern.
        Delegates directly to ABCBridge.sanitize_runaway_kern.
        """
        from core.abc_bridge import ABCBridge
        return ABCBridge.sanitize_runaway_kern(raw_kern, max_repeats=max_repeats)

    @staticmethod
    def sanitize_raw_tokens(raw_kern: str, max_repeats: int = 2) -> str:
        """Public static alias for runaway loop and repeat sanitization."""
        from core.abc_bridge import ABCBridge
        return ABCBridge.sanitize_runaway_kern(raw_kern, max_repeats=max_repeats)

    def purge_gpu_memory(self):
        """
        MANDATORY SEQUENTIAL GPU OWNERSHIP PROTOCOL:
        Unloads all OMR and Vision models from VRAM and triggers full garbage collection
        before Phase 3 (7GB VLM in LM Studio) begins.
        """
        if self.transcoda_model is not None:
            del self.transcoda_model
            self.transcoda_model = None
        if self.transcoda_tokenizer is not None:
            del self.transcoda_tokenizer
            self.transcoda_tokenizer = None
        if self.smt_model is not None:
            del self.smt_model
            self.smt_model = None
        if getattr(self, "enhancer", None) is not None:
            self.enhancer.purge_gpu_memory()
            
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
