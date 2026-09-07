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

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.abc_bridge import ABCBridge


class OMREngine:
    def __init__(
        self,
        device: str = "cpu",
        transcoda_repo: str = "btrkeks/transcoda-59M-zeroshot-v1",
        smt_repo: str = "antoniorv6/smt-grandstaff"
    ):
        self.device = device
        self.transcoda_repo = transcoda_repo
        self.smt_repo = smt_repo
        
        self.transcoda_model = None
        self.transcoda_tokenizer = None
        self.smt_model = None
        
        self.bridge = ABCBridge()

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

    def _ensure_smt_loaded(self):
        if self.smt_model is None:
            from SMT.smt_model import SMTModelForCausalLM
            self.smt_model = SMTModelForCausalLM.from_pretrained(self.smt_repo).to(self.device).eval()

    def _preprocess_transcoda(self, img_bgr: np.ndarray, target_w: int = 1050, target_h: int = 1485) -> torch.Tensor:
        """
        Transcoda expects normalized RGB float32 in [-1, 1] of shape (1, 3, target_h, target_w).
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        new_h = max(1, int(pil_img.height * (target_w / pil_img.width)))
        pil_resized = pil_img.resize((target_w, new_h), Image.BILINEAR)
        arr = np.array(pil_resized)
        
        if arr.shape[0] > target_h:
            arr = arr[:target_h]
        elif arr.shape[0] < target_h:
            pad = np.full((target_h - arr.shape[0], target_w, 3), 255, dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=0)
            
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        t = (t - 0.5) / 0.5
        return t.unsqueeze(0).to(self.device)

    def _collate_crops_batch(self, crops_bgr: List[np.ndarray], target_w: int = 1050) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dynamically resizes and collates a mini-batch of crops to the maximum height in the batch,
        rounded up to multiples of 32 for ConvNeXt, capped at 1485.
        Returns (pixel_values, image_sizes).
        """
        batch_size = len(crops_bgr)
        resized_crops = []
        item_heights = []

        for crop in crops_bgr:
            h, w = crop.shape[:2]
            scale = target_w / float(max(1, w))
            new_h = max(1, int(round(h * scale)))
            img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_resized = pil_img.resize((target_w, new_h), Image.BILINEAR)
            resized_crops.append(np.array(pil_resized))
            item_heights.append(new_h)

        max_h = max(item_heights) if item_heights else 32
        batch_h = min(1485, int(np.ceil(max_h / 32.0) * 32))

        batch_arr = np.full((batch_size, batch_h, target_w, 3), 255, dtype=np.uint8)
        for i, arr in enumerate(resized_crops):
            clip_h = min(arr.shape[0], batch_h)
            batch_arr[i, :clip_h, :, :] = arr[:clip_h, :, :]

        tensor = torch.from_numpy(batch_arr).permute(0, 3, 1, 2)
        is_cuda = self.device == "cuda" or "cuda" in str(self.device).lower()
        dtype = torch.float16 if (is_cuda and torch.cuda.is_available()) else torch.float32
        pixel_values = ((tensor.to(dtype=dtype) / 255.0) - 0.5) / 0.5
        pixel_values = pixel_values.to(self.device)

        image_sizes = torch.tensor([[h, target_w] for h in item_heights], device=self.device)
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

        try:
            self._ensure_transcoda_loaded()
            pixel_values, image_sizes = self._collate_crops_batch(crops_bgr)

            with torch.inference_mode():
                out = self.transcoda_model.generate(
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    max_length=max_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.1
                )

            decoded_kerns = self.transcoda_tokenizer.batch_decode(out, skip_special_tokens=True)
            results = []
            for raw_k, tit in zip(decoded_kerns, titles):
                try:
                    norm_k = self.bridge.normalize_humdrum(raw_k)
                    abc_str = self.bridge.kern_to_abc(norm_k, title=tit)
                    results.append({
                        "abc": abc_str,
                        "raw_kern": norm_k,
                        "model_used": "transcoda-59M",
                        "status": "success"
                    })
                except Exception as conv_err:
                    results.append({
                        "abc": f"% [OMR Conversion Error: {conv_err}]",
                        "raw_kern": raw_k,
                        "model_used": "transcoda-59M",
                        "status": "error",
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
