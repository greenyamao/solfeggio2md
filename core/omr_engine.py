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
from typing import Dict, Any, Optional, Tuple

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
            self.transcoda_model = AutoModelForCausalLM.from_pretrained(
                self.transcoda_repo,
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

    def transcribe_crop(
        self,
        crop_bgr: np.ndarray,
        notation_class: str = "staff",
        title: Optional[str] = None,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        """
        Routes crop to the optimal model:
        - 'staff': Transcoda-59M
        - 'grand_staff': SMT-GrandStaff (fallback to Transcoda)
        - 'system': Transcoda-59M
        Returns dictionary with:
            { 'abc': str, 'raw_kern': str, 'model_used': str, 'status': 'success' | 'error' }
        """
        cls_canonical = notation_class.lower().replace(" ", "_")
        
        # 1. Route to SMT for grand staff
        if cls_canonical in ("grand_staff", "grandstaff"):
            try:
                self._ensure_smt_loaded()
                from SMT.data_augmentation.data_augmentation import convert_img_to_tensor
                tensor = convert_img_to_tensor(crop_bgr, target_height=getattr(self.smt_model.config, "maxh", crop_bgr.shape[0]))
                tensor = tensor.unsqueeze(0).to(self.device)
                
                with torch.inference_mode():
                    preds, _ = self.smt_model.predict(
                        tensor,
                        convert_to_str=True,
                        max_tokens=max_tokens
                    )
                raw_kern = "".join(preds).replace("<s>", " ").replace("</s>", "").replace("<t>", "\t").replace("<b>", "\n")
                raw_kern = self.bridge.normalize_humdrum(raw_kern)
                abc = self.bridge.kern_to_abc(raw_kern, title=title)
                return {
                    "abc": abc,
                    "raw_kern": raw_kern,
                    "model_used": "smt-grandstaff",
                    "status": "success"
                }
            except Exception as smt_err:
                # Fallback to Transcoda if SMT encounters an edge case
                pass

        # 2. Transcoda for single staves, systems, or fallback
        try:
            self._ensure_transcoda_loaded()
            pixel_values = self._preprocess_transcoda(crop_bgr)
            image_sizes = torch.tensor([[1485, 1050]], device=self.device)
            
            with torch.inference_mode():
                out = self.transcoda_model.generate(
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    max_length=max_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.1
                )
                
            raw_kern = self.transcoda_tokenizer.decode(out[0], skip_special_tokens=True)
            raw_kern = self.bridge.normalize_humdrum(raw_kern)
            abc = self.bridge.kern_to_abc(raw_kern, title=title)
            return {
                "abc": abc,
                "raw_kern": raw_kern,
                "model_used": "transcoda-59M",
                "status": "success"
            }
        except Exception as e:
            return {
                "abc": f"% [OMR Error: {e}]",
                "raw_kern": "",
                "model_used": "error",
                "status": "error",
                "error": str(e)
            }

    def purge_gpu_memory(self):
        """
        MANDATORY SEQUENTIAL GPU OWNERSHIP PROTOCOL:
        Unloads all OMR and Vision models from VRAM and triggers full garbage collection
        before Phase 2 (7GB VLM in LM Studio) begins.
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
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
