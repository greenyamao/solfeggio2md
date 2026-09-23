"""
Client for LM Studio local REST API (OpenAI-compatible and native endpoints).
Handles connection health, model loading/unloading, and vision-language OCR requests.
"""

import base64
import json
import re
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List, Tuple


class LMStudioClient:
    """
    Communicates with local LM Studio instance for Vision-Language Models (e.g. Qwen 2.5-VL / Qwen 3.5 9B).
    """

    def __init__(self, host: str = "127.0.0.1", port: str = "1234", timeout: int = 900):
        self.host = host.strip()
        self.port = str(port).strip()
        self.timeout = timeout
        self.active_instance_id: Optional[str] = None
        self.was_loaded_by_client: bool = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _post(self, path: str, payload: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        req_timeout = timeout or self.timeout
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=req_timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get(self, path: str, timeout: int = 15) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, headers={"Content-Type": "application/json"}, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def check_connection(self) -> Tuple[bool, str]:
        """
        Checks if LM Studio server is reachable.
        Returns: (is_online, message)
        """
        try:
            res = self._get("/api/v1/models", timeout=5)
            models = res.get("models", [])
            return True, f"Сервер доступен (моделей в списке: {len(models)})"
        except urllib.error.URLError as e:
            return False, f"Ошибка подключения к {self.base_url}: {e.reason}"
        except Exception as e:
            return False, f"Не удалось связаться с сервером: {str(e)}"

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Returns list of all available models in LM Studio.
        """
        try:
            res = self._get("/api/v1/models", timeout=10)
            return res.get("models", [])
        except Exception:
            return []

    def get_loaded_instance(self, model_name: Optional[str] = None) -> Optional[str]:
        """
        Returns the instance_id if model_name (or any active model) is already loaded.
        Prevents spawning duplicate model instances in LM Studio VRAM.
        """
        try:
            models = self.list_models()
            target = (model_name or "").strip().lower()

            if target and target != "default":
                for m in models:
                    key = str(m.get("key", "")).strip().lower()
                    display = str(m.get("display_name", "")).strip().lower()
                    instances = m.get("loaded_instances", [])
                    if instances:
                        inst_ids = [str(inst.get("id", "")).strip().lower() for inst in instances]
                        if target == key or target == display or any(target == iid for iid in inst_ids):
                            return instances[0].get("id") or m.get("key")
                        if target in key or key in target:
                            return instances[0].get("id") or m.get("key")

            for m in models:
                instances = m.get("loaded_instances", [])
                if instances:
                    return instances[0].get("id") or m.get("key")
        except Exception:
            pass
        return None

    def load_model(
        self,
        model_name: str,
        context_length: int = 16196,
        eval_batch_size: int = 2048,
        flash_attention: bool = True,
        offload_kv_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Loads specified model into VRAM via native LM Studio API.
        Idempotent: if model is already loaded, reuses existing instance without spawning duplicates.
        """
        existing_inst = self.get_loaded_instance(model_name)
        if existing_inst:
            self.active_instance_id = existing_inst
            return {"instance_id": existing_inst, "status": "already_loaded"}

        payload = {
            "model": model_name,
            "context_length": int(context_length),
            "eval_batch_size": int(eval_batch_size),
            "flash_attention": bool(flash_attention),
            "offload_kv_cache_to_gpu": bool(offload_kv_cache),
            "echo_load_config": True,
        }
        res = self._post("/api/v1/models/load", payload, timeout=180)
        self.active_instance_id = res.get("instance_id") or model_name
        self.was_loaded_by_client = True
        return res

    def unload_model(self, instance_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Unloads model instance from VRAM.
        If instance_id is None, unloads the active instance.
        """
        target_id = instance_id or self.active_instance_id
        if not target_id:
            return None
        try:
            res = self._post("/api/v1/models/unload", {"instance_id": target_id}, timeout=60)
            if target_id == self.active_instance_id:
                self.active_instance_id = None
                self.was_loaded_by_client = False
            return res
        except Exception:
            return None

    def request_ocr(
        self,
        image_bytes: bytes,
        system_prompt: str,
        model_name: str = "default",
        temperature: float = 0.1,
        max_tokens: int = 8192,
        context_length: int = 16196,
    ) -> str:
        """
        Sends a masked book page image to the VLM with strict parameters.
        Forces reasoning: "off" to avoid token waste and latency.
        """
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        # Always route to the active or already loaded instance to prevent LM Studio JIT duplication
        target_model = self.active_instance_id or self.get_loaded_instance(model_name) or model_name or "default"
        payload = {
            "model": target_model,
            "system_prompt": system_prompt,
            "input": [
                {"type": "text", "content": "Распознай печатный текст на этой странице."},
                {"type": "image", "data_url": f"data:image/jpeg;base64,{b64}"},
            ],
            "reasoning": "off",
            "temperature": float(temperature),
            "max_output_tokens": int(max_tokens),
            "stream": False,
            "store": False,
        }

        result = self._post("/api/v1/chat", payload)
        output = [item.get("content", "") for item in result.get("output", []) if item.get("type") == "message"]
        raw_text = "\n".join(output).strip()

        # Clean outer markdown fences if returned
        return self._strip_markdown_fences(raw_text)

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```markdown"):
            cleaned = cleaned[11:].strip()
        elif cleaned.startswith("```md"):
            cleaned = cleaned[5:].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        return cleaned
