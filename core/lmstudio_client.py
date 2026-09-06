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
        """
        payload = {
            "model": model_name,
            "context_length": int(context_length),
            "eval_batch_size": int(eval_batch_size),
            "flash_attention": bool(flash_attention),
            "offload_kv_cache_to_gpu": bool(offload_kv_cache),
            "echo_load_config": True,
        }
        return self._post("/api/v1/models/load", payload, timeout=180)

    def unload_model(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """
        Unloads model instance from VRAM.
        """
        if not instance_id:
            return None
        try:
            return self._post("/api/v1/models/unload", {"instance_id": instance_id}, timeout=60)
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
        payload = {
            "model": model_name,
            "system_prompt": system_prompt,
            "input": [
                {"type": "text", "content": "Распознай печатный текст на этой странице."},
                {"type": "image", "data_url": f"data:image/jpeg;base64,{b64}"},
            ],
            "reasoning": "off",
            "temperature": float(temperature),
            "max_output_tokens": int(max_tokens),
            "context_length": int(context_length),
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
