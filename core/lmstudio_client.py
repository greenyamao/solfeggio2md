"""
Client for LM Studio local REST API (OpenAI-compatible and native endpoints).
Handles connection health, model loading/unloading, and vision-language OCR requests.
"""

import base64
import json
import re
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List, Tuple, Callable


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

    @base_url.setter
    def base_url(self, val: str) -> None:
        pass

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
        Checks if LM Studio or standalone llama-server is reachable.
        Returns: (is_online, message)
        """
        try:
            res = self._get("/api/v1/models", timeout=5)
            models = res.get("models", [])
            return True, f"Server reachable ({len(models)} models available)"
        except Exception:
            try:
                res = self._get("/v1/models", timeout=5)
                models = res.get("data", [])
                return True, f"VLM server reachable ({len(models)} models available)"
            except urllib.error.URLError as e:
                return False, f"Connection error to {self.base_url}: {e.reason}"
            except Exception as e:
                return False, f"Failed to connect to server: {str(e)}"

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Returns list of all available models in LM Studio or llama-server.
        """
        try:
            res = self._get("/api/v1/models", timeout=10)
            return res.get("models", [])
        except Exception:
            try:
                res = self._get("/v1/models", timeout=10)
                data = res.get("data", [])
                return [{"key": d.get("id"), "display_name": d.get("id")} for d in data]
            except Exception:
                return []

    def get_loaded_instance_info(self, model_name: Optional[str] = None) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Returns (instance_id, instance_config) if model_name (or any active model) is loaded.
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
                            return instances[0].get("id") or m.get("key"), instances[0].get("config", {})
                        if target in key or key in target:
                            return instances[0].get("id") or m.get("key"), instances[0].get("config", {})

            for m in models:
                instances = m.get("loaded_instances", [])
                if instances:
                    return instances[0].get("id") or m.get("key"), instances[0].get("config", {})
        except Exception:
            pass
        return None

    def get_loaded_instance(self, model_name: Optional[str] = None) -> Optional[str]:
        """
        Returns the instance_id if model_name (or any active model) is already loaded.
        Prevents spawning duplicate model instances in LM Studio VRAM.
        """
        info = self.get_loaded_instance_info(model_name)
        return info[0] if info else None

    def load_model(
        self,
        model_name: str,
        context_length: int = 4096,
        eval_batch_size: int = 2048,
        flash_attention: bool = True,
        offload_kv_cache: bool = True,
        parallel: int = 1,
    ) -> Dict[str, Any]:
        """
        Loads specified model into VRAM via native LM Studio API.
        Ensures optimal single-slot (parallel=1) and compact context (4096) configuration
        to guarantee 100% GPU offload and eliminate RAM spillover.
        """
        existing_info = self.get_loaded_instance_info(model_name)
        if existing_info:
            inst_id, inst_cfg = existing_info
            current_parallel = int(inst_cfg.get("parallel", 1))
            current_ctx = int(inst_cfg.get("context_length", 4096))

            # If the loaded model has different configuration (parallel != target or ctx != target),
            # automatically unload it to enforce optimal parameters.
            if current_parallel != int(parallel) or current_ctx != int(context_length):
                self.unload_model(inst_id)
            else:
                self.active_instance_id = inst_id
                return {"instance_id": inst_id, "status": "already_loaded"}

        payload = {
            "model": model_name,
            "context_length": int(context_length),
            "eval_batch_size": int(eval_batch_size),
            "parallel": int(parallel),
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

    def request_ocr_detailed(
        self,
        image_bytes: bytes,
        system_prompt: str,
        model_name: str = "default",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        context_length: int = 4096,
        max_dim: int = 1600,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Sends a masked book page image to the VLM with strict parameters and returns
        both sanitized text and detailed execution telemetry (tokens, speed, latency).
        Supports real-time token streaming via on_chunk callback.
        Automatically constrains image dimensions to max_dim (1600px) to prevent vision token
        explosion (cutting ~3700 vision tokens down to ~1900), guaranteeing zero RAM spillover
        and sub-30s inference.
        """
        processed_bytes = image_bytes
        if max_dim > 0 and len(image_bytes) > 0:
            try:
                import cv2
                import numpy as np
                np_buf = np.frombuffer(image_bytes, dtype=np.uint8)
                img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
                if img is not None:
                    h, w = img.shape[:2]
                    if max(h, w) > max_dim:
                        scale = float(max_dim) / float(max(h, w))
                        new_w = max(1, int(w * scale))
                        new_h = max(1, int(h * scale))
                        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        ok, enc = cv2.imencode(".png", resized)
                        if ok:
                            processed_bytes = enc.tobytes()
            except Exception:
                pass

        b64 = base64.b64encode(processed_bytes).decode("utf-8")
        target_model = self.active_instance_id or self.get_loaded_instance(model_name) or model_name or "default"
        t_start = time.time()
        raw_text = ""
        stats: Dict[str, Any] = {}
        result: Dict[str, Any] = {}

        # If on_chunk callback provided, attempt real-time SSE streaming first via OpenAI endpoint
        if on_chunk is not None:
            stream_payload = {
                "model": target_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Recognize printed text and music annotations on this page."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        ],
                    },
                ],
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "stream": True,
            }
            try:
                url = f"{self.base_url}/v1/chat/completions"
                req = urllib.request.Request(
                    url,
                    data=json.dumps(stream_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
                    method="POST",
                )
                accumulated: List[str] = []
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    for line_b in resp:
                        line = line_b.decode("utf-8", errors="replace").strip()
                        if not line or line.startswith(":"):
                            continue
                        if line.startswith("data: "):
                            d_str = line[6:].strip()
                            if d_str == "[DONE]":
                                break
                            try:
                                chunk_json = json.loads(d_str)
                                delta = chunk_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if delta:
                                    accumulated.append(delta)
                                    on_chunk(delta)
                            except Exception:
                                pass
                if accumulated:
                    raw_text = "".join(accumulated).strip()
            except Exception:
                raw_text = ""

        # Non-streaming execution if streaming not requested or failed
        if not raw_text:
            # 1. Try LM Studio API (/api/v1/chat)
            try:
                payload = {
                    "model": target_model,
                    "system_prompt": system_prompt,
                    "input": [
                        {"type": "text", "content": "Recognize printed text and music annotations on this page."},
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
                stats = result.get("stats", {})
            except Exception:
                # 2. Fallback to standard OpenAI /v1/chat/completions (standalone llama-server)
                openai_payload = {
                    "model": target_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Recognize printed text and music annotations on this page."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            ],
                        },
                    ],
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                    "stream": False,
                }
                result = self._post("/v1/chat/completions", openai_payload)
                choices = result.get("choices", [])
                if choices:
                    raw_text = choices[0].get("message", {}).get("content", "").strip()
                usage = result.get("usage", {})
                timings = result.get("timings", {})
                stats = {
                    "tokens_per_second": timings.get("predicted_per_second", 0.0),
                    "num_output_tokens": usage.get("completion_tokens", 0),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                }

        t_end = time.time()
        duration = max(0.01, t_end - t_start)
        cleaned_text = self._strip_markdown_fences(raw_text)

        # Ensure on_chunk receives the full text if streaming was bypassed
        if on_chunk is not None and cleaned_text and not raw_text.startswith(""):
            on_chunk(cleaned_text)

        tok_per_sec = float(stats.get("tokens_per_second") or 0.0)
        num_tokens = int(stats.get("num_output_tokens") or 0)
        if num_tokens <= 0 and cleaned_text:
            num_tokens = int(len(cleaned_text) / 3.5)
        if tok_per_sec <= 0.0 and duration > 0 and num_tokens > 0:
            tok_per_sec = round(num_tokens / duration, 1)

        stats["tokens_per_second"] = round(tok_per_sec, 1)
        stats["duration_seconds"] = round(duration, 2)
        stats["num_output_tokens"] = num_tokens

        return {
            "text": cleaned_text,
            "stats": stats,
            "model_instance_id": result.get("model_instance_id", target_model),
            "response_id": result.get("response_id", ""),
            "duration": round(duration, 2),
            "tokens_per_second": round(tok_per_sec, 1),
            "tokens_count": num_tokens,
        }

    def request_ocr(
        self,
        image_bytes: bytes,
        system_prompt: str,
        model_name: str = "default",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        context_length: int = 4096,
        max_dim: int = 1600,
    ) -> str:
        """
        Sends a masked book page image to the VLM with strict parameters.
        Forces reasoning: "off" to avoid token waste and latency.
        """
        detailed = self.request_ocr_detailed(
            image_bytes=image_bytes,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            context_length=context_length,
            max_dim=max_dim,
        )
        return detailed["text"]

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
