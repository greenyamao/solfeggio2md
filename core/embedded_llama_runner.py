"""
Embedded Llama.cpp Runner for Vision-Language Models (VLM).

Launches and supervises a standalone llama-server process directly from Python,
providing an OpenAI-compatible API on localhost without requiring LM Studio desktop.
Captures real-time stdout/stderr telemetry, parses tokens/second generation speed,
and feeds live diagnostics into the UI console.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).parent.parent.resolve()


def find_llama_server_binary() -> Optional[Path]:
    """
    Discovers the most capable llama-server executable available on the host system.
    Prioritizes project-local bin, then system PATH, then CUDA backends in ~/.lmstudio.
    """
    # 1. Project-local bin
    local_bin = ROOT_DIR / "bin" / "llama-server.exe"
    if local_bin.is_file():
        return local_bin

    # 2. System PATH
    which_bin = shutil.which("llama-server")
    if which_bin:
        p = Path(which_bin)
        if p.is_file():
            return p

    # 3. LM Studio backends (CUDA prioritized)
    try:
        user_home = Path.home()
        pattern = str(user_home / ".lmstudio" / "extensions" / "backends" / "**" / "llama-server.exe")
        candidates = glob.glob(pattern, recursive=True)
        if not candidates:
            # Check other Windows user profiles if available
            users_root = Path(os.environ.get("USERPROFILE", "C:\\Users")).parent
            pattern_all = str(users_root / "*" / ".lmstudio" / "extensions" / "backends" / "**" / "llama-server.exe")
            candidates = glob.glob(pattern_all, recursive=True)

        if candidates:
            # Sort candidates: cuda12 > cuda > vulkan > avx2
            def _score_candidate(path_str: str) -> int:
                low = path_str.lower()
                if "cuda12" in low:
                    return 40
                if "cuda" in low:
                    return 30
                if "vulkan" in low:
                    return 20
                if "avx2" in low:
                    return 10
                return 1

            candidates.sort(key=_score_candidate, reverse=True)
            return Path(candidates[0])
    except Exception:
        pass

    return None


class EmbeddedLlamaRunner:
    """Controls lifecycle of standalone llama-server with real-time telemetry streaming."""

    def __init__(
        self,
        port: int = 1234,
        host: str = "127.0.0.1",
        log_callback: Optional[Callable[[str, str], None]] = None,
    ):
        self.port = port
        self.host = host
        self.log_callback = log_callback

        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Telemetry metrics
        self.is_running = False
        self.last_eval_speed: float = 0.0  # tokens / sec
        self.last_prompt_eval_speed: float = 0.0
        self.last_eval_tokens: int = 0
        self.last_duration_seconds: float = 0.0
        self.vram_used_mb: float = 0.0

    def find_free_port(self, default_port: int = 1234) -> int:
        """Finds an open port starting from default_port."""
        for p in range(default_port, default_port + 50):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex((self.host, p)) != 0:
                    return p
        return default_port

    def is_port_listening(self) -> bool:
        """Checks if current port is actively accepting connections."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((self.host, self.port)) == 0

    def is_server_ready(self, timeout_sec: float = 2.0) -> bool:
        """Checks if /health or /v1/models returns HTTP 200."""
        try:
            req = urllib.request.Request(f"http://{self.host}:{self.port}/health")
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    # In llama.cpp, status is 'ok' or 'loading model'
                    return data.get("status") in ("ok", "success")
        except Exception:
            try:
                req2 = urllib.request.Request(f"http://{self.host}:{self.port}/v1/models")
                with urllib.request.urlopen(req2, timeout=timeout_sec) as resp:
                    return resp.status == 200
            except Exception:
                return False
        return False

    def start(
        self,
        model_path: str,
        mmproj_path: str,
        context_length: int = 4096,
        gpu_layers: int = 999,
        parallel: int = 1,
        flash_attention: bool = True,
        port: Optional[int] = None,
    ) -> bool:
        """Launches llama-server subprocess if not already running."""
        with self._lock:
            if self.is_running and self._proc and self._proc.poll() is None:
                if self.is_server_ready():
                    return True

            server_bin = find_llama_server_binary()
            if not server_bin or not server_bin.is_file():
                if self.log_callback:
                    self.log_callback("ERROR", "llama-server.exe executable not found on system")
                return False

            if port is not None:
                self.port = port

            # If port is already bound by an existing instance that is healthy, reuse it
            if self.is_server_ready():
                self.is_running = True
                if self.log_callback:
                    self.log_callback("VLM", f"Connecting to existing VLM server on port {self.port}")
                return True

            # Ensure port is free
            if self.is_port_listening():
                self.port = self.find_free_port(self.port + 1)

            cmd = [
                str(server_bin),
                "-m", str(model_path),
                "--mmproj", str(mmproj_path),
                "-c", str(int(context_length)),
                "-ngl", str(int(gpu_layers)),
                "-np", str(int(parallel)),
                "--port", str(self.port),
                "--host", self.host,
                "--ubatch-size", "512",
            ]
            if flash_attention:
                cmd.extend(["-fa", "1"])

            env = dict(os.environ)
            bin_dir = str(server_bin.parent)
            env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")

            if self.log_callback:
                self.log_callback("VLM", f"Starting standalone llama-server (port {self.port}, context {context_length})")

            # Launch headless subprocess without showing black console window on Windows
            startupinfo = None
            if os.name == "nt":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = 0  # SW_HIDE

            self._proc = subprocess.Popen(
                cmd,
                cwd=bin_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                startupinfo=startupinfo,
            )

            self.is_running = True
            self._reader_thread = threading.Thread(target=self._read_output_loop, daemon=True)
            self._reader_thread.start()

        # Wait for server readiness
        for _ in range(60):  # up to 30 seconds
            if self._proc.poll() is not None:
                self.is_running = False
                if self.log_callback:
                    self.log_callback("ERROR", f"llama-server terminated with exit code {self._proc.returncode}")
                return False

            if self.is_server_ready(timeout_sec=0.5):
                if self.log_callback:
                    self.log_callback("VLM", f"VLM server ready on http://{self.host}:{self.port}")
                return True
            time.sleep(0.5)

        return False

    def stop(self) -> None:
        """Terminates llama-server process and frees GPU memory."""
        with self._lock:
            if self._proc:
                try:
                    if self.log_callback:
                        self.log_callback("VLM", "Stopping VLM server and releasing VRAM...")
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                        self._proc.wait(timeout=1.0)
                except Exception:
                    pass
                self._proc = None

            self.is_running = False
            self.vram_used_mb = 0.0

    def _read_output_loop(self) -> None:
        """Reads stdout/stderr lines asynchronously, extracts speed timings, and routes logs."""
        if not self._proc or not self._proc.stdout:
            return

        for line in self._proc.stdout:
            clean_line = line.strip()
            if not clean_line:
                continue

            # Parse timings from llama.cpp logs:
            # print_timings: eval time = 520.12 ms / 21 runs ( 24.77 ms per token, 40.38 tokens per second )
            if "print_timings: eval time" in clean_line or "tokens per second" in clean_line:
                m_speed = re.search(r"([\d\.]+)\s+tokens per second", clean_line)
                if m_speed:
                    try:
                        self.last_eval_speed = float(m_speed.group(1))
                    except ValueError:
                        pass
                m_tokens = re.search(r"/\s+(\d+)\s+runs", clean_line)
                if m_tokens:
                    try:
                        self.last_eval_tokens = int(m_tokens.group(1))
                    except ValueError:
                        pass

            # Parse VRAM metrics: "total VRAM used = 5911.12 MiB" or "model size = ... MiB"
            if "total VRAM used" in clean_line or "model size =" in clean_line or "VRAM used:" in clean_line:
                m_vram = re.search(r"([\d\.]+)\s+(?:MiB|MB)", clean_line)
                if m_vram:
                    try:
                        self.vram_used_mb = round(float(m_vram.group(1)), 1)
                    except ValueError:
                        pass

            # Filter noisy ggml info, emit informative lines
            if any(term in clean_line for term in ("HTTP", "srv", "llama_model_load", "error", "warn", "eval time", "prompt eval time", "slot")):
                if self.log_callback:
                    # Strip verbose timestamps if present
                    msg = re.sub(r"^\d+\.\d+\.\d+\.\d+\s+[A-Z]\s+[a-z_]+:\s*", "", clean_line)
                    self.log_callback("VLM", msg[:160])

        self.is_running = False
