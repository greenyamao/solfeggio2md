"""
Isolated ML Pipeline Worker Process Manager.

Runs the heavy PyTorch/OpenCV/YOLO/OMR/VLM pipeline inside a dedicated OS process
spawned with a fresh CUDA context. This guarantees 100% responsiveness of the
PySide6 GUI event loop even when the GPU and CPU are saturated.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gui_qt.ipc.shared_frame import SHM_NAME, SharedFrameWriter


def _ml_worker_main(command_queue: mp.Queue, status_queue: mp.Queue, shm_name: str) -> None:
    """
    Child process entrypoint (spawned).
    Executes all CUDA and ML operations isolated from the GUI event loop.
    """
    # Ensure UTF-8 output encoding on Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    import warnings
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    from core.windows_perf import enable_windows_high_performance
    from core.pipeline_batch_runner import PipelineBatchRunner

    enable_windows_high_performance()

    # Initialize shared memory frame writer
    writer = SharedFrameWriter(name=shm_name)
    writer.connect_or_create()

    # Initialize pipeline batch runner
    runner = PipelineBatchRunner()

    def _on_log(entry: Dict[str, Any]) -> None:
        try:
            status_queue.put_nowait({"type": "log", "data": entry})
        except Exception:
            pass

    def _on_frame(img: Optional[Any], meta: Dict[str, Any]) -> None:
        try:
            frame_id = int(time.time() * 1000) % 1000000
            if img is not None:
                writer.write_frame(img, frame_id=frame_id)
                meta["shm_frame_id"] = frame_id
            status_queue.put_nowait({"type": "frame", "meta": meta})
        except Exception:
            pass

    runner.on_log_event = _on_log
    runner.on_frame_update = _on_frame

    # Initial scan
    runner.scan_inputs()

    last_heartbeat = 0.0

    while True:
        # 1. Process pending commands
        try:
            while True:
                msg = command_queue.get_nowait()
                cmd = msg.get("cmd")

                if cmd == "shutdown":
                    runner.stop()
                    writer.close()
                    return

                elif cmd == "start":
                    runner.start()
                    status_queue.put({"type": "log", "data": {
                        "time": time.strftime("%H:%M:%S"),
                        "tag": "SYSTEM",
                        "msg": "Конвейер запущен пользователем",
                        "level": "INFO",
                    }})

                elif cmd == "stop":
                    runner.stop()
                    status_queue.put({"type": "log", "data": {
                        "time": time.strftime("%H:%M:%S"),
                        "tag": "SYSTEM",
                        "msg": "Команда остановки отправлена конвейеру",
                        "level": "WARN",
                    }})

                elif cmd == "pause":
                    runner.pause()

                elif cmd == "resume":
                    runner.resume()

                elif cmd == "scan":
                    runner.scan_inputs()

                elif cmd == "download_model":
                    repo = msg.get("repo", "lmstudio-community/Qwen3.5-9B-GGUF")
                    model_file = msg.get("model_file", "Qwen3.5-9B-Q4_K_M.gguf")
                    mmproj_file = msg.get("mmproj_file", "mmproj-Qwen3.5-9B-BF16.gguf")
                    runner.model_manager.start_download(
                        repo_id=repo,
                        model_filename=model_file,
                        mmproj_filename=mmproj_file,
                    )

                elif cmd == "cancel_download":
                    runner.model_manager.cancel_download()

                elif cmd == "update_config":
                    cfg_updates = msg.get("config", {})
                    runner.config.update(cfg_updates)
                    try:
                        runner.config_path.write_text(
                            json.dumps(runner.config, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass
        except queue.Empty:
            pass
        except Exception:
            pass

        # 2. Periodic status emission (every 120 ms)
        now = time.time()
        if now - last_heartbeat >= 0.12:
            last_heartbeat = now
            try:
                summary = runner.get_queue_summary()
                dl_state = runner.model_manager.get_download_state()
                status_queue.put_nowait({
                    "type": "status",
                    "metrics": dict(runner.metrics),
                    "queue_summary": summary,
                    "download_state": dl_state,
                })
            except Exception:
                pass

        time.sleep(0.04)


class MLPipelineProcessClient:
    """Manages spawning, lifecycle, and non-blocking IPC with the ML Worker Process."""

    def __init__(self, shm_name: str = SHM_NAME):
        self.shm_name = shm_name
        self.ctx = mp.get_context("spawn")
        self.command_queue: mp.Queue = self.ctx.Queue()
        self.status_queue: mp.Queue = self.ctx.Queue()
        self.process: Optional[mp.Process] = None

    def start(self) -> None:
        """Spawns the child ML worker process."""
        if self.process is not None and self.process.is_alive():
            return

        self.process = self.ctx.Process(
            target=_ml_worker_main,
            args=(self.command_queue, self.status_queue, self.shm_name),
            name="MLPipelineWorkerProcess",
            daemon=True,
        )
        self.process.start()

    def send_command(self, cmd: str, **kwargs) -> None:
        """Sends an asynchronous command to the ML worker."""
        msg = {"cmd": cmd, **kwargs}
        try:
            self.command_queue.put_nowait(msg)
        except Exception:
            pass

    def poll_messages(self, max_messages: int = 50) -> List[Dict[str, Any]]:
        """Drains up to max_messages from the status queue non-blockingly."""
        messages: List[Dict[str, Any]] = []
        for _ in range(max_messages):
            try:
                msg = self.status_queue.get_nowait()
                messages.append(msg)
            except queue.Empty:
                break
            except Exception:
                break
        return messages

    def shutdown(self, timeout: float = 2.0) -> None:
        """Gracefully shuts down the worker process."""
        if self.process is None:
            return

        try:
            self.send_command("shutdown")
            self.process.join(timeout=timeout)
        except Exception:
            pass

        if self.process.is_alive():
            try:
                self.process.terminate()
                self.process.join(timeout=1.0)
            except Exception:
                pass
        self.process = None
