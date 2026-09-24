"""
Model Manager for Vision-Language Models (VLM).

Handles discovering, checking, downloading, and verifying GGUF weights
and multimodal projectors (mmproj) from Hugging Face or local caches.
Ensures zero hardcoded user paths and thread-safe background downloads
with fine-grained progress telemetry (speed, percentage, ETA).
"""

from __future__ import annotations

import glob
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).parent.parent.resolve()
DEFAULT_MODELS_DIR = ROOT_DIR / "models"


class ModelManager:
    """Manages local storage, cache discovery, and background downloading of GGUF VLM weights."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or DEFAULT_MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._download_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()

        # Telemetry state
        self.download_state: Dict[str, Any] = {
            "status": "idle",  # "idle" | "downloading" | "ready" | "error" | "cancelled"
            "repo": "",
            "current_file": "",
            "file_index": 0,
            "total_files": 0,
            "bytes_downloaded": 0,
            "total_bytes": 0,
            "percent": 0.0,
            "speed_mb_s": 0.0,
            "eta_seconds": 0.0,
            "error_message": "",
        }

        # Discovery caches to prevent repeated multi-gigabyte disk scans
        self._disk_cache: Dict[str, Optional[Path]] = {}
        self._resolved_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def _discover_local_cache(self, filename: str) -> Optional[Path]:
        """
        Scans known system directories (e.g. LM Studio models, HuggingFace cache)
        to find already-downloaded weights without re-downloading gigabytes of data.
        Caches results in memory to guarantee 0 ms repeated lookups.
        """
        if filename in self._disk_cache:
            cached = self._disk_cache[filename]
            if cached is not None and cached.is_file():
                return cached

        # 1. Project-local models directory
        for p in self.models_dir.rglob(filename):
            if p.is_file() and p.stat().st_size > 1024 * 1024:
                self._disk_cache[filename] = p
                return p


        # 2. Check ~/.lmstudio/settings.json downloadsFolder
        try:
            home = Path.home()
            lms_settings = home / ".lmstudio" / "settings.json"
            if lms_settings.is_file():
                cfg = json.loads(lms_settings.read_text(encoding="utf-8"))
                dl_folder = cfg.get("downloadsFolder")
                if dl_folder:
                    p_dl = Path(dl_folder)
                    for f in p_dl.rglob(filename):
                        if f.is_file() and f.stat().st_size > 1024 * 1024:
                            self._disk_cache[filename] = f
                            return f
        except Exception:
            pass

        # 3. Check ~/.lmstudio/models
        try:
            home = Path.home()
            lms_models = home / ".lmstudio" / "models"
            if lms_models.is_dir():
                for f in lms_models.rglob(filename):
                    if f.is_file() and f.stat().st_size > 1024 * 1024:
                        self._disk_cache[filename] = f
                        return f
        except Exception:
            pass

        # 4. Check user profile / other local windows accounts if applicable
        try:
            users_root = Path(os.environ.get("USERPROFILE", "C:\\Users")).parent
            if users_root.is_dir():
                pattern = str(users_root / "*" / ".lmstudio" / "models" / "**" / filename)
                matches = glob.glob(pattern, recursive=True)
                for m in matches:
                    p = Path(m)
                    if p.is_file() and p.stat().st_size > 1024 * 1024:
                        self._disk_cache[filename] = p
                        return p
        except Exception:
            pass

        # 5. Hugging Face hub cache
        try:
            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            if hf_cache.is_dir():
                for f in hf_cache.rglob(filename):
                    if f.is_file() and f.stat().st_size > 1024 * 1024:
                        self._disk_cache[filename] = f
                        return f
        except Exception:
            pass

        self._disk_cache[filename] = None
        return None

    def resolve_model_files(
        self,
        repo: str = "lmstudio-community/Qwen3.5-9B-GGUF",
        model_file: str = "Qwen3.5-9B-Q4_K_M.gguf",
        mmproj_file: str = "mmproj-Qwen3.5-9B-BF16.gguf",
    ) -> Dict[str, Any]:
        """
        Resolves physical disk paths for both model and vision projector.
        If found in external cache, creates hardlink/symlink to models_dir if possible,
        or returns direct path. Caches results in memory for 0 ms repeated lookups.
        """
        cache_key = (repo, model_file, mmproj_file)
        if cache_key in self._resolved_cache:
            cached_res = self._resolved_cache[cache_key]
            if cached_res.get("ready"):
                return cached_res

        target_dir = self.models_dir / repo.replace("/", "_")
        target_dir.mkdir(parents=True, exist_ok=True)


        res: Dict[str, Any] = {
            "repo": repo,
            "model_file": model_file,
            "mmproj_file": mmproj_file,
            "model_path": None,
            "mmproj_path": None,
            "model_exists": False,
            "mmproj_exists": False,
            "ready": False,
        }

        # Check model file
        local_model = target_dir / model_file
        if local_model.is_file() and local_model.stat().st_size > 1024 * 1024:
            res["model_path"] = str(local_model)
            res["model_exists"] = True
        else:
            cached = self._discover_local_cache(model_file)
            if cached and cached.is_file():
                # Attempt to link locally for zero-copy discovery
                try:
                    if not local_model.exists():
                        try:
                            os.link(str(cached), str(local_model))
                            res["model_path"] = str(local_model)
                        except Exception:
                            res["model_path"] = str(cached)
                    else:
                        res["model_path"] = str(local_model)
                except Exception:
                    res["model_path"] = str(cached)
                res["model_exists"] = True

        # Check mmproj file
        local_mmproj = target_dir / mmproj_file
        if local_mmproj.is_file() and local_mmproj.stat().st_size > 1024 * 1024:
            res["mmproj_path"] = str(local_mmproj)
            res["mmproj_exists"] = True
        else:
            cached = self._discover_local_cache(mmproj_file)
            if cached and cached.is_file():
                try:
                    if not local_mmproj.exists():
                        try:
                            os.link(str(cached), str(local_mmproj))
                            res["mmproj_path"] = str(local_mmproj)
                        except Exception:
                            res["mmproj_path"] = str(cached)
                    else:
                        res["mmproj_path"] = str(local_mmproj)
                except Exception:
                    res["mmproj_path"] = str(cached)
                res["mmproj_exists"] = True

        res["ready"] = bool(res["model_exists"] and res["mmproj_exists"])
        self._resolved_cache[cache_key] = res
        return res


    def get_status(
        self,
        repo: str = "lmstudio-community/Qwen3.5-9B-GGUF",
        model_file: str = "Qwen3.5-9B-Q4_K_M.gguf",
        mmproj_file: str = "mmproj-Qwen3.5-9B-BF16.gguf",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Returns unified model availability status and download telemetry."""
        if "repo_id" in kwargs:
            repo = kwargs["repo_id"]
        if "model_filename" in kwargs:
            model_file = kwargs["model_filename"]
        if "mmproj_filename" in kwargs:
            mmproj_file = kwargs["mmproj_filename"]

        with self._lock:
            st = dict(self.download_state)

        files = self.resolve_model_files(repo, model_file, mmproj_file)
        st["model_info"] = files
        st["ready"] = bool(files.get("ready", False))
        st["model_path"] = files.get("model_path")
        st["mmproj_path"] = files.get("mmproj_path")
        st["missing"] = files.get("missing", [])
        if files["ready"] and st["status"] != "downloading":
            st["status"] = "ready"
        elif not files["ready"] and st["status"] not in ("downloading", "error"):
            st["status"] = "missing"

        return st


    def get_download_state(self) -> Dict[str, Any]:
        """Returns a snapshot of the current background download state."""
        with self._lock:
            return dict(self.download_state)

    def start_download(
        self,
        repo: str = "lmstudio-community/Qwen3.5-9B-GGUF",
        model_file: str = "Qwen3.5-9B-Q4_K_M.gguf",
        mmproj_file: str = "mmproj-Qwen3.5-9B-BF16.gguf",
    ) -> bool:
        """Starts asynchronous download of required model files in background thread."""
        with self._lock:
            if self._download_thread and self._download_thread.is_alive():
                return False

            self._cancel_event.clear()
            self.download_state.update(
                {
                    "status": "downloading",
                    "repo": repo,
                    "current_file": "",
                    "file_index": 0,
                    "total_files": 2,
                    "bytes_downloaded": 0,
                    "total_bytes": 0,
                    "percent": 0.0,
                    "speed_mb_s": 0.0,
                    "eta_seconds": 0.0,
                    "error_message": "",
                }
            )

            self._download_thread = threading.Thread(
                target=self._download_worker,
                args=(repo, model_file, mmproj_file),
                daemon=True,
            )
            self._download_thread.start()
            return True

    def cancel_download(self) -> None:
        """Signals background downloader to cleanly stop."""
        self._cancel_event.set()
        with self._lock:
            self.download_state["status"] = "cancelled"

    def _download_worker(self, repo: str, model_file: str, mmproj_file: str) -> None:
        target_dir = self.models_dir / repo.replace("/", "_")
        target_dir.mkdir(parents=True, exist_ok=True)

        files_to_download = [
            (mmproj_file, target_dir / mmproj_file),
            (model_file, target_dir / model_file),
        ]

        total_files = len(files_to_download)

        try:
            for idx, (filename, dest_path) in enumerate(files_to_download, start=1):
                if self._cancel_event.is_set():
                    return

                # If already exists and valid, skip
                if dest_path.is_file() and dest_path.stat().st_size > 1024 * 1024:
                    continue

                url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
                self._download_single_file(url, dest_path, filename, idx, total_files)

            with self._lock:
                self.download_state["status"] = "ready"
                self.download_state["percent"] = 100.0
                self.download_state["speed_mb_s"] = 0.0
                self.download_state["eta_seconds"] = 0.0
        except Exception as e:
            with self._lock:
                self.download_state["status"] = "error"
                self.download_state["error_message"] = str(e)

    def _download_single_file(
        self,
        url: str,
        dest_path: Path,
        filename: str,
        file_index: int,
        total_files: int,
    ) -> None:
        part_path = dest_path.with_suffix(dest_path.suffix + ".part")
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "pdf_to_md_music/2.0 (ModelManager; Python/urllib)",
            },
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))

            with self._lock:
                self.download_state["current_file"] = filename
                self.download_state["file_index"] = file_index
                self.download_state["total_files"] = total_files
                self.download_state["total_bytes"] = total_size
                self.download_state["bytes_downloaded"] = 0

            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunk
            t0 = time.time()
            t_last_calc = t0
            bytes_since_calc = 0

            with open(part_path, "wb") as f_out:
                while True:
                    if self._cancel_event.is_set():
                        f_out.close()
                        if part_path.exists():
                            part_path.unlink()
                        return

                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break

                    f_out.write(chunk)
                    downloaded += len(chunk)
                    bytes_since_calc += len(chunk)

                    now = time.time()
                    if now - t_last_calc >= 0.5:
                        dt = now - t_last_calc
                        speed = (bytes_since_calc / (1024 * 1024)) / dt if dt > 0 else 0.0
                        pct = round((downloaded / max(1, total_size)) * 100.0, 1)
                        rem_bytes = max(0, total_size - downloaded)
                        eta = (rem_bytes / (speed * 1024 * 1024)) if speed > 0 else 0.0

                        with self._lock:
                            self.download_state["bytes_downloaded"] = downloaded
                            self.download_state["percent"] = pct
                            self.download_state["speed_mb_s"] = round(speed, 2)
                            self.download_state["eta_seconds"] = round(eta, 1)

                        t_last_calc = now
                        bytes_since_calc = 0

        # Rename atomic part to final file
        if part_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            part_path.rename(dest_path)
