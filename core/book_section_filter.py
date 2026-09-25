"""
Automated Book Section Filter.
Detects and filters non-content pages:
- Front-matter: Cover, copyright, preface, table of contents (Contents).
- Back-matter: Index, bibliography, publisher back-matter.
Focuses the conversion pipeline strictly on educational music chapters and scores.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
import re
import cv2
import numpy as np
import pymupdf


TOC_KEYWORDS = {
    "contents", "table of contents",
    "sommaire", "inhaltsverzeichnis", "table des matieres"
}

BACK_MATTER_KEYWORDS = {
    "index", "subject index", "general index",
    "bibliography", "selected bibliography", "literaturverzeichnis"
}

FRONT_MATTER_KEYWORDS = {
    "preface", "foreword", "copyright", "all rights reserved",
    "dedication", "acknowledgments", "title page", "introduction"
}


class BookSectionFilter:
    """
    Analyzes document structure to identify and skip front-matter (TOC, preface)
    and back-matter (index, bibliography) that do not contain musical score content.
    Works seamlessly on scanned PDFs and digital PDFs.
    """

    def __init__(
        self,
        skip_front_matter: bool = True,
        skip_back_matter: bool = True,
        front_ratio: float = 0.15,
        back_ratio: float = 0.12,
    ):
        self.skip_front_matter = skip_front_matter
        self.skip_back_matter = skip_back_matter
        self.front_ratio = front_ratio
        self.back_ratio = back_ratio
        self.music_started = False
        self.first_music_page: Optional[int] = None
        self.last_music_page: Optional[int] = None
        self.index_started = False

    def get_front_limit(self, total_pages: int) -> int:
        return max(5, int(total_pages * self.front_ratio))

    def get_back_start(self, total_pages: int) -> int:
        return max(self.get_front_limit(total_pages) + 1, total_pages - int(total_pages * self.back_ratio) + 1)

    def analyze_document_sections(self, pdf_path: Path) -> Dict[int, Dict[str, Any]]:
        """
        Fast upfront scan for digital PDFs with embedded text.
        Extracts table of contents and index pages in under 0.1s.
        """
        sections: Dict[int, Dict[str, Any]] = {}
        try:
            with pymupdf.open(pdf_path) as doc:
                total_pages = len(doc)
                front_limit = self.get_front_limit(total_pages)
                back_start = self.get_back_start(total_pages)

                for p_idx in range(1, total_pages + 1):
                    page = doc[p_idx - 1]
                    raw_text = page.get_text().lower().strip()
                    if not raw_text:
                        continue

                    # Digital TOC check
                    if self.skip_front_matter and any(kw in raw_text for kw in TOC_KEYWORDS):
                        sections[p_idx] = {
                            "skip": True,
                            "reason": "Table of Contents",
                            "section": "toc",
                        }
                    # Digital Front-matter check
                    elif self.skip_front_matter and p_idx <= front_limit and any(kw in raw_text for kw in FRONT_MATTER_KEYWORDS):
                        sections[p_idx] = {
                            "skip": True,
                            "reason": "Front-matter (Preface/Copyright)",
                            "section": "front_matter",
                        }
                    # Digital Index / Back-matter check
                    elif self.skip_back_matter and p_idx >= back_start and any(kw in raw_text for kw in BACK_MATTER_KEYWORDS):
                        sections[p_idx] = {
                            "skip": True,
                            "reason": "Back-matter (Index)",
                            "section": "index",
                        }
        except Exception:
            pass

        return sections

    def check_after_detection(
        self,
        p_num: int,
        total_pages: int,
        detections_count: int,
        gray: Optional[np.ndarray] = None
    ) -> Tuple[bool, str]:
        """
        Real-time evaluation after YOLO layout detection.
        Seamlessly skips non-music front-matter and index pages on scanned documents.
        """
        front_limit = self.get_front_limit(total_pages)
        back_start = self.get_back_start(total_pages)

        # 1. Front-matter: before music has started
        if not self.music_started and p_num <= front_limit:
            if detections_count == 0:
                return self.skip_front_matter, "Front-matter without music"
            else:
                self.music_started = True
                self.first_music_page = p_num
                self.last_music_page = p_num
                return False, ""

        # Update last known music page
        if detections_count > 0:
            self.music_started = True
            self.last_music_page = p_num
            return False, ""

        # 2. Back-matter: index and end pages
        if p_num >= back_start:
            if self.index_started:
                return self.skip_back_matter, "Back-matter / Appendix"

            # Check if multi-column index page
            if gray is not None and self.is_multi_column_index_page(gray):
                self.index_started = True
                return self.skip_back_matter, "Back-matter (Index)"

        return False, ""

    def is_multi_column_index_page(self, img_gray: np.ndarray) -> bool:
        """
        Visual layout heuristic: detects if a scanned page has dense multi-column index layout.
        """
        h, w = img_gray.shape
        if h < 400 or w < 300:
            return False

        small_w = 400
        small_h = int(h * (small_w / float(w)))
        small_gray = cv2.resize(img_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

        bin_ink = (small_gray < 180).astype(np.float32)
        col_proj = np.mean(bin_ink[int(small_h * 0.15):int(small_h * 0.85), :], axis=0)
        col_smooth = cv2.GaussianBlur(col_proj.reshape(1, -1), (15, 1), 0).flatten()

        med = float(np.median(col_smooth))
        if med < 0.02:
            return False

        valleys = []
        for x in range(int(small_w * 0.20), int(small_w * 0.80)):
            if col_smooth[x] < med * 0.35 and col_smooth[x] <= col_smooth[x-1] and col_smooth[x] <= col_smooth[x+1]:
                valleys.append(x)

        distinct_valleys = []
        for v in valleys:
            if not distinct_valleys or v - distinct_valleys[-1] > 35:
                distinct_valleys.append(v)

        return len(distinct_valleys) >= 2
