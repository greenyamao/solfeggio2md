"""
Developer Diagnostic and Inspection Toolkit for pdf_to_md_music.
Optimized for zero console bloat and minimal LLM token consumption.

Usage:
  python -m tools.debug_toolkit page --page 155 [--render]
  python -m tools.debug_toolkit page --pages 25,56,60,155
  python -m tools.debug_toolkit scan [--range 1-100]
  python -m tools.debug_toolkit pair --page 155 --pair 1,2
"""

import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Dynamic root resolution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import fitz  # PyMuPDF
from core.layout_detector import LayoutDetector
from core.page_preprocessor import deskew_page, detect_and_split_spread, normalize_staff_crop

# Persistent output directory for detailed logs and rendered debug images
SCRATCH_DIR = ROOT_DIR / "scratch"


def get_default_pdf() -> Path:
    in_dir = ROOT_DIR / "in"
    pdfs = list(in_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {in_dir}")
    return pdfs[0]


class BookPageResolver:
    """Maps 1-based book page numbers to physical PDF sheets and sub-page splits."""

    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.mapping: Dict[int, Tuple[int, int]] = {}
        self.total_pages = 0
        self._build_mapping()

    def _build_mapping(self):
        cur_bp = 1
        for s_idx in range(len(self.doc)):
            p = self.doc[s_idx]
            w, h = p.rect.width, p.rect.height
            is_spread = (1.22 <= (w / float(max(1.0, h))) <= 2.2 and h >= 300)
            if is_spread:
                self.mapping[cur_bp] = (s_idx, 0)
                self.mapping[cur_bp + 1] = (s_idx, 1)
                cur_bp += 2
            else:
                self.mapping[cur_bp] = (s_idx, 0)
                cur_bp += 1
        self.total_pages = cur_bp - 1

    def get_page_bgr(self, book_page: int, dpi: int = 200) -> np.ndarray:
        if book_page not in self.mapping:
            raise IndexError(f"Book page {book_page} out of bounds (1..{self.total_pages})")
        s_idx, side_idx = self.mapping[book_page]
        page = self.doc[s_idx]
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
        split_pages = detect_and_split_spread(img_bgr)
        sub_img, _ = split_pages[side_idx if side_idx < len(split_pages) else 0]
        deskewed_bgr, _ = deskew_page(sub_img)
        return deskewed_bgr


def inspect_page(
    resolver: BookPageResolver,
    detector: LayoutDetector,
    page_num: int,
    render: bool = False
) -> Dict[str, Any]:
    """Inspects layout on a single book page with compact aggregated metrics."""
    img_bgr = resolver.get_page_bgr(page_num)
    h_img, w_img = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    staves, staff_s = LayoutDetector.extract_physical_5line_staves(gray)
    detections = detector.detect(img_bgr)

    # Calculate concise summary metrics
    cls_counts: Dict[str, int] = {}
    heights: List[int] = []
    anomalies: List[str] = []

    for idx, d in enumerate(detections):
        c = d["class"]
        cls_counts[c] = cls_counts.get(c, 0) + 1
        h = d["padded_box"][3] - d["padded_box"][1]
        heights.append(h)
        # Check for monster aggregation
        if h > int(staff_s * 25.0) or d.get("staves_count", 1) >= 4:
            anomalies.append(f"block#{idx+1}:monster_system(h={h}px,staves={d.get('staves_count',1)})")

    # Check for adjacent split staves
    for i in range(len(detections) - 1):
        d1, d2 = detections[i], detections[i + 1]
        if d1["class"] == "staff" and d2["class"] == "staff":
            gap = d2["raw_box"][1] - d1["raw_box"][3]
            if 0 <= gap <= staff_s * 8.5:
                anomalies.append(f"staves#{i+1}+{i+2}:potential_split_grand_staff(gap={gap/staff_s:.1f}S)")

    render_path = None
    if render:
        SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
        render_path = SCRATCH_DIR / f"debug_p{page_num:04d}.png"
        dbg_img = detector.render_debug_image(img_bgr, detections)
        cv2.imwrite(str(render_path), dbg_img)

    return {
        "page": page_num,
        "dims": [w_img, h_img],
        "staff_s": round(staff_s, 2),
        "physical_staves": len(staves),
        "total_blocks": len(detections),
        "class_breakdown": cls_counts,
        "height_range": [min(heights), max(heights)] if heights else [0, 0],
        "anomalies": anomalies,
        "blocks": [
            {"cls": d["class"], "box": d["raw_box"], "pad": d["padded_box"], "conf": round(d.get("confidence", 1.0), 2)}
            for d in detections
        ],
        "render_file": str(render_path) if render_path else None,
    }


def scan_book_anomalies(
    resolver: BookPageResolver,
    detector: LayoutDetector,
    start_page: int,
    end_page: int
) -> Dict[str, Any]:
    """Scans a page range, collecting anomalous blocks without printing repetitive noise."""
    t0 = time.time()
    total_staves = 0
    total_blocks = 0
    cls_totals: Dict[str, int] = {}
    flagged_pages: List[Dict[str, Any]] = []

    for bp in range(start_page, end_page + 1):
        try:
            res = inspect_page(resolver, detector, bp, render=False)
            total_staves += res["physical_staves"]
            total_blocks += res["total_blocks"]
            for c, cnt in res["class_breakdown"].items():
                cls_totals[c] = cls_totals.get(c, 0) + cnt
            if res["anomalies"]:
                flagged_pages.append({
                    "page": bp,
                    "anomalies": res["anomalies"],
                    "classes": res["class_breakdown"]
                })
        except Exception as e:
            flagged_pages.append({"page": bp, "anomalies": [f"error:{str(e)[:60]}"]})

    elapsed = round(time.time() - t0, 2)
    return {
        "range": [start_page, end_page],
        "elapsed_sec": elapsed,
        "total_staves": total_staves,
        "total_blocks": total_blocks,
        "class_totals": cls_totals,
        "anomalous_pages_count": len(flagged_pages),
        "anomalies": flagged_pages,
    }


def inspect_staff_pair(
    resolver: BookPageResolver,
    page_num: int,
    staff_i: int,
    staff_j: int
) -> Dict[str, Any]:
    """Analyzes the exact spatial and morphological relation between two staves."""
    img_bgr = resolver.get_page_bgr(page_num)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    staves, staff_s = LayoutDetector.extract_physical_5line_staves(gray)

    if staff_i < 1 or staff_j > len(staves) or staff_i >= staff_j:
        raise ValueError(f"Invalid pair ({staff_i}, {staff_j}). Page has {len(staves)} staves (1-indexed).")

    s1 = staves[staff_i - 1]
    s2 = staves[staff_j - 1]
    v_gap = s2["y_top"] - s1["y_bot"]

    w1 = s1["x_right"] - s1["x_left"]
    w2 = s2["x_right"] - s2["x_left"]
    inter = max(0, min(s1["x_right"], s2["x_right"]) - max(s1["x_left"], s2["x_left"]))
    union = max(s1["x_right"], s2["x_right"]) - min(s1["x_left"], s2["x_left"])
    h_iou = inter / float(max(1, union))

    has_left_conn = LayoutDetector.has_continuous_vertical_connector(gray, s1, s2, staff_s)

    # Check for inter-staff measure barlines across the span
    inter_roi = gray[s1["y_bot"]:s2["y_top"], max(s1["x_left"], s2["x_left"]):min(s1["x_right"], s2["x_right"])]
    barline_count = 0
    if inter_roi.size > 0:
        bg = float(np.percentile(inter_roi, 90))
        bin_inv = (inter_roi < bg - 35).astype(np.uint8)
        k_h = max(3, int(v_gap * 0.70))
        vert_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_h))
        opened = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, vert_k)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(opened)
        for idx in range(1, num_labels):
            h_stat = stats[idx, cv2.CC_STAT_HEIGHT]
            w_stat = stats[idx, cv2.CC_STAT_WIDTH]
            if h_stat >= int(v_gap * 0.75) and (h_stat / max(1, w_stat)) >= 2.0:
                barline_count += 1

    can_merge_grand = (
        ((has_left_conn or barline_count >= 1) and 0 <= v_gap <= int(staff_s * 12.0) and (h_iou >= 0.40 or has_left_conn)) or
        (0 <= v_gap <= int(staff_s * 6.5) and h_iou >= 0.50)
    )

    return {
        "page": page_num,
        "pair": [staff_i, staff_j],
        "v_gap_px": v_gap,
        "v_gap_S": round(v_gap / staff_s, 2),
        "h_iou": round(h_iou, 3),
        "has_left_brace": has_left_conn,
        "inter_barlines_count": barline_count,
        "verdict": "MERGE (grand_staff)" if can_merge_grand else "SEPARATE (distinct systems)",
    }


def main():
    parser = argparse.ArgumentParser(description="Compact Developer Debug Toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Page command
    p_page = subparsers.add_parser("page", help="Inspect one or more pages")
    p_page.add_argument("--page", type=int, help="Single book page number (1-based)")
    p_page.add_argument("--pages", type=str, help="Comma-separated page numbers (e.g. 25,56,155)")
    p_page.add_argument("--render", action="store_true", help="Save debug image to scratch/")
    p_page.add_argument("--verbose", "-v", action="store_true", help="Print individual block coordinates")
    p_page.add_argument("--book", type=str, help="Optional PDF path")

    # Scan command
    p_scan = subparsers.add_parser("scan", help="Scan range for monster blocks and split staves")
    p_scan.add_argument("--range", type=str, default="1-30", help="Page range (e.g. 1-50, default: 1-30)")
    p_scan.add_argument("--book", type=str, help="Optional PDF path")

    # Pair command
    p_pair = subparsers.add_parser("pair", help="Inspect connection between two staves on a page")
    p_pair.add_argument("--page", type=int, required=True, help="Book page number")
    p_pair.add_argument("--pair", type=str, required=True, help="1-based staff indices, e.g. 1,2")
    p_pair.add_argument("--book", type=str, help="Optional PDF path")

    # Crop command
    p_crop = subparsers.add_parser("crop", help="Inspect normalization of a specific staff crop")
    p_crop.add_argument("--page", type=int, required=True, help="Book page number")
    p_crop.add_argument("--staff", type=int, default=1, help="1-based staff block index on page (default: 1)")
    p_crop.add_argument("--render", action="store_true", help="Save before/after comparison to scratch/")
    p_crop.add_argument("--book", type=str, help="Optional PDF path")

    args = parser.parse_args()

    pdf_path = Path(args.book) if args.book else get_default_pdf()
    resolver = BookPageResolver(pdf_path)
    import torch
    weights = ROOT_DIR / "weights" / "ola-layout-analysis-2.0-2025-03-09.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = LayoutDetector(weights_path=str(weights), device=device)

    if args.command == "page":
        pages = [args.page] if args.page else [int(p.strip()) for p in args.pages.split(",") if p.strip()]
        for p in pages:
            res = inspect_page(resolver, detector, p, render=args.render)
            cls_str = ", ".join(f"{k}:{v}" for k, v in res["class_breakdown"].items())
            anom_str = f" | ANOMALIES: {res['anomalies']}" if res["anomalies"] else ""
            render_str = f" | Img: {res['render_file']}" if res["render_file"] else ""
            print(f"P{res['page']:04d} [{res['dims'][0]}x{res['dims'][1]}] S={res['staff_s']}px | "
                  f"{res['physical_staves']} staves -> {res['total_blocks']} blocks ({cls_str}){anom_str}{render_str}")
            if getattr(args, "verbose", False):
                for idx, b in enumerate(res["blocks"]):
                    bw = b["box"][2] - b["box"][0]
                    bh = b["box"][3] - b["box"][1]
                    print(f"  #{idx+1} {b['cls']} ({bw}x{bh}px) raw={b['box']} conf={b['conf']}")

    elif args.command == "scan":
        parts = [int(x.strip()) for x in args.range.split("-")]
        start_p, end_p = parts[0], parts[1] if len(parts) > 1 else parts[0]
        end_p = min(end_p, resolver.total_pages)
        res = scan_book_anomalies(resolver, detector, start_p, end_p)
        cls_str = ", ".join(f"{k}:{v}" for k, v in res["class_totals"].items())
        print(f"SCAN P{res['range'][0]}..P{res['range'][1]} in {res['elapsed_sec']}s: "
              f"{res['total_staves']} staves, {res['total_blocks']} blocks ({cls_str}). "
              f"Anomalous pages: {res['anomalous_pages_count']}")
        if res["anomalies"]:
            SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
            report_file = SCRATCH_DIR / "scan_anomalies.json"
            report_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(f"Details saved: {report_file}")
            for a in res["anomalies"][:5]:
                print(f"  P{a['page']:04d}: {', '.join(a['anomalies'])}")
            if len(res["anomalies"]) > 5:
                print(f"  ... and {len(res['anomalies']) - 5} more")

    elif args.command == "pair":
        s1_idx, s2_idx = [int(x.strip()) for x in args.pair.split(",")]
        res = inspect_staff_pair(resolver, args.page, s1_idx, s2_idx)
        print(f"P{res['page']:04d} Pair {res['pair'][0]}->{res['pair'][1]}: "
              f"gap={res['v_gap_px']}px ({res['v_gap_S']}S), IoU={res['h_iou']}, "
              f"left_brace={res['has_left_brace']}, barlines={res['inter_barlines_count']} -> "
              f"{res['verdict']}")

    elif args.command == "crop":
        page_res = inspect_page(resolver, detector, args.page, render=False)
        blocks = page_res["blocks"]
        if args.staff < 1 or args.staff > len(blocks):
            print(f"Error: Page {args.page} has {len(blocks)} blocks (1-based).")
            return
        b = blocks[args.staff - 1]
        img_bgr = resolver.get_page_bgr(args.page)
        px1, py1, px2, py2 = b["pad"]
        crop = img_bgr[py1:py2, px1:px2]

        t0 = time.perf_counter()
        norm_crop, tilt, bend = normalize_staff_crop(crop, notation_class=b["cls"])
        dt_ms = (time.perf_counter() - t0) * 1000

        render_msg = ""
        if args.render:
            SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
            out_p = SCRATCH_DIR / f"crop_p{args.page:04d}_s{args.staff:02d}.png"
            h_c, w_c = crop.shape[:2]
            h_n, w_n = norm_crop.shape[:2]
            max_w = max(w_c, w_n)
            c_pad = cv2.copyMakeBorder(crop, 0, 0, 0, max_w - w_c, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            n_pad = cv2.copyMakeBorder(norm_crop, 0, 0, 0, max_w - w_n, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            sep = np.full((3, max_w, 3), (0, 0, 255), dtype=np.uint8)
            stacked = np.vstack([c_pad, sep, n_pad])
            cv2.imwrite(str(out_p), stacked)
            render_msg = f" | Img: {out_p}"

        action = "straightened" if bend >= 1.0 else "clean (unchanged)"
        print(f"P{args.page:04d} #{args.staff} {b['cls']} [{crop.shape[1]}x{crop.shape[0]}px]: "
              f"tilt={tilt:+.1f}° bend={bend:.1f}px -> {action} in {dt_ms:.1f}ms{render_msg}")


if __name__ == "__main__":
    main()
