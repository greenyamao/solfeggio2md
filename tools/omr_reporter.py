"""
Visual Side-by-Side OMR Diagnostic Reporter.
Generates an interactive, zero-external-dependency HTML dashboard comparing
original sheet music crops with Verovio-rendered vector notations, annotated
with structural invariant checks from NotationValidator.
"""

import base64
import html
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.abc_bridge import ABCBridge
from core.notation_validator import NotationValidator, ValidationReport
from core.omr_engine import OMREngine
from core.page_preprocessor import normalize_staff_crop


def _img_to_base64_png(img_bgr: np.ndarray) -> str:
    """Encodes BGR numpy image to base64 PNG string."""
    success, buffer = cv2.imencode(".png", img_bgr)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("ascii")
    return f"data:image/png;base64,{b64_str}"


def build_html_report(snippets: List[Dict[str, Any]], title: str = "OMR Visual Side-by-Side Diagnostic Report") -> str:
    """
    Constructs a dark-mode engineering dashboard with high-contrast layout,
    clean vector SVG icons (no emojis), and interactive filters.
    """
    total_count = len(snippets)
    anomaly_count = sum(1 for s in snippets if not s["is_valid"] or s["anomalies"])
    clean_count = total_count - anomaly_count
    avg_score = (sum(s["score"] for s in snippets) / max(1, total_count)) * 100.0

    cards_html = []
    for idx, s in enumerate(snippets):
        card_id = f"snippet-{idx+1}"
        is_clean = (s["is_valid"] and not s["anomalies"])
        card_class = "clean" if is_clean else "anomaly"

        # Score badge styling
        score_val = int(round(s["score"] * 100))
        if score_val >= 85:
            score_cls = "badge-success"
        elif score_val >= 65:
            score_cls = "badge-warning"
        else:
            score_cls = "badge-danger"

        # Anomalies chips
        if s["anomalies"]:
            anom_chips = "".join(
                f'<span class="anomaly-chip">{html.escape(str(a))}</span>'
                for a in s["anomalies"]
            )
        else:
            anom_chips = '<span class="chip-clean">No structural anomalies</span>'

        # Rendered notation container
        svg_content = s.get("svg_content", "").strip()
        if svg_content and "<svg" in svg_content:
            render_box = f'<div class="rendered-score-box">{svg_content}</div>'
        else:
            render_box = '<div class="render-error">Render failed or empty score</div>'

        # Safe code snippets
        abc_code = html.escape(s.get("abc", ""))
        kern_code = html.escape(s.get("kern", ""))

        card_html = f"""
        <div class="snippet-card {card_class}" id="{card_id}" data-category="{card_class}">
            <div class="card-header">
                <div class="card-title-group">
                    <span class="page-badge">P{s['page']:04d} #{s['staff_idx']}</span>
                    <span class="cls-badge">{html.escape(s['class'])}</span>
                    <span class="dim-badge">{s['dimensions']}</span>
                </div>
                <div class="card-metrics-group">
                    <span class="score-badge {score_cls}">Score: {score_val}%</span>
                    <span class="status-badge {'status-pass' if s['is_valid'] else 'status-fail'}">
                        {'PASS' if s['is_valid'] else 'FLAGGED'}
                    </span>
                </div>
            </div>

            <div class="anomaly-bar">
                {anom_chips}
            </div>

            <div class="comparison-grid">
                <div class="comparison-pane">
                    <div class="pane-label">
                        <svg class="icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                            <circle cx="8.5" cy="8.5" r="1.5"></circle>
                            <polyline points="21 15 16 10 5 21"></polyline>
                        </svg>
                        Original Staff Scan Crop
                    </div>
                    <div class="image-wrapper">
                        <img src="{s['crop_base64']}" alt="Original Crop P{s['page']:04d} #{s['staff_idx']}">
                    </div>
                </div>

                <div class="comparison-pane">
                    <div class="pane-label">
                        <svg class="icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M9 18V5l12-2v13"></path>
                            <circle cx="6" cy="18" r="3"></circle>
                            <circle cx="18" cy="16" r="3"></circle>
                        </svg>
                        Verovio Vector Notation Render (Decoded ABC)
                    </div>
                    <div class="image-wrapper verovio-wrapper">
                        {render_box}
                    </div>
                </div>
            </div>

            <details class="code-drawer">
                <summary>Inspect Raw ABC / Humdrum Kern</summary>
                <div class="drawer-content">
                    <div class="code-block">
                        <div class="code-title">ABC Notation:</div>
                        <pre><code>{abc_code}</code></pre>
                    </div>
                    <div class="code-block">
                        <div class="code-title">Humdrum **kern:</div>
                        <pre><code>{kern_code}</code></pre>
                    </div>
                </div>
            </details>
        </div>
        """
        cards_html.append(card_html)

    body_cards = "\n".join(cards_html) if cards_html else '<div class="empty-notice">No snippets processed.</div>'

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<style>
:root {{
    --bg-main: #0b0f17;
    --bg-card: #131b26;
    --bg-pane: #1b2636;
    --border-color: #27374d;
    --border-highlight: #3d567a;
    --text-primary: #e6edf3;
    --text-muted: #8b9bb4;
    --accent-blue: #388bfd;
    --color-success: #238636;
    --color-warning: #9e6a03;
    --color-danger: #da3633;
    --badge-bg: #21262d;
}}

* {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}}

body {{
    background-color: var(--bg-main);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 13px;
    line-height: 1.5;
    padding: 24px;
}}

.dashboard-container {{
    max-width: 1440px;
    margin: 0 auto;
}}

header {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 24px;
}}

.header-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}}

h1 {{
    font-size: 20px;
    font-weight: 600;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 10px;
}}

.metrics-strip {{
    display: flex;
    gap: 16px;
}}

.metric-pill {{
    background: var(--bg-pane);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 12px;
}}

.metric-val {{
    font-weight: 700;
    color: #ffffff;
    margin-left: 4px;
}}

.filter-bar {{
    display: flex;
    gap: 10px;
    align-items: center;
    padding-top: 12px;
    border-top: 1px solid var(--border-color);
}}

.filter-btn {{
    background: var(--bg-pane);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    padding: 6px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 500;
    transition: all 0.15s ease;
}}

.filter-btn:hover {{
    background: var(--border-highlight);
    color: #ffffff;
}}

.filter-btn.active {{
    background: var(--accent-blue);
    border-color: var(--accent-blue);
    color: #ffffff;
}}

.cards-stream {{
    display: flex;
    flex-direction: column;
    gap: 20px;
}}

.snippet-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    overflow: hidden;
    transition: border-color 0.2s;
}}

.snippet-card.anomaly {{
    border-left: 4px solid var(--color-danger);
}}

.snippet-card.clean {{
    border-left: 4px solid var(--color-success);
}}

.card-header {{
    background: rgba(255, 255, 255, 0.02);
    border-bottom: 1px solid var(--border-color);
    padding: 10px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}

.card-title-group, .card-metrics-group {{
    display: flex;
    align-items: center;
    gap: 10px;
}}

.page-badge {{
    font-family: monospace;
    font-weight: 700;
    font-size: 13px;
    color: #ffffff;
}}

.cls-badge, .dim-badge {{
    background: var(--badge-bg);
    border: 1px solid var(--border-color);
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    color: var(--text-muted);
}}

.score-badge {{
    font-family: monospace;
    font-size: 12px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 4px;
}}

.badge-success {{
    background: rgba(35, 134, 54, 0.2);
    color: #3fb950;
    border: 1px solid var(--color-success);
}}

.badge-warning {{
    background: rgba(158, 106, 3, 0.2);
    color: #d29922;
    border: 1px solid var(--color-warning);
}}

.badge-danger {{
    background: rgba(218, 54, 51, 0.2);
    color: #f85149;
    border: 1px solid var(--color-danger);
}}

.status-badge {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    padding: 2px 6px;
    border-radius: 4px;
}}

.status-pass {{
    color: #3fb950;
}}

.status-fail {{
    color: #f85149;
}}

.anomaly-bar {{
    padding: 8px 16px;
    background: rgba(0, 0, 0, 0.2);
    border-bottom: 1px solid var(--border-color);
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
}}

.anomaly-chip {{
    background: rgba(218, 54, 51, 0.15);
    border: 1px solid rgba(218, 54, 51, 0.4);
    color: #ff7b72;
    font-family: monospace;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
}}

.chip-clean {{
    color: #3fb950;
    font-size: 11px;
}}

.comparison-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    padding: 16px;
}}

@media (max-width: 1024px) {{
    .comparison-grid {{
        grid-template-columns: 1fr;
    }}
}}

.comparison-pane {{
    display: flex;
    flex-direction: column;
    gap: 8px;
}}

.pane-label {{
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 6px;
}}

.image-wrapper {{
    background: #ffffff;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    min-height: 120px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 8px;
    overflow-x: auto;
}}

.image-wrapper img {{
    max-width: 100%;
    max-height: 240px;
    object-fit: contain;
    display: block;
}}

.rendered-score-box {{
    width: 100%;
    overflow-x: auto;
    display: flex;
    justify-content: center;
}}

.rendered-score-box svg {{
    max-height: 240px;
    width: auto !important;
}}

.code-drawer {{
    border-top: 1px solid var(--border-color);
    padding: 8px 16px;
    background: rgba(0, 0, 0, 0.15);
    font-size: 11px;
}}

.code-drawer summary {{
    cursor: pointer;
    color: var(--text-muted);
    font-weight: 500;
    user-select: none;
}}

.code-drawer summary:hover {{
    color: var(--text-primary);
}}

.drawer-content {{
    margin-top: 10px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}}

.code-block pre {{
    background: #080c12;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 8px;
    font-family: monospace;
    font-size: 11px;
    max-height: 160px;
    overflow-y: auto;
    color: #8be9fd;
}}

.code-title {{
    font-weight: 600;
    margin-bottom: 4px;
    color: var(--text-muted);
}}
</style>
<script>
function filterCards(cat) {{
    document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    document.querySelectorAll('.snippet-card').forEach(card => {{
        if (cat === 'all') {{
            card.style.display = 'block';
        }} else if (cat === 'anomaly') {{
            card.style.display = card.classList.contains('anomaly') ? 'block' : 'none';
        }} else if (cat === 'clean') {{
            card.style.display = card.classList.contains('clean') ? 'block' : 'none';
        }}
    }});
}}
</script>
</head>
<body>
<div class="dashboard-container">
    <header>
        <div class="header-top">
            <h1>
                <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M9 18V5l12-2v13"></path>
                    <circle cx="6" cy="18" r="3"></circle>
                    <circle cx="18" cy="16" r="3"></circle>
                </svg>
                {html.escape(title)}
            </h1>
            <div class="metrics-strip">
                <div class="metric-pill">Total Snippets: <span class="metric-val">{total_count}</span></div>
                <div class="metric-pill">Structural Passes: <span class="metric-val" style="color: #3fb950;">{clean_count}</span></div>
                <div class="metric-pill">Anomalies Flagged: <span class="metric-val" style="color: #f85149;">{anomaly_count}</span></div>
                <div class="metric-pill">Avg Score: <span class="metric-val">{avg_score:.1f}%</span></div>
            </div>
        </div>
        <div class="filter-bar">
            <span style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; font-weight: 700;">Filter:</span>
            <button class="filter-btn active" onclick="filterCards('all')">All Snippets ({total_count})</button>
            <button class="filter-btn" onclick="filterCards('anomaly')">Flagged Anomalies ({anomaly_count})</button>
            <button class="filter-btn" onclick="filterCards('clean')">Clean Pass ({clean_count})</button>
        </div>
    </header>

    <div class="cards-stream">
        {body_cards}
    </div>
</div>
</body>
</html>
"""
    return html_content


def run_visual_report_pipeline(
    resolver: Any,
    detector: Any,
    pages: List[int],
    output_html_path: Path,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Executes OMR transcription and Verovio rendering across requested book pages,
    evaluating structural invariants and generating the interactive HTML dashboard.
    """
    omr = OMREngine(device=device)
    validator = NotationValidator(bridge=omr.bridge)
    snippets: List[Dict[str, Any]] = []

    assets_dir = output_html_path.parent / "omr_report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    from tools.debug_toolkit import inspect_page

    t0 = time.perf_counter()
    for p_num in pages:
        # 1. Resolve and deskew page
        img_bgr = resolver.get_page_bgr(p_num)
        h, w = img_bgr.shape[:2]

        # 2. Detect layout blocks via standard toolkit inspect_page
        page_res = inspect_page(resolver, detector, p_num, render=False)
        blocks = page_res.get("blocks", [])

        # Collect and normalize crops on this page for batched processing
        page_crops: List[np.ndarray] = []
        page_classes: List[str] = []
        page_indices: List[int] = []

        for s_idx, b in enumerate(blocks):
            px1, py1, px2, py2 = b["pad"]
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(w, px2), min(h, py2)

            crop = img_bgr[py1:py2, px1:px2]
            if crop.size == 0:
                continue

            norm_crop, _, _ = normalize_staff_crop(crop, notation_class=b["cls"])
            page_crops.append(norm_crop)
            page_classes.append(b["cls"])
            page_indices.append(s_idx + 1)

        if not page_crops:
            continue

        titles = [f"P{p_num:04d}_S{idx:02d}" for idx in page_indices]
        omr_results = omr.transcribe_crops_batch(
            crops_bgr=page_crops,
            notation_classes=page_classes,
            titles=titles
        )

        for s_num, norm_crop, cls_name, res in zip(page_indices, page_crops, page_classes, omr_results):
            crop_h, crop_w = norm_crop.shape[:2]
            raw_kern = res.get("raw_kern", "")
            abc_text = res.get("abc", "")

            # Run structural invariant validation
            rep: ValidationReport = validator.validate(
                raw_kern=raw_kern,
                abc_text=abc_text,
                crop_width=crop_w,
                crop_height=crop_h,
                notation_class=cls_name
            )

            # Render to SVG using Verovio
            svg_str = ""
            if raw_kern:
                try:
                    svg_str = omr.bridge.render_svg(raw_kern, scale=80)
                except Exception:
                    svg_str = ""

            crop_b64 = _img_to_base64_png(norm_crop)
            asset_png = assets_dir / f"crop_p{p_num:04d}_s{s_num:02d}.png"
            cv2.imwrite(str(asset_png), norm_crop)

            if svg_str:
                asset_svg = assets_dir / f"render_p{p_num:04d}_s{s_num:02d}.svg"
                asset_svg.write_text(svg_str, encoding="utf-8")

            snippets.append({
                "page": p_num,
                "staff_idx": s_num,
                "class": cls_name,
                "dimensions": f"{crop_w}x{crop_h}px",
                "crop_base64": crop_b64,
                "svg_content": svg_str,
                "score": rep.score,
                "is_valid": rep.is_valid,
                "anomalies": rep.anomalies,
                "abc": abc_text,
                "kern": raw_kern,
            })

    # Memory purge barrier
    omr.purge_gpu_memory()
    elapsed = time.perf_counter() - t0

    # Build and write HTML
    html_content = build_html_report(snippets, title=f"OMR Visual Diagnostic Report (Pages {min(pages)}..{max(pages)})")
    output_html_path.write_text(html_content, encoding="utf-8")

    total_snippets = len(snippets)
    anomalies_count = sum(1 for s in snippets if not s["is_valid"] or s["anomalies"])
    pass_count = total_snippets - anomalies_count

    return {
        "report_path": str(output_html_path),
        "total_snippets": total_snippets,
        "pass_count": pass_count,
        "anomalies_count": anomalies_count,
        "elapsed_sec": round(elapsed, 2),
        "snippets": snippets
    }
