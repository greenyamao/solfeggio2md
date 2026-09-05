import base64
import gc
import json
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.request

import cv2
import music21
import numpy as np
import pymupdf as fitz
import streamlit as st
import torch

torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Подключение SMT (Sheet Music Transformer)
SMT_IMPORT_ERROR = None
try:
    from SMT.data_augmentation.data_augmentation import convert_img_to_tensor
    from SMT.smt_model import SMTModelForCausalLM
    SMT_AVAILABLE = True
except Exception as exc:
    SMT_AVAILABLE = False
    SMT_IMPORT_ERROR = str(exc)

st.set_page_config(layout="wide", page_title="OMR & Book Converter")

st.markdown(
    """
    <style>
    .block-container { max-width: 1500px; padding-top: 2rem; padding-bottom: 4rem; }
    [data-testid="stMetric"] { background: #f5f7fa; border: 1px solid #e4e8ee; padding: 0.8rem; border-radius: 10px; }
    .queue-card { background: #f8fafc; border: 1px solid #e4e8ee; border-radius: 10px; padding: 0.8rem 1rem; margin: 0.35rem 0; }
    .muted { color: #657184; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

CONFIG_FILE = Path("config.json")
INPUT_DIR = Path.cwd() / "in"
DEFAULT_CONFIG = {
    "lm_host": "127.0.0.1",
    "lm_port": "1234",
    "lm_model": "default",
    "lm_temperature": 0.1,
    "lm_max_tokens": 8192,
    "smt_max_tokens": 512,
    "output_dir": str(Path.cwd() / "output"),
    "dpi": 200,
    "delay": 0.5,
    "overwrite": False,
    "smt_device": "cuda" if torch.cuda.is_available() else "cpu",
    "smt_model": "antoniorv6/smt-grandstaff",
    "qwen_context_length": 16196,
    "qwen_eval_batch_size": 2048,
    "qwen_flash_attention": True,
    "qwen_offload_kv_cache_to_gpu": True,
    "system_prompt": (
        "Ты — строгий OCR-транскрибатор. Перенеси весь печатный текст страницы в чистый Markdown дословно.\n"
        "Сохраняй иерархию заголовков (#, ##, ###), таблицы и списки.\n"
        "ВАЖНО: Если на странице встречаются технические метки вида <!-- MUSIC_STUB_ID:... -->, "
        "ОБЯЗАТЕЛЬНО оставь их в тексте на тех же местах без малейших изменений. Ничего не додумывай от себя."
    ),
}


def load_config():
    if CONFIG_FILE.is_file():
        try:
            return {**DEFAULT_CONFIG, **json.loads(CONFIG_FILE.read_text(encoding="utf-8"))}
        except Exception:
            return DEFAULT_CONFIG
    return DEFAULT_CONFIG


def save_config(cfg):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


cfg = load_config()

if "pdf_queue" not in st.session_state:
    st.session_state.pdf_queue = []
if "pdf_sources" not in st.session_state:
    st.session_state.pdf_sources = {}
if "pdf_labels" not in st.session_state:
    st.session_state.pdf_labels = {}
if "output_dir" not in st.session_state:
    st.session_state.output_dir = cfg.get("output_dir", str(Path.cwd() / "output"))
if "is_running" not in st.session_state:
    st.session_state.is_running = False
Path(st.session_state.output_dir).mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)


def stage_pdf(path):
    source = Path(path).resolve()
    if not source.is_file():
        return None

    source_key = str(source).lower()
    existing = st.session_state.pdf_sources.get(source_key)
    if existing and Path(existing).is_file():
        return existing

    numbered_files = []
    for candidate in INPUT_DIR.iterdir():
        if not candidate.is_file() or candidate.suffix.lower() != ".pdf":
            continue
        try:
            numbered_files.append(int(candidate.stem))
        except ValueError:
            continue
    next_number = max(numbered_files, default=0) + 1
    target = INPUT_DIR / f"{next_number}.pdf"
    shutil.copy2(source, target)
    target_key = str(target.resolve()).lower()
    st.session_state.pdf_sources[source_key] = str(target)
    st.session_state.pdf_labels[target_key] = source.name
    return str(target)


def add_pdf_paths(paths):
    existing = list(st.session_state.pdf_queue)
    known = {str(Path(path).resolve()).lower() for path in existing}
    added = 0
    for path in paths:
        staged = stage_pdf(path)
        if staged and str(Path(staged).resolve()).lower() not in known:
            existing.append(staged)
            known.add(str(Path(staged).resolve()).lower())
            added += 1
    st.session_state.pdf_queue = existing
    return added


def remove_pdf(path):
    st.session_state.pdf_queue = [item for item in st.session_state.pdf_queue if item != path]
    staged_key = str(Path(path).resolve()).lower()
    st.session_state.pdf_labels.pop(staged_key, None)
    for source_key, staged_path in list(st.session_state.pdf_sources.items()):
        if str(Path(staged_path).resolve()).lower() == staged_key:
            del st.session_state.pdf_sources[source_key]


def pdf_page_count(path):
    try:
        with fitz.open(path) as doc:
            return len(doc)
    except Exception:
        return "?"


def has_valid_abc_cache(path):
    if not path.exists():
        return False
    content = path.read_text(encoding="utf-8").strip()
    return bool(content) and not content.startswith("% [SMT не подключен]") and not content.startswith("% Ошибка нотации:")


def has_valid_final_cache(path):
    if not path.exists():
        return False
    content = path.read_text(encoding="utf-8")
    return "% [SMT не подключен]" not in content and "% Ошибка нотации:" not in content


# ----------------- ДЕТЕКТОР НОТНЫХ СТАНОВ ----------------- #
def detect_staff_regions(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 120:
        gray = cv2.bitwise_not(gray)

    img_h, img_w = img_bgr.shape[:2]
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, img_w // 80), 1))
    staff_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

    contours, _ = cv2.findContours(staff_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width >= img_w * 0.08 and height <= max(12, img_h * 0.02):
            segments.append((x, y + height / 2, width, height))

    # Connect line segments that belong to the same staff. Horizontal overlap
    # keeps left/right columns separate even when their Y coordinates match.
    parent = list(range(len(segments)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first, left in enumerate(segments):
        for second in range(first + 1, len(segments)):
            right = segments[second]
            vertical_distance = abs(left[1] - right[1])
            overlap = min(left[0] + left[2], right[0] + right[2]) - max(left[0], right[0])
            if vertical_distance <= max(32, img_h * 0.025) and overlap >= min(left[2], right[2]) * 0.25:
                union(first, second)

    component_segments = {}
    for index, segment in enumerate(segments):
        component_segments.setdefault(find(index), []).append(segment)

    candidates = []
    for component in component_segments.values():
        centers = sorted({round(segment[1]) for segment in component})
        if len(centers) < 5:
            continue
        gaps = np.diff(centers)
        typical_gap = float(np.median(gaps)) if gaps.size else 0
        if typical_gap < 2 or np.any(gaps > max(32, typical_gap * 2.2)):
            continue
        left = min(segment[0] for segment in component)
        right = max(segment[0] + segment[2] for segment in component)
        if right - left < img_w * 0.08:
            continue
        pad_x = max(30, int((right - left) * 0.06))
        pad_y = max(14, int(typical_gap * 2.2))
        x1 = max(0, int(left - pad_x))
        y1 = max(0, int(min(centers) - pad_y))
        x2 = min(img_w, int(right + pad_x))
        y2 = min(img_h, int(max(centers) + pad_y))
        roi = thresh[y1:y2, x1:x2]
        horizontal = cv2.morphologyEx(roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, (x2 - x1) // 80), 1)))
        detail_density = np.mean(cv2.bitwise_xor(roi, horizontal) > 0)
        if detail_density > 0.09:
            continue
        candidates.append((x1, y1, x2 - x1, y2 - y1))

    boxes = sorted(candidates, key=lambda box: box[1])
    merged = []
    for box in boxes:
        should_merge = False
        if merged:
            previous = merged[-1]
            union_top = min(previous[1], box[1])
            union_bottom = max(previous[1] + previous[3], box[1] + box[3])
            vertical_gap = max(0, box[1] - (previous[1] + previous[3]), previous[1] - (box[1] + box[3]))
            horizontal_overlap = min(previous[0] + previous[2], box[0] + box[2]) - max(previous[0], box[0])
            should_merge = horizontal_overlap >= max(previous[2], box[2]) * 0.5 and vertical_gap <= img_h * 0.08 and union_bottom - union_top <= img_h * 0.30
        if should_merge:
            previous = merged[-1]
            top = min(previous[1], box[1])
            bottom = max(previous[1] + previous[3], box[1] + box[3])
            left = min(previous[0], box[0])
            right = max(previous[0] + previous[2], box[0] + box[2])
            merged[-1] = (left, top, right - left, bottom - top)
        else:
            merged.append(box)
    return merged


# ----------------- СИНГЛТОН SMT МОДЕЛИ ----------------- #
@st.cache_resource
def get_smt_engine(device: str, model_id: str):
    if not SMT_AVAILABLE:
        return None
    try:
        model = SMTModelForCausalLM.from_pretrained(model_id).to(device)
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Не удалось загрузить SMT: {e}")
        return None


@torch.inference_mode()
def transcribe_crop(model, crop_bgr, device, progress_callback=None, max_tokens=512, raw_output_path=None):
    if model is None:
        return "% [SMT не подключен]"
    try:
        max_height = int(getattr(model.config, "maxh", crop_bgr.shape[0]))
        max_width = int(getattr(model.config, "maxw", crop_bgr.shape[1]))
        scale = min(1.0, max_height / crop_bgr.shape[0], max_width / crop_bgr.shape[1])
        if scale < 1.0:
            crop_bgr = cv2.resize(
                crop_bgr,
                (max(1, int(crop_bgr.shape[1] * scale)), max(1, int(crop_bgr.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        tensor = convert_img_to_tensor(crop_bgr).unsqueeze(0).to(device)
        autocast_context = torch.autocast("cuda", dtype=torch.float16) if str(device).startswith("cuda") else torch.autocast("cpu", enabled=False)
        with autocast_context:
            predictions, _ = model.predict(
                tensor,
                convert_to_str=True,
                progress_callback=progress_callback,
                max_tokens=max_tokens,
            )
        raw_bekern = "".join(predictions).replace("<s>", " ").replace("</s>", "").replace("<t>", "\t").replace("<b>", "\n")
        raw_bekern = normalize_bekern(raw_bekern)
        if raw_output_path:
            raw_output_path.write_text(raw_bekern, encoding="utf-8")
        
        score = music21.converter.parse(raw_bekern, format='humdrum')
        score.makeNotation(inPlace=True)
        abc = music21.converter.subConverters.ConverterABC().subdataABC(score)
        return abc.strip()
    except Exception as exc:
        return f"% Ошибка нотации: {exc}"


def normalize_bekern(raw_bekern):
    text = raw_bekern.strip()
    if not text:
        raise ValueError("SMT вернул пустую нотацию")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("SMT вернул пустую нотацию")

    if not any(line.startswith("**") for line in lines):
        spine_count = max(len(line.split("\t")) for line in lines)
        header = "\t".join(["**kern"] * spine_count)
        terminator = "\t".join(["*-"] * spine_count)
        text = "\n".join([header, *lines, terminator])

    return text


def strip_markdown_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```markdown") and t.endswith("```"):
        return t[11:-3].strip()
    if t.startswith("```md") and t.endswith("```"):
        return t[5:-3].strip()
    return t


def stream_vlm(img_bytes: bytes, prompt: str, host: str, port: str, model: str, temp: float, max_tokens: int, out_placeholder):
    url = f"http://{host.strip()}:{port.strip()}/v1/chat/completions"
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}
        ],
        "temperature": temp,
        "max_tokens": max_tokens,
        "stream": True
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    full_content = []

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line_bytes in resp:
                line = line_bytes.decode("utf-8").strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    try:
                        delta = json.loads(line[6:])["choices"][0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            full_content.append(delta["content"])
                            out_placeholder.markdown("".join(full_content) + " ▌")
                    except Exception:
                        continue
    finally:
        del b64
        del payload

    cleaned = strip_markdown_fences("".join(full_content).strip())
    out_placeholder.markdown(cleaned)
    return cleaned


def lmstudio_request(host, port, path, payload, timeout=900):
    url = f"http://{host.strip()}:{port.strip()}{path}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def load_qwen(host, port, model, context_length, eval_batch_size, flash_attention, offload_kv_cache):
    return lmstudio_request(
        host,
        port,
        "/api/v1/models/load",
        {
            "model": model,
            "context_length": int(context_length),
            "eval_batch_size": int(eval_batch_size),
            "flash_attention": bool(flash_attention),
            "offload_kv_cache_to_gpu": bool(offload_kv_cache),
            "echo_load_config": True,
        },
    )


def unload_lm_model(host, port, instance_id):
    if instance_id:
        return lmstudio_request(host, port, "/api/v1/models/unload", {"instance_id": instance_id}, timeout=120)
    return None


def request_vlm_native(img_bytes, prompt, host, port, model, temp, max_tokens, context_length):
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    result = lmstudio_request(
        host,
        port,
        "/api/v1/chat",
        {
            "model": model,
            "system_prompt": prompt,
            "input": [
                {"type": "text", "content": "Распознай печатный текст на этой странице."},
                {"type": "image", "data_url": f"data:image/jpeg;base64,{b64}"},
            ],
            "reasoning": "off",
            "temperature": temp,
            "max_output_tokens": int(max_tokens),
            "context_length": int(context_length),
            "stream": False,
            "store": False,
        },
    )
    output = [item.get("content", "") for item in result.get("output", []) if item.get("type") == "message"]
    return strip_markdown_fences("\n".join(output).strip())


# ----------------- НАСТРОЙКИ ----------------- #
with st.sidebar:
    st.title("Настройки")
    st.caption("LM Studio отвечает за текст, SMT распознает ноты на CPU.")
    if not SMT_AVAILABLE:
        st.error(f"SMT недоступен: {SMT_IMPORT_ERROR}")
    with st.expander("LM Studio", expanded=True):
        ch_h, ch_p = st.columns([2, 1])
        lm_host = ch_h.text_input("Хост", value=cfg["lm_host"])
        lm_port = ch_p.text_input("Порт", value=cfg["lm_port"])
        lm_model = st.text_input("ID модели", value=cfg["lm_model"])
        lm_temperature = st.number_input("Температура", min_value=0.0, value=float(cfg["lm_temperature"]), step=0.05)
        lm_max_tokens = st.number_input("Максимум токенов", min_value=512, value=int(cfg["lm_max_tokens"]), step=512)
    with st.expander("Обработка", expanded=False):
        dpi = st.number_input("Качество PDF (DPI)", min_value=150, max_value=300, value=int(cfg["dpi"]), step=10)
        delay = st.number_input("Пауза между страницами", min_value=0.0, value=float(cfg["delay"]), step=0.5)
        qwen_context_length = st.number_input("Контекст Qwen", min_value=2048, max_value=262144, value=int(cfg.get("qwen_context_length", 16196)), step=1024)
        qwen_eval_batch_size = st.number_input("Eval batch Qwen", min_value=128, max_value=4096, value=int(cfg.get("qwen_eval_batch_size", 2048)), step=128)
        qwen_flash_attention = st.checkbox("Flash Attention Qwen", value=bool(cfg.get("qwen_flash_attention", True)))
        qwen_offload_kv_cache = st.checkbox("KV cache на GPU", value=bool(cfg.get("qwen_offload_kv_cache_to_gpu", True)))
        smt_model_id = st.text_input("OMR модель", value=cfg.get("smt_model", "antoniorv6/smt-grandstaff"))
        smt_max_tokens = st.number_input(
            "Максимум токенов OMR",
            min_value=64,
            max_value=1281,
            value=min(1281, max(64, int(cfg.get("smt_max_tokens", 512)))),
            step=64,
            help="Меньшее значение ускоряет CPU-распознавание, но может обрезать длинную нотацию.",
        )
        overwrite = st.checkbox("Перезаписывать готовые страницы", value=cfg["overwrite"])
    with st.expander("Инструкция модели", expanded=False):
        system_prompt = st.text_area("Системный промпт", value=cfg["system_prompt"], height=180)
    smt_device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"SMT будет работать на: `{smt_device}`. Перед Qwen GPU будет очищена.")

st.title("PDF → учебник в Markdown")
st.caption("Загрузите книги, дождитесь проверки очереди и запустите обработку. Результаты будут разложены по отдельным папкам автоматически.")

tab_main, tab_state, tab_results = st.tabs(["Рабочая очередь", "История обработки", "Готовые книги"])

with tab_main:
    st.subheader("Добавить книги")
    st.file_uploader(
        "Перетащите сюда один или несколько PDF или нажмите для выбора",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )
    up_files = st.session_state.get("pdf_uploader") or []
    valid_uploads = [
        uploaded for uploaded in up_files
        if getattr(uploaded, "name", None) and callable(getattr(uploaded, "getvalue", None))
    ]
    if valid_uploads:
        td = Path.cwd() / ".queue_temp"
        td.mkdir(exist_ok=True)
        uploaded_paths = []
        for uploaded in valid_uploads:
            target = td / uploaded.name
            target.write_bytes(uploaded.getvalue())
            uploaded_paths.append(str(target))
        add_pdf_paths(uploaded_paths)

    folder_col, folder_button_col = st.columns([5, 1])
    with folder_col:
        folder_path = st.text_input(
            "Или путь к папке с PDF",
            placeholder=r"Например: C:\Учебники\Музыка",
            label_visibility="visible",
        )
    with folder_button_col:
        st.write("")
        add_folder_clicked = st.button("Добавить папку", icon="📁", width="stretch")
    if add_folder_clicked:
        folder = Path(folder_path.strip().strip('"'))
        folder_files = [str(path) for path in sorted(folder.iterdir()) if path.is_file() and path.suffix.lower() == ".pdf"] if folder.is_dir() else []
        added_count = add_pdf_paths(folder_files)
        if added_count:
            st.success(f"Добавлено PDF: {added_count}")
        elif folder_path:
            st.warning("В этой папке не найдено PDF-файлов.")

    active_queue = [p for p in st.session_state.pdf_queue if Path(p).is_file()]
    st.session_state.pdf_queue = active_queue
    queue_count = len(active_queue)
    page_counts = {path: pdf_page_count(path) for path in active_queue}
    output_root = Path(st.session_state.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    m1, m2, m3 = st.columns(3)
    m1.markdown(f"**{queue_count}**  \nВ очереди")
    m2.markdown(f"**{sum(count for count in page_counts.values() if isinstance(count, int))}**  \nСтраниц к обработке")
    m3.markdown(f"**Результаты**  \n`{output_root}`")

    st.divider()
    if active_queue:
        st.subheader("Очередь книг")
        for idx, path in enumerate(active_queue):
            pdf_path = Path(path)
            original_name = st.session_state.pdf_labels.get(str(pdf_path.resolve()).lower(), pdf_path.name)
            item_col, pages_col, remove_col = st.columns([5, 1.5, 0.8])
            with item_col:
                st.markdown(
                    f"**{idx + 1}. {pdf_path.name}**<br><span class='muted'>{original_name}</span>",
                    unsafe_allow_html=True,
                )
            with pages_col:
                st.caption(f"{page_counts[path]} страниц")
            with remove_col:
                if st.button("Убрать", key=f"remove_pdf_{idx}", width="stretch"):
                    remove_pdf(path)
                    st.rerun()
    else:
        st.info("Очередь пуста. Добавьте PDF-файлы или папку с книгами выше.")

    st.divider()
    action_col, stop_col, save_col = st.columns([2, 1, 1])
    btn_start = action_col.button("Запустить обработку", icon="▶️", type="primary", disabled=st.session_state.is_running or not active_queue, width="stretch")
    btn_stop = stop_col.button("Остановить", icon="⏹️", disabled=not st.session_state.is_running, width="stretch")
    save_clicked = save_col.button("Сохранить настройки", icon="💾", width="stretch")
    config_payload = {
        "lm_host": lm_host, "lm_port": lm_port, "lm_model": lm_model,
        "lm_temperature": lm_temperature, "lm_max_tokens": lm_max_tokens,
        "output_dir": str(output_root), "dpi": int(dpi), "delay": float(delay),
        "smt_max_tokens": int(smt_max_tokens),
        "qwen_context_length": int(qwen_context_length),
        "qwen_eval_batch_size": int(qwen_eval_batch_size),
        "qwen_flash_attention": qwen_flash_attention,
        "qwen_offload_kv_cache_to_gpu": qwen_offload_kv_cache,
        "smt_model": smt_model_id,
        "overwrite": overwrite, "smt_device": smt_device, "system_prompt": system_prompt
    }
    if save_clicked:
        save_config(config_payload)
        st.success("Настройки сохранены.")
    if btn_stop:
        st.session_state.is_running = False
        st.rerun()
    if btn_start:
        save_config(config_payload)
        st.session_state.is_running = True
        st.rerun()

    if st.session_state.is_running:
        st.divider()
        view_stat = st.empty()
        preview_col, result_col = st.columns(2)
        with preview_col:
            st.subheader("Предпросмотр страницы")
            view_img = st.empty()
        with result_col:
            st.subheader("Результат Markdown")
            view_res = st.empty()
    else:
        st.caption("Предпросмотр появится здесь после запуска обработки.")

with tab_state:
    st.subheader("История обработки")
    base_out = Path(st.session_state.output_dir)
    books_dirs = [d for d in base_out.iterdir() if d.is_dir()] if base_out.exists() else []
    if books_dirs:
        for book_dir in sorted(books_dirs):
            final_count = len(list((book_dir / "4_final_pages").glob("*.md")))
            crop_count = len(list((book_dir / "1_crops").glob("*.png")))
            complete = (book_dir / f"{book_dir.name}_complete.md").exists()
            status = "Готово" if complete else "В процессе / незавершено"
            st.markdown(f"**{book_dir.name}** · {status} · страниц: {final_count} · нотных фрагментов: {crop_count}")
    else:
        st.info("Здесь появится история после первого запуска.")

with tab_results:
    st.subheader("Готовые книги")
    base_out = Path(st.session_state.output_dir)
    completed = list(base_out.glob("*/*_complete.md")) if base_out.exists() else []
    if completed:
        selected_book = st.selectbox("Выберите книгу", completed, format_func=lambda x: x.parent.name)
        st.download_button("Скачать Markdown", selected_book.read_bytes(), file_name=selected_book.name, mime="text/markdown", width="stretch")
        st.code(selected_book.read_text(encoding="utf-8")[:6000] + "\n\n...[превью обрезано]...", language="markdown")
    else:
        st.info("Готовых книг пока нет.")

# ----------------- ДВУХФАЗНЫЙ ИСПОЛНИТЕЛЬНЫЙ ЦИКЛ ----------------- #
if st.session_state.is_running:
    base_out = Path(st.session_state.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    smt_model = None
    qwen_instance_id = None

    try:
        # Phase 1: unload Qwen, then run all OMR pages on the GPU.
        view_stat.info("Фаза 1/2: освобождаем LM Studio и загружаем SMT на GPU...")
        try:
            models = json.loads(urllib.request.urlopen(
                f"http://{lm_host.strip()}:{lm_port.strip()}/api/v1/models", timeout=30
            ).read().decode("utf-8")).get("models", [])
            qwen_info = next((item for item in models if item.get("key") == lm_model), None)
            for loaded_instance in (qwen_info or {}).get("loaded_instances", []):
                unload_lm_model(lm_host, lm_port, loaded_instance.get("id"))
        except Exception as exc:
            view_stat.warning(f"Qwen не была выгружена перед OMR: {exc}")

        smt_model = get_smt_engine(smt_device, smt_model_id)
        if smt_model is None:
            raise RuntimeError("SMT не загрузилась, OMR-фаза остановлена")

        total_pages_all = sum(count for count in page_counts.values() if isinstance(count, int))
        processed_pages = 0
        phase_progress = st.progress(0.0, text="Фаза 1/2: OMR GPU")
        phase_detail = st.empty()

        for pdf_path_str in active_queue:
            if not st.session_state.is_running:
                break
            pdf_path = Path(pdf_path_str)
            book_title = pdf_path.stem
            book_dir = base_out / book_title
            crops_dir = book_dir / "1_crops"
            masked_dir = book_dir / "2_masked_pages"
            raw_md_dir = book_dir / "3_raw_md"
            final_dir = book_dir / "4_final_pages"
            for directory in [crops_dir, masked_dir, raw_md_dir, final_dir]:
                directory.mkdir(parents=True, exist_ok=True)

            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                for p_num in range(1, total_pages + 1):
                    if not st.session_state.is_running:
                        break
                    view_stat.warning(f"Фаза 1/2 OMR: {book_title} | страница {p_num}/{total_pages}")
                    page = doc[p_num - 1]
                    pix = page.get_pixmap(dpi=int(dpi))
                    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
                    del pix
                    boxes = detect_staff_regions(img_bgr)
                    masked_img = img_bgr.copy()
                    crop_jobs = []

                    for s_id, (x, y, w, h) in enumerate(boxes, start=1):
                        crop_tag = f"{book_title}_P{p_num:04d}_S{s_id:02d}"
                        crop_png = crops_dir / f"{crop_tag}.png"
                        crop_abc_file = crops_dir / f"{crop_tag}.abc"
                        crop_krn_file = crops_dir / f"{crop_tag}.krn"
                        crop_mat = img_bgr[y:y+h, x:x+w]
                        cached_crop = cv2.imread(str(crop_png)) if crop_png.exists() else None
                        geometry_changed = cached_crop is None or cached_crop.shape[:2] != crop_mat.shape[:2]
                        if geometry_changed or overwrite:
                            cv2.imwrite(str(crop_png), crop_mat)
                        crop_jobs.append((crop_tag, crop_abc_file, crop_krn_file, crop_mat, geometry_changed))
                        cv2.rectangle(masked_img, (x, y), (x + w, y + h), (255, 255, 255), -1)
                        cv2.putText(masked_img, f"<!-- MUSIC_STUB_ID:{crop_tag} -->", (x + 10, y + max(25, int(h / 2))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)

                    masked_page_png = masked_dir / f"page_{p_num:04d}_masked.png"
                    cv2.imwrite(str(masked_page_png), masked_img)
                    _, masked_jpg_bytes = cv2.imencode(".jpg", masked_img)
                    view_img.image(masked_jpg_bytes.tobytes(), caption=f"OMR: страница {p_num}, систем: {len(boxes)}", width="stretch")
                    for crop_index, (_, crop_abc_file, crop_krn_file, crop_mat, geometry_changed) in enumerate(crop_jobs, start=1):
                        total_crops = max(1, len(crop_jobs))
                        def update_omr_progress(token_index, token_total, finished=False):
                            crop_fraction = ((crop_index - 1) + min(1.0, token_index / max(1, token_total))) / total_crops
                            page_fraction = (processed_pages + crop_fraction) / max(1, total_pages_all)
                            phase_progress.progress(page_fraction, text=f"Фаза 1/2 OMR GPU | стр. {processed_pages + 1}/{total_pages_all} | кроп {crop_index}/{total_crops} | токен {token_index}/{token_total}")
                            phase_detail.caption(f"Текущий crop: {crop_index}/{total_crops} | токен {token_index}/{token_total}")
                        if has_valid_abc_cache(crop_abc_file) and not overwrite and not geometry_changed:
                            update_omr_progress(1, 1, True)
                        else:
                            abc_content = transcribe_crop(smt_model, crop_mat, smt_device, update_omr_progress, int(smt_max_tokens), crop_krn_file)
                            crop_abc_file.write_text(abc_content, encoding="utf-8")
                            update_omr_progress(1, 1, True)
                        del crop_mat

                    del img_bgr, masked_img, masked_jpg_bytes
                    gc.collect()
                    processed_pages += 1
                    phase_progress.progress(processed_pages / max(1, total_pages_all), text=f"Фаза 1/2 OMR GPU | страниц готово {processed_pages}/{total_pages_all}")

        # Explicitly release all SMT CUDA allocations before loading Qwen.
        view_stat.info("Фаза 1/2 завершена. Освобождаем VRAM SMT...")
        del smt_model
        smt_model = None
        get_smt_engine.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Phase 2: load Qwen once, process all masked pages, then unload it.
        view_stat.info("Фаза 2/2: загружаем Qwen с reasoning=off...")
        loaded_qwen = load_qwen(lm_host, lm_port, lm_model, qwen_context_length, qwen_eval_batch_size, qwen_flash_attention, qwen_offload_kv_cache)
        qwen_instance_id = loaded_qwen.get("instance_id")
        view_stat.success(f"Qwen загружена: {loaded_qwen.get('load_config', {})}")
        processed_text_pages = 0
        phase_progress.progress(0.0, text="Фаза 2/2: Qwen text/VLM")

        for pdf_path_str in active_queue:
            pdf_path = Path(pdf_path_str)
            book_title = pdf_path.stem
            book_dir = base_out / book_title
            crops_dir = book_dir / "1_crops"
            masked_dir = book_dir / "2_masked_pages"
            raw_md_dir = book_dir / "3_raw_md"
            final_dir = book_dir / "4_final_pages"
            with fitz.open(pdf_path) as doc:
                for p_num in range(1, len(doc) + 1):
                    if not st.session_state.is_running:
                        break
                    final_page_file = final_dir / f"page_{p_num:04d}.md"
                    raw_md_file = raw_md_dir / f"page_{p_num:04d}_raw.md"
                    masked_page_png = masked_dir / f"page_{p_num:04d}_masked.png"
                    if has_valid_final_cache(final_page_file) and not overwrite:
                        view_res.markdown(final_page_file.read_text(encoding="utf-8"))
                        processed_text_pages += 1
                        phase_progress.progress(processed_text_pages / max(1, total_pages_all), text=f"Фаза 2/2 Qwen | страниц готово {processed_text_pages}/{total_pages_all}")
                        continue
                    view_stat.warning(f"Фаза 2/2 текст: {book_title} | страница {p_num}/{len(doc)}")
                    masked_bytes = masked_page_png.read_bytes()
                    view_img.image(masked_bytes, caption=f"Текстовая фаза: страница {p_num}", width="stretch")
                    if raw_md_file.exists() and not overwrite:
                        raw_text = raw_md_file.read_text(encoding="utf-8")
                    else:
                        raw_text = request_vlm_native(masked_bytes, system_prompt, lm_host, lm_port, lm_model, lm_temperature, lm_max_tokens, qwen_context_length)
                        raw_md_file.write_text(raw_text, encoding="utf-8")

                    def inject_abc(match):
                        cid = match.group(1).strip()
                        abc_file = crops_dir / f"{cid}.abc"
                        abc = abc_file.read_text(encoding="utf-8") if abc_file.exists() else "% [Ноты не найдены]"
                        return f"\n\n```abc\n{abc}\n```\n\n"

                    final_text = re.sub(r"<!--\s*MUSIC_STUB_ID:\s*(.*?)\s*-->", inject_abc, raw_text)
                    final_page_file.write_text(final_text, encoding="utf-8")
                    view_res.markdown(final_text)
                    processed_text_pages += 1
                    phase_progress.progress(processed_text_pages / max(1, total_pages_all), text=f"Фаза 2/2 Qwen | страниц готово {processed_text_pages}/{total_pages_all}")

            all_pages = sorted(final_dir.glob("page_*.md"))
            full_content = "\n\n---\n\n".join([f"<!-- PAGE {page.stem} -->\n" + page.read_text(encoding="utf-8") for page in all_pages])
            (book_dir / f"{book_title}_complete.md").write_text(full_content, encoding="utf-8")
    finally:
        if smt_model is not None:
            del smt_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if qwen_instance_id:
            try:
                unload_lm_model(lm_host, lm_port, qwen_instance_id)
            except Exception as exc:
                st.error(f"Не удалось выгрузить Qwen: {exc}")
        st.session_state.is_running = False
        st.success("Обе фазы завершены, модели выгружены из памяти.")