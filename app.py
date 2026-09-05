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
    "output_dir": str(Path.cwd() / "output"),
    "dpi": 200,
    "delay": 0.5,
    "overwrite": False,
    "smt_device": "cpu",
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
    
    # Инвертируем изображение, если страница темная
    if np.mean(gray) < 120:
        gray = cv2.bitwise_not(gray)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)

    img_h, img_w = img_bgr.shape[:2]
    # Оставляем только длинные горизонтальные штрихи. Текстовые строки обычно
    # не образуют несколько параллельных линий с регулярным шагом.
    h_len = max(25, int(img_w / 30))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    staff_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

    # Вертикальная склейка (строго в пределах одной системы)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(img_h / 80))))
    dilated = cv2.dilate(staff_lines, v_kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > img_w * 0.40 and (img_h * 0.015 < h < img_h * 0.30):
            region = staff_lines[y:min(img_h, y + h), x:min(img_w, x + w)]
            row_projection = np.count_nonzero(region, axis=1)
            active_rows = row_projection > max(8, int(w * 0.12))

            # Count distinct horizontal lines and their regularity. This rejects
            # long borders, underlines, and most ordinary text blocks.
            line_groups = []
            for row_idx, is_active in enumerate(active_rows):
                if is_active and (not line_groups or row_idx > line_groups[-1][-1] + 1):
                    line_groups.append([row_idx])
                elif is_active:
                    line_groups[-1].append(row_idx)
            line_centers = [sum(group) / len(group) for group in line_groups]
            gaps = np.diff(line_centers)
            regular_gaps = gaps[(gaps >= 2) & (gaps <= max(12, h * 0.20))]
            line_coverage = float(np.count_nonzero(row_projection > max(8, int(w * 0.12)))) / max(h, 1)

            if len(line_centers) < 4 or len(regular_gaps) < 3 or line_coverage > 0.45:
                continue

            pad_y = int(h * 0.15)
            y1 = max(0, y - pad_y)
            y2 = min(img_h, y + h + pad_y)
            boxes.append((x, y1, w, y2 - y1))

    boxes.sort(key=lambda b: b[1])
    merged = []
    for box in boxes:
        if merged and box[1] <= merged[-1][1] + merged[-1][3] * 0.25:
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
def get_smt_engine(device: str):
    if not SMT_AVAILABLE:
        return None
    try:
        # LM Studio owns the GPU. Keeping OMR on CPU prevents a second model
        # from competing with the 8 GB VRAM budget.
        device = "cpu"
        model = SMTModelForCausalLM.from_pretrained("antoniorv6/smt-camera-grandstaff").to(device)
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Не удалось загрузить SMT: {e}")
        return None


@torch.inference_mode()
def transcribe_crop(model, crop_bgr, device):
    if model is None:
        return "% [SMT не подключен]"
    try:
        # This process must never move OMR tensors onto the GPU used by LM Studio.
        tensor = convert_img_to_tensor(crop_bgr).unsqueeze(0).to("cpu")
        predictions, _ = model.predict(tensor, convert_to_str=True)
        raw_bekern = "".join(predictions).replace("<s>", "").replace("</s>", "").replace("<t>", "\t").replace("<b>", "\n")
        
        score = music21.converter.parse(raw_bekern, format='humdrum')
        score.makeNotation(inPlace=True)
        abc = music21.converter.subConverters.ConverterABC().subdataABC(score)
        return abc.strip()
    except Exception as exc:
        return f"% Ошибка нотации: {exc}"


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
        overwrite = st.checkbox("Перезаписывать готовые страницы", value=cfg["overwrite"])
    with st.expander("Инструкция модели", expanded=False):
        system_prompt = st.text_area("Системный промпт", value=cfg["system_prompt"], height=180)
    smt_device = "cpu"

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
    if up_files:
        td = Path.cwd() / ".queue_temp"
        td.mkdir(exist_ok=True)
        uploaded_paths = []
        for uploaded in up_files:
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

# ----------------- ИСПОЛНИТЕЛЬНЫЙ ЦИКЛ ----------------- #
if st.session_state.is_running:
    base_out = Path(st.session_state.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    smt_model = get_smt_engine(smt_device)

    for b_idx, pdf_path_str in enumerate(active_queue):
        if not st.session_state.is_running:
            break

        pdf_path = Path(pdf_path_str)
        if not pdf_path.is_file():
            continue

        book_title = pdf_path.stem
        book_dir = base_out / book_title
        crops_dir = book_dir / "1_crops"
        masked_dir = book_dir / "2_masked_pages"
        raw_md_dir = book_dir / "3_raw_md"
        final_dir = book_dir / "4_final_pages"

        for d in [crops_dir, masked_dir, raw_md_dir, final_dir]:
            d.mkdir(parents=True, exist_ok=True)

        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)

            for p_num in range(1, total_pages + 1):
                if not st.session_state.is_running:
                    break

                final_page_file = final_dir / f"page_{p_num:04d}.md"
                masked_page_png = masked_dir / f"page_{p_num:04d}_masked.png"
                raw_md_file = raw_md_dir / f"page_{p_num:04d}_raw.md"

                # Проверка чекпоинта
                if has_valid_final_cache(final_page_file) and not overwrite:
                    view_stat.info(f"Стр. {p_num} загружена из кэша.")
                    view_res.markdown(final_page_file.read_text(encoding="utf-8"))
                    continue

                view_stat.warning(f"Книга: {book_title} | Обработка страницы {p_num}/{total_pages}...")

                # 1. Рендер страницы PDF
                page = doc[p_num - 1]
                pix = page.get_pixmap(dpi=int(dpi))
                img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
                del pix

                # 2. Детекция нотных станов (с автоинверсией фона)
                boxes = detect_staff_regions(img_bgr)
                masked_img = img_bgr.copy()
                scores_on_page = {}

                for s_id, (x, y, w, h) in enumerate(boxes, start=1):
                    crop_tag = f"{book_title}_P{p_num:04d}_S{s_id:02d}"
                    crop_png = crops_dir / f"{crop_tag}.png"
                    crop_abc_file = crops_dir / f"{crop_tag}.abc"

                    crop_mat = img_bgr[y:y+h, x:x+w]
                    if not crop_png.exists() or overwrite:
                        cv2.imwrite(str(crop_png), crop_mat)

                    # 3. Инференс SMT
                    if has_valid_abc_cache(crop_abc_file) and not overwrite:
                        abc_content = crop_abc_file.read_text(encoding="utf-8")
                    else:
                        abc_content = transcribe_crop(smt_model, crop_mat, smt_device)
                        crop_abc_file.write_text(abc_content, encoding="utf-8")

                    scores_on_page[crop_tag] = abc_content

                    # Белая заплата и ID-метка на очищенном скане
                    cv2.rectangle(masked_img, (x, y), (x + w, y + h), (255, 255, 255), -1)
                    cv2.putText(
                        masked_img,
                        f"<!-- MUSIC_STUB_ID:{crop_tag} -->",
                        (x + 10, y + max(25, int(h / 2))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (80, 80, 80),
                        2
                    )

                cv2.imwrite(str(masked_page_png), masked_img)
                _, masked_jpg_bytes = cv2.imencode(".jpg", masked_img)
                view_img.image(masked_jpg_bytes.tobytes(), caption=f"Стр. {p_num}: {len(boxes)} нотных систем найдено", width="stretch")

                # 4. Распознавание текста через LM Studio (без нот модель не галлюцинирует)
                if raw_md_file.exists() and not overwrite:
                    raw_text = raw_md_file.read_text(encoding="utf-8")
                else:
                    raw_text = stream_vlm(
                        img_bytes=masked_jpg_bytes.tobytes(),
                        prompt=system_prompt,
                        host=lm_host,
                        port=lm_port,
                        model=lm_model,
                        temp=lm_temperature,
                        max_tokens=lm_max_tokens,
                        out_placeholder=view_res
                    )
                    raw_md_file.write_text(raw_text, encoding="utf-8")

                # 5. Подстановка ABC-нотации вместо меток
                def inject_abc(match):
                    cid = match.group(1).strip()
                    abc = scores_on_page.get(cid, "")
                    if not abc:
                        c_file = crops_dir / f"{cid}.abc"
                        abc = c_file.read_text(encoding="utf-8") if c_file.exists() else "% [Ноты не найдены]"
                    return f"\n\n```abc\n{abc}\n```\n\n"

                final_text = re.sub(r"<!--\s*MUSIC_STUB_ID:\s*(.*?)\s*-->", inject_abc, raw_text)
                final_page_file.write_text(final_text, encoding="utf-8")
                view_res.markdown(final_text)

                # Очистка памяти
                del img_bgr, masked_img
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                time.sleep(float(delay))

            # Склейка всей книги в единый файл
            all_pages = sorted(final_dir.glob("page_*.md"))
            full_content = "\n\n---\n\n".join([f"<!-- PAGE {p.stem} -->\n" + p.read_text(encoding="utf-8") for p in all_pages])
            (book_dir / f"{book_title}_complete.md").write_text(full_content, encoding="utf-8")

    st.session_state.is_running = False
    st.success("Все книги из очереди успешно обработаны!")