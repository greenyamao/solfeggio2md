import base64
import gc
import json
from pathlib import Path
import re
import sys
import time
import tkinter as tk
from tkinter import filedialog
import urllib.request

import cv2
import music21
import numpy as np
import pymupdf as fitz
import streamlit as st
import torch

# Подключение SMT репозитория
sys.path.append(str(Path.cwd() / "SMT"))
try:
    from data_augmentation import convert_img_to_tensor
    from smt_model import SMTModelForCausalLM
    SMT_AVAILABLE = True
except ImportError:
    SMT_AVAILABLE = False

st.set_page_config(layout="wide", page_title="4-Stage OMR & Book Pipeline")

CONFIG_FILE = Path("config.json")
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
    "smt_device": "cuda" if torch.cuda.is_available() else "cpu",
    "system_prompt": (
        "Ты — строгий OCR-транскрибатор. Перенеси весь текст страницы книги в Markdown дословно.\n"
        "Сохраняй заголовки, списки и таблицы.\n"
        "ВАЖНО: Если на странице встречаются технические метки вида <!-- MUSIC_STUB_ID:XXXX -->, "
        "ОБЯЗАТЕЛЬНО оставь их в тексте на тех же местах без изменений. Ничего не додумывай."
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
if "output_dir" not in st.session_state:
    st.session_state.output_dir = cfg.get("output_dir", str(Path.cwd() / "output"))
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# ----------------- СИНГЛТОН SMT МОДЕЛИ ----------------- #
@st.cache_resource
def get_smt_engine(device: str):
    if not SMT_AVAILABLE:
        return None
    try:
        model = SMTModelForCausalLM.from_pretrained("antoniorv6/smt-camera-grandstaff").to(device)
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Не удалось инициализировать веса SMT: {e}")
        return None

# ----------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------------- #
def strip_markdown_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```markdown") and t.endswith("```"):
        return t[11:-3].strip()
    if t.startswith("```md") and t.endswith("```"):
        return t[5:-3].strip()
    return t

def detect_staff_regions(img_bgr):
    """Поиск координат нотных станов на CPU морфологией OpenCV."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    
    h_size = max(10, int(img_bgr.shape[1] / 30))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
    lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=1)
    
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(img_bgr.shape[0] / 40))))
    dilated = cv2.dilate(lines, v_kernel, iterations=3)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    img_h, img_w = img_bgr.shape[:2]
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > img_w * 0.35 and 25 < h < img_h * 0.65:
            pad_y = int(h * 0.12)
            pad_x = int(w * 0.02)
            y1 = max(0, y - pad_y)
            y2 = min(img_h, y + h + pad_y)
            x1 = max(0, x - pad_x)
            x2 = min(img_w, x + w + pad_x)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
            
    boxes.sort(key=lambda b: b[1])
    return boxes

@torch.inference_mode()
def transcribe_crop(model, crop_bgr, device):
    """Распознавание кропа в SMT + валидация через music21."""
    if model is None:
        return "% [SMT модель не загружена]"
    try:
        tensor = convert_img_to_tensor(crop_bgr).unsqueeze(0).to(device)
        predictions, _ = model.predict(tensor, convert_to_str=True)
        raw_bekern = "".join(predictions).replace("<s>", "").replace("</s>", "").replace("<t>", "\t").replace("<b>", "\n")
        
        score = music21.converter.parse(raw_bekern, format='humdrum')
        score.makeNotation(inPlace=True)
        abc = music21.converter.subConverters.ConverterABC().subdataABC(score)
        return abc.strip()
    except Exception as exc:
        return f"% Ошибка нотации: {exc}"

def stream_vlm(img_bytes: bytes, prompt: str, host: str, port: str, model: str, temp: float, max_tokens: int, out_placeholder):
    url = f"http://{host.strip()}:{port.strip()}/v1/chat/completions"
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}
        ],
        "temperature": temp,
        "max_tokens": max_tokens,
        "stream": True
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    full_content = []
    
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
                    
    cleaned = strip_markdown_fences("".join(full_content).strip())
    out_placeholder.markdown(cleaned)
    return cleaned

# ----------------- БОКОВАЯ ПАНЕЛЬ ----------------- #
with st.sidebar:
    st.title("Параметры инференса")
    ch_h, ch_p = st.columns([2, 1])
    lm_host = ch_h.text_input("Хост", value=cfg["lm_host"])
    lm_port = ch_p.text_input("Порт", value=cfg["lm_port"])
    lm_model = st.text_input("ID модели", value=cfg["lm_model"])
    
    ct, ctok = st.columns(2)
    lm_temperature = ct.number_input("Temp", min_value=0.0, value=float(cfg["lm_temperature"]), step=0.05)
    lm_max_tokens = ctok.number_input("Max Tokens", min_value=512, value=int(cfg["lm_max_tokens"]), step=512)
    
    st.subheader("Параметры OMR / SMT")
    smt_device = st.selectbox("Устройство для SMT", ["cuda", "cpu"], index=0 if cfg["smt_device"] == "cuda" and torch.cuda.is_available() else 1)
    
    st.subheader("Рендер")
    dpi = st.number_input("DPI", min_value=150, max_value=300, value=int(cfg["dpi"]), step=10)
    delay = st.number_input("Пауза (сек)", min_value=0.0, value=float(cfg["delay"]), step=0.5)
    overwrite = st.checkbox("Перезаписывать готовое", value=cfg["overwrite"])
    
    with st.expander("Системный промпт текста"):
        system_prompt = st.text_area("Инструкция", value=cfg["system_prompt"], height=160)

# ----------------- ВКЛАДКИ GUI ----------------- #
tab_main, tab_state, tab_results = st.tabs(["🚀 Панель запуска", "⚙️ Статус и кэш", "📖 Готовые книги"])

with tab_main:
    st.subheader("Очередь обработки документов")
    
    c_btn1, c_btn2, c_path = st.columns([1.5, 1.5, 3])
    with c_btn1:
        if st.button("📂 Выбрать PDF файлы", width="stretch"):
            r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
            files = filedialog.askopenfilenames(filetypes=[("PDF", "*.pdf")]); r.destroy()
            if files: st.session_state.pdf_queue = list(files)
    with c_btn2:
        if st.button("📁 Выбрать папку", width="stretch"):
            r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
            folder = filedialog.askdirectory(); r.destroy()
            if folder: st.session_state.pdf_queue = [str(p) for p in Path(folder).glob("*.pdf")]
    with c_path:
        up_files = st.file_uploader("Загрузка через браузер", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
        if up_files:
            td = Path.cwd() / ".queue_temp"; td.mkdir(exist_ok=True)
            st.session_state.pdf_queue = [str(td / f.name) for f in up_files]
            for f in up_files: (td / f.name).write_bytes(f.read())
            
    c_out_btn, c_out_str = st.columns([1.5, 4.5])
    with c_out_btn:
        if st.button("💾 Папка вывода", width="stretch"):
            r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
            f_out = filedialog.askdirectory(); r.destroy()
            if f_out: st.session_state.output_dir = f_out
    with c_out_str:
        st.session_state.output_dir = st.text_input("Директория", value=st.session_state.output_dir, label_visibility="collapsed")

    active_queue = [p for p in st.session_state.pdf_queue if Path(p).is_file()]
    if active_queue:
        st.caption(f"Книг в очереди: {len(active_queue)}")
        
    c_start, c_stop, c_cfg = st.columns([2, 2, 1])
    btn_start = c_start.button("▶️ СТАРТ ПАЙПЛАЙНА", type="primary", disabled=st.session_state.is_running or not active_queue, width="stretch")
    btn_stop = c_stop.button("⏹️ СТОП", type="secondary", disabled=not st.session_state.is_running, width="stretch")
    
    if c_cfg.button("💾 Сохранить конфиг", width="stretch"):
        save_config({
            "lm_host": lm_host, "lm_port": lm_port, "lm_model": lm_model,
            "lm_temperature": lm_temperature, "lm_max_tokens": lm_max_tokens,
            "output_dir": st.session_state.output_dir, "dpi": int(dpi), "delay": float(delay),
            "overwrite": overwrite, "smt_device": smt_device, "system_prompt": system_prompt
        })
        st.success("Настройки сохранены!")

    if btn_stop:
        st.session_state.is_running = False
        st.rerun()

    if btn_start:
        save_config({
            "lm_host": lm_host, "lm_port": lm_port, "lm_model": lm_model,
            "lm_temperature": lm_temperature, "lm_max_tokens": lm_max_tokens,
            "output_dir": st.session_state.output_dir, "dpi": int(dpi), "delay": float(delay),
            "overwrite": overwrite, "smt_device": smt_device, "system_prompt": system_prompt
        })
        st.session_state.is_running = True
        st.rerun()

    st.divider()
    col_v_in, col_v_out = st.columns([1, 1])
    with col_v_in:
        st.caption("Очищенный скан с ID-заплатками")
        view_stat = st.empty()
        with st.container(height=600):
            view_img = st.empty()
    with col_v_out:
        st.caption("Финальный объединенный Markdown")
        with st.container(height=600):
            view_res = st.empty()

with tab_state:
    st.subheader("Проверка артефактов на диске")
    base_out = Path(st.session_state.output_dir)
    if base_out.exists():
        books_dirs = [d for d in base_out.iterdir() if d.is_dir()]
        for bd in books_dirs:
            p_final = len(list((bd / "4_final_pages").glob("*.md"))) if (bd / "4_final_pages").exists() else 0
            p_crops = len(list((bd / "1_crops").glob("*.png"))) if (bd / "1_crops").exists() else 0
            st.write(f"📖 **{bd.name}**: собрано страниц `{p_final}`, распознано кропов нот `{p_crops}`")
    else:
        st.info("Выходная папка еще не создана.")

with tab_results:
    st.subheader("Скомпилированные файлы")
    if base_out.exists():
        completed = list(base_out.glob("*/*_complete.md"))
        if completed:
            selected_book = st.selectbox("Открыть книгу", completed, format_func=lambda x: x.name)
            if selected_book:
                st.code(selected_book.read_text(encoding="utf-8")[:3000] + "\n\n...[Обрезано для превью]...", language="markdown")
        else:
            st.info("Нет завершенных книг.")

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
                
                # Чекпоинт страницы
                if final_page_file.exists() and not overwrite:
                    view_stat.info(f"Стр. {p_num} загружена из кэша.")
                    view_res.markdown(final_page_file.read_text(encoding="utf-8"))
                    continue

                view_stat.warning(f"Книга: {book_title} | Обработка страницы {p_num}/{total_pages}...")
                
                # ШАГ 1: Рендер оригинального скана
                page = doc[p_num - 1]
                pix = page.get_pixmap(dpi=int(dpi))
                img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
                del pix

                # ШАГ 2: Детекция станов и создание заплат (CPU)
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
                        
                    # ШАГ 3: Распознавание нот через SMT (кэшируется отдельно)
                    if crop_abc_file.exists() and not overwrite:
                        abc_content = crop_abc_file.read_text(encoding="utf-8")
                    else:
                        abc_content = transcribe_crop(smt_model, crop_mat, smt_device)
                        crop_abc_file.write_text(abc_content, encoding="utf-8")
                        
                    scores_on_page[crop_tag] = abc_content

                    # Рисуем белую заплату и ID-маркер прямо на изображении
                    cv2.rectangle(masked_img, (x, y), (x + w, y + h), (255, 255, 255), -1)
                    cv2.putText(masked_img, f"<!-- MUSIC_STUB_ID:{crop_tag} -->", 
                                (x + 10, y + max(25, int(h / 2))), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)

                # Сохранение очищенного изображения
                cv2.imwrite(str(masked_page_png), masked_img)
                _, masked_jpg_bytes = cv2.imencode(".jpg", masked_img)
                view_img.image(masked_jpg_bytes.tobytes(), caption=f"Стр. {p_num}: {len(boxes)} нотных заплат", width="stretch")

                # ШАГ 4: Инференс LM Studio по очищенному скану (без нот модель не галлюцинирует)
                if raw_md_file.exists() and not overwrite:
                    raw_text = raw_md_file.read_text(encoding="utf-8")
                else:
                    raw_text = stream_vlm(
                        img_bytes=masked_jpg_bytes.tobytes(),
                        prompt=system_prompt,
                        host=lm_host, port=lm_port, model=lm_model,
                        temp=lm_temperature, max_tokens=lm_max_tokens,
                        out_placeholder=view_res
                    )
                    raw_md_file.write_text(raw_text, encoding="utf-8")

                # ШАГ 5: Финальная линковка — вставка ABC вместо заплат
                def inject_abc(match):
                    cid = match.group(1).strip()
                    abc = scores_on_page.get(cid, "")
                    if not abc:
                        c_file = crops_dir / f"{cid}.abc"
                        abc = c_file.read_text(encoding="utf-8") if c_file.exists() else "% [Not found]"
                    return f"\n\n```abc\n{abc}\n```\n\n"

                final_text = re.sub(r"<!--\s*MUSIC_STUB_ID:([^\s>]+)\s*-->", inject_abc, raw_text)
                final_page_file.write_text(final_text, encoding="utf-8")
                view_res.markdown(final_text)

                # Очистка памяти после страницы
                del img_bgr, masked_img
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                time.sleep(float(delay))

            # Склейка книги в один файл
            all_pages = sorted(final_dir.glob("page_*.md"))
            full_content = "\n\n---\n\n".join([f"<!-- PAGE {p.stem} -->\n" + p.read_text(encoding="utf-8") for p in all_pages])
            (book_dir / f"{book_title}_complete.md").write_text(full_content, encoding="utf-8")

    st.session_state.is_running = False
    st.success("Все книги в очереди успешно обработаны!")