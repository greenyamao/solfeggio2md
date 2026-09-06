"""
Solfeggio OCR Workbench & Batch Studio — Standalone Windows Desktop GUI
Powered by pywebview (Microsoft Edge WebView2) and local Bottle server.
"""

import json
import os
import shutil
import socket
import sys
import threading
import time
from pathlib import Path
import bottle
import webview
from socketserver import ThreadingMixIn
from wsgiref.simple_server import make_server, WSGIServer, WSGIRequestHandler

class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True

class QuietHandler(WSGIRequestHandler):
    def log_message(self, format, *args):
        # Silence periodic polling logs to prevent terminal spam
        pass

class ThreadedWSGIAdapter(bottle.ServerAdapter):
    def run(self, handler):
        server = make_server(self.host, self.port, handler, server_class=ThreadingWSGIServer, handler_class=QuietHandler)
        server.serve_forever()

ROOT_DIR = Path(__file__).parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.pipeline_worker import PipelineWorker
from core.pipeline_batch_runner import PipelineBatchRunner
from core.lmstudio_client import LMStudioClient


def find_available_port(start_port: int = 18492) -> int:
    for port in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start_port


def create_app(worker: PipelineWorker, runner: PipelineBatchRunner) -> bottle.Bottle:
    app = bottle.Bottle()
    gui_dir = ROOT_DIR / "gui"
    production_output = ROOT_DIR / "output"
    in_dir = ROOT_DIR / "in"

    # Auto-create all required directories if missing
    in_dir.mkdir(parents=True, exist_ok=True)
    production_output.mkdir(parents=True, exist_ok=True)
    
    # ---------------- Static Asset Serving ---------------- #
    @app.route('/')
    def index():
        return bottle.static_file('index.html', root=str(gui_dir))

    @app.route('/static/<filepath:path>')
    def serve_static(filepath):
        return bottle.static_file(filepath, root=str(gui_dir))

    @app.route('/output/<filepath:path>')
    def serve_output(filepath):
        return bottle.static_file(filepath, root=str(production_output))

    # ---------------- Inspection Workbench APIs ---------------- #
    @app.route('/api/pages')
    def api_pages():
        bottle.response.content_type = 'application/json; charset=utf-8'
        pages = worker.list_pages()
        return json.dumps({"pages": pages}, ensure_ascii=False)

    @app.route('/api/page/<page_id>')
    def api_page(page_id):
        bottle.response.content_type = 'application/json; charset=utf-8'
        try:
            data = worker.get_page_data(page_id)
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            bottle.response.status = 404
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    @app.route('/api/hardware')
    def api_hardware():
        bottle.response.content_type = 'application/json; charset=utf-8'
        return json.dumps(worker.get_hardware_status(), ensure_ascii=False)

    @app.route('/api/save_markdown', method='POST')
    def api_save_markdown():
        bottle.response.content_type = 'application/json; charset=utf-8'
        try:
            data = bottle.request.json or {}
            page_id = data.get("page_id")
            md = data.get("markdown")
            ok = worker.save_page_markdown(page_id, md)
            return json.dumps({"status": "success" if ok else "error"}, ensure_ascii=False)
        except Exception as e:
            bottle.response.status = 500
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    @app.route('/api/transcribe_crop', method='POST')
    def api_transcribe_crop():
        bottle.response.content_type = 'application/json; charset=utf-8'
        try:
            data = bottle.request.json or {}
            page_id = data.get("page_id")
            crop_stem = data.get("crop_stem")
            res = worker.transcribe_crop_live(page_id, crop_stem)
            return json.dumps(res, ensure_ascii=False)
        except Exception as e:
            bottle.response.status = 500
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    # ---------------- Batch Pipeline Runner APIs ---------------- #
    @app.route('/api/queue', method='GET')
    def api_queue_get():
        bottle.response.content_type = 'application/json; charset=utf-8'
        return json.dumps({"queue": runner.get_queue()}, ensure_ascii=False)

    @app.route('/api/queue/add', method='POST')
    def api_queue_add():
        bottle.response.content_type = 'application/json; charset=utf-8'
        try:
            data = bottle.request.json or {}
            path_str = data.get("path", "").strip().strip('"')
            if not path_str:
                return json.dumps({"status": "error", "message": "Путь к файлу не указан"}, ensure_ascii=False)

            p = Path(path_str)
            added_count = 0
            in_dir = ROOT_DIR / "in"
            in_dir.mkdir(exist_ok=True)
            if p.is_dir():
                for sub_pdf in sorted(p.glob("*.pdf")):
                    dest = in_dir / sub_pdf.name
                    if sub_pdf.resolve() != dest.resolve():
                        import shutil
                        shutil.copy2(str(sub_pdf), str(dest))
                    if runner.add_pdf(str(dest)):
                        added_count += 1
            elif p.is_file() and p.suffix.lower() == ".pdf":
                dest = in_dir / p.name
                if p.resolve() != dest.resolve():
                    import shutil
                    shutil.copy2(str(p), str(dest))
                if runner.add_pdf(str(dest)):
                    added_count += 1

            return json.dumps({"status": "success", "added": added_count, "queue": runner.get_queue()}, ensure_ascii=False)
        except Exception as e:
            bottle.response.status = 500
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    @app.route('/api/queue/upload', method='POST')
    def api_queue_upload():
        bottle.response.content_type = 'application/json; charset=utf-8'
        try:
            upload = bottle.request.files.get('file')
            if not upload:
                return json.dumps({"status": "error", "message": "Файл не получен"}, ensure_ascii=False)

            in_dir = ROOT_DIR / "in"
            in_dir.mkdir(exist_ok=True)
            raw_name = getattr(upload, 'raw_filename', None) or getattr(upload, 'filename', '') or 'document.pdf'
            safe_name = Path(raw_name).name
            if not safe_name.lower().endswith('.pdf'):
                return json.dumps({"status": "error", "message": "Поддерживаются только .pdf файлы"}, ensure_ascii=False)

            save_path = in_dir / safe_name
            upload.save(str(save_path), overwrite=True)

            runner.add_pdf(str(save_path))
            return json.dumps({"status": "success", "queue": runner.get_queue()}, ensure_ascii=False)
        except Exception as e:
            bottle.response.status = 500
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    @app.route('/api/queue/remove', method='POST')
    def api_queue_remove():
        bottle.response.content_type = 'application/json; charset=utf-8'
        try:
            data = bottle.request.json or {}
            item_id = str(data.get("id", ""))
            ok = runner.remove_pdf(item_id)
            return json.dumps({"status": "success" if ok else "not_found", "queue": runner.get_queue()}, ensure_ascii=False)
        except Exception as e:
            bottle.response.status = 500
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    @app.route('/api/queue/clear', method='POST')
    def api_queue_clear():
        bottle.response.content_type = 'application/json; charset=utf-8'
        runner.clear_queue()
        return json.dumps({"status": "success", "queue": runner.get_queue()}, ensure_ascii=False)

    @app.route('/api/pipeline/start', method='POST')
    def api_pipeline_start():
        bottle.response.content_type = 'application/json; charset=utf-8'
        ok = runner.start()
        return json.dumps({"status": "success" if ok else "already_running"}, ensure_ascii=False)

    @app.route('/api/pipeline/pause', method='POST')
    def api_pipeline_pause():
        bottle.response.content_type = 'application/json; charset=utf-8'
        runner.pause()
        return json.dumps({"status": "success"}, ensure_ascii=False)

    @app.route('/api/pipeline/resume', method='POST')
    def api_pipeline_resume():
        bottle.response.content_type = 'application/json; charset=utf-8'
        runner.resume()
        return json.dumps({"status": "success"}, ensure_ascii=False)

    @app.route('/api/pipeline/stop', method='POST')
    def api_pipeline_stop():
        bottle.response.content_type = 'application/json; charset=utf-8'
        runner.stop()
        return json.dumps({"status": "success"}, ensure_ascii=False)

    @app.route('/api/pipeline/status', method='GET')
    def api_pipeline_status():
        bottle.response.content_type = 'application/json; charset=utf-8'
        return json.dumps(runner.get_metrics(), ensure_ascii=False)

    # ---------------- Configuration & LM Studio Test ---------------- #
    @app.route('/api/config', method='GET')
    def api_config_get():
        bottle.response.content_type = 'application/json; charset=utf-8'
        return json.dumps(runner.config, ensure_ascii=False)

    @app.route('/api/config', method='POST')
    def api_config_post():
        bottle.response.content_type = 'application/json; charset=utf-8'
        try:
            data = bottle.request.json or {}
            runner.save_config(data)
            return json.dumps({"status": "success", "config": runner.config}, ensure_ascii=False)
        except Exception as e:
            bottle.response.status = 500
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    @app.route('/api/lmstudio/test', method='GET')
    def api_lmstudio_test():
        bottle.response.content_type = 'application/json; charset=utf-8'
        client = LMStudioClient(
            host=runner.config.get("lm_host", "127.0.0.1"),
            port=runner.config.get("lm_port", "1234"),
        )
        online, message = client.check_connection()
        models = client.list_models() if online else []
        return json.dumps(
            {"online": online, "message": message, "models": models}, ensure_ascii=False
        )

    return app


def main():
    worker = PipelineWorker()
    runner = PipelineBatchRunner()
    port = find_available_port(18492)
    app = create_app(worker, runner)

    server_thread = threading.Thread(
        target=lambda: bottle.run(app, host='127.0.0.1', port=port, quiet=True, server=ThreadedWSGIAdapter),
        daemon=True
    )
    server_thread.start()
    time.sleep(0.4)

    server_url = f"http://127.0.0.1:{port}"
    print(f"============================================================")
    print(f"Solfeggio OCR Workbench & Batch Studio")
    print(f"Local Server: {server_url}")
    print(f"Launching Dedicated Desktop Window (WebView2)...")
    print(f"============================================================")

    try:
        window = webview.create_window(
            title="Solfeggio OCR Workbench & Batch Studio",
            url=server_url,
            width=1600,
            height=950,
            min_size=(1280, 800),
            easy_drag=True,
            zoomable=True
        )
        webview.start(debug=False)
    except Exception as e:
        print(f"pywebview GUI launch error: {e}")
        print(f"Falling back to system browser: {server_url}")
        import webbrowser
        webbrowser.open(server_url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down.")


if __name__ == "__main__":
    main()
