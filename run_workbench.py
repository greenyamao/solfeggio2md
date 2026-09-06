"""
Solfeggio OCR Workbench — Standalone Windows Desktop GUI
Powered by pywebview (Microsoft Edge WebView2) and local Bottle server.
"""

import os
import sys
import json
import time
import socket
import threading
from pathlib import Path
import bottle
import webview

# Add project root to sys.path
ROOT_DIR = Path(__file__).parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.pipeline_worker import PipelineWorker


def find_available_port(start_port: int = 18492) -> int:
    for port in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start_port


def create_app(worker: PipelineWorker) -> bottle.Bottle:
    app = bottle.Bottle()
    gui_dir = ROOT_DIR / "gui"
    output_dir = ROOT_DIR / "test_bench" / "output"

    # Static Assets
    @app.route('/')
    def index():
        return bottle.static_file('index.html', root=str(gui_dir))

    @app.route('/static/<filepath:path>')
    def serve_static(filepath):
        return bottle.static_file(filepath, root=str(gui_dir))

    @app.route('/output/<filepath:path>')
    def serve_output(filepath):
        return bottle.static_file(filepath, root=str(output_dir))

    # REST APIs
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

    return app


def main():
    worker = PipelineWorker()
    port = find_available_port(18492)
    app = create_app(worker)

    server_thread = threading.Thread(
        target=lambda: bottle.run(app, host='127.0.0.1', port=port, quiet=True),
        daemon=True
    )
    server_thread.start()
    time.sleep(0.4)

    server_url = f"http://127.0.0.1:{port}"
    print(f"============================================================")
    print(f"🎵 Solfeggio OCR Workbench Visual Inspection Studio")
    print(f"   Local Server: {server_url}")
    print(f"   Launching Dedicated Desktop Window (WebView2)...")
    print(f"============================================================")

    try:
        window = webview.create_window(
            title="🎵 Solfeggio OCR Workbench — Visual Inspection Studio",
            url=server_url,
            width=1600,
            height=950,
            min_size=(1280, 800),
            easy_drag=True,
            zoomable=True
        )
        webview.start(debug=False)
    except Exception as e:
        print(f"⚠️ pywebview GUI launch error: {e}")
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
