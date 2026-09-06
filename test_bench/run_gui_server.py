"""
Runs the Workbench HTTP backend for browser testing and inspection.
"""

import sys
from pathlib import Path
import bottle

ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from run_workbench import create_app
from core.pipeline_worker import PipelineWorker

def run():
    worker = PipelineWorker()
    app = create_app(worker)
    print("Serving Workbench at http://127.0.0.1:18492 ...")
    bottle.run(app, host='127.0.0.1', port=18492, quiet=False)

if __name__ == "__main__":
    run()
