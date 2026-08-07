from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os
root = Path(__file__).resolve().parents[1]
os.chdir(root)
print("Open http://localhost:8000/steps/step_09_portfolio_copilot/step_09_portfolio_copilot_final.html")
ThreadingHTTPServer(("0.0.0.0", 8000), SimpleHTTPRequestHandler).serve_forever()
