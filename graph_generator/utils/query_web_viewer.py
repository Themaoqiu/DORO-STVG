#!/usr/bin/env python3
import argparse
import json
import os
import posixpath
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Query Viewer</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --card: #ffffff;
      --text: #1b2430;
      --muted: #5a6878;
      --line: #dbe3ee;
      --accent: #1769aa;
      --accent-2: #0f4c75;
    }
    body { margin: 0; font-family: "Segoe UI", "PingFang SC", sans-serif; background: var(--bg); color: var(--text); }
    .wrap { max-width: 1400px; margin: 0 auto; padding: 16px; }
    .toolbar { display: grid; grid-template-columns: 1.6fr 1fr 1fr 1fr 1fr; gap: 10px; margin-bottom: 12px; }
    .toolbar label { font-size: 12px; color: var(--muted); display: block; margin-bottom: 4px; }
    .toolbar input, .toolbar select, .toolbar button {
      width: 100%; box-sizing: border-box; padding: 8px 10px; border: 1px solid var(--line);
      border-radius: 8px; background: #fff; color: var(--text);
    }
    .toolbar button { background: var(--accent); color: #fff; border: none; cursor: pointer; }
    .toolbar button:hover { background: var(--accent-2); }
    .main { display: grid; grid-template-columns: 1.4fr 1fr; gap: 12px; }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
    .card h3 { margin: 0; font-size: 14px; padding: 10px 12px; border-bottom: 1px solid var(--line); background: #f9fbff; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid #edf2f7; padding: 8px 10px; text-align: left; font-size: 12px; vertical-align: top; }
    th { color: var(--muted); font-weight: 600; position: sticky; top: 0; background: #fff; }
    .table-wrap { max-height: calc(100vh - 210px); overflow: auto; }
    .pill { display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 11px; border: 1px solid var(--line); }
    .pill.easy { background: #eef8f0; }
    .pill.medium { background: #fff6e8; }
    .pill.hard { background: #ffecec; }
    .pill.very_hard { background: #fde7ff; }
    .video-wrap { position: relative; background: #000; }
    video { width: 100%; display: block; max-height: 56vh; }
    canvas { position: absolute; left: 0; top: 0; pointer-events: none; }
    .meta { padding: 10px 12px; font-size: 12px; line-height: 1.5; }
    .muted { color: var(--muted); }
    .query { font-size: 13px; background: #f4f9ff; border: 1px solid #d8e9ff; border-radius: 8px; padding: 8px 10px; margin: 8px 0; }
    .row-btn { cursor: pointer; }
    .row-btn:hover { background: #f7fbff; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="toolbar">
      <div>
        <label>Video</label>
        <select id="videoFilter"></select>
      </div>
      <div>
        <label>Difficulty</label>
        <select id="difficultyFilter">
          <option value="">All</option>
          <option value="easy">easy</option>
          <option value="medium">medium</option>
          <option value="hard">hard</option>
          <option value="very_hard">very_hard</option>
        </select>
      </div>
      <div>
        <label>Search Query</label>
        <input id="searchInput" placeholder="keyword..." />
      </div>
      <div>
        <label>FPS for overlay</label>
        <input id="fpsInput" type="number" step="0.1" value="2" />
      </div>
      <div>
        <label>&nbsp;</label>
        <button id="refreshBtn">Refresh</button>
      </div>
    </div>

    <div class="main">
      <div class="card">
        <h3>Query Records</h3>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>#</th><th>Video</th><th>Target</th><th>Difficulty</th><th>Query</th></tr>
            </thead>
            <tbody id="rows"></tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <h3>Preview</h3>
        <div class="video-wrap" id="videoWrap">
          <video id="video" controls></video>
          <canvas id="overlay"></canvas>
        </div>
        <div class="meta">
          <div><span class="muted">Video:</span> <span id="metaVideo"></span></div>
          <div><span class="muted">Target:</span> <span id="metaTarget"></span></div>
          <div><span class="muted">Difficulty:</span> <span id="metaDifficulty"></span></div>
          <div><span class="muted">Boxes:</span> <span id="metaBoxes"></span></div>
          <div class="query" id="metaQuery"></div>
        </div>
      </div>
    </div>
  </div>

<script>
let allRecords = [];
let current = null;

const els = {
  rows: document.getElementById('rows'),
  videoFilter: document.getElementById('videoFilter'),
  difficultyFilter: document.getElementById('difficultyFilter'),
  searchInput: document.getElementById('searchInput'),
  refreshBtn: document.getElementById('refreshBtn'),
  fpsInput: document.getElementById('fpsInput'),
  video: document.getElementById('video'),
  overlay: document.getElementById('overlay'),
  metaVideo: document.getElementById('metaVideo'),
  metaTarget: document.getElementById('metaTarget'),
  metaDifficulty: document.getElementById('metaDifficulty'),
  metaBoxes: document.getElementById('metaBoxes'),
  metaQuery: document.getElementById('metaQuery'),
};

function safe(s) {
  return (s || '').toString().replace(/[&<>\"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c]));
}

async function fetchRecords() {
  const res = await fetch('/api/records');
  allRecords = await res.json();
  populateVideoFilter();
  renderRows();
}

function populateVideoFilter() {
  const vids = Array.from(new Set(allRecords.map(r => r.video_path))).sort();
  const currentValue = els.videoFilter.value || '';
  let html = '<option value="">All</option>';
  for (const v of vids) html += `<option value="${safe(v)}">${safe(v.split('/').pop())}</option>`;
  els.videoFilter.innerHTML = html;
  if (vids.includes(currentValue)) els.videoFilter.value = currentValue;
}

function filtered() {
  const v = els.videoFilter.value;
  const d = els.difficultyFilter.value;
  const q = els.searchInput.value.trim().toLowerCase();
  return allRecords.filter(r => {
    if (v && r.video_path !== v) return false;
    if (d && r.difficulty_bucket !== d) return false;
    if (q && !(r.query || '').toLowerCase().includes(q)) return false;
    return true;
  });
}

function renderRows() {
  const rows = filtered();
  let html = '';
  rows.forEach((r, idx) => {
    html += `<tr class="row-btn" data-idx="${idx}">
      <td>${idx + 1}</td>
      <td>${safe((r.video_path || '').split('/').pop())}</td>
      <td>${safe(r.target_node_id || '')}</td>
      <td><span class="pill ${safe(r.difficulty_bucket || '')}">${safe(r.difficulty_bucket || '')}</span></td>
      <td>${safe(r.query || '')}</td>
    </tr>`;
  });
  els.rows.innerHTML = html;

  const trs = els.rows.querySelectorAll('tr');
  trs.forEach(tr => {
    tr.addEventListener('click', () => {
      const rec = rows[parseInt(tr.dataset.idx, 10)];
      preview(rec);
    });
  });

  if (rows.length) preview(rows[0]);
}

function preview(rec) {
  current = rec;
  const src = '/video?path=' + encodeURIComponent(rec.video_path || '');
  if (els.video.src !== location.origin + src) {
    els.video.src = src;
  }
  els.metaVideo.textContent = rec.video_path || '';
  els.metaTarget.textContent = rec.target_node_id || '';
  els.metaDifficulty.textContent = rec.difficulty_bucket || '';
  const boxCount = rec.boxes ? Object.keys(rec.boxes).length : 0;
  els.metaBoxes.textContent = boxCount.toString();
  els.metaQuery.textContent = rec.query || '';
  resizeCanvas();
  drawOverlay();
}

function resizeCanvas() {
  const rect = els.video.getBoundingClientRect();
  els.overlay.width = Math.max(1, Math.floor(rect.width));
  els.overlay.height = Math.max(1, Math.floor(rect.height));
  els.overlay.style.width = rect.width + 'px';
  els.overlay.style.height = rect.height + 'px';
}

function pickBox(boxes, frame) {
  if (!boxes) return null;
  const key = String(frame);
  if (Object.prototype.hasOwnProperty.call(boxes, key)) return boxes[key];
  return null;
}


function drawOverlay() {
  const ctx = els.overlay.getContext('2d');
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  if (!current || !current.boxes || !els.video.videoWidth || !els.video.videoHeight) return;

  const fps = parseFloat(els.fpsInput.value || '2') || 2;
  const frame = Math.round(els.video.currentTime * fps);
  const box = pickBox(current.boxes, frame);
  if (!box || box.length < 4) return;

  const scaleX = els.overlay.width / els.video.videoWidth;
  const scaleY = els.overlay.height / els.video.videoHeight;
  const x1 = box[0] * scaleX;
  const y1 = box[1] * scaleY;
  const x2 = box[2] * scaleX;
  const y2 = box[3] * scaleY;

  ctx.strokeStyle = '#00ff6a';
  ctx.lineWidth = 2;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

  const label = `${current.target_node_id || 'target'} | f=${frame}`;
  ctx.font = '12px sans-serif';
  const w = ctx.measureText(label).width + 8;
  ctx.fillStyle = 'rgba(0,0,0,0.6)';
  ctx.fillRect(x1, Math.max(0, y1 - 18), w, 16);
  ctx.fillStyle = '#fff';
  ctx.fillText(label, x1 + 4, Math.max(12, y1 - 6));
}

function tick() {
  drawOverlay();
  requestAnimationFrame(tick);
}

[els.videoFilter, els.difficultyFilter].forEach(el => el.addEventListener('change', renderRows));
els.searchInput.addEventListener('input', renderRows);
els.refreshBtn.addEventListener('click', fetchRecords);
els.fpsInput.addEventListener('change', drawOverlay);
window.addEventListener('resize', () => { resizeCanvas(); drawOverlay(); });
els.video.addEventListener('loadedmetadata', () => { resizeCanvas(); drawOverlay(); });

fetchRecords();
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


class QueryViewerHandler(BaseHTTPRequestHandler):
    records = []
    base_dir = Path("/")

    def _send_json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text, code=200):
        data = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html):
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/":
            self._send_html(HTML_PAGE)
            return

        if path == "/api/records":
            self._send_json(self.records)
            return

        if path == "/video":
            p = qs.get("path", [""])[0]
            p = unquote(p)
            self._serve_video_file(p)
            return

        self._send_text("Not Found", 404)

    def _serve_video_file(self, file_path: str):
        try:
            path = Path(file_path).resolve()
        except Exception:
            self._send_text("Bad path", 400)
            return

        if not str(path).startswith(str(self.base_dir)):
            self._send_text("Forbidden", 403)
            return
        if not path.exists() or not path.is_file():
            self._send_text("File not found", 404)
            return

        ctype = "video/mp4"
        size = path.stat().st_size
        range_header = self.headers.get("Range")

        if not range_header:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(path, "rb") as f:
                self.wfile.write(f.read())
            return

        try:
            unit, rng = range_header.strip().split("=", 1)
            if unit != "bytes":
                raise ValueError("unsupported range unit")
            start_str, end_str = rng.split("-", 1)
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else size - 1
            start = max(0, start)
            end = min(size - 1, end)
            if start > end:
                raise ValueError("invalid range")
        except Exception:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return

        length = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            chunk = 1024 * 1024
            while remaining > 0:
                read_size = min(chunk, remaining)
                data = f.read(read_size)
                if not data:
                    break
                self.wfile.write(data)
                remaining -= len(data)


def load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            records.append(
                {
                    "idx": i,
                    "video_path": obj.get("video_path", ""),
                    "target_node_id": obj.get("target_node_id", ""),
                    "query": obj.get("query", ""),
                    "difficulty_bucket": obj.get("difficulty_bucket", ""),
                    "boxes": obj.get("boxes", {}) or {},
                }
            )
    return records


def main():
    parser = argparse.ArgumentParser(description="Lightweight web viewer for query_minimal.jsonl")
    parser.add_argument(
        "--jsonl",
        default="/home/wangxingjian/DORO-STVG/graph_generator/output/query_minimal.jsonl",
        help="Path to query jsonl",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl).resolve()
    if not jsonl_path.exists():
        raise FileNotFoundError(f"jsonl not found: {jsonl_path}")

    QueryViewerHandler.records = load_jsonl(jsonl_path)
    QueryViewerHandler.base_dir = Path("/")

    server = ThreadingHTTPServer((args.host, args.port), QueryViewerHandler)
    print(f"Loaded {len(QueryViewerHandler.records)} records from {jsonl_path}")
    print(f"Open: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
