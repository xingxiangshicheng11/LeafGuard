import argparse
import base64
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

from scripts.predict import (
    build_model,
    load_checkpoint,
    load_class_to_idx,
    resolve_device,
    strip_module_prefix,
)
from src.data.transforms import build_transforms

STATE = {
    "model": None,
    "transform": None,
    "device": None,
    "idx_to_class": None,
    "num_classes": None,
    "model_name": None,
    "zh": None,
    "crops": None,
    "crop_diseases": None,
    "msp_threshold": None,
    "energy_threshold": None,
    "advice": None,
    "lock": threading.Lock(),
}

INDEX_HTML = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>LeafGuard 叶片病害识别 Demo</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --accent:#22c55e; --text:#e2e8f0; --muted:#94a3b8; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: "Microsoft YaHei", system-ui, sans-serif; background:var(--bg); color:var(--text); }
  .wrap { max-width: 720px; margin: 0 auto; padding: 32px 20px 64px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 24px; }
  .card { background: var(--card); border-radius: 14px; padding: 20px; box-shadow: 0 8px 30px rgba(0,0,0,.3); }
  .drop { border: 2px dashed #334155; border-radius: 12px; padding: 28px; text-align: center; cursor: pointer; transition: .15s; }
  .drop:hover { border-color: var(--accent); }
  .drop.drag { border-color: var(--accent); background: rgba(34,197,94,.08); }
  input[type=file] { display:none; }
  #preview { max-width: 100%; max-height: 260px; border-radius: 10px; margin-top: 16px; display:none; }
  .result { margin-top: 22px; display:none; }
  .pred { font-size: 20px; font-weight: 700; }
  .pred .tag { color: var(--accent); }
  .pred .en { color: var(--muted); font-size: 13px; font-weight: 400; }
  .bar-row { margin: 10px 0; }
  .bar-label { display:flex; justify-content:space-between; font-size: 13px; margin-bottom: 4px; }
  .bar-track { background:#0b1220; border-radius: 8px; height: 14px; overflow:hidden; }
  .bar-fill { height:100%; background: linear-gradient(90deg,#16a34a,#22c55e); width:0; transition: width .5s ease; }
  .note { color: var(--muted); font-size: 12px; margin-top: 18px; }
  .reject { background: rgba(245,158,11,.12); border:1px solid #f59e0b; color:#fbbf24; border-radius:10px; padding:12px 14px; font-size:14px; line-height:1.6; margin:8px 0; }
  .advice-btn { margin-top:14px; background:#16a34a; color:#fff; border:none; border-radius:10px; padding:10px 16px; font-size:14px; cursor:pointer; }
  .advice-btn:hover { background:#15803d; }
  .advice { margin-top:12px; background:#0b1220; border:1px solid #334155; border-radius:10px; padding:14px; font-size:14px; line-height:1.7; display:none; }
  .advice .a-sum { color:var(--text); margin-bottom:6px; }
  .advice .a-sec { color:var(--accent); font-weight:600; margin-top:8px; }
  .advice ul { margin:4px 0 4px 18px; padding:0; }
  .advice li { margin:2px 0; }
  .advice .a-src { color:var(--muted); font-size:12px; margin-top:10px; }
  .status { color: var(--muted); font-size: 13px; margin-top: 12px; min-height: 18px; }
  .supported { background: var(--card); border-radius: 14px; padding: 16px 20px; margin-bottom: 18px; box-shadow: 0 8px 30px rgba(0,0,0,.2); }
  .sup-title { font-size: 13px; color: var(--muted); margin-bottom: 10px; }
  .chips { display:flex; flex-wrap:wrap; gap:8px; }
  .chip { background:#0b1220; color:var(--text); border:1px solid #334155; border-radius:999px; padding:4px 12px; font-size:13px; }
  .fulllist { margin-top:12px; color:var(--muted); font-size:13px; }
  .fulllist summary { cursor:pointer; color:var(--accent); }
  .grp { margin:6px 0; line-height:1.7; }
  .grp-crop { color:var(--text); font-weight:600; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🌿 LeafGuard · 叶片病害识别</h1>
  <div class="sub">上传一张植物叶片照片，识别它是否患病及可能的病害类别（本页为本地演示，结果仅供参考）。</div>
  <div class="supported">
    <div class="sup-title">✅ 能识别的植物（{{CROPS_N}} 种）· 共 {{NUM_CLASSES}} 类（含健康与病害）</div>
    <div class="chips">{{CROPS}}</div>
    <details class="fulllist"><summary>查看每种植物能识别哪些病害/健康类别</summary>{{DISEASE_GROUPS}}</details>
  </div>
  <div class="card">
    <label class="drop" id="drop">
      <div>点击选择，或把图片拖到这里</div>
      <div class="sub" style="margin:8px 0 0">支持 JPG / PNG</div>
      <input type="file" id="file" accept="image/*" />
      <img id="preview" alt="preview" />
    </label>
    <div class="status" id="status"></div>
    <div class="result" id="result">
      <div class="pred">预测：<span class="tag" id="predName">-</span> <span class="en" id="predEn"></span></div>
      <div class="reject" id="rejected" style="display:none"></div>
      <div id="bars"></div>
      <button class="advice-btn" id="adviceBtn" style="display:none">查看防治建议</button>
      <div class="advice" id="advice"></div>
      <div class="note">分数为 softmax 置信度，不代表校准后的诊断概率；本项目为课程/研究演示，不能作为真实农业诊断依据。</div>
    </div>
  </div>
</div>
<script>
const fileInput = document.getElementById('file');
const drop = document.getElementById('drop');
const preview = document.getElementById('preview');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const predName = document.getElementById('predName');
const bars = document.getElementById('bars');
const adviceBtn = document.getElementById('adviceBtn');
const adviceBox = document.getElementById('advice');
adviceBtn.addEventListener('click', function () {
  adviceBox.style.display = (adviceBox.style.display === 'block') ? 'none' : 'block';
});

function handleFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
    preview.style.display = 'block';
    statusEl.textContent = '正在识别…';
    resultEl.style.display = 'none';
    fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({image_b64: reader.result})
    }).then(r => r.json()).then(data => {
      if (data.error) { statusEl.textContent = '出错：' + data.error; return; }
      statusEl.textContent = '';
      render(data);
      resultEl.style.display = 'block';
    }).catch(e => { statusEl.textContent = '请求失败：' + e; });
  };
  reader.readAsDataURL(file);
}

function render(data) {
  var predEn = document.getElementById('predEn');
  var rej = document.getElementById('rejected');
  if (data.rejected) {
    predName.textContent = '无法可靠识别';
    predEn.textContent = '';
    bars.innerHTML = '';
    rej.textContent = data.message || '识别不确定，建议人工确诊。';
    rej.style.display = 'block';
    adviceBtn.style.display = 'none';
    adviceBox.style.display = 'none';
    adviceBox.innerHTML = '';
    return;
  }
  rej.style.display = 'none';
  predName.textContent = data.prediction;
  predEn.textContent = data.prediction_en ? '(' + data.prediction_en + ')' : '';
  bars.innerHTML = '';
  data.top_k.forEach(item => {
    const pct = (item.score * 100);
    var label = item.class_name_zh || item.class_name;
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = '<div class="bar-label"><span>' + label +
      '</span><span>' + pct.toFixed(1) + '%</span></div>' +
      '<div class="bar-track"><div class="bar-fill"></div></div>';
    bars.appendChild(row);
    requestAnimationFrame(() => { row.querySelector('.bar-fill').style.width = pct + '%'; });
  });
  renderAdvice(data.advice);
}

function renderAdvice(a) {
  if (!a) { adviceBtn.style.display = 'none'; adviceBox.style.display = 'none'; adviceBox.innerHTML = ''; return; }
  var html = '';
  if (a['摘要']) { html += '<div class="a-sum">' + a['摘要'] + '</div>'; }
  if (a['防治措施'] && a['防治措施'].length) {
    html += '<div class="a-sec">防治措施</div><ul>';
    a['防治措施'].forEach(function (x) { html += '<li>' + x + '</li>'; });
    html += '</ul>';
  }
  if (a['预防'] && a['预防'].length) {
    html += '<div class="a-sec">预防</div><ul>';
    a['预防'].forEach(function (x) { html += '<li>' + x + '</li>'; });
    html += '</ul>';
  }
  if (a['来源']) { html += '<div class="a-src">来源：' + a['来源'] + '</div>'; }
  adviceBox.innerHTML = html;
  adviceBox.style.display = 'none';
  adviceBtn.style.display = 'inline-block';
}

fileInput.addEventListener('change', e => handleFile(e.target.files[0]));
['dragover','dragenter'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('drag'); }));
['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('drag'); }));
drop.addEventListener('drop', e => { if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]); });
</script>
</body>
</html>
"""


def setup(checkpoint, classes, device_arg, zh_map_path="splits/class_to_zh.json", ood_thresholds_path=None, advice_map_path="assets/disease_advice_zh.json"):
    device = resolve_device(device_arg)
    ckpt = load_checkpoint(checkpoint, device)
    if "model_name" not in ckpt:
        raise SystemExit("checkpoint missing required field: model_name")
    class_to_idx, _ = load_class_to_idx(ckpt, classes)
    num_classes = int(ckpt.get("num_classes", len(class_to_idx)))
    model = build_model(ckpt["model_name"], num_classes)
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if state_dict is None:
        raise SystemExit("checkpoint missing required field: model_state_dict")
    model.load_state_dict(strip_module_prefix(state_dict))
    model.to(device).eval()

    zh = {}
    zh_path = Path(zh_map_path)
    if zh_path.is_file():
        with open(zh_path, encoding="utf-8") as f:
            zh = json.load(f)

    crops = []
    crop_diseases = {}
    for name, idx in sorted(class_to_idx.items(), key=lambda kv: int(kv[1])):
        full = zh.get(name, name)
        crop, sep, disease = full.partition("-")
        if crop not in crop_diseases:
            crop_diseases[crop] = []
            crops.append(crop)
        crop_diseases[crop].append(disease if sep else full)

    msp_threshold = None
    energy_threshold = None
    if ood_thresholds_path is None:
        ood_thresholds_path = str(Path(checkpoint).parent.parent / "metrics" / "ood_thresholds.json")
    ood_path = Path(ood_thresholds_path)
    if ood_path.is_file():
        with open(ood_path, encoding="utf-8") as f:
            ood = json.load(f)
        msp_threshold = ood.get("msp_threshold")
        energy_threshold = ood.get("energy_threshold")

    advice = {}
    advice_path = Path(advice_map_path)
    if advice_path.is_file():
        with open(advice_path, encoding="utf-8") as f:
            advice = json.load(f)

    STATE.update(
        model=model,
        transform=build_transforms(int(ckpt.get("input_size", 224)), train=False, augment=False),
        device=device,
        idx_to_class={int(v): k for k, v in class_to_idx.items()},
        num_classes=num_classes,
        model_name=ckpt["model_name"],
        zh=zh,
        crops=crops,
        crop_diseases=crop_diseases,
        msp_threshold=msp_threshold,
        energy_threshold=energy_threshold,
        advice=advice,
    )


def predict_pil(image):
    with STATE["lock"]:
        tensor = STATE["transform"](image).unsqueeze(0).to(STATE["device"])
        with torch.no_grad():
            logits = STATE["model"](tensor)
            probs = torch.softmax(logits, dim=1)
        msp_val = float(probs.max(dim=1).values[0])
        energy_val = float((-torch.logsumexp(logits, dim=1))[0])
        k = min(5, STATE["num_classes"])
        scores, indices = probs[0].topk(k)
    zh = STATE.get("zh") or {}
    top = []
    for s, i in zip(scores.tolist(), indices.tolist()):
        en = STATE["idx_to_class"][int(i)]
        top.append({
            "class_name": en,
            "class_name_zh": zh.get(en, en),
            "label": int(i),
            "score": float(s),
        })

    msp_th = STATE.get("msp_threshold")
    energy_th = STATE.get("energy_threshold")
    reasons = []
    if msp_th is not None and msp_val < msp_th:
        reasons.append("low_confidence")
    if energy_th is not None and energy_val > energy_th:
        reasons.append("out_of_distribution")

    if reasons:
        return {
            "rejected": True,
            "reasons": reasons,
            "prediction": "无法可靠识别",
            "prediction_en": None,
            "confidence": msp_val,
            "energy": energy_val,
            "message": "这张图可能不属于系统支持的植物/病害范围，或与训练分布差异较大。请人工确诊，不要据此用药或判断病害。",
            "model_name": STATE["model_name"],
            "top_k": [],
            "note": "rejected by OOD gate (msp/energy)",
        }

    advice_map = STATE.get("advice") or {}
    pred_en = top[0]["class_name"]
    return {
        "rejected": False,
        "prediction": top[0]["class_name_zh"],
        "prediction_en": pred_en,
        "confidence": msp_val,
        "energy": energy_val,
        "model_name": STATE["model_name"],
        "top_k": top,
        "advice": advice_map.get(pred_en),
        "note": "scores are softmax, not calibrated diagnostic probabilities",
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            crops = STATE.get("crops") or []
            crop_diseases = STATE.get("crop_diseases") or {}
            chips = "".join('<span class="chip">{}</span>'.format(c) for c in crops)
            groups = "".join(
                '<div class="grp"><span class="grp-crop">{}</span>：{}</div>'.format(
                    c, "、".join(crop_diseases.get(c, []))
                )
                for c in crops
            )
            page = (
                INDEX_HTML.replace("{{CROPS_N}}", str(len(crops)))
                .replace("{{CROPS}}", chips)
                .replace("{{DISEASE_GROUPS}}", groups)
                .replace("{{NUM_CLASSES}}", str(STATE.get("num_classes") or len(crops)))
            )
            self._send(200, page, "text/html")
        elif self.path == "/health":
            self._send(200, "ok", "text/plain")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/predict":
            self._send(404, "not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 25_000_000:
            self._send(400, json.dumps({"error": "invalid content length"}, ensure_ascii=False))
            return
        try:
            payload = json.loads(self.rfile.read(length))
            b64 = payload["image_b64"]
            if b64.strip().startswith("data:"):
                b64 = b64.split(",", 1)[1]
            image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            self._send(200, json.dumps(predict_pil(image), ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            self._send(400, json.dumps({"error": str(exc)}, ensure_ascii=False))

    def log_message(self, *args):  # quiet
        pass


def main():
    parser = argparse.ArgumentParser(description="LeafGuard local web demo (zero dependency)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--classes", default="splits/class_to_idx.json")
    parser.add_argument("--zh-map", default="splits/class_to_zh.json")
    parser.add_argument("--ood-thresholds", default=None)
    parser.add_argument("--advice-map", default="assets/disease_advice_zh.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    setup(checkpoint, args.classes, args.device, args.zh_map, args.ood_thresholds, args.advice_map)

    server = None
    port = args.port
    for candidate in range(args.port, args.port + 10):
        try:
            server = ThreadingHTTPServer((args.host, candidate), Handler)
            port = candidate
            break
        except OSError as exc:
            print(f"port {candidate} unavailable ({exc}); trying next...")
    if server is None:
        raise SystemExit(f"could not bind any port in {args.port}..{args.port + 9}")
    print(f"LeafGuard demo running: http://{args.host}:{port}")
    print(f"model: {STATE['model_name']}  device: {STATE['device']}  classes: {STATE['num_classes']}")
    print("Open the URL above in your browser. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
