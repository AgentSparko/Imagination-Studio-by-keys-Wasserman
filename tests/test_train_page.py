"""Looks → Train a LoRA (app/static/train.js), run in node with the page's helpers stood in for:
python3 -m pytest tests/test_train_page.py. Skipped where node is not installed."""
import json
import subprocess
import tempfile
from pathlib import Path

from test_shell_page import need_node

ROOT = Path(__file__).resolve().parent.parent
HARNESS = r"""
const vm = require("vm"), fs = require("fs");
globalThis.renders = 0; globalThis.render = () => { renders++; };
globalThis.toasts = []; globalThis.toast = t => toasts.push(String(t));
globalThis.calls = []; globalThis.answers = {};
globalThis.api = async (path, body, method) => { calls.push([path, body === undefined ? null : body, method || ""]); if (path in answers) { const a = answers[path]; if (a instanceof Error) throw a; return typeof a === "function" ? a(body) : a; } return {}; };
globalThis.jobs = []; globalThis.polls = 0; globalThis.poll = async () => { polls++; };
globalThis.confirms = []; globalThis.confirm = m => { confirms.push(String(m)); return true; };
globalThis.LIBPICK = new Set(); globalThis.LORAS = []; globalThis.loads = 0; globalThis.loadLoras = async () => { loads++; };
globalThis.mmss = s => s >= 3600 ? Math.floor(s / 3600) + " h " + Math.round(s % 3600 / 60) + " min" : Math.round(s) + " s";
globalThis.media = p => el("video", { src: p });
globalThis.jobCard = j => el("div", { class: "job", "data-job": j.id }, j.title, el("button", {}, "Cancel"));
const text = node => node == null || node === false ? "" : typeof node !== "object" ? String(node) : Array.isArray(node) ? node.map(text).join(" ") : [node.attrs && node.attrs.placeholder, node.attrs && node.attrs.title, text(node.kids)].filter(Boolean).join(" ");
globalThis.text = text;
globalThis.el = (tag, attrs = {}, ...kids) => { const n = { tag, attrs, kids: kids.flat(), value: attrs.value || "", textContent: "" }; return n; };
globalThis.field = (label, node) => el("div", {}, el("label", {}, label), node);
globalThis.choice = (obj, key, options, after) => el("div", { class: "choice" }, Object.entries(options).map(([v, [title, sub]]) => el("button", { class: String(obj[key]) === v ? "on" : "", onclick: () => { obj[key] = v; if (after) after(); } }, el("b", {}, title), sub)));
const find = (node, test) => !node || typeof node !== "object" ? null : Array.isArray(node) ? node.map(n => find(n, test)).find(Boolean) || null : test(node) ? node : find(node.kids, test);
globalThis.find = find; globalThis.button = (root, words) => find(root, n => n.tag === "button" && text(n).includes(words));
globalThis.settle = () => new Promise(r => setTimeout(r, 0));
vm.runInThisContext(fs.readFileSync(process.argv[3], "utf8"));
vm.runInThisContext(`(async () => { const out = {}; ${fs.readFileSync(process.argv[2], "utf8")}; console.log(JSON.stringify(out)); })().catch(e => { console.error(e); process.exit(1); });`);
"""

READY = """answers["/api/train"] = { ready: true, spark: true, missing: [], setup: "bash oneshot-setup-spark.sh --train", datasets: [], runs: [],
  sizes: { standard: { label: "928x512, 124 frames (measured)", per_step: 110 }, small: { label: "704x384", per_step: 63 } }, limits: { items: 8 } };
answers["/api/library"] = Array.from({ length: 10 }, (_, i) => ({ path: "renders/c" + i + ".mp4", kind: "video" })).concat([{ path: "voice/a.wav", kind: "audio" }]);
"""


def run(code: str, setup: str = READY) -> dict:
    need_node()
    with tempfile.TemporaryDirectory() as scratch:
        (Path(scratch) / "harness.js").write_text(HARNESS)
        (Path(scratch) / "test.js").write_text(setup + code)
        done = subprocess.run(["node", str(Path(scratch) / "harness.js"), str(Path(scratch) / "test.js"), str(ROOT / "app/static/train.js")],
                              capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_not_set_up_says_the_command_and_what_is_missing():
    out = run("""trainPanel(); await settle(); out.page = text(trainPanel());""",
              """answers["/api/train"] = { ready: false, spark: true, missing: ["ostris's de-distillation adapter v2 (/m/a)"], setup: "bash oneshot-setup-spark.sh --train" };""")
    assert "bash oneshot-setup-spark.sh --train" in out["page"] and "adapter v2" in out["page"] and "Start training" not in out["page"]
    mac = run("""trainPanel(); await settle(); out.page = text(trainPanel());""", """answers["/api/train"] = { ready: false, spark: false, missing: [], setup: "bash oneshot-setup-spark.sh --train" };""")
    assert "runs on a DGX Spark" in mac["page"]


def test_the_form_shows_the_time_and_starts_a_run_from_picked_clips():
    out = run("""trainPanel(); await settle(); let page = trainPanel(); out.page = text(page);
      out.disabled = button(page, "Start training").attrs.disabled;
      button(page, "Pick from the Library").attrs.onclick(); page = trainPanel(); await settle(); page = trainPanel();
      out.tiles = 0; for (let i = 0; i < 9; i++) { const t = find(page, n => n.attrs && n.attrs["data-path"] === "renders/c" + i + ".mp4"); if (t) { out.tiles++; t.attrs.onclick(); } }
      out.audio = !!find(page, n => n.attrs && n.attrs["data-path"] === "voice/a.wav");
      TR.form.name = "Night Rain"; TR.form.trigger = "wk_rain"; button(page, "A character").attrs.onclick();
      const slide = find(page, n => n.tag === "input" && n.attrs.type === "range"); slide.attrs.oninput({ target: { value: "2000" } });
      answers["/api/train/datasets"] = { id: "night-rain", items: 9 }; answers["/api/train/start"] = { id: "j1", kind: "train" };
      page = trainPanel(); out.page2 = text(page); out.enabled = !button(page, "Start training").attrs.disabled;
      await button(page, "Start training").attrs.onclick();
      out.calls = calls.filter(c => c[0].startsWith("/api/train/")).map(c => [c[0], c[1]]); out.confirm = confirms[0]; out.picked = TR.pick.size; out.toasts = toasts;""")
    assert "1000 steps · about 30 h 33 min" in out["page"] and "nothing else renders" in out["page"] and out["disabled"]
    assert out["tiles"] == 9 and not out["audio"] and out["enabled"] and "9 picked" in out["page2"]
    (path1, body1), (path2, body2) = out["calls"][-2:]
    assert path1 == "/api/train/datasets" and body1 == {"name": "Night Rain", "trigger": "wk_rain", "kind": "character", "captions": "auto",
                                                        "items": [f"renders/c{i}.mp4" for i in range(9)]}
    assert path2 == "/api/train/start" and body2 == {"dataset": "night-rain", "name": "Night Rain", "steps": 2000, "size": "standard"}
    assert "about 61 h 7 min" in out["confirm"] and out["picked"] == 0 and "Training queued: Night Rain" in out["toasts"]


def test_too_few_or_no_trigger_is_said_and_nothing_is_sent():
    out = run("""trainPanel(); await settle(); for (let i = 0; i < 3; i++) TR.pick.add("renders/c" + i + ".mp4");
      await trainStart(); TR.pick.clear(); for (let i = 0; i < 8; i++) TR.pick.add("renders/c" + i + ".mp4"); await trainStart();
      out.toasts = toasts; out.sent = calls.filter(c => c[0] === "/api/train/datasets").length;""")
    assert "at least 8" in out["toasts"][0] and "trigger word" in out["toasts"][1] and out["sent"] == 0


def test_a_running_run_shows_its_card_a_stopped_one_resumes_and_a_finished_one_is_tried():
    out = run("""answers["/api/train"].runs = [{ lora: "fog", name: "Fog", trigger: "wk_fog", steps: 1000, saved: 400, can_resume: true, done: false },
        { lora: "rain", name: "Rain", trigger: "wk_rain", steps: 1000, saved: 1000, can_resume: false, done: true }];
      LORAS = [{ id: "rain", name: "Rain", installed: true, trigger: "wk_rain", styles: { look: "wk_rain" }, strength: 1, trained: { steps: 1000 } }];
      jobs = [{ id: "j9", kind: "train", status: "running", title: "Train LoRA · Mist", params: { lora: "mist" } }, { id: "j8", kind: "train", status: "done", title: "Train LoRA · Rain", params: { lora: "rain" } }];
      trainPanel(); await settle(); const page = trainPanel(); out.page = text(page);
      out.card = !!find(page, n => n.attrs && n.attrs["data-job"] === "j9"); out.start = button(page, "Start training").attrs.disabled;
      out.loads = loads;
      await button(find(page, n => n.attrs && n.attrs["data-run"] === "fog"), "Resume").attrs.onclick();
      await button(find(page, n => n.attrs && n.attrs["data-run"] === "rain"), "Try it").attrs.onclick();
      out.calls = calls.filter(c => c[0] !== "/api/train" && c[0] !== "/api/library").map(c => [c[0], c[1]]);""")
    assert out["card"] and out["start"] and "stopped at step 400 of 1000" in out["page"] and "ready in Looks" in out["page"] and out["loads"] >= 1
    assert out["calls"][0] == ["/api/train/start", {"resume": "fog"}]
    path, body = out["calls"][1]
    assert path == "/api/jobs" and body["kind"] == "video" and body["params"]["loras"] == [{"id": "rain", "strength": 1, "style": "look"}] and body["params"]["prompt"]
