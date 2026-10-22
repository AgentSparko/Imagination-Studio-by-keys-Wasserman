"""The rough cut's page code (app/static/roughcut.js): Slate's Deliver panel and the Library's "Trailer from these",
run in node with the page's helpers stood in for: python3 -m pytest tests/test_roughcut_page.py. Skipped where node is
not installed; no browser, no GPU, no network."""
import json
import subprocess
import tempfile
from pathlib import Path

from test_shell_page import need_node

ROOT = Path(__file__).resolve().parent.parent
HARNESS = r"""
const vm = require("vm"), fs = require("fs");
globalThis.render = () => { globalThis.renders = (globalThis.renders || 0) + 1; }; globalThis.toasts = []; globalThis.toast = t => toasts.push(String(t));
globalThis.calls = []; globalThis.answers = {};
globalThis.api = async (path, body) => { calls.push([path, body]); const a = answers[path]; if (a instanceof Error) throw a; return a || {}; };
globalThis.jobs = []; globalThis.poll = async () => {}; globalThis.LIBPICK = new Set(); globalThis.timers = [];
globalThis.setInterval = (f, ms) => { timers.push(f); return timers.length; }; globalThis.clearInterval = () => {};
globalThis.mmss = s => Math.round(s) + " s"; globalThis.file = p => "/files/" + p;
const text = node => node == null || node === false ? "" : typeof node !== "object" ? String(node) : Array.isArray(node) ? node.map(text).join(" ") : [node.attrs && node.attrs.title, text(node.kids)].filter(Boolean).join(" ");
globalThis.text = text;
globalThis.el = (tag, attrs = {}, ...kids) => ({ tag, attrs, kids: kids.flat(Infinity).filter(k => k != null && k !== false), value: attrs.value || "" });
globalThis.field = (label, node) => el("div", {}, el("label", {}, label), node);
globalThis.choice = (obj, key, options, after) => el("div", { class: "choice", role: "radiogroup" }, Object.entries(options).map(([v, [t, sub]]) =>
  el("button", { type: "button", class: String(obj[key]) === v ? "on" : "", onclick: () => { obj[key] = v; if (after) after(); } }, el("b", {}, t), sub ? el("span", {}, sub) : null)));
globalThis.jobCard = job => el("div", { class: "job" }, "job " + job.id + " " + job.status);
const find = (node, test) => !node || typeof node !== "object" ? null : Array.isArray(node) ? node.map(n => find(n, test)).find(Boolean) || null : test(node) ? node : find(node.kids, test);
const all = (node, test, out = []) => { if (!node || typeof node !== "object") return out; if (Array.isArray(node)) { node.forEach(n => all(n, test, out)); return out; } if (test(node)) out.push(node); all(node.kids, test, out); return out; };
globalThis.find = find; globalThis.all = all; globalThis.button = (root, words) => find(root, n => n.tag === "button" && text(n).includes(words));
vm.runInThisContext(fs.readFileSync(process.argv[2], "utf8"));
vm.runInThisContext(`(async () => { const out = {}; ${fs.readFileSync(process.argv[3], "utf8")}; console.log(JSON.stringify(out)); })().catch(e => { console.error(e); process.exit(1); });`);
"""
FILM = {"id": "f1", "name": "Night Bus", "scenes": [{"shots": [{"takes": [{"mediaPath": "renders/a.mp4", "rating": "good"}]},
                                                               {"takes": [{"mediaPath": "renders/b.mp4", "rating": "no-good"}]},
                                                               {"takes": [{"mediaPath": "renders/c.mp4", "rating": "circled"}]}]}]}


def run(code: str) -> dict:
    need_node()
    with tempfile.TemporaryDirectory() as scratch:
        (Path(scratch) / "harness.js").write_text(HARNESS)
        (Path(scratch) / "test.js").write_text(code)
        done = subprocess.run(["node", str(Path(scratch) / "harness.js"), str(ROOT / "app/static/roughcut.js"), str(Path(scratch) / "test.js")],
                              capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_deliver_panel_picks_a_song_and_makes_a_rough_cut():
    got = run("""
      const film = %s;
      answers["/api/library"] = [{ path: "music/m1.wav", kind: "audio", sound: "music", title: "Neon run" }, { path: "voice/v1.wav", kind: "audio", sound: "voice", title: "Line" },
                                 { path: "renders/a.mp4", kind: "video" }];
      answers["/api/music/beats"] = { bpm: 121.6, bars: 40, duration: 80, confidence: 0.8 };
      let view = rcPanel(film); await new Promise(r => setTimeout(r, 0)); view = rcPanel(film);
      out.text = text(view);
      out.make_off = button(view, "Make a rough cut").attrs.disabled;
      out.groups = all(view, n => n.tag === "optgroup").map(n => n.attrs.label);
      out.cuts = all(find(view, n => n.attrs && n.attrs.class === "choice"), n => n.tag === "b").map(n => text(n));
      out.seg = !!find(view, n => n.attrs && n.attrs.class === "seg-ctl fit");
      const pick = find(view, n => n.tag === "select"); pick.value = "music/m1.wav"; pick.attrs.onchange(); await new Promise(r => setTimeout(r, 0));
      view = rcPanel(film); out.beats = text(view);
      button(view, "Every two bars").attrs.onclick(); button(view, "Foley for each clip").attrs.onclick(); button(view, "Calm to busy").attrs.onclick();
      answers["/api/roughcut"] = { project: "p1", job: "j9", foley: [], title: "Rough cut of Night Bus", beats: { bpm: 121.6 }, dropped: [],
                                   timeline: [{ path: "renders/a.mp4", in: 1, out: 3, at: 0 }, { path: "renders/c.mp4", in: 0, out: 1.5, at: 2, short: true }] };
      view = rcPanel(film); await button(view, "Make a rough cut").attrs.onclick();
      out.sent = calls.find(c => c[0] === "/api/roughcut")[1]; out.toast = toasts.slice(-1)[0];
      jobs = [{ id: "j9", status: "running" }];
      view = rcPanel(film); out.running = text(view);
      answers["/api/roughcut/p1"] = { project: "p1", stage: "done", job: "j9", video: "exports/j9.mp4", files: { fcpxml: "exports/x.fcpxml", edl: "exports/x.edl", zip: "exports/x-resolve.zip" } };
      await timers[0](); view = rcPanel(film); out.done = text(view);
      out.links = all(view, n => n.tag === "a").map(n => [text(n.kids), n.attrs.href]);
      out.strip = find(view, n => n.attrs && n.attrs.class === "strip").kids.length;
    """ % json.dumps(FILM))
    if got is None:
        return
    assert "A rough cut on the beat" in got["text"] and "2 shots with a take" in got["text"] and got["make_off"]
    assert got["groups"] == ["Songs", "Voice lines"]
    assert got["cuts"] == ["Every beat", "Every bar", "Every two bars", "Natural"] and got["seg"]
    assert "122 BPM · 40 bars · 80 s · a steady beat." in got["beats"]
    assert got["sent"] == {"film": "f1", "title": "Rough cut of Night Bus", "song": "music/m1.wav", "cut": "two-bars", "order": "energy", "foley": True}
    assert "2 cuts at 122 BPM" in got["toast"]
    assert "job j9 running" in got["running"] and "Exporting the cut" in got["running"]
    assert "Ready" in got["done"] and "job j9" not in got["done"]
    assert dict(got["links"]) == {"Play": "/files/exports/j9.mp4", "Open in Resolve (zip)": "/files/exports/x-resolve.zip",
                                  "FCPXML": "/files/exports/x.fcpxml", "Cut list · EDL": "/files/exports/x.edl"}
    assert got["strip"] == 2


def test_the_library_makes_a_trailer_from_the_ticked_clips():
    got = run("""
      answers["/api/library"] = [{ path: "music/m1.wav", kind: "audio", sound: "music", title: "Neon run" }];
      let view = rcLibrary(); button(view, "Trailer from these").attrs.onclick(); out.empty = toasts.slice(-1)[0];
      LIBPICK.add("renders/a.mp4"); LIBPICK.add("scouts/s1.png"); LIBPICK.add("exports/b.mp4");
      button(view, "Trailer from these").attrs.onclick();
      view = rcLibrary(); await new Promise(r => setTimeout(r, 0)); view = rcLibrary(); out.text = text(view);
      await button(view, "Make the trailer").attrs.onclick(); out.refused = toasts.slice(-1)[0];
      RC.song = "music/m1.wav"; answers["/api/roughcut"] = new Error("no steady beat was found in that song");
      view = rcLibrary(); await button(view, "Make the trailer").attrs.onclick(); out.error = toasts.slice(-1)[0];
      out.sent = calls.find(c => c[0] === "/api/roughcut")[1];
      button(rcLibrary(), "Close").attrs.onclick(); out.closed = text(rcLibrary());
    """)
    if got is None:
        return
    assert got["empty"] == "Tick the clips first"
    assert "Trailer from these" in got["text"] and "2 clips" in got["text"] and "Make the trailer" in got["text"]
    assert got["refused"] == "Pick a song first"
    assert got["sent"]["clips"] == ["renders/a.mp4", "exports/b.mp4"] and got["sent"]["title"] == "Trailer"
    assert got["error"] == "no steady beat was found in that song"
    assert "Make the trailer" not in got["closed"] and "Trailer from these" in got["closed"]
