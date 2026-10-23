"""The night shift's page code (app/static/nightshift.js) and Settings → Notifications, run in node with the page's
helpers stood in for (tests/test_shell_page.py's harness): no browser, no GPU, no network."""
import json
import subprocess
import tempfile
from pathlib import Path

from test_shell_page import FILES, HARNESS, need_node

ROOT = Path(__file__).resolve().parent.parent
SHIFTS = [
    {"id": "s3", "name": "Lighthouse, the rest", "state": "running", "start": "idle", "count": 3, "progress": {"done": 1, "total": 3, "current": "j2"}, "results": []},
    {"id": "s2", "name": "Tuesday", "state": "planned", "start": "23:00", "count": 2, "progress": {"done": 0, "total": 2}, "results": []},
    {"id": "s1", "name": "Monday", "state": "done", "start": "idle", "count": 2, "message": "Night shift done: 1 of 2 clips, 1 to look at",
     "reel": "dailies/2026-10-21-s1.mp4", "zip": "dailies/2026-10-21-s1.zip", "notes": "dailies/2026-10-21-s1.md",
     "results": [{"item": 0, "status": "done"}, {"item": 1, "status": "failed"}], "problems": []},
]


def run(code: str) -> dict:
    need_node()
    with tempfile.TemporaryDirectory() as scratch:
        (Path(scratch) / "harness.js").write_text(HARNESS)
        (Path(scratch) / "test.js").write_text(code)
        done = subprocess.run(["node", str(Path(scratch) / "harness.js"), str(Path(scratch) / "test.js"),
                               *[str(ROOT / "app/static" / f) for f in dict.fromkeys(FILES + ["nightshift.js"])]], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_panel_shows_each_shift_and_plans_a_film():
    out = run("""globalThis.jobCard = job => el("div", { class: "job" }, "card " + job.id); globalThis.confirm = () => true;
      globalThis.jobs = [{ id: "j2", status: "running", title: "Dusk · Shot 02" }];
      NS.list = """ + json.dumps(SHIFTS) + """; NS.films = [{ id: "p-1", name: "Lighthouse", shotCount: 4 }];
      let page = nightPanel(); out.page = text(page);
      out.seg = find(page, n => n.attrs && n.attrs.class === "seg-ctl").kids.map(text);
      out.running = text(find(page, n => n.attrs && n.attrs["data-shift"] === "s3"));
      out.done = text(find(page, n => n.attrs && n.attrs["data-shift"] === "s1"));
      out.zip = find(page, n => n.tag === "a" && text(n) === "Download the dailies").attrs.href;
      button(find(page, n => n.attrs && n.attrs.class === "seg-ctl"), "At 23:00").attrs.onclick();
      await button(nightPanel(), "Plan the shots without a good take").attrs.onclick();
      out.plan = calls.filter(c => c[0] === "/api/nightshift" && c[1]).slice(-1)[0][1];
      answers["/api/nightshift/s2/start"] = {}; await button(nightPanel(), "Start now").attrs.onclick();
      out.started = calls.some(c => c[0] === "/api/nightshift/s2/start");
      await button(find(nightPanel(), n => n.attrs && n.attrs["data-shift"] === "s3"), "Cancel").attrs.onclick();
      out.cancelled = calls.some(c => c[0] === "/api/nightshift/s3/cancel");""")
    assert "Night shift" in out["page"] and "Settings → Notifications" in out["page"]
    assert out["seg"] == ["When idle", "At 23:00", "Now"]
    assert "1 of 3 finished" in out["running"] and "card j2" in out["running"]
    assert "Night shift done: 1 of 2 clips, 1 to look at" in out["done"] and "Read the notes" in out["done"] and "1 clip not made" in out["done"]
    assert out["zip"] == "/files/dailies/2026-10-21-s1.zip"
    assert out["plan"] == {"start": "23:00", "film": "p-1"}
    assert out["started"] and out["cancelled"]


def test_slate_plans_the_rest_of_a_film_for_tonight():
    out = run("""globalThis.confirm = () => false; NS.list = []; NS.films = [];
      answers["/api/nightshift"] = { id: "s9", count: 3, state: "planned", start: "idle" };
      const b = nightFromFilm({ id: "p-7", name: "Harbour" }); out.label = text(b);
      await b.attrs.onclick(); out.sent = calls.filter(c => c[0] === "/api/nightshift" && c[1]).slice(-1)[0][1]; out.toast = toasts.slice(-1)[0];""")
    assert out["label"].endswith("Render the rest tonight")
    assert out["sent"] == {"start": "idle", "film": "p-7"}
    assert "3 clips" in out["toast"] and "when the queue is idle" in out["toast"]


def test_settings_has_a_notifications_section_with_plain_iphone_words():
    out = run("""answers["/api/push/key"] = { ready: true, key: "BAAA", devices: 2 };
      await settingsLoad(); go("Settings"); SET.section = "notify"; viewSettings(); await new Promise(r => setTimeout(r, 0));
      const page = viewSettings(); out.page = text(page);
      out.rail = find(page, n => n.attrs && n.attrs.class === "st-rail").kids.map(text);
      await button(page, "Send a test").attrs.onclick(); out.test = calls.some(c => c[0] === "/api/push/test");""")
    assert any("Notifications" in r for r in out["rail"])
    page = out["page"]
    assert "Turn on notifications on this device" in page and "Send a test" in page and "Add to Home Screen" in page
    assert "2 devices get notifications" in page
    assert out["test"]
