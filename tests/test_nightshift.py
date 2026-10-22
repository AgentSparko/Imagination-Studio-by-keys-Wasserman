"""The night shift (app/nightshift.py) and phone notifications (app/push.py): plans kept and resumed, the scheduler's
idle and time starts, the morning's reel, dailies and notes from tiny clips made here, Web Push keys, encryption and
the pruning of gone subscriptions. No job is rendered: the queue is a stand-in, and no request leaves the machine."""
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp()) / "ImaginationStudio"))
os.environ.setdefault("STUDIO_GPU_LOCK", str(Path(tempfile.mkdtemp()) / "studio-gpu.lock.d"))
sys.path.insert(0, str(ROOT / "app"))
import nightshift  # noqa: E402
import push  # noqa: E402
import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)
HOME = server.HOME
FFMPEG = shutil.which("ffmpeg")
need_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg is not installed")
try:
    import cryptography  # noqa: F401
    HAVE_CRYPTO = True
except ImportError:
    HAVE_CRYPTO = False
need_crypto = pytest.mark.skipif(not HAVE_CRYPTO, reason="the cryptography package is not installed")


@pytest.fixture
def queue(monkeypatch):
    """The queue stood in for: jobs are recorded, never run; the plans folder starts empty."""

    shutil.rmtree(HOME / "nightshift", ignore_errors=True)
    (HOME / "nightshift").mkdir(parents=True)
    jobs, made, cancelled, told = {}, [], [], []

    def queue_job(body):
        job = {"id": f"j{len(made) + 1}", "kind": body["kind"], "status": "queued", "title": body.get("title") or body["kind"],
               "params": dict(body.get("params") or {}), "outputs": []}
        jobs[job["id"]] = job
        made.append(job)
        return dict(job)

    def cancel(job_id):
        cancelled.append(job_id)
        jobs[job_id]["status"] = "cancelled"

    monkeypatch.setitem(nightshift.Q, "jobs", jobs)
    monkeypatch.setitem(nightshift.Q, "lock", threading.Lock())
    monkeypatch.setitem(nightshift.Q, "queue_job", queue_job)
    monkeypatch.setitem(nightshift.Q, "cancel", cancel)
    monkeypatch.setitem(nightshift.Q, "notify", lambda title, body, url: told.append((title, body, url)) or {"sent": 1})
    monkeypatch.setitem(nightshift.IDLE, "since", None)
    return {"jobs": jobs, "made": made, "cancelled": cancelled, "told": told}


def clip(name: str, seconds: float = 1.0, size: str = "320x240", sound: bool = True) -> str:
    """A tiny clip in the studio folder, made by ffmpeg: a test pattern, a tone when it has sound."""

    target = HOME / "renders" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    tone = ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"] if sound else []
    subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=24:duration={seconds}", *tone,
                    "-pix_fmt", "yuv420p", *(["-c:a", "aac", "-shortest"] if sound else []), str(target)], check=True)
    return str(target.relative_to(HOME))


def finish(queue, job_id, output=None, **more):
    job = queue["jobs"][job_id]
    job.update(status="done", outputs=[output] if output else [], finished=time.time(), **more)


def jobs_body(n):
    return [{"kind": "job", "body": {"kind": "video", "title": f"{i + 1:02d} shot", "params": {"prompt": f"a lighthouse, beat {i + 1}", "seed": 40 + i}}}
            for i in range(n)]


# ---------- plans, the scheduler, a restart ----------

def test_a_plan_is_kept_queues_two_ahead_and_resumes_after_a_restart(queue, monkeypatch):
    monkeypatch.setattr(nightshift, "morning", lambda plan: plan.update(state="done", finished=time.time()))
    made = client.post("/api/nightshift", json={"name": "Tonight", "start": "23:00", "items": jobs_body(4)}).json()
    assert made["state"] == "planned" and made["count"] == 4 and made["start"] == "23:00"
    kept = json.loads((HOME / "nightshift" / f"{made['id']}.json").read_text())
    assert kept["items"][0]["body"]["params"]["seed"] == 40 and kept["state"] == "planned"
    assert [p["id"] for p in client.get("/api/nightshift").json()] == [made["id"]]

    started = client.post(f"/api/nightshift/{made['id']}/start").json()
    assert started["state"] == "running"
    assert [r["job"] for r in started["results"]] == ["j1", "j2"]  # never more than two waiting
    assert client.post(f"/api/nightshift/{made['id']}/start").status_code == 409

    queue["jobs"]["j1"]["status"] = "running"
    nightshift.tick()
    plan = client.get(f"/api/nightshift/{made['id']}").json()
    assert [r["job"] for r in plan["results"]] == ["j1", "j2", "j3"]  # j1 runs: two wait behind it
    assert plan["progress"]["current"] == "j1"

    finish(queue, "j1", seed=40)
    queue["jobs"]["j1"]["params"]["seed"] = 40
    nightshift.tick()
    assert client.delete(f"/api/nightshift/{made['id']}").status_code == 409  # running: cancel it first

    # the studio restarts: its queue is empty again, the plan is read back from its file
    queue["jobs"].clear()
    nightshift.tick()
    plan = client.get(f"/api/nightshift/{made['id']}").json()
    assert plan["state"] == "running"
    assert plan["results"][0]["status"] == "done" and plan["results"][0]["seed"] == 40  # what finished is kept
    assert [r["job"] for r in plan["results"][1:]] == ["j4", "j5"]  # the items that had no clip, queued again
    for job_id in ("j4", "j5"):
        finish(queue, job_id)
    nightshift.tick()
    finish(queue, "j6")
    nightshift.tick()
    plan = client.get(f"/api/nightshift/{made['id']}").json()
    assert plan["state"] == "done" and plan["progress"]["done"] == 4


def test_a_cleared_job_that_wrote_its_clip_is_not_rendered_again(queue, monkeypatch):
    monkeypatch.setattr(nightshift, "morning", lambda plan: plan.update(state="done"))
    body = jobs_body(1)
    body[0]["body"]["params"]["output"] = "renders/kept.mp4"
    (HOME / "renders").mkdir(exist_ok=True)
    (HOME / "renders" / "kept.mp4").write_bytes(b"clip")
    made = client.post("/api/nightshift", json={"items": body, "start": "now"}).json()
    assert made["state"] == "running" and made["results"][0]["job"] == "j1"
    queue["jobs"].clear()
    nightshift.tick()
    plan = client.get(f"/api/nightshift/{made['id']}").json()
    assert plan["state"] == "done" and plan["results"][0]["outputs"] == ["renders/kept.mp4"] and len(queue["made"]) == 1


def test_the_scheduler_starts_an_idle_plan_after_two_quiet_minutes_and_a_timed_one_at_its_time(queue):
    idle = client.post("/api/nightshift", json={"start": "idle", "items": jobs_body(1)}).json()
    queue["jobs"]["other"] = {"id": "other", "status": "running"}
    nightshift.tick(1000.0)
    assert nightshift.IDLE["since"] is None and client.get(f"/api/nightshift/{idle['id']}").json()["state"] == "planned"
    queue["jobs"]["other"]["status"] = "done"
    nightshift.tick(1010.0)
    nightshift.tick(1010.0 + nightshift.IDLE_AFTER - 1)
    assert client.get(f"/api/nightshift/{idle['id']}").json()["state"] == "planned"  # quiet, but not for two minutes yet
    nightshift.tick(1010.0 + nightshift.IDLE_AFTER)
    assert client.get(f"/api/nightshift/{idle['id']}").json()["state"] == "running"

    made = time.mktime((2026, 10, 22, 21, 30, 0, 0, 0, -1))
    assert not nightshift.time_due("23:00", made, made + 3600)
    assert nightshift.time_due("23:00", made, made + 90 * 60)
    late = time.mktime((2026, 10, 22, 23, 30, 0, 0, 0, -1))  # planned after 23:00: the next night's
    assert not nightshift.time_due("23:00", late, late + 3600) and nightshift.time_due("23:00", late, late + 24 * 3600)
    assert nightshift.time_due("2026-10-22T23:00", made, made + 90 * 60) and not nightshift.time_due("2026-10-22T23:00", made, made)
    assert nightshift.time_due("now", made, made)


def test_one_shift_runs_at_a_time_and_a_cancelled_one_cancels_its_jobs(queue):
    first = client.post("/api/nightshift", json={"start": "now", "items": jobs_body(3)}).json()
    second = client.post("/api/nightshift", json={"start": "now", "items": jobs_body(1)}).json()
    assert first["state"] == "running" and second["state"] == "planned"
    nightshift.tick()
    assert client.get(f"/api/nightshift/{second['id']}").json()["state"] == "planned"
    stopped = client.post(f"/api/nightshift/{first['id']}/cancel").json()
    assert stopped["state"] == "cancelled" and queue["cancelled"] == ["j1", "j2"]
    nightshift.tick()
    assert client.get(f"/api/nightshift/{second['id']}").json()["state"] == "running"
    assert client.delete(f"/api/nightshift/{first['id']}").json() == {"ok": True}
    assert client.get(f"/api/nightshift/{first['id']}").status_code == 404


def test_a_plan_is_refused_without_items_or_with_a_bad_start(queue):
    assert client.post("/api/nightshift", json={"items": []}).status_code == 400
    assert client.post("/api/nightshift", json={"items": jobs_body(1), "start": "tonight"}).status_code == 400
    assert client.post("/api/nightshift", json={"items": [{"kind": "shot", "film": "x"}]}).status_code == 400
    assert client.get("/api/nightshift/..%2Fsecret").status_code in (400, 404)


def test_a_film_plans_its_shots_without_a_good_take_and_each_take_is_logged(queue):
    film = client.post("/api/slate/projects", json={"name": "Lighthouse at dusk"}).json()
    (HOME / "renders").mkdir(exist_ok=True)
    (HOME / "renders" / "good.mp4").write_bytes(b"clip")
    film["scenes"] = [{"id": "scene-1", "name": "Dusk", "shots": [
        {"id": "shot-1", "name": "Shot 01", "prompt": "# Subject: The keeper climbs the stairs.", "spec": {"durationSec": 5}, "takes": []},
        {"id": "shot-2", "name": "Shot 02", "prompt": "# Subject: The lamp turns.", "spec": {"durationSec": 5},
         "takes": [{"id": "t", "mediaPath": "renders/good.mp4", "rating": "good"}]},
        {"id": "shot-3", "name": "Shot 03", "prompt": "# Subject: The sea below.", "spec": {"durationSec": 5},
         "takes": [{"id": "t", "mediaPath": "renders/good.mp4", "rating": "no-good"}]},
        {"id": "shot-4", "name": "Shot 04", "prompt": "", "spec": {}, "takes": []}]}]
    client.put(f"/api/slate/projects/{film['id']}", json=film)
    plan = client.post("/api/nightshift", json={"film": "lighthouse at dusk", "start": "now"}).json()
    assert [i["shot"] for i in plan["items"]] == ["shot-1", "shot-3"] and plan["name"] == "Lighthouse at dusk, the rest"
    assert plan["state"] == "running" and [r["job"] for r in plan["results"]] == ["j1", "j2"]
    sent = queue["made"][0]
    assert sent["kind"] == "video" and sent["title"] == "Dusk · Shot 01" and "keeper" in sent["params"]["prompt"]
    assert sent["params"]["slate"] == {"project": film["id"], "scene": "scene-1", "shot": "shot-1"}
    assert sent["params"]["output"].startswith(f"slate/{film['id']}/takes/shot-1-")
    saved = client.get(f"/api/slate/projects/{film['id']}").json()
    take = saved["scenes"][0]["shots"][0]["takes"][0]
    assert take["jobId"] == "j1" and take["mediaPath"] == sent["params"]["output"] and take["origin"] == "Night shift"
    assert take["rating"] == ""  # rated when it finishes, from its checks

    # its checks are written beside the take when it finishes
    finish(queue, "j1", qa={"passed": False, "takes": [{"audio": {"result": "fail", "note": "no speech heard"}, "identity": {"result": "pass", "note": ""}}]})
    nightshift.tick()
    saved = client.get(f"/api/slate/projects/{film['id']}").json()
    assert saved["scenes"][0]["shots"][0]["takes"][0]["qa"]["passed"] is False
    assert saved["scenes"][0]["shots"][0]["takes"][0]["rating"] == "no-good"
    wrote = HOME / queue["made"][1]["params"]["output"]
    wrote.parent.mkdir(parents=True, exist_ok=True)
    wrote.write_bytes(b"clip")
    finish(queue, "j2", str(wrote.relative_to(HOME)))  # its clip written, no checks ran: good
    nightshift.tick()
    assert client.get(f"/api/slate/projects/{film['id']}").json()["scenes"][0]["shots"][2]["takes"][0]["rating"] == "good"
    named = client.post("/api/nightshift", json={"film": film["id"], "shots": ["Shot 02"]}).json()
    assert [i["shot"] for i in named["items"]] == ["shot-2"]
    assert client.post("/api/nightshift", json={"film": film["id"], "shots": ["Shot 99"]}).status_code == 404


def test_only_a_good_or_circled_take_on_disk_is_a_good_take():
    (HOME / "renders").mkdir(exist_ok=True)
    (HOME / "renders" / "good.mp4").write_bytes(b"clip")
    take = lambda rating, path="renders/good.mp4": {"takes": [{"mediaPath": path, "rating": rating}]}
    assert nightshift.good_take(take("good")) and nightshift.good_take(take("circled"))
    assert not nightshift.good_take(take("")) and not nightshift.good_take(take("no-good"))  # rendering, or failed
    assert not nightshift.good_take(take("good", "renders/missing.mp4"))


# ---------- the morning ----------

@need_ffmpeg
def test_the_morning_makes_a_reel_the_dailies_and_notes_failures_first(queue):
    passed = clip("night-a.mp4", 1.0, "320x240", sound=True)
    failed = clip("night-b.mp4", 1.0, "240x320", sound=False)
    plan = {"id": "0101-000000-abcdef", "name": "Tonight", "items": jobs_body(3), "state": "running", "started": time.time() - 3600,
            "results": [
                {"item": 0, "job": "j1", "status": "done", "title": "01 shot", "seed": 40, "outputs": [passed], "params": {"prompt": "a", "seed": 40},
                 "qa": {"passed": True, "takes": []}, "shot": {"passed": True, "takes": []}},
                {"item": 1, "job": "j2", "status": "done", "title": "02 shot", "seed": 41, "outputs": [failed], "params": {"prompt": "b", "seed": 41},
                 "shot": {"passed": False, "takes": [{"first": {"held": False}, "last": {"held": True}, "cuts": [{"seconds": 0.5}]}]}},
                {"item": 2, "job": "j3", "status": "failed", "title": "03 shot", "error": "out of memory", "outputs": []}]}
    nightshift.morning(plan)
    assert plan["state"] == "done" and not plan["problems"], plan["problems"]
    reel = HOME / plan["reel"]
    assert plan["reel"].startswith("dailies/") and plan["reel"].endswith(f"-{plan['id']}.mp4")
    seen = json.loads(subprocess.run([shutil.which("ffprobe"), "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate:format=duration",
                                      "-of", "json", str(reel)], capture_output=True, text=True, check=True).stdout)
    video = next(s for s in seen["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1280, 720) and video["r_frame_rate"] == "24/1"
    assert any(s["codec_type"] == "audio" for s in seen["streams"])
    assert abs(float(seen["format"]["duration"]) - (2 * nightshift.SLATE_SECONDS + 2.0)) < 0.4

    with zipfile.ZipFile(HOME / plan["zip"]) as bundle:
        names = bundle.namelist()
    assert "dailies/01_take01.mp4" in names and "dailies/02_take01.json" in names

    notes = (HOME / plan["notes"]).read_text()
    assert notes.index("To look at") < notes.index("Passed or not checked")
    assert notes.index("02 shot") < notes.index("01 shot") and "first frame not held" in notes and "1 cut inside the shot" in notes
    assert "03 shot" in notes and "out of memory" in notes and passed in notes
    assert queue["told"] == [("Night shift done", "2 of 3 clips, 2 to look at", "/#Home")]
    assert plan["message"] == "Night shift done: 2 of 3 clips, 2 to look at"


def test_a_clip_is_judged_by_its_checks():
    assert nightshift.verdict({"status": "failed", "error": "stopped"}) == (False, "stopped")
    assert nightshift.verdict({"status": "done", "outputs": []}) == (False, "no clip was written")


@need_ffmpeg
def test_the_reel_has_no_titles_without_a_font_and_is_made_all_the_same(queue, monkeypatch, tmp_path):
    monkeypatch.setattr(nightshift, "font", lambda: None)
    one = clip("night-c.mp4", 0.5)
    target = nightshift.build_reel([{"path": str(HOME / one), "lines": ["Shot: it's 50% done"]}], tmp_path / "reel.mp4")
    assert target.is_file() and target.stat().st_size > 1000


# ---------- push ----------

@pytest.fixture
def pushed(monkeypatch, tmp_path):
    monkeypatch.setattr(push, "HOME", tmp_path)
    monkeypatch.setattr(push, "pywebpush", None)  # the studio's own implementation, whether or not the library is there
    posts = []
    answers = {}

    def post(url, body, headers):
        posts.append((url, body, headers))
        return answers.get(url, 201)
    monkeypatch.setattr(push, "post", post)
    return {"posts": posts, "answers": answers}


def browser():
    from cryptography.hazmat.primitives.asymmetric import ec
    key = ec.generate_private_key(ec.SECP256R1())
    return key, {"p256dh": push.b64(push.raw_public(key.public_key())), "auth": push.b64(os.urandom(16))}


@need_crypto
def test_the_key_pair_is_made_once_and_kept_private(pushed, tmp_path):
    first = push.keys()["public"]
    assert push.keys()["public"] == first and len(push.unb64(first)) == 65 and push.unb64(first)[0] == 4
    assert oct((tmp_path / "push" / "vapid.json").stat().st_mode & 0o777) == "0o600"


@need_crypto
def test_encryption_matches_rfc_8291_and_a_browser_can_read_it():
    from cryptography.hazmat.primitives.asymmetric import ec

    def key(text):
        return ec.derive_private_key(int.from_bytes(push.unb64(text), "big"), ec.SECP256R1())
    # RFC 8291, Appendix A: the example's keys, salt and message
    sender, receiver = key("yfWPiYE-n46HLnH0KqZOF1fJJU3MYrct3AELtAQ-oRw"), key("q1dXpw3UpT5VOmu_cf_v6ih07Aems3njxI-JWgLcM94")
    auth = "BTBZMqHH6r4Tts7J_aSIgg"
    out = push.encrypt(b"When I grow up, I want to be a watermelon", push.b64(push.raw_public(receiver.public_key())), auth,
                       salt=push.unb64("DGv6ra1nlYgDCS1FRnbzlw"), sender=sender)
    assert push.b64(out) == ("DGv6ra1nlYgDCS1FRnbzlwAAEABBBP4z9KsN6nGRTbVYI_c7VJSPQTBtkgcy27mlmlMoZIIgDll6e3vCYLocInmYWAmS6TlzAC8wEqKK6PBru3jl7A_"
                             "yl95bQpu6cVPTpK4Mqgkf1CXztLVBSt2Ks3oZwbuwXPXLWyouBWLVWGNWQexSgSxsj_Qulcy4a-fN")
    other, keys = browser()
    sealed = push.encrypt(b'{"title": "Night shift done"}', keys["p256dh"], keys["auth"])
    assert push.decrypt(sealed, other, keys["auth"]) == b'{"title": "Night shift done"}'
    assert push.encrypt(b"x", keys["p256dh"], keys["auth"]) != push.encrypt(b"x", keys["p256dh"], keys["auth"])  # new salt and key each time


@need_crypto
def test_the_vapid_token_is_signed_for_the_push_service(pushed):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
    signing = push.keys()
    header = push.vapid_header("https://web.push.apple.com/abc", signing["private"], signing["public"], now=1000)
    token, k = header.removeprefix("vapid t=").split(", k=")
    assert k == signing["public"]
    head, claims, signature = token.split(".")
    said = json.loads(base64.urlsafe_b64decode(claims + "=="))
    assert said["aud"] == "https://web.push.apple.com" and said["exp"] == 1000 + 12 * 3600 and said["sub"].startswith(("mailto:", "https:"))
    raw = push.unb64(signature)
    public = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), push.unb64(k))
    public.verify(encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")), f"{head}.{claims}".encode(), ec.ECDSA(hashes.SHA256()))


@need_crypto
def test_a_message_goes_to_every_device_and_gone_ones_are_dropped(pushed):
    phones = [browser() for _ in range(3)]
    endpoints = ["https://web.push.apple.com/one", "https://fcm.googleapis.com/fcm/send/two", "https://updates.push.services.mozilla.com/three"]
    for (_, keys), endpoint in zip(phones, endpoints):
        assert client.post("/api/push/subscribe", json={"endpoint": endpoint, "keys": keys}).status_code == 200
    pushed["answers"][endpoints[1]] = 410
    pushed["answers"][endpoints[2]] = 500
    assert push.send("Night shift done", "7 of 8 clips, 2 to look at", "/#Home") == {"sent": 1, "dropped": 1, "failed": 1}
    assert [s["endpoint"] for s in push.subscriptions()] == [endpoints[0], endpoints[2]]
    url, body, headers = pushed["posts"][0]
    assert url == endpoints[0] and headers["Content-Encoding"] == "aes128gcm" and headers["Authorization"].startswith("vapid t=")
    said = json.loads(push.decrypt(body, phones[0][0], phones[0][1]["auth"]))
    assert said == {"title": "Night shift done", "body": "7 of 8 clips, 2 to look at", "url": "/#Home"}


@need_crypto
def test_the_push_routes(pushed):
    key = client.get("/api/push/key").json()
    assert key["ready"] and key["key"] == push.keys()["public"] and key["devices"] == 0
    assert client.post("/api/push/test", json={}).status_code == 400  # nobody subscribed yet
    _, keys = browser()
    for bad in ({"endpoint": "http://127.0.0.1:7870/x", "keys": keys}, {"endpoint": "https://example.com/push", "keys": keys},
                {"endpoint": "https://fcm.googleapis.com/x", "keys": {"p256dh": "AAAA", "auth": "BBBB"}}):
        assert client.post("/api/push/subscribe", json=bad).status_code == 400
    sub = {"endpoint": "https://fcm.googleapis.com/fcm/send/abc", "keys": keys}
    assert client.post("/api/push/subscribe", json=sub).json()["devices"] == 1
    assert client.post("/api/push/subscribe", json=sub).json()["devices"] == 1  # the same browser once
    assert client.post("/api/push/test", json={}).json() == {"sent": 1, "dropped": 0, "failed": 0}
    assert client.request("DELETE", "/api/push/subscribe", json={"endpoint": sub["endpoint"]}).json()["devices"] == 0
    refused = TestClient(server.app, base_url="http://studio.example.com").post("/api/push/subscribe", json=sub)
    assert refused.status_code == 403  # from this computer or its Tailscale address only
