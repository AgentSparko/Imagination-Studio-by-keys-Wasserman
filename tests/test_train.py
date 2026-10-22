"""Train a LoRA (app/train.py, spark/train_lora.py): datasets from the Library, the job's command, resuming from a
checkpoint, the hub record, readiness. Run: python3 -m pytest tests/test_train.py"""

import json
import os
import struct
import sys
import tempfile
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "spark"))
import train  # noqa: E402
from fastapi import HTTPException  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("train_lora", ROOT / "spark/train_lora.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def safetensors(path: Path, names: list[str]) -> None:
    """A small safetensors file with these tensors (2x4 float32 each)."""

    header, at = {}, 0
    for name in names:
        header[name] = {"dtype": "F32", "shape": [2, 4], "data_offsets": [at, at + 32]}
        at += 32
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * at)


COMFY = [f"diffusion_model.blocks.{n}.attn.qkv_proj.lora_{side}.weight" for n in range(2) for side in "AB"]


class Studio:
    def __init__(self, ready=True):
        root = Path(tempfile.mkdtemp())
        self.home, self.prefix, self.models = root / "ImaginationStudio", root / "prefix", root / "models"
        for name in ("renders", "uploads", "loras", "sparks"):
            (self.home / name).mkdir(parents=True)
        os.environ.update(PREFIX=str(self.prefix), SPARK_MODELS=str(self.models))
        if ready:
            self.install()
        self.lengths = {}
        self.jobs, self.lock, self.queued = {}, threading.Lock(), []

    def install(self):
        musubi = self.prefix / "train/musubi-tuner"
        (musubi / ".git").mkdir(parents=True, exist_ok=True)
        (musubi / ".git/HEAD").write_text(train.MUSUBI_COMMIT + "\n")
        (self.prefix / "train/venv/bin").mkdir(parents=True, exist_ok=True)
        for tool in ("python", "accelerate"):
            (self.prefix / "train/venv/bin" / tool).write_text("")
        (self.prefix / "train/optimizer").write_text("adamw8bit\n")
        for one in (train.DIT, train.TEXT_ENCODERS[1], train.AUDIO_VAE, train.VIDEO_VAE, train.ADAPTER):
            (self.models / one).parent.mkdir(parents=True, exist_ok=True)
            (self.models / one).write_bytes(b"x")

    def clip(self, name, seconds=5.17):
        path = self.home / "renders" / name
        path.write_bytes(b"clip")
        self.lengths[str(path.resolve())] = seconds
        return f"renders/{name}"

    def still(self, name):
        (self.home / "uploads" / name).write_bytes(b"png")
        return f"uploads/{name}"

    def inside(self, path):
        resolved = (self.home / path).resolve()
        if self.home.resolve() not in resolved.parents:
            raise HTTPException(400, "path is outside the studio folder")
        return resolved

    def seconds(self, path):
        return self.lengths.get(str(Path(path).resolve()), self.lengths.get(path)), True

    def attach(self, spark=True):
        def enqueue(kind, title, params, estimate=None):
            job = {"id": f"j{len(self.queued) + 1}", "kind": kind, "title": title, "params": params, "estimate": estimate, "status": "queued", "created": len(self.queued)}
            self.queued.append(job)
            self.jobs[job["id"]] = job
            return job
        router, build, free = train.attach(self.home, ROOT, spark, self.inside, self.seconds, enqueue, self.jobs, self.lock)
        routes = {(sorted(r.methods)[0], r.path): r.endpoint for r in router.routes}
        return routes, build, free

    def dataset(self, items, **extra):
        body = {"name": "Night Rain", "trigger": "wk_rain", "kind": "look", "captions": "auto", "items": items, **extra}
        return train.build_dataset(self.home, body, self.inside, self.seconds)


@pytest.fixture(autouse=True)
def spark_platform(monkeypatch):
    monkeypatch.setattr(train, "spark", lambda: True)


def refused(call, *args, **kwargs):
    try:
        call(*args, **kwargs)
    except HTTPException as error:
        return error.detail
    return None


def test_captions_come_from_prompts_then_the_index_then_the_trigger():
    s = Studio()
    clips = [s.clip(f"c{n}.mp4") for n in range(6)] + [s.still("a.png"), s.still("b.png")]
    with open(s.home / "sparks/takes.jsonl", "w") as out:
        out.write(json.dumps({"path": clips[0], "prompt": "[Shot 1] A wet street at night, neon in the puddles."}) + "\n")
        out.write(json.dumps({"path": clips[1], "settings": {"prompt": "wk_rain, a tram passes in the rain."}}) + "\n")
        out.write("not json\n")
    (s.home / "library").mkdir()
    (s.home / "library/captions.json").write_text(json.dumps({clips[2]: "An umbrella crosses a square.", "uploads/a.png": {"caption": "A window streaked with rain."}}))
    record = s.dataset(clips)
    caps = [one["caption"] for one in record["items"]]
    assert caps[0] == "wk_rain, A wet street at night, neon in the puddles."
    assert caps[1] == "wk_rain, a tram passes in the rain."  # the trigger is not said twice
    assert caps[2] == "wk_rain, An umbrella crosses a square." and caps[6] == "wk_rain, A window streaked with rain."
    assert caps[3] == "wk_rain" and caps[7] == "wk_rain"
    assert [one["from"] for one in record["items"]] == ["prompt", "prompt", "index", "trigger", "trigger", "trigger", "index", "trigger"]
    assert all(c.startswith("wk_rain") for c in caps)
    # prompts only: the index is not read; none: the trigger alone
    folder = Path(s.home / "train" / s.dataset(clips, name="Night Rain B", captions="prompts")["id"])
    assert [json.loads(l)["caption"] for l in (folder / "metadata.jsonl").read_text().splitlines()][2] == "wk_rain"
    record = s.dataset(clips, name="Night Rain C", captions="none")
    assert {one["caption"] for one in record["items"]} == {"wk_rain"}


def test_a_character_says_what_each_item_is_of_and_the_files_are_written():
    s = Studio()
    clips = [s.clip(f"c{n}.mp4") for n in range(5)] + [s.still(f"s{n}.jpg") for n in range(3)]
    (s.home / "library").mkdir()
    (s.home / "library/captions.json").write_text(json.dumps([{"path": clips[0], "caption": "She turns to the window."}]))
    record = s.dataset(clips, name="Mara", trigger="mara_k", kind="character")
    folder = s.home / "train" / record["id"]
    assert record["id"] == "mara" and record["items"][0]["caption"] == "mara_k, a video of mara_k, She turns to the window."
    assert record["items"][5]["caption"] == "mara_k, a photo of mara_k" and record["items"][5]["file"] == "media/006.jpg"
    videos = [json.loads(l) for l in (folder / "videos.jsonl").read_text().splitlines()]
    images = [json.loads(l) for l in (folder / "images.jsonl").read_text().splitlines()]
    assert len(videos) == 5 and len(images) == 3 and set(videos[0]) == {"video_path", "caption"} and set(images[0]) == {"image_path", "caption"}
    assert Path(videos[0]["video_path"]).read_bytes() == b"clip" and Path(videos[0]["video_path"]).parent == folder / "media"
    assert len((folder / "metadata.jsonl").read_text().splitlines()) == 8
    toml = (folder / "dataset.toml").read_text()
    for line in ("resolution = [928, 512]", "batch_size = 1", "enable_bucket = true", "bucket_no_upscale = false",
                 f'video_jsonl_file = "{folder / "videos.jsonl"}"', "target_frames = [124]", 'frame_extraction = "uniform"',
                 "frame_sample = 2", f'image_jsonl_file = "{folder / "images.jsonl"}"', f'cache_directory = "{folder / "cache/standard/image"}"'):
        assert line in toml, line
    assert toml.count("[[datasets]]") == 2
    assert "image_jsonl_file" not in train.toml_text(folder, {"video"}) and "resolution = [704, 384]" in train.toml_text(folder, {"video"}, "small")
    assert train.summary(record)["videos"] == 5 and train.summary(record)["stills"] == 3 and train.summary(record)["from"]["index"] == 1
    assert s.dataset(clips, name="Mara", trigger="mara_k", kind="character")["id"] == "mara-2"  # a second one does not overwrite


def test_a_dataset_is_refused_when_it_is_too_small_too_short_or_badly_named():
    s = Studio()
    clips = [s.clip(f"c{n}.mp4") for n in range(8)]
    assert "at least 8" in refused(s.dataset, clips[:7])
    s.lengths[str((s.home / clips[3]).resolve())] = 4.0
    assert "at least 5 seconds" in refused(s.dataset, clips)
    s.lengths[str((s.home / clips[3]).resolve())] = 5.17
    assert "trigger" in refused(s.dataset, clips, trigger="two words")
    assert "name" in refused(s.dataset, clips, name="../escape")
    assert "kind" in refused(s.dataset, clips, kind="sound")
    assert "outside" in refused(s.dataset, clips[:7] + ["../../etc/passwd"])
    (s.home / "renders/notes.txt").write_text("x")
    assert "not a clip or a still" in refused(s.dataset, clips[:7] + ["renders/notes.txt"])
    assert not (s.home / "train").exists() or not list((s.home / "train").iterdir())  # nothing half made


def test_readiness_names_what_is_missing():
    s = Studio(ready=False)
    said = train.ready()
    assert not said["ready"] and said["setup"] == "bash oneshot-setup-spark.sh --train"
    joined = " ".join(said["missing"])
    for words in ("musubi-tuner at f8a1b03", "training environment", "FL2VA", "video decoder in fp16", "adapter v2", "text encoder"):
        assert words in joined, words
    s.install()
    assert train.ready() == {"ready": True, "spark": True, "missing": [], "setup": train.SETUP, "optimizer": "adamw8bit"}
    (s.prefix / "train/musubi-tuner/.git/HEAD").write_text("0" * 40)
    assert not train.ready()["ready"]


def test_readiness_on_a_mac(monkeypatch):
    monkeypatch.setattr(train, "spark", lambda: False)
    assert train.ready()["spark"] is False and not train.ready()["ready"]


def test_the_job_runs_the_runner_with_its_settings_and_writes_into_the_hub():
    s = Studio()
    record = s.dataset([s.clip(f"c{n}.mp4") for n in range(8)])
    routes, build, free = s.attach()
    params = {"dataset": record["id"], "lora": "night-rain", "name": "Night Rain", "steps": 1200, "rank": 16, "lr": 1e-4, "size": "standard", "save_every": 100}
    command, env, outputs = build(params, "j1")
    assert command[0] == sys.executable and command[1].endswith("spark/train_lora.py")
    joined = " ".join(command)
    for part in ("--lora night-rain", "--trigger wk_rain", "--kind look", "--steps 1200", "--rank 16", "--lr 0.0001", "--size standard", "--save-every 100"):
        assert part in joined, part
    assert "--resume" not in command and build({**params, "resume": True}, "j1")[0][-1] == "--resume"
    assert outputs == [str(s.home / "loras/night-rain.safetensors")] and env["PYTHONUNBUFFERED"] == "1"
    assert "between 300 and 3000" in refused(build, {**params, "steps": 50}, "j1")
    assert "rank" in refused(build, {**params, "rank": 64}, "j1")
    assert "size" in refused(build, {**params, "size": "huge"}, "j1")
    assert "no dataset" in refused(build, {**params, "dataset": "missing"}, "j1")
    _, mac_build, _ = s.attach(spark=False)
    assert "DGX Spark" in refused(mac_build, params, "j1")
    os.remove(s.models / train.ADAPTER)
    assert "oneshot-setup-spark.sh --train" in refused(build, params, "j1")


def test_start_queues_a_run_with_its_estimate_and_a_free_id_and_resume_goes_on(monkeypatch):
    s = Studio()
    (s.home / "loras/hub.json").write_text(json.dumps([{"id": "night-rain", "name": "Night Rain"}]))
    record = s.dataset([s.clip(f"c{n}.mp4") for n in range(8)])
    routes, build, free = s.attach()
    job = routes[("POST", "/api/train/start")]({"dataset": record["id"], "steps": 1000})
    assert job["kind"] == "train" and job["estimate"] == 1000 * 110 and job["title"] == "Train LoRA · Night Rain"
    assert job["params"]["lora"] == "night-rain-2" and job["params"]["rank"] == 16 and job["params"]["lr"] == 1e-4
    run = json.loads((s.home / "train" / record["id"] / "runs/night-rain-2/run.json").read_text())
    assert run["trigger"] == "wk_rain" and run["items"] == 8 and run["job"] == job["id"]
    assert routes[("GET", "/api/train/status/{key}")](job["id"])["job"]["status"] == "queued"
    assert "in the queue already" in refused(routes[("POST", "/api/train/start")], {"resume": "night-rain-2"})
    job["status"] = "cancelled"
    state = s.home / "train" / record["id"] / "runs/night-rain-2/night-rain-2-step00000300-state"
    state.mkdir()
    (state / "optimizer.bin").write_bytes(b"x")
    overview = routes[("GET", "/api/train")]()
    assert overview["ready"] and overview["runs"][0]["saved"] == 300 and overview["runs"][0]["can_resume"]
    again = routes[("POST", "/api/train/start")]({"resume": "night-rain-2"})
    assert again["params"]["resume"] is True and again["estimate"] == 700 * 110 and again["params"]["dataset"] == record["id"]
    assert routes[("GET", "/api/train/status/{key}")]("night-rain-2")["job"]["id"] == again["id"]
    assert "no run" in refused(routes[("POST", "/api/train/start")], {"resume": "nothing"})


def test_resume_finds_the_last_checkpoint_across_parts(tmp_path):
    out = tmp_path / "runs/night"
    for name in ("night-step00000100-state", "night-step00000200-state", "night-from00200-step00000100-state", "night-from00200-step00000200-state"):
        (out / name).mkdir(parents=True)
        (out / name / "model.safetensors").write_bytes(b"x")
    (out / "night-from00200-step00000300-state").mkdir()  # cut off while it was written: no file in it
    (out / "night-step00000100.safetensors").write_bytes(b"x")
    assert [step for step, _ in train.checkpoints(out)] == [100, 200, 300, 400]
    assert train.last_checkpoint(out)[1].name == "night-from00200-step00000200-state"
    assert train.segment_name("night", 0) == "night" and train.segment_name("night", 400) == "night-from00400"
    assert train.last_checkpoint(tmp_path / "none") is None


def test_the_runner_resumes_with_the_steps_left_under_a_new_name():
    s = Studio()
    record = s.dataset([s.clip(f"c{n}.mp4") for n in range(6)] + [s.still("a.png"), s.still("b.png")])
    folder = s.home / "train" / record["id"]
    args = ["--home", str(s.home), "--dataset", str(folder), "--lora", "night", "--name", "Night", "--trigger", "wk_rain",
            "--steps", "1000", "--rank", "8", "--lr", "2e-4", "--save-every", "100"]
    plan = runner.plan(runner.parse(args))
    command = plan["commands"]["train"]
    value = lambda flag: command[command.index(flag) + 1]  # noqa: E731
    assert (value("--max_train_steps"), value("--output_name"), value("--network_dim"), value("--network_alpha")) == ("1000", "night", "8", "8")
    assert value("--learning_rate") == "0.0002" and value("--optimizer_type") == "adamw8bit" and value("--base_weights").endswith("ostris_dd_adapter_v2.safetensors")
    assert "--convrot_int8" in command and "--resume" not in command and "--one_frame" in command and "--video_only" not in command
    assert value("--dit").endswith("minimax_h3_fl2va_pruned_bf16.safetensors") and value("--save_every_n_steps") == "100" and "--save_state" in command
    assert "--one_frame" in plan["commands"]["latents"] and "--skip_broken" in plan["commands"]["latents"]
    assert plan["commands"]["text"][plan["commands"]["text"].index("--text_encoder_blocks_to_swap") + 1] == "50"
    state = folder / "runs/night/night-step00000300-state"
    state.mkdir(parents=True)
    (state / "optimizer.bin").write_bytes(b"x")
    plan = runner.plan(runner.parse(args + ["--resume"]))
    command = plan["commands"]["train"]
    assert (value("--max_train_steps"), value("--output_name"), value("--resume")) == ("700", "night-from00300", str(state))
    assert plan["offset"] == 300 and plan["final"] == folder / "runs/night/night-from00300.safetensors"
    (s.prefix / "train/optimizer").write_text("adamw\n")
    assert "adamw" == runner.plan(runner.parse(args))["commands"]["train"][command.index("--optimizer_type") + 1]


def test_stills_only_train_without_sound():
    s = Studio()
    record = s.dataset([s.still(f"s{n}.png") for n in range(8)])
    plan = runner.plan(runner.parse(["--home", str(s.home), "--dataset", str(s.home / "train" / record["id"]), "--lora", "x",
                                     "--name", "X", "--trigger", "wk_rain"]))
    assert "--video_only" in plan["commands"]["train"] and plan["kinds"] == ["image"]


def test_the_progress_bar_becomes_step_lines_counted_from_the_checkpoint():
    shown = [-1]
    bar = b"steps:  10%|\xe2\x96\x88         | 70/700 [2:08:20<19:15:00, 110.0s/it, avr_loss=0.1]"
    assert runner.progress(bar, 300, 1000, shown) == ["[tensorfold] train 370/1000"]
    assert runner.progress(bar, 300, 1000, shown) == []  # the same step is said once
    assert runner.progress(b"caching: 100%|###| 8/8", 0, 1000, [-1]) == []
    assert runner.TQDM.search(b"steps:   0%|          | 0/1000 [00:00<?, ?it/s]")


def test_the_server_reads_train_lines_as_progress():
    import re
    source = (ROOT / "app/server.py").read_text()
    pattern = re.search(r'STEP = re\.compile\(r"(.+)"\)', source).group(1)
    found = re.search(pattern, "[tensorfold] train 370/1000")
    assert found and found.group(1) == "train" and '"train": "training step"' in source
    assert "train_free(job)" in source and "return train_build(params, job_id)" in source


def test_a_finished_run_is_converted_and_added_to_the_hub(tmp_path):
    s = Studio()
    final = tmp_path / "night.safetensors"
    safetensors(final, [f"lora_unet_blocks_{n}_attn_qkv_proj.{part}" for n in range(2) for part in ("lora_down.weight", "lora_up.weight")])
    converted = []

    def fake_convert(source, part):  # musubi-tuner's convert_lora.py --target other, stood in for
        converted.append(source)
        safetensors(part, COMFY)

    target = runner.finish(s.home, "night", "Night Rain", "wk_rain", "look", 1000, 8, final, fake_convert, {"rank": 16})
    assert target == s.home / "loras/night.safetensors" and converted == [final]
    assert not list((s.home / "loras").glob("*part*")) and not (s.home / "loras/night.comfy.safetensors").exists()
    hub = json.loads((s.home / "loras/hub.json").read_text())
    assert len(hub) == 1
    one = hub[0]
    assert (one["id"], one["name"], one["trigger"], one["styles"], one["strength"], one["status"], one["target"], one["base"]) == \
        ("night", "Night Rain", "wk_rain", {"look": "wk_rain"}, 1.0, "ready", "video", "minimax-h3")
    assert one["trained"]["steps"] == 1000 and one["trained"]["items"] == 8 and one["trained"]["at"] > 0 and one["trained"]["rank"] == 16
    # again with the same id: the record is replaced, not doubled; the hub lists it as installed on a Spark
    runner.finish(s.home, "night", "Night Rain", "wk_rain", "character", 1200, 9, final, fake_convert)
    hub = json.loads((s.home / "loras/hub.json").read_text())
    assert len(hub) == 1 and hub[0]["trained"]["steps"] == 1200 and hub[0]["styles"] == {"character": "a video of wk_rain"}
    import loras
    router, resolve, lead = loras.attach(s.home, ROOT, True)
    listed = next(r.endpoint for r in router.routes if r.path == "/api/loras" and "GET" in r.methods)()
    mine = next(r for r in listed if r["id"] == "night")
    assert mine["installed"] and mine["usable"] and mine["trained"]["steps"] == 1200
    assert resolve([{"id": "night", "strength": 1}])[0]["path"] == str(s.home / "loras/night.safetensors")


def test_a_converter_that_keeps_kohya_names_is_rewritten_and_a_bad_file_is_refused(tmp_path):
    s = Studio()
    final = tmp_path / "night.safetensors"
    safetensors(final, ["x"])
    kohya = [f"lora_unet_blocks_{n}_attn_qkv_proj.{part}" for n in range(2) for part in ("lora_down.weight", "lora_up.weight")]
    runner.finish(s.home, "night", "Night", "wk_rain", "look", 300, 8, final, lambda src, part: safetensors(part, kohya))
    import lora_layout
    assert lora_layout.layout(list(lora_layout.header(s.home / "loras/night.safetensors")[0])) == "comfy"
    with pytest.raises(ValueError):
        runner.finish(s.home, "bad", "Bad", "wk_rain", "look", 300, 8, final, lambda src, part: safetensors(part, ["something.else"]))
    assert not (s.home / "loras/bad.safetensors").exists()
    assert [one["id"] for one in json.loads((s.home / "loras/hub.json").read_text())] == ["night"]


def test_the_agents_tools_are_listed():
    import agents
    source = (ROOT / "mcp/studio-mcp.py").read_text()
    for name in ("train_lora", "training_status"):
        assert f'"name": "{name}"' in source and f'if name == "{name}"' in source and name in agents.STUDIO_TOOLS
    assert "train_lora" in agents.MAKERS


def test_the_server_serves_the_routes_and_builds_the_kind():
    import server
    from fastapi.testclient import TestClient

    client = TestClient(server.app)
    said = client.get("/api/train").json()
    assert {"ready", "missing", "datasets", "runs", "sizes", "limits"} <= set(said) and said["sizes"]["standard"]["per_step"] == 110
    assert client.post("/api/train/start", json={"dataset": "nothing-here"}).status_code == 404
    assert client.post("/api/train/datasets", json={"name": "X", "trigger": "wk_x", "items": []}).status_code == 400
    assert "train" in server.NOT_ON_COMFY
    if not server.SPARK:
        assert "DGX Spark" in refused(server.build, "train", {"dataset": "x", "lora": "x"}, "j1")
