"""The take lane's own rules: a hidden file stays out of a film, and a name given as a local model server is
checked by what it resolves to. python3 tests/test_slate_take.py. No GPU, no network: the resolving is stood in
for with a monkeypatched getaddrinfo."""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp()) / "TensorFoldStudio"))
sys.path.insert(0, str(ROOT / "app"))
import pytest  # noqa: E402
import slate  # noqa: E402
import server  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)


def film(project_id: str) -> None:
    """A film with one scene and one shot, written the way the app writes them."""

    file = slate.project_file(project_id)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps({"id": project_id, "name": "one", "scenes": [{"id": "sc", "shots": [{"id": "sh1", "name": "opening"}]}]}))


def test_a_hidden_clip_of_the_studio_folder_is_refused_as_a_take():
    """A take is copied into the film's folder, which the page downloads freely: a file /files keeps hidden must
    not arrive there by the side door (an upload's path, a stray .cache, the key folder #101 keeps there)."""

    pid = "p-hidden"
    film(pid)
    home = Path(server.HOME)
    (home / "projects").mkdir(parents=True, exist_ok=True)
    (home / "projects" / "x.mp4").write_bytes(b"mp4 of a take")
    (home / ".secrets").mkdir(parents=True, exist_ok=True)
    (home / ".secrets" / "id_ed25519.mp4").write_bytes(b"not really a clip")
    ok = client.post("/api/slate/takes", json={"projectId": pid, "shot": "sh1", "path": "projects/x.mp4"})
    assert ok.status_code == 200, ok.text
    refused = client.post("/api/slate/takes", json={"projectId": pid, "shot": "sh1", "path": ".secrets/id_ed25519.mp4"})
    assert refused.status_code == 400 and "hidden" in refused.text
    assert not list((Path(server.HOME) / "slate" / pid / "takes").glob("*id_ed25519*"))  # nothing was copied


def test_a_name_for_a_local_model_server_must_resolve_nearby(monkeypatch):
    """The page types an address; the server then fetches from it. A *.local or one-word name is only a neighbour
    if DNS says so: an answer that points at the public internet (or nothing) is the 400 the docstring promises."""

    def answering(*ips):
        return lambda host, port, **kw: [(0, 0, 0, "", (ip, 0)) for ip in ips]

    monkeypatch.setattr(slate.socket, "getaddrinfo", answering("192.168.1.9"))
    assert slate.nearby_endpoint("http://studio-nas.local:8080") == "http://studio-nas.local:8080"
    monkeypatch.setattr(slate.socket, "getaddrinfo", answering("93.184.2.1"))  # a public answer to a local-sounding name
    with pytest.raises(HTTPException):
        slate.nearby_endpoint("http://evil.local:8080")
    monkeypatch.setattr(slate.socket, "getaddrinfo", answering("169.254.169.254"))  # the metadata address, named
    with pytest.raises(HTTPException):
        slate.nearby_endpoint("http://metadata.local:8080")

    def nowhere(host, port, **kw):
        raise OSError("name does not resolve")

    monkeypatch.setattr(slate.socket, "getaddrinfo", nowhere)
    with pytest.raises(HTTPException):
        slate.nearby_endpoint("http://gone.local:8080")
    assert slate.nearby_endpoint("") == ""  # nothing typed is still nothing checked
    assert slate.nearby_endpoint("studio:bolt") == "studio:bolt"  # the app's own engine is still named without DNS
