"""A production pack ZIP whose headers lie about sizes must not be allowed to fill the disk
(app/motion.py import_pack — the same guard suite.py unpack() already carries).
python3 -m pytest tests/test_motion_pack_zip_guard.py"""
import io
import json
import os
import struct
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("STUDIO_HOME", str(Path(tempfile.mkdtemp()) / "TensorFoldStudio"))
sys.path.insert(0, str(ROOT / "app"))
import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)


def lying_zip(lie_to=10):
    """A valid pack whose preview.jpg claims a tiny size while its stream carries 5 MB."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr("bundle_manifest.json", json.dumps({"files": {}}))
        z.writestr("reference.mp4", b"x" * 10)
        z.writestr("preview.jpg", b"y" * (5 << 20))
    data = bytearray(buf.getvalue())
    hits = 0
    for sig, off in ((b"PK\x03\x04", 22), (b"PK\x01\x02", 24)):  # local, then central: uncompressed size field
        i = -1
        while (i := data.find(sig, i + 1)) >= 0:
            name_at = i + (30 if sig == b"PK\x03\x04" else 46)
            if data[name_at:name_at + 11] == b"preview.jpg":
                struct.pack_into("<I", data, i + off, lie_to)
                hits += 1
    assert hits == 2, "the patch missed a header"
    return bytes(data)


def test_a_zip_that_lies_about_its_size_is_refused():
    """Either guard may catch it — a lie in the central directory breaks the CRC, a lie only in the
    stream trips the written-bytes ceiling (which suite.py has and motion.py lacked)."""
    r = client.post("/api/motion/packs/import", files={"file": ("pack.zip", lying_zip(), "application/zip")})
    assert r.status_code == 400, r.text[:200]
    said = r.json()["error"].lower()
    assert "declared" in said or "crc" in said, r.text[:200]
    stray = [p for p in (server.HOME / "motion").rglob("preview.jpg")]
    assert not stray, "the lie was extracted anyway"


def test_a_stream_bigger_than_the_central_directory_admits_is_refused():
    """The written-bytes ceiling proper: headers honest, one chunk's stream carries more than claimed."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr("bundle_manifest.json", json.dumps({"files": {}}))
        z.writestr("reference.mp4", b"x" * 10)
        z.writestr("preview.jpg", b"y" * (5 << 20))
    data = bytearray(buf.getvalue())
    i = -1
    while (i := data.find(b"PK\x01\x02", i + 1)) >= 0:  # central entry only: zipfile.open trusts the local header
        if data[i + 46:i + 57] == b"preview.jpg":
            struct.pack_into("<I", data, i + 24, 10)  # uncompressed size
            struct.pack_into("<I", data, i + 20, 10)  # compressed size (stored: same bytes)
    r = client.post("/api/motion/packs/import", files={"file": ("pack.zip", bytes(data), "application/zip")})
    assert r.status_code == 400, r.text[:200]
    assert not list((server.HOME / "motion").rglob("preview.jpg"))


def test_a_honest_pack_still_gets_in():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("bundle_manifest.json", json.dumps({"files": {}}))
        z.writestr("reference.mp4", b"abc")
        z.writestr("preview.jpg", b"y" * 1000)
    r = client.post("/api/motion/packs/import", files={"file": ("pack.zip", buf.getvalue(), "application/zip")})
    assert r.status_code == 200, r.text[:200]
