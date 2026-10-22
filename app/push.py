"""Notifications on a phone: Web Push from this studio, with no account anywhere.

The studio makes its own VAPID key pair once (EC P-256, <studio folder>/push/vapid.json, 0600) and keeps the browsers
that asked for notifications in <studio folder>/push/subscriptions.json. A notification is encrypted for each browser
(RFC 8291, aes128gcm) and handed to that browser's push service (Apple, Google or Mozilla), which wakes the phone;
the service worker (static/sw.js) shows it. pywebpush sends it when it is installed; otherwise the studio's own
implementation below, on the ``cryptography`` package, does the same. A browser whose subscription is gone
(404 or 410) is dropped.

Routes: GET /api/push/key (the public key and how many devices), POST /api/push/subscribe, DELETE /api/push/subscribe,
POST /api/push/test."""
from __future__ import annotations

import base64
import json
import os
import struct
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request

try:  # the same work, done by the maintained library when it is there
    import pywebpush
except ImportError:  # noqa: SIM105
    pywebpush = None

HOME: Path = Path(".")
SUBJECT = os.environ.get("STUDIO_PUSH_SUBJECT", "mailto:imagination-studio@example.com")  # the contact a push service may use
TTL = 24 * 3600  # a phone that is off for the night still gets the morning's message
RECORD = 4096  # the record size given in the aes128gcm header
# the push services browsers use: a subscription to any other address is refused, so the studio never posts elsewhere
SERVICES = ("push.apple.com", "fcm.googleapis.com", "android.googleapis.com", "push.services.mozilla.com", "notify.windows.com")
_lock = threading.Lock()


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def unb64(text: str) -> bytes:
    text = str(text).strip()
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def crypto():
    """The ``cryptography`` package's parts, or an error that says what is missing."""
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    except ImportError:
        raise RuntimeError("notifications need the Python package cryptography (run the studio's setup again)") from None
    return hashes, serialization, ec, decode_dss_signature, AESGCM, HKDF


def ready() -> dict:
    try:
        crypto()
    except RuntimeError as error:
        return {"ready": False, "why": str(error)}
    return {"ready": True, "why": ""}


# ---------- keys and subscriptions ----------

def folder() -> Path:
    return HOME / "push"


def raw_public(public_key) -> bytes:
    _, serialization, *_ = crypto()
    return public_key.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)


def keys() -> dict:
    """{"private": the key object, "public": its uncompressed point, base64url}; made once and kept."""

    _, serialization, ec, *_ = crypto()
    file = folder() / "vapid.json"
    with _lock:
        if file.is_file():
            kept = json.loads(file.read_text())
            private = serialization.load_pem_private_key(kept["private_pem"].encode(), password=None)
        else:
            private = ec.generate_private_key(ec.SECP256R1())
            pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
            file.parent.mkdir(parents=True, exist_ok=True)
            scratch = file.with_suffix(".tmp")
            scratch.write_text(json.dumps({"private_pem": pem, "public": b64(raw_public(private.public_key())), "made": time.time()}, indent=1))
            os.chmod(scratch, 0o600)
            scratch.replace(file)
    return {"private": private, "public": b64(raw_public(private.public_key()))}


def subscriptions() -> list[dict]:
    file = folder() / "subscriptions.json"
    try:
        return json.loads(file.read_text()) if file.is_file() else []
    except ValueError:
        return []


def keep(subs: list[dict]) -> None:
    file = folder() / "subscriptions.json"
    file.parent.mkdir(parents=True, exist_ok=True)
    scratch = file.with_suffix(".tmp")
    scratch.write_text(json.dumps(subs, indent=1))
    os.chmod(scratch, 0o600)
    scratch.replace(file)


def check_subscription(body: dict) -> dict:
    endpoint = str(body.get("endpoint") or "")
    parts = urlsplit(endpoint)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not any(host == s or host.endswith("." + s) for s in SERVICES):
        raise HTTPException(400, "this is not a browser's push address")
    given = body.get("keys") or {}
    try:
        p256dh, auth = unb64(given.get("p256dh", "")), unb64(given.get("auth", ""))
    except (ValueError, TypeError):
        raise HTTPException(400, "the subscription's keys are not readable")
    if len(p256dh) != 65 or p256dh[0] != 4 or len(auth) != 16:
        raise HTTPException(400, "the subscription's keys are not a browser's")
    return {"endpoint": endpoint, "keys": {"p256dh": b64(p256dh), "auth": b64(auth)}}


# ---------- RFC 8291 (message encryption) and RFC 8292 (VAPID) ----------

def hkdf(salt: bytes, ikm: bytes, info: bytes, length: int) -> bytes:
    hashes, *_, HKDF = crypto()
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(ikm)


def encrypt(payload: bytes, p256dh: str, auth: str, salt: bytes | None = None, sender=None) -> bytes:
    """``payload`` for one browser, as the aes128gcm body of a push message (RFC 8291 section 3 and RFC 8188):
    salt, record size, the sender's public key, then one record. ``salt`` and ``sender`` (an EC private key) are new
    for every message; a test gives them to compare with the RFC's own example."""

    _, _, ec, _, AESGCM, _ = crypto()
    receiver_bytes = unb64(p256dh)
    receiver = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), receiver_bytes)
    sender = sender or ec.generate_private_key(ec.SECP256R1())
    sender_bytes = raw_public(sender.public_key())
    salt = salt or os.urandom(16)
    shared = sender.exchange(ec.ECDH(), receiver)
    ikm = hkdf(unb64(auth), shared, b"WebPush: info\x00" + receiver_bytes + sender_bytes, 32)
    key = hkdf(salt, ikm, b"Content-Encoding: aes128gcm\x00", 16)
    nonce = hkdf(salt, ikm, b"Content-Encoding: nonce\x00", 12)
    record = AESGCM(key).encrypt(nonce, payload + b"\x02", None)  # \x02: the last (and only) record, no padding
    return salt + struct.pack("!IB", RECORD, len(sender_bytes)) + sender_bytes + record


def decrypt(body: bytes, receiver, auth: str) -> bytes:
    """What a browser does with a message (``receiver`` its EC private key): for the tests, and to check encrypt."""

    _, _, ec, _, AESGCM, _ = crypto()
    salt, _, length = body[:16], struct.unpack("!I", body[16:20])[0], body[20]
    sender_bytes, record = body[21:21 + length], body[21 + length:]
    sender = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), sender_bytes)
    receiver_bytes = raw_public(receiver.public_key())
    ikm = hkdf(unb64(auth), receiver.exchange(ec.ECDH(), sender), b"WebPush: info\x00" + receiver_bytes + sender_bytes, 32)
    plain = AESGCM(hkdf(salt, ikm, b"Content-Encoding: aes128gcm\x00", 16)).decrypt(hkdf(salt, ikm, b"Content-Encoding: nonce\x00", 12), record, None)
    return plain.rstrip(b"\x00")[:-1]  # the delimiter, and any padding after it


def vapid_header(endpoint: str, private, public: str, now: float | None = None) -> str:
    """The Authorization header of RFC 8292: a JWT signed with ES256 for the push service's origin, and the key."""

    hashes, _, ec, decode_dss_signature, *_ = crypto()
    parts = urlsplit(endpoint)
    claims = {"aud": f"{parts.scheme}://{parts.netloc}", "exp": int((now or time.time()) + 12 * 3600), "sub": SUBJECT}
    signing = b64(json.dumps({"typ": "JWT", "alg": "ES256"}, separators=(",", ":")).encode()) + "." + \
        b64(json.dumps(claims, separators=(",", ":")).encode())
    r, s = decode_dss_signature(private.sign(signing.encode(), ec.ECDSA(hashes.SHA256())))
    token = signing + "." + b64(r.to_bytes(32, "big") + s.to_bytes(32, "big"))
    return f"vapid t={token}, k={public}"


def post(url: str, body: bytes, headers: dict) -> int:
    """One message to a push service; its HTTP status. Replaced in the tests."""
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=20) as answer:
            return answer.status
    except urllib.error.HTTPError as error:
        return error.code


def deliver(sub: dict, payload: bytes, signing: dict) -> int:
    if pywebpush is not None:
        try:
            from py_vapid import Vapid01
            answer = pywebpush.webpush(sub, payload.decode(), vapid_private_key=Vapid01(private_key=signing["private"]),
                                       vapid_claims={"sub": SUBJECT}, ttl=TTL, content_encoding="aes128gcm")
            return answer.status_code
        except pywebpush.WebPushException as error:
            return error.response.status_code if error.response is not None else 0
    body = encrypt(payload, sub["keys"]["p256dh"], sub["keys"]["auth"])
    return post(sub["endpoint"], body, {"Authorization": vapid_header(sub["endpoint"], signing["private"], signing["public"]),
                                        "Content-Encoding": "aes128gcm", "Content-Type": "application/octet-stream",
                                        "TTL": str(TTL), "Urgency": "normal"})


def send(title: str, body: str, url: str = "/") -> dict:
    """The message to every subscribed browser: {"sent", "dropped", "failed"}. Gone subscriptions are dropped."""

    subs = subscriptions()
    if not subs:
        return {"sent": 0, "dropped": 0, "failed": 0}
    signing = keys()
    payload = json.dumps({"title": title, "body": body, "url": url}).encode()
    sent, gone, failed = 0, [], 0
    for sub in subs:
        try:
            status = deliver(sub, payload, signing)
        except Exception as error:  # noqa: BLE001 - one phone's service never stops the others
            print(f"[push] {urlsplit(sub['endpoint']).hostname}: {type(error).__name__}: {str(error)[:200]}", flush=True)
            failed += 1
            continue
        if status in (404, 410):
            gone.append(sub["endpoint"])
        elif 200 <= status < 300:
            sent += 1
        else:
            failed += 1
    if gone:
        with _lock:
            keep([s for s in subscriptions() if s["endpoint"] not in gone])
    return {"sent": sent, "dropped": len(gone), "failed": failed}


# ---------- the routes ----------

def attach(home: Path, local=None) -> APIRouter:
    """``local(request)`` refuses a request that is not from this computer or its Tailscale address (agents.local)."""

    global HOME
    HOME = Path(home)
    router = APIRouter()
    guard = local or (lambda request: None)

    @router.get("/api/push/key")
    def public_key() -> dict:
        state = ready()
        return {**state, "key": keys()["public"] if state["ready"] else None, "devices": len(subscriptions())}

    @router.post("/api/push/subscribe")
    def subscribe(body: dict, request: Request) -> dict:
        guard(request)
        if not ready()["ready"]:
            raise HTTPException(503, ready()["why"])
        sub = check_subscription(body)
        with _lock:
            subs = [s for s in subscriptions() if s["endpoint"] != sub["endpoint"]]
            keep(subs + [{**sub, "added": time.time()}])
        return {"ok": True, "devices": len(subs) + 1}

    @router.delete("/api/push/subscribe")
    def unsubscribe(body: dict, request: Request) -> dict:
        guard(request)
        endpoint = str((body or {}).get("endpoint") or "")
        with _lock:
            subs = subscriptions()
            left = [s for s in subs if s["endpoint"] != endpoint]
            keep(left)
        return {"ok": True, "removed": len(subs) - len(left), "devices": len(left)}

    @router.post("/api/push/test")
    def test(request: Request) -> dict:
        guard(request)
        if not ready()["ready"]:
            raise HTTPException(503, ready()["why"])
        if not subscriptions():
            raise HTTPException(400, "no device has turned notifications on yet")
        return send("Imagination Studio", "Notifications work on this device.", "/#Settings")

    return router
