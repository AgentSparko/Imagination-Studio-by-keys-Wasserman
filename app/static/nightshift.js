"use strict";
// The night shift (app/nightshift.py): a batch of clips rendered overnight, and dailies to wake up to: a reel, the
// Circle Take package and notes. nightPanel() is the panel on Home; nightFromFilm(film) is Slate's "Render the rest
// tonight"; nightNotify() is Settings → Notifications (app/push.py, static/sw.js).
const NS = { list: null, films: null, film: "", start: "idle", at: "23:00", busy: false, timer: null, notes: {}, open: null };
const NS_STARTS = { idle: "When idle", at: "At 23:00", now: "Now" };
const NS_STATES = { planned: "planned", running: "running", morning: "making the dailies", done: "done", cancelled: "cancelled" };
const nsLive = () => (NS.list || []).some(p => ["running", "planned", "morning"].includes(p.state));
async function nsLoad() { try { const got = await api("/api/nightshift"); NS.list = Array.isArray(got) ? got : NS.list || []; } catch { NS.list = NS.list || []; }
  if (NS.films === null) { try { const got = await api("/api/slate/projects"); NS.films = Array.isArray(got) ? got : []; } catch { NS.films = []; } }
  render(); }
function nsWatch() { if (NS.timer) return; NS.timer = setInterval(async () => { if (tab !== "Home" || !nsLive()) { clearInterval(NS.timer); NS.timer = null; return; }
  const before = JSON.stringify(NS.list); try { const got = await api("/api/nightshift"); if (Array.isArray(got)) NS.list = got; } catch {} if (JSON.stringify(NS.list) !== before) render(); }, 10000);
  if (NS.timer && NS.timer.unref) NS.timer.unref(); }
// the start a plan is given: "idle", "now", or the time set beside "At"
const nsStart = () => NS.start === "at" ? (NS.at || "23:00") : NS.start;
const nsWhen = p => p.start === "idle" ? "when the queue is idle" : p.start === "now" ? "now" : /^\d\d:\d\d$/.test(p.start) ? "at " + p.start : "at " + String(p.start).replace("T", " ").slice(0, 16);
const nsCount = (n, word) => n + " " + word + (n === 1 ? "" : "s");
function nsStartPick() { const at = NS_STARTS.at.replace("23:00", NS.at || "23:00");
  return el("div", { class: "row", style: "align-items:center;gap:10px;flex-wrap:wrap" },
    el("div", { class: "seg-ctl", role: "radiogroup", "aria-label": "When the night shift starts" }, Object.entries({ ...NS_STARTS, at }).map(([k, label]) =>
      el("button", { type: "button", role: "radio", "aria-checked": NS.start === k ? "true" : "false", class: NS.start === k ? "on" : "", onclick: () => { NS.start = k; render(); } }, label))),
    NS.start === "at" ? el("input", { type: "time", value: NS.at, style: "width:auto", "aria-label": "Start at", onchange: e => { NS.at = e.target.value || "23:00"; render(); } }) : null); }
async function nsPlan(body, quiet) { NS.busy = true; render();
  try { const p = await api("/api/nightshift", { start: nsStart(), ...body });
    if (!quiet) toast("Night shift planned: " + nsCount(p.count, "clip") + ", " + (p.state === "running" ? "started now" : "starts " + nsWhen(p)));
    await nsLoad(); nsWatch(); NS.busy = false; render(); return p; }
  catch (e) { toast(e.message, 7000); NS.busy = false; render(); return null; } }
async function nsAct(p, what) { try { await api("/api/nightshift/" + p.id + "/" + what, {}); if (typeof poll === "function") await poll(); await nsLoad(); nsWatch(); } catch (e) { toast(e.message); } }
async function nsForget(p) { try { await api("/api/nightshift/" + p.id, {}, "DELETE"); await nsLoad(); } catch (e) { toast(e.message); } }
async function nsNotes(p) { if (NS.open === p.id) { NS.open = null; return render(); } NS.open = p.id;
  if (!NS.notes[p.id]) { try { NS.notes[p.id] = await (await fetch(file(p.notes))).text(); } catch { NS.notes[p.id] = "The notes could not be read."; } }
  render(); }
function nsCurrent(p) { const id = p.progress && p.progress.current, job = id && (typeof jobs !== "undefined" ? jobs : []).find(j => j.id === id);
  return job && typeof jobCard === "function" ? jobCard(job) : id ? el("div", { class: "note" }, "Waiting for job " + id) : null; }
function nsShift(p) { const prog = p.progress || { done: 0, total: p.count }, share = prog.total ? Math.round(100 * prog.done / prog.total) : 0;
  const looks = (p.results || []).filter(r => r.status !== "done").length;
  return el("div", { class: "job", "data-shift": p.id },
    el("div", { class: "seghead" }, el("b", {}, p.name), el("span", { class: "pill " + (p.state === "planned" ? "queued" : p.state === "morning" ? "running" : p.state) }, NS_STATES[p.state] || p.state), el("span", { style: "flex:1" }),
      p.state === "planned" ? el("button", { class: "small primary", onclick: () => nsAct(p, "start") }, "Start now") : null,
      ["planned", "running"].includes(p.state) ? el("button", { class: "small", onclick: () => confirm("Cancel the night shift “" + p.name + "”?") && nsAct(p, "cancel") }, "Cancel") : null,
      ["done", "cancelled"].includes(p.state) || p.state === "planned" ? el("button", { class: "small ghost", title: "Forget this night shift (its dailies stay in the dailies folder)", onclick: () => nsForget(p) }, "Forget") : null),
    el("div", { class: "note" }, nsCount(p.count, "clip") + (p.state === "planned" ? " · starts " + nsWhen(p) : p.state === "running" ? " · " + prog.done + " of " + prog.total + " finished" : p.state === "morning" ? " · all finished: making the reel, the dailies and the notes" : p.state === "done" ? " · " + (p.message || "done") : " · cancelled")),
    p.state === "running" ? [el("div", { class: "bar" }, el("i", { style: "width:" + Math.max(3, share) + "%" })), nsCurrent(p)] : null,
    p.state === "done" ? el("div", { class: "stack" },
      p.reel ? media(p.reel, { controls: true, preload: "metadata", style: "width:100%;border-radius:var(--r2)" }) : null,
      el("div", { class: "m-acts" },
        p.zip ? el("a", { class: "tag2", href: file(p.zip), download: "" }, "Download the dailies") : null,
        p.notes ? el("button", { class: "small", onclick: () => nsNotes(p) }, NS.open === p.id ? "Hide the notes" : "Read the notes") : null,
        looks ? el("span", { class: "tag2 warn" }, nsCount(looks, "clip") + " not made") : null),
      NS.open === p.id ? el("pre", { style: "white-space:pre-wrap;max-height:360px;overflow:auto" }, NS.notes[p.id] || "Reading…") : null,
      (p.problems || []).map(x => el("div", { class: "note", style: "color:var(--warn)" }, x))) : null); }
// Home: the shifts, newest first, and a film's remaining shots planned for tonight
function nightPanel() { if (NS.list === null) nsLoad(); else if (nsLive()) nsWatch();
  const list = NS.list || [], shown = list.filter(p => p.state !== "cancelled").slice(0, 4), films = NS.films || [];
  if (!NS.film && films.length) NS.film = films[0].id;
  return el("section", { class: "panel ns-panel", "data-panel": "nightshift" },
    el("div", { class: "sectionhead" }, el("h2", {}, "Night shift"), el("span", { class: "note m-flat" }, "Render a batch overnight; wake up to a reel, the dailies and notes on your phone.")),
    films.length ? el("div", { class: "stack" },
      el("div", { class: "row", style: "align-items:flex-end;gap:10px;flex-wrap:wrap" },
        field("Film", el("select", { "aria-label": "Film", onchange: e => { NS.film = e.target.value; } }, films.map(f => el("option", { value: f.id, selected: f.id === NS.film }, f.name + " · " + nsCount(f.shotCount || 0, "shot"))))),
        nsStartPick(),
        el("button", { class: "primary fit", disabled: NS.busy || !NS.film, onclick: () => nsPlan({ film: NS.film }) }, "Plan the shots without a good take"))) :
      el("div", { class: "note" }, "Plan a night shift from a film in Production (Slate): every shot without a good take is rendered overnight."),
    shown.length ? el("div", { class: "stack" }, shown.map(nsShift)) : el("div", { class: "note" }, "No night shift planned."),
    el("div", { class: "note" }, "When it is done, a notification says how many clips are ready and how many to look at. Turn notifications on in ",
      el("a", { href: "#Settings", onclick: e => { e.preventDefault(); goSettings("notify"); } }, "Settings → Notifications"), ".")); }
// Slate → Deliver: every shot of this film without a good take, planned for tonight
function nightFromFilm(film) {
  return el("button", { class: "fit", disabled: NS.busy, title: "Every shot with a prompt and no good take is rendered overnight, then a reel, the dailies and notes are made",
    onclick: async () => { if (typeof slTimer !== "undefined" && typeof SL !== "undefined" && SL.p && SL.p.id === film.id) { clearTimeout(slTimer); try { await api("/api/slate/projects/" + film.id, SL.p, "PUT"); } catch (e) { return toast(e.message); } }
      const p = await nsPlan({ film: film.id, start: NS.start === "now" ? "idle" : nsStart() }); if (p && typeof go === "function" && confirm("Planned " + nsCount(p.count, "shot") + " for tonight. Open the night shift on Home?")) go("Home"); } },
    "Render the rest tonight"); }

/* ---------- Settings → Notifications ---------- */
const NT = { state: null, sub: null, busy: false, said: "" };
const ntSupported = () => typeof navigator !== "undefined" && "serviceWorker" in navigator && typeof window !== "undefined" && "PushManager" in window && typeof Notification !== "undefined";
const ntKey = text => { const pad = "=".repeat((4 - text.length % 4) % 4), raw = atob((text + pad).replace(/-/g, "+").replace(/_/g, "/")); return Uint8Array.from(raw, c => c.charCodeAt(0)); };
async function ntLoad() { try { NT.state = await api("/api/push/key"); } catch (e) { NT.state = { ready: false, why: e.message, devices: 0 }; }
  if (ntSupported()) { try { const reg = await navigator.serviceWorker.getRegistration(); NT.sub = reg ? await reg.pushManager.getSubscription() : null; } catch { NT.sub = null; } }
  render(); }
async function ntOn() { if (!ntSupported()) return toast("This browser cannot show notifications from the studio");
  NT.busy = true; render();
  try { const reg = await navigator.serviceWorker.register("sw.js"); await navigator.serviceWorker.ready;
    if (await Notification.requestPermission() !== "granted") throw new Error("Notifications are blocked for the studio in this browser's settings");
    const key = (NT.state && NT.state.key) || (await api("/api/push/key")).key;
    const sub = await reg.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: ntKey(key) });
    await api("/api/push/subscribe", sub.toJSON()); NT.sub = sub; NT.said = "Notifications are on for this device"; toast(NT.said); }
  catch (e) { NT.said = e.message; toast(e.message, 7000); }
  NT.busy = false; await ntLoad(); }
async function ntOff() { NT.busy = true; render();
  try { if (NT.sub) { const endpoint = NT.sub.endpoint; await NT.sub.unsubscribe(); await api("/api/push/subscribe", { endpoint }, "DELETE"); } NT.sub = null; NT.said = "Notifications are off for this device"; }
  catch (e) { NT.said = e.message; }
  NT.busy = false; await ntLoad(); }
async function ntTest() { try { const r = await api("/api/push/test", {}); NT.said = r.sent ? "Sent to " + nsCount(r.sent, "device") + (r.dropped ? ", " + nsCount(r.dropped, "device") + " no longer subscribed" : "") : "Nothing was delivered" + (r.dropped ? ": " + nsCount(r.dropped, "device") + " no longer subscribed" : "");
  } catch (e) { NT.said = e.message; } render(); }
function nightNotify() { if (NT.state === null) { NT.state = { loading: true }; ntLoad(); }
  const s = NT.state || {}, here = !!NT.sub, can = ntSupported();
  const status = s.loading ? "Looking…" : s.ready === false ? s.why : (here ? "On for this device" : "Off for this device") + " · " + nsCount(s.devices || 0, "device") + " get notifications from this studio";
  return [setPanel("Notifications", "A notification on your phone when a night shift is done: how many clips are ready, and how many to look at.",
    setRow("Turn on notifications on this device", can ? "This browser asks once whether the studio may show notifications." : "This browser cannot show notifications from the studio.",
      here ? el("button", { class: "small", disabled: NT.busy, onclick: ntOff }, "Turn off") : el("button", { class: "small primary", disabled: NT.busy || !can || s.ready === false, onclick: ntOn }, "Turn on")),
    setRow("Test", "Sends a notification to every device that turned them on.", el("button", { class: "small", disabled: !s.devices, onclick: ntTest }, "Send a test")),
    el("p", { class: "note", "data-status": "notify" }, status + (NT.said ? " · " + NT.said : "")),
    el("p", { class: "note" }, "On an iPhone or iPad: open the studio in Safari at its HTTPS address (see Using it on your phone), tap Share, then Add to Home Screen. Open the studio from that icon and turn notifications on there; Safari in a tab cannot show them."),
    el("p", { class: "note" }, "Notifications go through your phone's own push service (Apple's or Google's). The studio makes its own keys; there is no account to sign in to."))]; }
