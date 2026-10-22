"use strict";
/* A rough cut on the beat (app/roughcut.py, app/beats.py): a film's best takes, or clips ticked in the Library, cut
   to a song so every cut falls on a beat or a bar, exported with the song under it, and handed to DaVinci Resolve as
   a zip of the FCPXML and its media. rcPanel(film) is Slate's Deliver panel; rcLibrary() is the Library's
   "Trailer from these". */
const RC = { song: "", cut: "bar", order: "given", foley: false, songs: null, beats: {}, busy: false, made: {}, state: {}, trailer: null, timers: {} };
const RC_CUTS = { beat: ["Every beat", "a fast montage"], bar: ["Every bar", "one clip a bar"], "two-bars": ["Every two bars", "longer, calmer shots"], natural: ["Natural", "each clip near its own length, cut on a beat"] };
const RC_ORDERS = { given: "As filmed", shuffle: "Shuffled", energy: "Calm to busy" };
const RC_SOUNDS = { music: "Songs", voice: "Voice lines", sfx: "Sound effects" };
const RC_STAGES = { foley: "Making the foley", export: "Exporting the cut", song: "Laying the song under it", files: "Writing the files for Resolve", done: "Ready", failed: "Stopped" };
// the sounds a cut can go to: every song, voice line and sound of the Library, newest first
async function rcLoadSongs() { try { RC.songs = (await api("/api/library")).filter(m => m.kind === "audio"); } catch { RC.songs = []; } render(); }
async function rcLoadBeats(path) { if (!path || RC.beats[path]) return; RC.beats[path] = { pending: true };
  try { RC.beats[path] = await api("/api/music/beats", { path }); } catch (e) { RC.beats[path] = { error: e.message }; } render(); }
function rcBeatNote() { const b = RC.beats[RC.song]; if (!RC.song || !b) return null;
  if (b.pending) return el("div", { class: "note" }, "Finding the beats…");
  if (b.error) return el("div", { class: "note", style: "color:var(--warn)" }, b.error);
  const steady = b.confidence >= 0.6 ? "a steady beat" : b.confidence >= 0.35 ? "a fair beat" : "a loose beat: the cuts may drift";
  return el("div", { class: "note" }, Math.round(b.bpm) + " BPM · " + b.bars + " bars · " + mmss(b.duration) + " · " + steady + "."); }
// a cut in progress: its state from the server every two seconds until it is ready or stopped
function rcWatch(key) { const made = RC.made[key]; if (!made || RC.timers[key]) return;
  RC.timers[key] = setInterval(async () => { let s; try { s = await api("/api/roughcut/" + made.project); } catch { return; }
    const was = (RC.state[key] || {}).stage; RC.state[key] = s;
    if (["done", "failed"].includes(s.stage)) { clearInterval(RC.timers[key]); delete RC.timers[key]; }
    if (was !== s.stage) render(); }, 2000); }
async function rcMake(key, body) { if (!RC.song) return toast("Pick a song first");
  RC.busy = true; render();
  try { const r = await api("/api/roughcut", { ...body, song: RC.song, cut: RC.cut, order: RC.order, foley: RC.foley });
    RC.made[key] = { ...r, cut: RC.cut }; RC.state[key] = { stage: r.foley.length ? "foley" : "export", project: r.project, job: r.job, foley: {} };
    toast("Queued: " + r.title + " · " + r.timeline.length + " cuts at " + Math.round(r.beats.bpm) + " BPM"); rcWatch(key); await poll(); }
  catch (e) { toast(e.message, 6000); }
  RC.busy = false; render(); }
// what the cut made: its jobs while they run, then the film and its files
function rcResult(key) { const made = RC.made[key], s = RC.state[key]; if (!made) return null;
  const ids = [...(made.foley || []), ...(s && s.job ? [s.job] : made.job ? [made.job] : [])], cards = ids.map(id => jobs.find(j => j.id === id)).filter(Boolean);
  const files = s && s.files, stage = (s && s.stage) || "export";
  return el("div", { style: "margin-top:12px" },
    el("div", { class: "strip" }, made.timeline.map((c, i) => el("div", { style: "flex:" + Math.max(0.2, c.out - c.in), title: c.path + " · " + (c.out - c.in).toFixed(2) + " s" + (c.short ? " (whole clip)" : "") }, String(i + 1)))),
    el("div", { class: "note" }, made.timeline.length + " cuts on " + RC_CUTS[made.cut][0].toLowerCase() + " at " + Math.round(made.beats.bpm) + " BPM" + (made.dropped.length ? " · " + made.dropped.length + " clips left out: the song ended" : "") + " · " + RC_STAGES[stage] + (stage === "failed" && s.error ? ": " + s.error : "")),
    stage === "done" ? null : cards.map(jobCard),
    (s && s.notes || []).map(n => el("div", { class: "note", style: "color:var(--warn)" }, n)),
    stage === "done" && s.video ? el("div", {}, el("video", { src: file(s.video), controls: true, preload: "metadata", style: "width:100%;max-width:560px;border-radius:8px;margin-top:8px" }),
      el("div", { class: "row", style: "margin-top:8px" },
        el("a", { class: "fit tag2", href: file(s.video), target: "_blank", rel: "noopener" }, "Play"),
        files ? el("a", { class: "fit tag2", href: file(files.zip), download: "", title: "The cut and every clip and the song, for DaVinci Resolve: unzip it, then File > Import > Timeline" }, "Open in Resolve (zip)") : null,
        files ? el("a", { class: "fit tag2", href: file(files.fcpxml), download: "", title: "A Final Cut XML of the cut, naming the clips where they are on this machine" }, "FCPXML") : null,
        files ? el("a", { class: "fit tag2", href: file(files.edl), download: "", title: "A CMX 3600 cut list" }, "Cut list · EDL") : null, el("span"))) : null); }
// the settings both panels share: the song, where the cuts fall, the order, foley
function rcControls() { if (RC.songs === null) { RC.songs = []; rcLoadSongs(); }
  const pick = el("select", { "aria-label": "The song", onchange: () => { RC.song = pick.value; rcLoadBeats(RC.song); render(); } },
    el("option", { value: "", selected: !RC.song }, RC.songs.length ? "Pick a song…" : "No songs yet: make one in Music"),
    Object.entries(RC_SOUNDS).map(([k, label]) => { const list = RC.songs.filter(m => (m.sound || "sfx") === k);
      return list.length ? el("optgroup", { label }, list.map(m => el("option", { value: m.path, selected: RC.song === m.path }, m.title || m.path.split("/").pop()))) : null; }));
  if (RC.song) rcLoadBeats(RC.song);
  return [field("Song", pick), rcBeatNote(),
    field("Cut on", choice(RC, "cut", RC_CUTS, render, true)),
    el("div", { class: "row", style: "align-items:center;margin-top:10px" },
      el("div", { class: "seg-ctl fit", role: "radiogroup", "aria-label": "Order of the clips" }, Object.entries(RC_ORDERS).map(([k, t]) => el("button", { type: "button", role: "radio", "aria-checked": RC.order === k ? "true" : "false", class: RC.order === k ? "on" : "", onclick: () => { RC.order = k; render(); } }, t))),
      el("button", { type: "button", class: "toggle fit" + (RC.foley ? " on" : ""), "aria-pressed": RC.foley ? "true" : "false", title: "Each clip of 15 s or less gets its sound made for its picture (MMAudio) before the cut. Slower.", onclick: () => { RC.foley = !RC.foley; render(); } },
        el("span", { class: "dotl" + (RC.foley ? "" : " o") }), "Foley for each clip"), el("span"))]; }
// Slate's Deliver tab: the film's best take of each shot (circled, else good, else the newest), in scene and shot order
function rcPanel(film) { if (!film) return null; const key = "film:" + film.id;
  const shots = (film.scenes || []).reduce((n, sc) => n + (sc.shots || []).filter(s => (s.takes || []).some(t => t.mediaPath && t.rating !== "no-good")).length, 0);
  return el("div", { class: "seg" }, el("div", { class: "seghead" }, el("b", {}, "A rough cut on the beat"), el("span", { class: "note", style: "margin:0" }, shots + " shots with a take"), el("span", { style: "flex:1" })),
    el("div", { class: "note", style: "margin-top:0" }, "The best take of every shot, cut to a song so each cut lands on a beat or a bar. The song goes under the cut, lowered while someone speaks, and the cut opens in DaVinci Resolve with its clips and the song."),
    rcControls(),
    el("div", { class: "row", style: "margin-top:12px" }, el("button", { class: "primary fit", disabled: RC.busy || !RC.song || !shots, onclick: () => rcMake(key, { film: film.id, title: "Rough cut of " + (film.name || "the film") }) }, RC.busy ? "Cutting…" : "Make a rough cut"), el("span")),
    rcResult(key)); }
// the Library: the ticked clips cut to a song as a trailer
function rcLibrary() { const key = "library", clips = () => [...LIBPICK].filter(p => /\.(mp4|mov)$/i.test(p));
  const open = el("button", { class: "fit", title: "Cut the ticked clips to a song, on the beat", onclick: () => { const ticked = clips(); if (!ticked.length) return toast("Tick the clips first"); RC.trailer = ticked; render(); } }, "Trailer from these");
  if (!RC.trailer) return el("div", { class: "row", style: "margin-top:10px" }, open, el("span"));
  return el("div", { class: "seg", style: "margin-top:12px" }, el("div", { class: "seghead" }, el("b", {}, "Trailer from these"), el("span", { class: "note", style: "margin:0" }, RC.trailer.length + " clips"), el("span", { style: "flex:1" }),
      el("button", { class: "small", onclick: () => { RC.trailer = null; render(); } }, "Close")),
    el("div", { class: "note", style: "margin-top:0" }, "The ticked clips cut to a song, every cut on the beat, with the song under them. Shuffled or calm to busy makes a different trailer from the same clips."),
    rcControls(),
    el("div", { class: "row", style: "margin-top:12px" }, el("button", { class: "primary fit", disabled: RC.busy || !RC.song, onclick: () => rcMake(key, { clips: RC.trailer, title: "Trailer" }) }, RC.busy ? "Cutting…" : "Make the trailer"), el("span")),
    rcResult(key)); }
