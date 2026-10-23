"use strict";
/* Work on a clip you already have (app/revise.py): Restyle, Extend and Fix, three Library actions on a clip. They open
   in the Library's action panel (actions.js: actOpen, ACT.open) and queue the studio's own jobs. Restyle and Fix are
   three steps: the first frame redrawn (in a new look, or from an instruction), the clip's motion read by Motion
   Previs, then the clip made again from its depth layer and the redrawn frame. Extend makes a clip opening on its last
   frame (joined to it by default). */
const REV = { chains: {}, timer: null,
  restyle: { prompt: "", image_model: "qwen", seconds: "", keep_sound: true, seed: 1 },
  extend: { seconds: "5", prompt: "", join: true, seed: 1 },
  fix: { instruction: "", image_model: "qwen", keep_sound: true, seed: 1 } };
const REV_KINDS = ["restyle", "extend", "fix"];
const REV_TITLES = { restyle: "Restyle", extend: "Extend", fix: "Fix" };
const REV_MODELS = ["qwen", "qwen-full", "flux3k-fast", "flux3k-full"];
const REV_STEPS = ["the first frame", "the motion", "the clip"];
const REV_STAGES = { prepare: "preparing the clip", pose: "following the body", camera: "solving the camera move", depth: "making the depth layer", encode: "writing the layers", pack: "packing" };
// the buttons a clip's Library tile gets
function revButtons(m) { if (m.kind !== "video") return [];
  const one = (kind, title) => el("button", { class: "small", title, onclick: () => actOpen(kind, m.path) }, REV_TITLES[kind]);
  return [one("restyle", "Make this clip again in a new look: the first frame redrawn, the motion and camera from the clip"),
    one("extend", "Carry this clip on from its last frame, and join the two"),
    one("fix", "Change one thing in this clip from a sentence: the first frame is redrawn, then the clip is made again with its motion and camera")]; }
const revModels = () => { const have = typeof S !== "undefined" && S && S.image_models && S.image_models.length ? S.image_models.filter(m => REV_MODELS.includes(m.id)) : [];
  return have.length ? Object.fromEntries(have.map(m => [m.id, m.label])) : { qwen: "Qwen 2.1 Turbo" }; };
async function revQueue(kind, params) { try { const r = await api("/api/revise", { kind, params }); ACT.jobs[kind] = r.id; REV.chains[kind] = r.revise;
  toast("Queued: " + r.title); for (const n of (r.revise && r.revise.notes) || []) toast(n, 6000); revFollow(); poll(); render(); } catch (e) { toast(e.message, 6000); } }
// a chain's later steps are queued by the studio when the one before finishes: followed here until it is over
function revFollow() { if (REV.timer) return;
  REV.timer = setInterval(async () => { const open = Object.entries(REV.chains).filter(([, c]) => c && c.status === "running");
    if (!open.length) { clearInterval(REV.timer); REV.timer = null; return; }
    let changed = false;
    for (const [kind, c] of open) { try { const fresh = await api("/api/revise/" + c.id); if (JSON.stringify(fresh) !== JSON.stringify(c)) { REV.chains[kind] = fresh; changed = true; } } catch { REV.chains[kind] = null; changed = true; } }
    if (changed && tab === "Library" && !(typeof busyHere === "function" && busyHere())) render(); }, 3000); }
// the motion analysis is a Motion Previs job, not one of the studio's queue: a card of its own
const revAnalysis = s => el("div", { class: "job" }, el("div", { class: "seghead" }, el("b", {}, s.title || "The motion"), el("span", { class: "pill " + s.status }, s.status)),
  el("div", { class: "note" }, s.cached ? "Read before: this clip's depth layer is used again." : s.status === "done" ? "The depth layer and the camera move are ready."
    : s.status === "failed" ? "The analysis failed" + (s.error ? ": " + s.error : ".") : s.stage ? "Motion Previs is " + (REV_STAGES[s.stage] || s.stage) + "." : "Motion Previs reads the clip's depth and camera move, on the processor."));
// each step's job card and, when the last is done, what it made beside the clip as it was
function revShow(kind, path, made) { const c = REV.chains[kind]; if (!c) return actShow(kind, made);
  const named = i => c.of === 3 && kind !== "extend" ? ": " + REV_STEPS[i] : "";
  const cards = c.steps.map((s, i) => { const job = s.kind === "analysis" ? null : jobs.find(j => j.id === s.id); if (!job && s.kind !== "analysis") return null;
    return el("div", { style: "margin-top:12px" }, c.of > 1 ? el("label", {}, "Step " + (i + 1) + " of " + c.of + named(i)) : null, job ? jobCard(job) : revAnalysis(s)); });
  const waiting = c.of - c.steps.length, last = jobs.find(j => j.id === c.steps[c.steps.length - 1].id);
  return el("div", {}, cards, waiting > 0 && c.status === "running" ? el("div", { class: "note" }, "Step " + (c.steps.length + 1) + " of " + c.of + named(c.steps.length) + " is queued when the one before finishes.") : null,
    c.error ? el("div", { class: "note", style: "color:var(--warn)" }, c.error) : null,
    c.status === "done" && last && last.status === "done" ? made(last) : null); }
const revBeside = (job, label) => el("div", { class: "row", style: "align-items:flex-start;margin-top:8px" },
  el("div", {}, el("label", {}, label), el("video", { class: "main", src: file(job.outputs[0]), controls: true })),
  el("div", {}, el("label", {}, "The clip as it was"), el("video", { class: "main", src: file(ACT.open ? ACT.open.path : job.outputs[0]), controls: true })));
const revSound = f => el("label", { style: "display:flex;gap:6px;align-items:center;flex:none", title: "The clip’s own sound goes with its depth layer as a reference. Untick it to have sound made for the new picture." },
  bind(f, "keep_sound", el("input", { type: "checkbox", style: "width:auto" })), "Keep the clip’s sound");
function revPanel(o) { const f = REV[o.kind], redraw = () => render();
  const box = el("div", { class: "seg", id: "actpanel", style: "margin-top:12px" },
    el("div", { class: "seghead" }, el("b", {}, REV_TITLES[o.kind]), el("span", { class: "note", style: "margin:0" }, o.path.split("/").pop()), el("span", { style: "flex:1" }),
      el("button", { class: "small", onclick: () => { ACT.open = null; render(); } }, "Close")));
  const preview = el("div", { style: "flex:0 0 220px" }, media(o.path, { style: "width:220px;border-radius:var(--r2)" }));
  if (o.kind === "restyle") {
    put(box, el("div", { class: "row", style: "align-items:flex-start" }, preview,
      el("div", {}, el("div", { class: "note", style: "margin-top:0" }, "Makes this clip again in a new look, in three steps. 1: the clip’s first frame is redrawn in that look by the picture model. 2: Motion Previs reads the clip’s motion as a depth layer. 3: the clip is made again from the depth layer and the redrawn frame: the motion, camera and timing come from the clip’s depth, the look and the setting from the new frame. The whole frame is redrawn, so small details can change. At most 15 seconds: a longer clip is cut to its first 15. Needs MiniMax H3 Ref2VA, Motion Previs and a picture model."),
        field("The new look", bind(f, "prompt", el("input", { placeholder: "a hand-painted watercolour, soft paper texture · 1970s 16 mm film, warm grain · claymation" }))),
        el("div", { class: "row" }, field("Picture model", select(f, "image_model", revModels())),
          field("Seconds (optional)", bind(f, "seconds", el("input", { type: "number", min: 2, max: 15, placeholder: "whole clip", style: "max-width:110px" }))),
          field("Seed", bind(f, "seed", el("input", { type: "number", min: 0, style: "max-width:110px" }))), revSound(f)),
        el("button", { class: "hot", onclick: () => { if (!String(f.prompt).trim()) return toast("Say what the clip should look like now");
          revQueue("restyle", { video: o.path, prompt: f.prompt, image_model: f.image_model, keep_sound: !!f.keep_sound, seed: Number(f.seed) || 0, ...(f.seconds ? { seconds: Number(f.seconds) } : {}) }); } }, "Restyle this clip"))),
      revShow("restyle", o.path, job => revBeside(job, "Restyled")));
  } else if (o.kind === "extend") {
    put(box, el("div", { class: "row", style: "align-items:flex-start" }, preview,
      el("div", {}, el("div", { class: "note", style: "margin-top:0" }, "Carries the clip on: a new clip opens on its last frame and goes on from there, then the two are joined into one file (the repeated frame at the join is dropped). Over 15 seconds the new part is a long take, a chain of clips."),
        field("How much longer", choice(f, "seconds", { 5: ["5 s", ""], 8: ["8 s", ""], 10: ["10 s", ""], 15: ["15 s", "one clip at most"], 20: ["20 s", "a long take"], 30: ["30 s", "a long take"] }, redraw, true)),
        field("What happens next (optional)", bind(f, "prompt", el("input", { placeholder: "empty: the clip’s own prompt, or “continue the shot”" }))),
        el("div", { class: "row" }, field("Seed", bind(f, "seed", el("input", { type: "number", min: 0, style: "max-width:110px" }))),
          el("label", { style: "display:flex;gap:6px;align-items:center;flex:none", title: "Untick to keep only the new part" }, bind(f, "join", el("input", { type: "checkbox", style: "width:auto" })), "Join to the clip")),
        el("button", { class: "hot", onclick: () => revQueue("extend", { video: o.path, seconds: Number(f.seconds), seed: Number(f.seed) || 0, join: !!f.join, ...(String(f.prompt).trim() ? { prompt: f.prompt } : {}) }) }, "Extend this clip"))),
      revShow("extend", o.path, job => revBeside(job, job.outputs.length > 1 ? "Extended, joined" : "The new part")));
  } else {
    put(box, el("div", { class: "row", style: "align-items:flex-start" }, preview,
      el("div", {}, el("div", { class: "note", style: "margin-top:0" }, "Changes one thing in the clip from a sentence, in three steps. 1: the clip’s first frame is redrawn by the picture model from your words. 2: Motion Previs reads the clip’s motion as a depth layer. 3: the clip is made again from the depth layer and the redrawn frame, so the motion and camera are the clip’s and the look the new frame’s. There is no mask: the whole frame is redrawn from the instruction, so other small details can change too. At most 15 seconds; needs MiniMax H3 Ref2VA, Motion Previs and a picture model."),
        field("What to change", bind(f, "instruction", el("input", { placeholder: "change the sky to a stormy sunset · make her coat red · put snow on the ground" }))),
        el("div", { class: "row" }, field("Picture model", select(f, "image_model", revModels())),
          field("Seed", bind(f, "seed", el("input", { type: "number", min: 0, style: "max-width:110px" }))), revSound(f)),
        el("button", { class: "hot", onclick: () => { if (!String(f.instruction).trim()) return toast("Say what to change");
          revQueue("fix", { video: o.path, instruction: f.instruction, image_model: f.image_model, keep_sound: !!f.keep_sound, seed: Number(f.seed) || 0 }); } }, "Fix this clip"))),
      revShow("fix", o.path, job => revBeside(job, "Fixed")));
  }
  return box; }
