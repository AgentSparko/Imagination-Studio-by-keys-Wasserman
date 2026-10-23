"use strict";
// The rooms of their own in the sidebar for what lives inside a bigger page: Cast & Places, the Director and the Night
// shift (each on a film from Production, the open Slate film by default, else the newest), and Finish under Edit. Every
// panel is drawn by its own module (slate.js, director.js, nightshift.js, finish.js); this file only gives it a page
// and a film to work on, by opening that film in Slate (SL.p), exactly as Production does.
const RM = { films: null, asked: false, opening: "", kind: "cast", nsFor: "" };
async function rmFilms() { try { const got = await api("/api/slate/projects"); RM.films = Array.isArray(got) ? got : []; } catch { RM.films = []; } render(); }
const rmNewest = () => (RM.films || []).slice().sort((a, b) => String(b.updatedAt || "").localeCompare(String(a.updatedAt || "")))[0] || null;
// a film opened here is the film Production shows too: the one open there is saved first
async function rmOpen(id) { if (!id || (SL.p && SL.p.id === id) || RM.opening) return; RM.opening = id; render();
  try { if (SL.p && typeof slTimer !== "undefined") { clearTimeout(slTimer); await slPut(SL.p).catch(() => {}); }
    SL.p = await api("/api/slate/projects/" + id); SL.sceneId = SL.shotId = null; SL.have = {}; SL.plan = {}; }
  catch (e) { RM.failed = id; toast("The film did not open: " + e.message); }
  RM.opening = ""; render(); }
// the film picker on each of these pages: the open film, else the newest one
function rmPicker() { if (!RM.asked) { RM.asked = true; rmFilms(); }
  if (!SL.p && !RM.opening) { const n = rmNewest(); if (n && n.id !== RM.failed) rmOpen(n.id); }
  const films = RM.films || [], open = SL.p ? SL.p.id : RM.opening;
  if (!films.length && !SL.p) return null;
  const list = SL.p && !films.some(f => f.id === SL.p.id) ? [{ id: SL.p.id, name: SL.p.name }, ...films] : films;
  return el("div", { class: "fit", style: "min-width:220px" }, el("label", {}, "Film"),
    el("select", { "aria-label": "Film", onchange: e => rmOpen(e.target.value) }, list.map(f => el("option", { value: f.id, selected: f.id === open }, f.name || "Untitled")))); }
// no film yet: start one in Production
const rmNone = what => { const waiting = RM.opening || RM.films === null || (RM.films.length && !SL.p && !RM.failed);
  return el("section", { class: "panel" }, el("div", { class: "note m-flat" }, RM.opening ? "Opening the film…" : waiting ? "Loading…" : "No film yet. " + what),
  waiting ? null : el("div", { class: "row", style: "margin-top:10px" }, el("button", { class: "primary fit", onclick: () => rmProduction(null) }, "Start a film in Production"), el("span"))); };
const rmProduction = view => { if (typeof slateMode === "function") slateMode("slate", view); else go("Production"); };
const rmBrain = () => { if (!SL.brain) { SL.brain = {}; api("/api/slate/brain").then(b => { SL.brain = b; render(); }).catch(() => {}); } };

// Cast & Places: the bible and dialogue choices, then the characters' and the locations' sheets, as Slate's Studios tab
function viewCast() { const head = pageHead("Creative Engines", "Cast & Places", "The people and places of a film, a sheet for each with its pictures and LoRAs, kept the same in every shot.", rmPicker());
  if (!SL.p) return el("div", {}, head, rmNone("The cast and the places belong to a film."));
  rmBrain(); const p = SL.p;
  return el("div", { "data-room-view": "cast" }, head, el("section", { class: "card" },
    el("h3", { style: "margin-top:0" }, "Faces, places and voices"), slBibleChoices(),
    el("div", { class: "choice tight", style: "margin:18px 0 14px" }, ["cast", "locations"].map(k => el("button", { type: "button", class: RM.kind === k ? "on" : "", onclick: () => { RM.kind = k; render(); } },
      el("b", {}, SL_SHEETS[k].title), el("span", {}, (p[SL_SHEETS[k].key] || []).length + " sheets")))),
    slSheets(RM.kind),
    el("div", { class: "row", style: "margin-top:14px" }, el("button", { class: "small fit", onclick: () => rmProduction("studios") }, "Open the film in Production"), el("span")))); }

// the Director: a logline to a checked cut, on the picked film; a run that is live on another film says so
function viewDirector() { const head = pageHead("Creative Engines", "Director", "Give it a logline: it writes, casts, renders, checks and cuts the whole film while you watch.", rmPicker());
  if (!SL.p) return el("div", {}, head, rmNone("The Director works on a film: start one, or give it a logline here once it is made."));
  rmBrain();
  const live = typeof DR !== "undefined" ? Object.keys(DR.st).find(id => id !== SL.p.id && DR.st[id] && DR.st[id].running) : null;
  const other = live && (RM.films || []).find(f => f.id === live);
  return el("div", { "data-room-view": "director" }, head,
    other ? el("div", { class: "row", style: "margin-bottom:12px" }, el("span", { class: "note m-flat" }, "The Director is running on " + (other.name || "another film") + "."),
      el("button", { class: "small fit", onclick: () => rmOpen(live) }, "Show that run")) : null,
    el("section", { class: "card" }, directorPanel(SL.p),
      el("div", { class: "row", style: "margin-top:14px" }, el("button", { class: "small fit", onclick: () => rmProduction("deliver") }, "See the takes in Production"), el("span")))); }

// the Night shift: the shifts and their planning (nightshift.js), and the rest of the picked film rendered tonight
function viewNight() { const head = pageHead("Creative Engines", "Night shift", "Render a batch overnight and wake up to a reel, the dailies and notes on your phone.", rmPicker());
  if (SL.p && RM.nsFor !== SL.p.id && typeof NS !== "undefined") { RM.nsFor = SL.p.id; NS.film = SL.p.id; }  // the panel's own picker starts on the same film
  return el("div", { "data-room-view": "night" }, head,
    SL.p ? el("div", { class: "row", style: "margin-bottom:14px;align-items:center" }, el("span", { class: "note m-flat" }, "Every shot of " + (SL.p.name || "this film") + " with a prompt and no good take, rendered overnight:"), nightFromFilm(SL.p), el("span")) : null,
    nightPanel()); }

// Edit → Finish: the panel the Timeline carries, on a page of its own
function viewFinish() { if (!project && typeof ensureProject === "function") ensureProject().then(() => render()).catch(e => toast(e.message));
  return el("div", { "data-room-view": "finish" }, pageHead("Edit", "Finish", "The final cut and mix of the timeline, to DaVinci Resolve, Premiere Pro or After Effects."),
    project ? finishPanel(project) : el("div", { class: "note" }, "Opening the timeline…")); }
