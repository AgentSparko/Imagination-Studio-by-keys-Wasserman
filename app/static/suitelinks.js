"use strict";
// Suite links: what the filmmaker suite made goes into a film that already exists, or into the Home composer.
//   suiteAttachPanel(importId)   "Send into a film": pick a film, say which shot each import shot goes to, Attach
//   suitePromptPicker(onPick, importId?)   a sheet of recent imports and their shots; onPick({title, prompt, seconds, image})
//   suiteAppIcon(appId)          the app's icon, for a list of imports
// Both read /api/suite/imports; nothing here is needed to bring a file in.
const SLK = { panels: {} };
const SLK_ICON = { blockout: "blockout", scriptbreak: "scriptbreak", "cork-board": "cork-board", "master-canvas": "master-canvas", "master-canvas-handoff": "master-canvas",
  "storyboard-reference": "storyboard-reference", slate: "slate", "motion-previs-studio": "motion-previs-studio" };
const SLK_NAME = { blockout: "Blockout", scriptbreak: "ScriptBreak", "cork-board": "Cork Board", "master-canvas": "Master Canvas", "master-canvas-handoff": "Master Canvas", "storyboard-reference": "Storyboard Reference",
  slate: "Slate", "motion-previs-studio": "Motion Previs Studio", fountain: "A screenplay" };
const suiteAppName = id => SLK_NAME[id] || id || "A file";
function suiteAppIcon(appId) { const slug = SLK_ICON[appId];
  return slug ? el("img", { class: "su-icon", src: "art/suite/" + slug + ".webp", alt: "", width: 40, height: 40, loading: "lazy", decoding: "async" })
    : el("span", { class: "su-icon blank" }, typeof icon === "function" ? icon(appId === "fountain" ? "doc" : "suite") : ""); }
const slkPick = (value, options, onchange, label) => { const n = el("select", { "aria-label": label || "Choose", onchange: e => onchange(e.target.value) },
  options.map(([v, t]) => el("option", { value: v, selected: v === value }, t))); n.value = value; return n; };
const slkShots = film => (film.scenes || []).flatMap(sc => (sc.shots || []).map(s => ({ id: s.id, scene: sc.id, label: (sc.name || "Scene") + " · " + (s.name || "Shot") })));

// ---- Send into a film ----
function suiteAttachPanel(importId) {
  const st = SLK.panels[importId] || (SLK.panels[importId] = { imp: null, films: null, film: "", proj: null, map: {}, scene: "", busy: false, said: "", loading: false });
  const root = el("div", { class: "panel stack", "data-attach": importId });
  const draw = () => root.replaceChildren(...body());
  const loadFilm = async id => { st.film = id; st.proj = null; st.map = {}; if (!id) return draw();
    try { st.proj = await api("/api/slate/projects/" + id); } catch (e) { st.said = e.message; }
    const flat = st.proj ? slkShots(st.proj) : [];
    (st.imp.shots || []).forEach((s, i) => { st.map[s.id] = flat[i] ? flat[i].id : "new"; });
    st.scene = st.proj && st.proj.scenes && st.proj.scenes.length ? st.proj.scenes[st.proj.scenes.length - 1].id : ""; draw(); };
  const send = async () => { const mapping = (st.imp.shots || []).filter(s => st.map[s.id]).map(s => ({ shot: s.id, to: st.map[s.id], ...(st.map[s.id] === "new" && st.scene ? { scene: st.scene } : {}) }));
    if (!mapping.length) return toast("Pick where at least one shot goes");
    st.busy = true; st.said = ""; draw();
    try { const film = await api("/api/suite/imports/" + importId + "/attach", { film: st.film, mapping });
      st.proj = film; if (typeof SL !== "undefined" && SL.p && SL.p.id === film.id) SL.p = film;
      toast(count(mapping.length, "shot") + " sent into " + (film.name || "the film")); }
    catch (e) { st.said = e.message; }
    st.busy = false; draw(); };
  const body = () => {
    const head = el("h2", {}, "Send into a film");
    if (st.loading) return [head, el("div", { class: "note" }, "Looking…")];
    if (!st.imp) return [head, el("div", { class: "note" }, st.said || "Nothing to send.")];
    const films = st.films || [];
    if (!films.length) return [head, el("div", { class: "note su-flat" }, "There is no film to send into yet. Make a Slate film first, then come back.")];
    const shots = st.imp.shots || [], where = st.proj ? slkShots(st.proj) : [];
    const options = [["", "Do not send"], ["new", "A new shot"], ...where.map(w => [w.id, w.label])];
    const scenes = st.proj ? st.proj.scenes || [] : [];
    return [head, el("div", { class: "note su-flat" }, "The still becomes the shot's first frame, the previs clip its motion reference, and the words fill the prompt when it is empty."),
      field("Film", slkPick(st.film, [["", "Choose a film"], ...films.map(f => [f.id, f.name || f.id])], loadFilm, "Film")),
      st.proj ? el("div", { class: "stack" }, shots.map(s => el("div", { class: "kv", "data-shot": s.id },
        el("span", {}, s.title || s.id), slkPick(st.map[s.id] || "", options, v => { st.map[s.id] = v; draw(); }, "Where " + (s.title || s.id) + " goes")))) : null,
      st.proj && Object.values(st.map).includes("new") ? field("New shots go in", slkPick(st.scene, scenes.length ? scenes.map(sc => [sc.id, sc.name || "Scene"]) : [["", "A new scene"]], v => { st.scene = v; draw(); }, "Scene for new shots")) : null,
      st.said ? el("div", { class: "note su-flat vt-bad" }, st.said) : null,
      el("div", { class: "chips" }, el("button", { class: "primary", disabled: !st.proj || st.busy, onclick: send }, st.busy ? "Sending…" : "Attach"),
        st.proj && typeof SL !== "undefined" && typeof go === "function" ? el("button", { onclick: () => { SL.mode = "slate"; SL.p = st.proj; SL.metas = null; SL.view = "shots"; go("Production"); } }, "Open the film") : null)]; };
  if (!st.imp && !st.loading) { st.loading = true;
    Promise.all([api("/api/suite/imports/" + importId).catch(e => { st.said = e.message; return null; }), api("/api/slate/projects").catch(() => [])]).then(([imp, films]) => {
      st.imp = imp; st.films = films || []; st.loading = false; draw(); }); }
  draw(); return root; }

// ---- From the suite (the Home composer) ----
let SLK_SHEET = null;
function suitePromptPicker(onPick, importId) {
  if (typeof document === "undefined" || !document.body) return null;
  if (SLK_SHEET && SLK_SHEET.remove) SLK_SHEET.remove();
  const st = { imports: null, open: importId || "", shots: {}, said: "" };
  const shut = () => { if (SLK_SHEET && SLK_SHEET.remove) SLK_SHEET.remove(); SLK_SHEET = null; };
  const choose = p => { shut(); onPick(p); };
  const draw = () => { const list = (st.imports || []).slice(0, 12);
    sheet.replaceChildren(el("div", { class: "kv" }, el("h2", {}, "From the suite"), el("button", { class: "small ghost", onclick: shut }, "Close")),
      el("div", { class: "note su-flat" }, "Pick a shot from something you brought in. Its words fill the box, and its still becomes the first frame."),
      st.imports === null ? el("div", { class: "note" }, "Looking…") : !list.length ? el("div", { class: "note" }, st.said || "Nothing has been brought in yet. Drop a file on the Suite page first.")
        : el("div", { class: "stack" }, list.map(m => el("div", { class: "stack su-tight", "data-import": m.id },
          el("button", { class: "su-app", type: "button", onclick: () => open(m.id) }, suiteAppIcon(m.app),
            el("div", { class: "su-txt" }, el("b", {}, m.title || m.id), el("span", {}, suiteAppName(m.app) + " · " + count(m.shots || 0, "shot")))),
          st.open === m.id ? el("div", { class: "list" }, (st.shots[m.id] || []).length ? st.shots[m.id].map(p => el("button", { type: "button", "data-pick": p.title, onclick: () => choose(p) },
            el("div", {}, el("b", {}, p.title), el("span", {}, (p.prompt || "").slice(0, 120) + (p.seconds ? " · " + p.seconds + " s" : "")))))
            : el("div", { class: "note" }, st.shots[m.id] ? "No shots in this one." : "Looking…")) : null))));
    };
  const load = async id => { if (st.shots[id]) return;
    try { st.shots[id] = await api("/api/suite/imports/" + id + "/prompts"); } catch (e) { st.shots[id] = []; st.said = e.message; } draw(); };
  const open = id => { st.open = st.open === id ? "" : id; draw(); if (st.open) load(id); };
  const sheet = el("div", { class: "gate-sheet", role: "dialog", "aria-modal": "true", "aria-label": "From the suite" });
  const wrap = el("div", { class: "gate-wrap", "data-sheet": "suite", onclick: e => { if (e.target === wrap) shut(); }, onkeydown: e => { if (e.key === "Escape") shut(); } }, sheet);
  SLK_SHEET = wrap; document.body.append(wrap); draw();
  api("/api/suite/imports").then(list => { st.imports = importId ? list.filter(m => m.id === importId).concat(list.filter(m => m.id !== importId)) : list; draw(); if (importId) load(importId); },
    e => { st.imports = []; st.said = e.message; draw(); });
  return wrap; }
