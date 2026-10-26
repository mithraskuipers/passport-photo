// Shared passport photo guidance: overlay, live checks, countdown, gallery.
const Guide = (() => {
  const MIN_W = 551, MIN_H = 709;        // 400 dpi at 35 x 45 mm
  const HEAD_H = 1.6, HEAD_W = 1.07;     // calibration: face detector box -> chin to crown / ear to ear
  const $ = (id) => document.getElementById(id);

  const OVERLAY = `
    <svg id="overlay" viewBox="0 0 35 45" preserveAspectRatio="none">
      <line id="midLine" x1="17.5" y1="0" x2="17.5" y2="45"/>
      <line id="eyeLine" x1="0" y1="22" x2="35" y2="22"/>
      <ellipse id="gMax" cx="17.5" cy="22.5" rx="10" ry="15"/>
      <ellipse id="gMin" cx="17.5" cy="22.5" rx="8" ry="13"/>
      <ellipse id="est" cx="17.5" cy="22.5" rx="8" ry="13"/>
      <circle id="eye0" r="0.5" cx="-5" cy="-5"/>
      <circle id="eye1" r="0.5" cx="-5" cy="-5"/>
    </svg>
    <div id="count"></div>`;

  const INFO = `
    <details>
      <summary>Requirements (Rijksoverheid)</summary>
      <ul>
        <li>Fit the head between the two dashed ovals: chin to crown 26 to 30 mm (19 to 30 mm up to age 10), ear to ear 16 to 20 mm.</li>
        <li>Head centred, straight, eyes level, looking into the camera, neutral expression, mouth closed.</li>
        <li>Plain, even, light grey, light blue or white background without gradient or shadow.</li>
        <li>Even light on the face, no shadows, no reflections, no glare on glasses, eyes fully visible.</li>
        <li>No hat or cap unless for religious reasons. Face fully visible.</li>
        <li>Do not edit the photo. Print on glossy photo paper at 100% scale, 400 dpi or more. Photo is valid for 6 months.</li>
      </ul>
    </details>`;

  // ---- Optional subtle retouch (off by default, settings remembered per browser) ----
  const RT = { on: false, smooth: 30, slim: 20 };
  try { Object.assign(RT, JSON.parse(localStorage.getItem("retouch") || "{}")); } catch (e) { /* ignore */ }
  RT.on = !!RT.on;
  const pct = (v) => Math.max(0, Math.min(100, Math.round(Number(v) || 0)));

  function retouchQuery() {
    return RT.on ? "smooth=" + pct(RT.smooth) + "&slim=" + pct(RT.slim) : "smooth=0&slim=0";
  }
  function retouchActive() { return RT.on && (pct(RT.smooth) > 0 || pct(RT.slim) > 0); }

  function mountRetouch(el, onChange) {
    el.innerHTML = `
      <details id="retouchBox" ${RT.on ? "open" : ""}>
        <summary>Retouch (subtle, optional)</summary>
        <label class="rt-row"><input type="checkbox" id="rtOn"> Apply retouching to the photo</label>
        <label class="rt-row">Smooth skin <input type="range" id="rtSmooth" min="0" max="100" step="5"><output id="rtSmoothV"></output></label>
        <label class="rt-row">Slimmer face <input type="range" id="rtSlim" min="0" max="100" step="5"><output id="rtSlimV"></output></label>
        <p class="rt-warn">Dutch passport photos must not be edited. A retouched photo can be rejected at the counter. Use this only for non passport purposes, or keep it off.</p>
      </details>`;
    const on = $("rtOn"), sm = $("rtSmooth"), sl = $("rtSlim");
    const sync = () => {
      on.checked = RT.on;
      sm.value = pct(RT.smooth); sl.value = pct(RT.slim);
      $("rtSmoothV").textContent = pct(RT.smooth) + "%";
      $("rtSlimV").textContent = pct(RT.slim) + "%";
      sm.disabled = sl.disabled = !RT.on;
    };
    const changed = () => {
      RT.on = on.checked; RT.smooth = pct(sm.value); RT.slim = pct(sl.value);
      try { localStorage.setItem("retouch", JSON.stringify(RT)); } catch (e) { /* ignore */ }
      sync();
      if (onChange) onChange();
    };
    on.onchange = changed;
    sm.oninput = changed;
    sl.oninput = changed;
    sync();
  }

  function mount(stage) { stage.insertAdjacentHTML("beforeend", OVERLAY); }
  function mountInfo(el) { el.innerHTML = INFO; }

  // Centre crop of a video frame to 7:9 (35 x 45 mm)
  function cropRect(vw, vh) {
    let cw = vw, ch = vw * 9 / 7;
    if (ch > vh) { ch = vh; cw = vh * 7 / 9; }
    return { sx: (vw - cw) / 2, sy: (vh - ch) / 2, cw, ch };
  }

  function limits(age) {
    return { minH: age === "child" ? 19 : 26, maxH: 30, minW: 16, maxW: 20 };
  }

  function setEllipse(id, cx, cy, rx, ry) {
    const e = $(id);
    e.setAttribute("cx", cx); e.setAttribute("cy", cy);
    e.setAttribute("rx", rx); e.setAttribute("ry", ry);
  }

  function updateGuides(age) {
    const l = limits(age);
    setEllipse("gMax", 17.5, 22.5, l.maxW / 2, l.maxH / 2);
    setEllipse("gMin", 17.5, 22.5, l.minW / 2, l.minH / 2);
  }

  // j: analysis from the server, mir: preview is mirrored, age: "adult" or "child"
  function evaluate(j, mir, age) {
    const L = limits(age), c = [];
    const add = (label, state, text) => c.push({ label, state, text });

    if (!j.face) {
      add("Face", "bad", "No face found. Look straight into the camera.");
    } else {
      const f = j.face;
      add("Face", "ok", "Face found");
      const bottom = (f.y + f.h) * 45;
      const headH = f.h * 45 * HEAD_H;
      const cx = (f.x + f.w / 2) * 35 * (mir ? -1 : 1) + (mir ? 35 : 0);   // display space
      const dx = cx - 17.5, dy = bottom - headH / 2 - 22.5;
      const grade = (v, ok, warn) => v < ok ? "ok" : v < warn ? "warn" : "bad";
      const sx = grade(Math.abs(dx), 1.5, 3), sy = grade(Math.abs(dy), 2, 4);
      const sp = sx === "bad" || sy === "bad" ? "bad" : sx === "warn" || sy === "warn" ? "warn" : "ok";
      let t = "Centred";
      if (sp !== "ok") {
        const parts = [];
        if (sx !== "ok") parts.push(dx > 0 ? "left" : "right");
        if (sy !== "ok") parts.push(dy > 0 ? "up" : "down");
        t = "Move head " + parts.join(" and ") + " on screen";
      }
      add("Position", sp, t);

      const over = headH < L.minH ? L.minH - headH : headH > L.maxH ? headH - L.maxH : 0;
      add("Size", over === 0 ? "ok" : over < 2.5 ? "warn" : "bad",
          over === 0 ? "Head size fits (about " + headH.toFixed(0) + " mm)" : headH < L.minH ? "Too small, move closer" : "Too big, move back");

      if (j.eyes.length === 2) {
        add("Eyes", "ok", "Both eyes visible");
        const a = Math.abs(j.tilt);
        add("Head straight", a <= 2 ? "ok" : a <= 4 ? "warn" : "bad", a <= 2 ? "Eyes level" : "Level your head so the eyes are horizontal");
      } else {
        add("Eyes", "warn", "Both eyes must be clearly visible");
      }

      add("Sharpness", j.sharp >= 40 ? "ok" : j.sharp >= 18 ? "warn" : "bad",
          j.sharp >= 40 ? "Sharp" : "Blurry, hold still or improve light");
      const expo = j.lum < 90 ? "Too dark" : j.lum > 190 ? "Too bright" : j.clip_hi > 0.03 ? "Bright spots or reflections" : j.clip_lo > 0.05 ? "Very dark areas" : "";
      add("Exposure", expo ? "bad" : "ok", expo || "Good");
      add("Even light", j.lr < 0.12 ? "ok" : j.lr < 0.25 ? "warn" : "bad", j.lr < 0.12 ? "No shadow on one side" : "Shadow on one side of the face");
    }

    const bg = j.bg, chroma = Math.max(bg.r, bg.g, bg.b) - Math.min(bg.r, bg.g, bg.b);
    const colourOk = bg.lum >= 150 && (chroma < 30 || (bg.b >= bg.r + 10 && bg.b >= bg.g && chroma < 90));
    if (!colourOk) add("Background", "bad", "Must be light grey, light blue or white");
    else if (bg.std >= 22 || bg.lr >= 22) add("Background", "bad", "Not even: shadow, gradient or pattern");
    else if (bg.std >= 12 || bg.lr >= 12) add("Background", "warn", "Slightly uneven");
    else add("Background", "ok", "Plain and light");
    return c;
  }

  function render(j, mir, age) {
    const checks = evaluate(j, mir, age);
    const ul = $("checks");
    ul.textContent = "";
    for (const k of checks) {
      const li = document.createElement("li");
      const d = document.createElement("span"); d.className = "dot " + k.state;
      const b = document.createElement("b"); b.textContent = k.label;
      const t = document.createElement("span"); t.textContent = k.text;
      li.append(d, b, t);
      ul.appendChild(li);
    }
    const worst = checks.some((k) => k.state === "bad") ? "bad" : checks.some((k) => k.state === "warn") ? "warn" : "ok";
    $("overlay").setAttribute("class", worst);

    const est = $("est");
    if (j.face) {
      const f = j.face, mx = (x) => (mir ? 35 - x : x);
      const headH = f.h * 45 * HEAD_H, bottom = (f.y + f.h) * 45;
      setEllipse("est", mx((f.x + f.w / 2) * 35), bottom - headH / 2, f.w * 35 * HEAD_W / 2, headH / 2);
      est.style.display = "block";
      for (let i = 0; i < 2; i++) {
        const e = j.eyes[i];
        $("eye" + i).setAttribute("cx", e ? mx(e[0] * 35) : -5);
        $("eye" + i).setAttribute("cy", e ? e[1] * 45 : -5);
      }
    } else {
      est.style.display = "none";
      for (let i = 0; i < 2; i++) $("eye" + i).setAttribute("cx", -5);
    }
  }

  function countdown(n, done) {
    const el = $("count");
    el.textContent = n;
    el.style.display = "flex";
    const t = setInterval(() => {
      n--;
      if (n > 0) { el.textContent = n; return; }
      clearInterval(t);
      el.style.display = "none";
      done();
    }, 1000);
  }

  function sizeNote(w, h) {
    const low = w < MIN_W || h < MIN_H;
    return w + " x " + h + " px" + (low ? ". Too small for 400 dpi at 35 x 45 mm, try a higher camera resolution." : ". Resolution OK.");
  }

  function showSaved(el, name) {
    el.textContent = retouchActive() ? "Saved (retouched). " : "Saved. ";
    const a = document.createElement("a");
    a.href = "/api/sheet/" + encodeURIComponent(name);
    a.target = "_blank";
    a.textContent = "Open print sheet (10 x 15 cm, 8 photos)";
    el.appendChild(a);
  }

  async function loadGallery(g) {
    try {
      const names = await (await fetch("/api/photos")).json();
      g.textContent = "";
      names.forEach((n) => {
        const cell = document.createElement("div");
        const a = document.createElement("a");
        a.href = "/photos/" + encodeURIComponent(n);
        a.target = "_blank";
        const img = document.createElement("img");
        img.loading = "lazy";
        img.src = a.href;
        a.appendChild(img);
        const sh = document.createElement("a");
        sh.className = "sheet";
        sh.href = "/api/sheet/" + encodeURIComponent(n);
        sh.target = "_blank";
        sh.textContent = "print sheet";
        cell.append(a, sh);
        g.appendChild(cell);
      });
    } catch (e) { /* ignore */ }
  }

  return { mount, mountInfo, mountRetouch, retouchQuery, retouchActive, cropRect, updateGuides, evaluate, render, countdown, sizeNote, showSaved, loadGallery };
})();
