// Builds HomeFM_architecture_and_design.pptx from project/deck.json and project/slides/*.html.
// Each slide is laid out in headless Chromium at 1920x1080, then every box, text block, arrow and
// icon is written as a native, editable PowerPoint object at the same position. Fonts are mapped to
// Arial and Courier New so the browser layout and PowerPoint agree on line breaks.
//
//   npm install pptxgenjs playwright   (Chromium must be available to Playwright)
//   node results/2026-09-29/homefm_slides/build_pptx.js [out.pptx]
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { chromium } = require("playwright");

const ROOT = __dirname;
const PROJECT = path.join(ROOT, "project");
const OUT = process.argv[2] || path.join(ROOT, "HomeFM_architecture_and_design.pptx");
const PX = 144; // 1920 px = 13.333 in
const inch = (px) => px / PX;
const pt = (px) => px * 0.5;

const FONT_MAP = [
  [/'Space Grotesk', Arial, sans-serif/g, "Arial"],
  [/'IBM Plex Sans', Arial, sans-serif/g, "Arial"],
  [/'JetBrains Mono', 'Courier New', monospace/g, "'Courier New'"],
];

// Stroke icons in the style of the artifact's icon set (24x24 viewBox, currentColor).
const ICONS = {
  Check: '<path d="M20 6 9 17l-5-5"/>',
  Warning: '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
  Lightbulb: '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>',
  Search: '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
  Chart: '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
};

const BASE_CSS = `
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:1920px;height:1080px;overflow:hidden;background:#fff}
section{width:1920px;height:1080px;overflow:hidden;font-family:Arial}
table{border-collapse:collapse}
th{font-weight:700}
x-shape,x-icon{display:block}
x-icon svg{width:100%;height:100%;display:block}
aside{display:none}
`;

function pageHtml(section) {
  for (const [re, to] of FONT_MAP) section = section.replace(re, to);
  section = section.replace(/<x-icon name="(\w+)"([^>]*)>/g, (m, name, rest) => {
    const svg = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS[name] || ICONS.Check}</svg>`;
    return `<x-icon name="${name}"${rest}>${svg}`;
  });
  return `<!doctype html><html><head><meta charset="utf-8"><style>${BASE_CSS}</style></head><body>${section}</body></html>`;
}

// Runs in the page: returns the slide as a flat list of positioned items.
function extract() {
  const INLINE = new Set(["B", "I", "STRONG", "EM", "SPAN", "BR", "CODE", "SUB", "SUP"]);
  const items = [];
  let iconId = 0;
  const section = document.querySelector("section");
  const rgb = (c) => {
    const m = c.match(/rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?\)/);
    if (!m || (m[4] !== undefined && +m[4] === 0)) return null;
    return [m[1], m[2], m[3]].map((v) => (+v).toString(16).padStart(2, "0")).join("").toUpperCase();
  };
  const box = (r) => ({ x: r.left, y: r.top, w: r.width, h: r.height });

  function runsOf(node, out) {
    for (const ch of node.childNodes) {
      if (ch.nodeType === 3) {
        const cs = getComputedStyle(ch.parentElement);
        let t = ch.textContent.replace(/\s+/g, " ");
        if (cs.textTransform === "uppercase") t = t.toUpperCase();
        if (t) out.push({
          text: t, color: rgb(cs.color), size: parseFloat(cs.fontSize),
          bold: parseInt(cs.fontWeight, 10) >= 600, italic: cs.fontStyle === "italic",
          font: /courier|mono/i.test(cs.fontFamily) ? "Courier New" : "Arial",
          spacing: parseFloat(cs.letterSpacing) || 0,
        });
      } else if (ch.nodeType === 1 && ch.tagName === "BR") {
        out.push({ br: true });
      } else if (ch.nodeType === 1) {
        runsOf(ch, out);
      }
    }
    return out;
  }

  function walk(el) {
    const cs = getComputedStyle(el);
    if (cs.display === "none") return;
    const r = el.getBoundingClientRect();
    if (el.tagName === "X-SHAPE") {
      items.push({ type: "arrow", kind: el.getAttribute("kind"), ...box(r), fill: rgb(cs.backgroundColor) });
      return;
    }
    if (el.tagName === "X-ICON") {
      el.setAttribute("data-icon", String(iconId));
      items.push({ type: "icon", id: iconId++, ...box(r) });
      return;
    }
    if (el !== section) {
      const fill = rgb(cs.backgroundColor);
      const bw = parseFloat(cs.borderTopWidth) || 0;
      const line = bw && cs.borderTopStyle !== "none" ? rgb(cs.borderTopColor) : null;
      if (fill || line) {
        items.push({ type: "box", ...box(r), fill, line, bw: line ? bw : 0,
          radius: parseFloat(cs.borderTopLeftRadius) || 0 });
      }
    }
    const kids = [...el.childNodes];
    const hasText = kids.some((n) => n.nodeType === 3 && n.textContent.trim()) ||
      kids.some((n) => n.nodeType === 1 && INLINE.has(n.tagName) && n.textContent.trim());
    if (hasText && el !== section) {
      const range = document.createRange();
      range.selectNodeContents(el);
      const tr = range.getBoundingClientRect();
      const pl = parseFloat(cs.paddingLeft) + parseFloat(cs.borderLeftWidth);
      const pr = parseFloat(cs.paddingRight) + parseFloat(cs.borderRightWidth);
      const lh = cs.lineHeight === "normal" ? parseFloat(cs.fontSize) * 1.15 : parseFloat(cs.lineHeight);
      items.push({
        type: "text", x: r.left + pl, w: r.width - pl - pr, y: tr.top, h: tr.height,
        align: cs.textAlign === "center" ? "center" : cs.textAlign === "right" ? "right" : "left",
        lineHeight: lh, runs: runsOf(el, []),
      });
      return;
    }
    for (const ch of el.children) walk(ch);
  }

  walk(section);
  return { bg: rgb(getComputedStyle(section).backgroundColor) || "FFFFFF", items };
}

// Trim collapsed whitespace at paragraph edges and turn runs into pptxgenjs text objects.
function textObjects(runs) {
  const lines = [[]];
  for (const r of runs) (r.br ? lines.push([]) : lines[lines.length - 1].push({ ...r }));
  const out = [];
  lines.forEach((line, li) => {
    if (line.length) {
      line[0].text = line[0].text.replace(/^ +/, "");
      line[line.length - 1].text = line[line.length - 1].text.replace(/ +$/, "");
    }
    const kept = line.filter((r) => r.text);
    if (!kept.length) kept.push({ ...(runs.find((r) => !r.br) || {}), text: "" });
    kept.forEach((r, i) => {
      const o = { fontFace: r.font, fontSize: pt(r.size), color: r.color || "000000", bold: r.bold, italic: r.italic };
      if (r.spacing) o.charSpacing = pt(r.spacing);
      if (i === kept.length - 1 && li < lines.length - 1) o.breakLine = true;
      out.push({ text: r.text, options: o });
    });
  });
  return out;
}

function addItem(pres, slide, it, iconPngs) {
  if (it.type === "box") {
    const half = it.bw / 2;
    const w = it.w - it.bw, h = it.h - it.bw;
    const r = Math.min(it.radius, it.w / 2, it.h / 2);
    const opts = {
      x: inch(it.x + half), y: inch(it.y + half), w: inch(w), h: inch(h),
      fill: it.fill ? { color: it.fill } : { type: "none" },
      line: it.line ? { color: it.line, width: pt(it.bw) } : { type: "none" },
    };
    if (r > 0) slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { ...opts, rectRadius: inch(Math.max(r - half, 0)) });
    else slide.addShape(pres.shapes.RECTANGLE, opts);
  } else if (it.type === "arrow") {
    const shape = it.kind === "arrow-down" ? pres.shapes.DOWN_ARROW : pres.shapes.RIGHT_ARROW;
    slide.addShape(shape, { x: inch(it.x), y: inch(it.y), w: inch(it.w), h: inch(it.h),
      fill: { color: it.fill || "9AA3B2" }, line: { type: "none" } });
  } else if (it.type === "icon") {
    slide.addImage({ data: "image/png;base64," + iconPngs[it.id], x: inch(it.x), y: inch(it.y), w: inch(it.w), h: inch(it.h) });
  } else if (it.type === "text") {
    // A little slack so PowerPoint's wrapping never breaks a line the browser kept whole.
    const slack = 8 + it.w * 0.005;
    let x = it.x;
    if (it.align === "center") x -= slack / 2;
    else if (it.align === "right") x -= slack;
    slide.addText(textObjects(it.runs), {
      x: inch(x), y: inch(it.y), w: inch(it.w + slack), h: inch(Math.max(it.h, it.lineHeight)),
      margin: 0, valign: "top", align: it.align, lineSpacing: pt(it.lineHeight),
      fit: "none", wrap: true, isTextBox: true,
    });
  }
}

async function main() {
  const deck = JSON.parse(fs.readFileSync(path.join(PROJECT, "deck.json"), "utf8"));
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
  pres.title = deck.title;

  const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  for (const id of deck.order) {
    const src = fs.readFileSync(path.join(PROJECT, "slides", `${id}.html`), "utf8");
    const notes = (src.match(/<aside>([\s\S]*?)<\/aside>/) || [])[1];
    await page.setContent(pageHtml(src));
    const { bg, items } = await page.evaluate(extract);
    const iconPngs = {};
    for (const it of items.filter((i) => i.type === "icon")) {
      const buf = await page.locator(`[data-icon="${it.id}"]`).screenshot({ omitBackground: true, scale: "device" });
      iconPngs[it.id] = buf.toString("base64");
    }
    const slide = pres.addSlide();
    slide.background = { color: bg };
    for (const it of items) addItem(pres, slide, it, iconPngs);
    if (notes) slide.addNotes(notes.replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").trim());
  }
  await browser.close();
  await pres.writeFile({ fileName: OUT });
  console.log(`Wrote ${OUT} (${deck.order.length} slides)`);
}

main().catch((e) => { console.error(e); process.exit(1); });
