// Builds DomusFM_results.pptx: npm install pptxgenjs && node build_pptx.js ../DomusFM_results.pptx
const pptxgen = require("pptxgenjs");
const out = process.argv[2];

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "DomusFM Reproduction: Results";

// Palette and type
const NAVY = "14213D", BLUE = "2F6FB0", BLUE_D = "1F5A96", ORANGE = "C8691E", RUST = "A4520F";
const BODY = "3F4A5A", MUTED = "5F6B7A", LINE = "D5D9E0";
const BG = "FFFFFF", BG2 = "EEF1F5", CARD_ON_BG = "F3F5F8", CARD_ON_BG2 = "FFFFFF", TINT = "E3ECF7";
const HEAD = "Cambria", TEXT = "Calibri";
const nbsp = t => t.replace(/(\d) %/g, "$1\u00A0%");
const W = 13.333, M = 0.7, CW = W - 2 * M;

function header(s, eyebrow, title, eyeColor = BLUE) {
  s.addText(eyebrow.toUpperCase(), { x: M, y: 0.5, w: CW, h: 0.3, margin: 0, fontFace: TEXT, fontSize: 12, bold: true, color: eyeColor, charSpacing: 2, isTextBox: true });
  s.addText(title, { x: M, y: 0.82, w: CW, h: 0.75, margin: 0, fontFace: HEAD, fontSize: 36, bold: true, color: NAVY, isTextBox: true });
}
function footer(s, n, extra, color = MUTED) {
  s.addText(`DomusFM reproduction · ${n}` + (extra ? ` · ${extra}` : ""), { x: M, y: 6.95, w: CW, h: 0.3, margin: 0, fontFace: TEXT, fontSize: 11, color, isTextBox: true });
}
function card(s, x, y, w, h, fill, line = LINE) {
  const o = { x, y, w, h, fill: { color: fill }, rectRadius: 0.1 };
  if (line) o.line = { color: line, width: 0.75 }; else o.line = { color: fill, width: 0 };
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, o);
}
function numCircle(s, x, y, n, fill = BLUE, color = "FFFFFF", d = 0.42) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0 } });
  s.addText(String(n), { x, y, w: d, h: d, margin: 0, align: "center", valign: "middle", fontFace: TEXT, fontSize: 14, bold: true, color, isTextBox: true });
}
const nb = nbsp;
function txt(s, text, o) {
  if (typeof text === "string") text = nb(text); else text = text.map(r => Object.assign({}, r, { text: nb(r.text) }));
  s.addText(text, Object.assign({ margin: 0, fontFace: TEXT, fontSize: 16, color: BODY, valign: "top", isTextBox: true }, o));
}
// Table helpers
const HDR = (t, align = "left") => ({ text: nb(t), options: { bold: true, fill: { color: "DDE3EC" }, color: NAVY, align } });
const C = (t, o = {}) => ({ text: nb(t), options: o });
function table(s, rows, o) {
  s.addTable(rows, Object.assign({ x: M, fontFace: TEXT, fontSize: 13, color: NAVY, border: { type: "solid", pt: 0.75, color: "C9CFD8" }, margin: [0.04, 0.08, 0.04, 0.08], valign: "middle" }, o));
}

// 1 · Cover
{
  const s = pres.addSlide(); s.background = { color: NAVY };
  txt(s, "EXPERIMENT REPORT · SEPTEMBER 2026", { x: M, y: 2.2, w: CW, h: 0.35, fontSize: 13, bold: true, color: "8FB4E0", charSpacing: 2 });
  txt(s, "DomusFM reproduction", { x: M, y: 2.65, w: CW, h: 1.1, fontFace: HEAD, fontSize: 54, bold: true, color: "F7F6F2" });
  txt(s, "Setup, results and limitations of our DomusFM pretraining experiment, and how HomeFM addresses DomusFM's limitations", { x: M, y: 3.85, w: 10, h: 0.9, fontSize: 22, color: "C9D6E8" });
  footer(s, 1, "77-home pretraining · 7 held-out homes · one RTX 4090", "9FB0C8");
  s.addNotes("Independent reproduction of DomusFM (arXiv 2602.01910), pretrained on 77 homes and tested on 7 homes it never saw. Run finished 25 September 2026.");
}

// 2 · Takeaway
{
  const s = pres.addSlide(); s.background = { color: BLUE };
  txt(s, "THE SHORT VERSION", { x: M, y: 0.9, w: CW, h: 0.3, fontSize: 13, bold: true, color: "E3EDF8", charSpacing: 2 });
  txt(s, "26 of 28", { x: M, y: 1.3, w: CW, h: 1.6, fontFace: HEAD, fontSize: 110, bold: true, color: "FFFFFF", valign: "middle" });
  txt(s, "settings where pretrained DomusFM beats the same model trained from scratch, on 7 homes it never saw during pretraining.", { x: M, y: 3.05, w: 10.5, h: 1.0, fontSize: 22, color: "FFFFFF" });
  const stats = [["+0.076", "Activity F1 gain, 5 % labels"], ["+0.053", "Next-30 F1 gain, 5 % labels"], ["0.70–0.82", "Training loss held, no collapse"]];
  stats.forEach(([v, l], i) => {
    const x = M + i * 3.6;
    txt(s, v, { x, y: 4.45, w: 3.4, h: 0.75, fontFace: HEAD, fontSize: 38, bold: true, color: "FFFFFF", valign: "middle" });
    txt(s, l, { x, y: 5.25, w: 3.4, h: 0.35, fontSize: 13, color: "E3EDF8" });
  });
  footer(s, 2, "", "E3EDF8");
  s.addNotes("7 held-out homes, 2 tasks, 2 label fractions = 28 comparisons. Pretrained wins 26; the two losses are about 0.01 each, within noise. Gains are means over the 7 homes.");
}

// 3 · What DomusFM is
{
  const s = pres.addSlide(); s.background = { color: BG };
  header(s, "Introduction", "What DomusFM is");
  txt(s, [
    { text: "DomusFM is a foundation model for smart-home sensor events, by Fiori, Civitarese, Salim and Bettini (arXiv 2602.01910).", options: { breakLine: true, paraSpaceAfter: 14 } },
    { text: "It reads streams of binary ON/OFF sensor events, learns general patterns from unlabelled homes, then adapts to a new home using only a few labels.", options: { breakLine: true, paraSpaceAfter: 14 } },
    { text: "The authors' code is not public, so this is an independent implementation built from the paper's description." },
  ], { x: M, y: 2.0, w: 6.9, h: 3.6, fontSize: 18, lineSpacingMultiple: 1.15 });
  card(s, 8.1, 2.0, 4.53, 2.9, CARD_ON_BG);
  txt(s, "Why it matters here", { x: 8.45, y: 2.3, w: 3.9, h: 0.4, fontSize: 18, bold: true, color: NAVY });
  txt(s, "DomusFM is the main baseline for HomeFM. Before comparing the two, the reproduction must be sound, which means its pretraining has to actually help.", { x: 8.45, y: 2.8, w: 3.9, h: 1.9, fontSize: 15, lineSpacingMultiple: 1.15 });
  footer(s, 3);
}

// 4 · Model
{
  const s = pres.addSlide(); s.background = { color: BG };
  header(s, "Introduction · the model", "How the model reads a home");
  const steps = [
    ["One sensor event", "Item, type and room as text, embedded by a frozen MiniLM. Status ON or OFF. Time: day, hour, second."],
    ["Attribute attention", "One transformer layer fuses the 5 attributes into a single event vector."],
    ["Context encoder", "A 12-layer transformer, 12 heads, width 384, reads a window of 30 events. No positional encoding."],
    ["Task heads", "ADL: the activity at the last event. Next-30: which events come next, and how many."],
  ];
  const cw = 2.65, gap = 0.47;
  steps.forEach(([t, d], i) => {
    const x = M + i * (cw + gap);
    card(s, x, 2.0, cw, 3.0, CARD_ON_BG);
    numCircle(s, x + 0.25, 2.25, i + 1);
    txt(s, t, { x: x + 0.25, y: 2.8, w: cw - 0.5, h: 0.65, fontSize: 16, bold: true, color: NAVY });
    txt(s, d, { x: x + 0.25, y: 3.5, w: cw - 0.5, h: 1.4, fontSize: 13, lineSpacingMultiple: 1.1 });
    if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + cw + 0.08, y: 3.35, w: 0.31, h: 0.3, fill: { color: "9AA1AD" }, line: { color: "9AA1AD", width: 0 } });
  });
  txt(s, "28.6M parameters in total. The paper reports 36.1M; the gap is unexplained, since layer widths such as the feed-forward size (2048 here) are not reported.", { x: M, y: 5.35, w: CW, h: 0.7, fontSize: 15 });
  footer(s, 4);
}

// 5 · Tasks
{
  const s = pres.addSlide(); s.background = { color: BG };
  header(s, "Introduction · training and testing", "Pretraining, then two tests");
  const cards = [
    ["Pretraining: a matching game", "Two masked copies of the same 30-event window must find each other among 128 candidates (InfoNCE loss). Phase 1 hides attributes; phase 2 hides whole events.", TINT],
    ["Test 1 · Activity (ADL)", "Name the activity at the last event of a window, such as cooking or sleeping. Scored with weighted F1.", CARD_ON_BG],
    ["Test 2 · Next-30", "Predict which event types occur in the next 30 events, and how often. Scored with multiset F1.", CARD_ON_BG],
  ];
  const cw = (CW - 0.6) / 3;
  cards.forEach(([t, d, f], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 2.0, cw, 2.75, f, null);
    txt(s, t, { x: x + 0.3, y: 2.25, w: cw - 0.6, h: 0.7, fontSize: 17, bold: true, color: NAVY });
    txt(s, d, { x: x + 0.3, y: 3.0, w: cw - 0.6, h: 1.65, fontSize: 14, lineSpacingMultiple: 1.12 });
  });
  card(s, M, 5.1, CW, 1.1, "F3F5F8", LINE);
  txt(s, "Every score is compared with the same model trained from random weights (\"no pretraining\"). Pretraining is only worth having if it beats that. F1 runs from 0 to 1; higher is better.", { x: M + 0.3, y: 5.25, w: CW - 0.6, h: 0.85, fontSize: 15, color: NAVY, valign: "middle" });
  footer(s, 5);
}

// 6 · Data and protocol
{
  const s = pres.addSlide(); s.background = { color: BG2 };
  header(s, "Experimental setup", "Data and protocol");
  [["77", "homes used for pretraining"], ["27.3M", "sensor events, unlabelled"], ["7", "held-out homes for testing"]].forEach(([v, l], i) => {
    const y = 2.0 + i * 1.5;
    txt(s, v, { x: M, y, w: 3.4, h: 0.85, fontFace: HEAD, fontSize: 48, bold: true, color: BLUE, valign: "middle" });
    txt(s, l, { x: M, y: y + 0.85, w: 3.4, h: 0.35, fontSize: 13 });
  });
  const items = [
    "Pretrain once on 77 homes: every labelled CASAS home except the 7 test homes, plus Milan and Aruba.",
    "No data from a test home is seen during pretraining.",
    "Test homes: UCI Home B and CASAS hh101, hh103, hh105, hh110, hh119, hh122.",
    "Fine-tune on 5 % or 30 % of a test home's labels, for 10 epochs.",
    "3 time-contiguous folds per home; results are mean ± std.",
    "Up to 20,000 test windows per fold.",
  ];
  card(s, 4.6, 2.0, 8.03, 4.45, CARD_ON_BG2);
  txt(s, items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 10 } })), { x: 4.9, y: 2.25, w: 7.45, h: 4.0, fontSize: 16 });
  footer(s, 6);
  s.addNotes("No data from any of the 7 test homes is used in pretraining, which is stricter than the paper's leave-one-dataset-out setup.");
}

// 7 · Parameters
{
  const s = pres.addSlide(); s.background = { color: BG2 };
  header(s, "Experimental setup", "Experimental parameters");
  const ours = { color: BLUE_D, bold: true };
  const rows = [
    [HDR("Parameter"), HDR("Setting"), HDR("Source")],
    ["Model", "28.6M params · 12 layers · 12 heads · d 384", "Paper (widths guessed)"],
    ["Input window", "30 events, sliding by 1", "Paper"],
    ["Pretraining", "20,000 + 20,000 steps, batch 128, AdamW 1e-4", "Our choice (not reported)"],
    ["Batch make-up", "4 homes × 32 windows", C("Our addition", ours)],
    ["Masking", "40 % of both copies", C("Our choice (not reported)", ours)],
    ["Temperature", "0.2", C("Our choice (not reported)", ours)],
    ["Clock jitter", "±15 minutes per copy", C("Our addition", ours)],
    ["Projection head", "MLP, dropped after pretraining", C("Our addition", ours)],
    ["Extra loss", "Predict hidden attributes (weight 1.0)", C("Our addition", ours)],
    ["Fine-tuning", "10 epochs, batch 64; backbone LR 5e-5, head 1e-3, 10 % warm-up", "Paper (epochs) + our choice"],
  ].map(r => r.map(c => typeof c === "string" ? C(c) : c));
  table(s, rows, { y: 1.85, w: CW, colW: [2.6, 6.3, 3.03], fontSize: 13, fill: { color: "FFFFFF" }, rowH: 0.4 });
  footer(s, 7, "Config: configs/domusfm_corpus_fixed.yaml · Bold blue = our change to the paper's pretraining game");
  s.addNotes("The paper does not report masking rate, temperature, pretraining length or learning rates, so those are our choices. The additions are explained on slide 9.");
}

// 8 · Paper vs this work
{
  const s = pres.addSlide(); s.background = { color: BG2 };
  header(s, "Paper vs this work", "How this differs from the paper");
  const rows = [
    [HDR("Aspect"), HDR("Paper"), HDR("This reproduction")],
    ["Code", "Authors' implementation (not public)", "Independent, from the paper text"],
    ["Model size", "36.1M parameters", "28.6M (unreported widths guessed)"],
    ["Test datasets", "Milan, Aruba, UCI B, Kasteren A/C, MuRAL, Orange4Home", "UCI B + 6 CASAS homes (others unavailable)"],
    ["Pretraining data", "Leave-one-out: the other 6 datasets", "Once, on 77 other homes (27.3M events)"],
    ["CASAS data", "Original: one id per sensor, labelled", "2025 release: sensors merged per room"],
    ["CV folds", "5-fold cross-validation", "3 time-contiguous folds (no leakage)"],
    ["Label amounts", "5, 10, 15 and 30 %", "5 and 30 %"],
    ["Tasks", "ADL, Next-10, Next-30, clustering", "ADL and Next-30"],
    ["Pretraining game", "Masking, temperature, steps not reported", "Our choices, plus a harder game (slide 9)"],
    ["Comparisons", "DeepCASAS, GPT-2, Chronos, ablations", "No-pretraining ablation only"],
  ].map(r => r.map(c => typeof c === "string" ? C(c) : c));
  table(s, rows, { y: 1.85, w: CW, colW: [2.4, 4.77, 4.76], fontSize: 13, fill: { color: "FFFFFF" }, rowH: 0.4 });
  footer(s, 8, "Only UCI Home B is directly comparable with the paper (slide 15)");
  s.addNotes("In the 2025 CASAS release, Milan and Aruba have no activity labels and their sensors are merged into 9 to 10 room names, so they are used for pretraining only. Contiguous folds matter: consecutive windows share 29 of 30 events, so random folds put near-copies of test windows into training. On UCI B, random folds raised no-pretraining ADL from 0.28 to 0.40.");
}

// 9 · Harder matching game
{
  const s = pres.addSlide(); s.background = { color: BG2 };
  header(s, "Pretraining design", "A harder matching game");
  const rows = [
    [HDR("Change"), HDR("What it does"), HDR("Config setting")],
    ["1 · Same-home candidates", "Each batch holds 4 homes × 32 windows, so sensor names give nothing away", "homes_per_batch: 4"],
    ["2 · Clock jitter", "Shifts each copy's clock by up to ±15 min; gaps and time of day are kept", "time_jitter_s: 900"],
    ["3 · Mask both copies", "Hides 40 % of both copies, with a softer temperature", "mask_both_views, 0.4, τ 0.2"],
    ["4 · Projection head", "A small MLP plays the game and is dropped after pretraining", "projector: true"],
    ["5 · Fill in the blanks", "Predicts the hidden item, type, room and ON/OFF status", "mlm_weight: 1.0"],
    ["6 · Gentler fine-tuning", "Backbone LR 5e-5, head 1e-3, 10 % warm-up; also used for the baseline", "backbone_lr, head_lr"],
  ].map(r => r.map(c => typeof c === "string" ? C(c) : c));
  table(s, rows, { y: 1.85, w: CW, colW: [3.0, 6.13, 2.8], fontSize: 14, fill: { color: "FFFFFF" }, rowH: 0.5 });
  txt(s, "Each change stops the model from matching copies by surface clues, so it has to learn how the home behaves. With all six off, the code runs the paper's recipe.", { x: M, y: 5.95, w: CW, h: 0.7, fontSize: 15 });
  footer(s, 9);
  s.addNotes("Change 6 is applied to the from-scratch baseline too, so the comparison stays fair. Why each change is needed is written up in the HomeFM limitations document (docs/LIMITATIONS_AND_REMEDIES.md, L11).");
}

// 10 · Loss chart
{
  const s = pres.addSlide(); s.background = { color: BG };
  header(s, "Training health · pretraining loss, log scale", "Pretraining stayed healthy");
  const contrastive = [4.1113,0.8586,0.7761,0.7827,0.7914,0.7355,0.7355,0.7125,0.7307,0.7249,0.7228,0.7095,0.7097,0.7151,0.7366,0.7107,0.7049,0.7344,0.7206,0.6977,
    1.6662,0.8486,0.8038,0.7873,0.7756,0.8029,0.8182,0.829,0.7854,0.7798,0.784,0.8209,0.7622,0.7762,0.8177,0.8002,0.7874,0.797,0.7912,0.8174];
  const mlm = [3.6014,0.3935,0.3073,0.2135,0.1953,0.181,0.1888,0.2212,0.222,0.2409,0.2,0.2109,0.191,0.1984,0.1771,0.2269,0.1582,0.2431,0.1932,0.1622,
    2.2802,0.6213,0.6008,0.6942,0.5138,0.609,0.5292,0.6549,0.5745,0.6589,0.5073,0.5368,0.6403,0.5132,0.5432,0.6147,0.6703,0.5041,0.5806,0.5765];
  const labels = contrastive.map((_, i) => i < 20 ? `P1 ${i}k` : `P2 ${i - 20}k`);
  s.addChart(pres.charts.LINE, [
    { name: "Matching (contrastive) loss", labels, values: contrastive },
    { name: "Fill-in-the-blanks loss", labels, values: mlm },
  ], {
    x: M, y: 1.75, w: 9.0, h: 4.9, chartColors: [BLUE, ORANGE], lineSize: 2.5, lineDataSymbol: "none",
    valAxisLogScaleBase: 10, valAxisMinVal: 0.1, valAxisMaxVal: 10,
    valAxisLabelColor: MUTED, catAxisLabelColor: MUTED, valAxisLabelFontSize: 11, catAxisLabelFontSize: 10, catAxisLabelFrequency: 5,
    valGridLine: { color: "E1E5EA", size: 0.75 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: BODY,
    showTitle: true, title: "Loss per 1,000 steps (P1 = attribute masking, P2 = event masking)", titleFontSize: 12, titleColor: BODY,
  });
  card(s, 10.0, 1.9, 2.63, 4.6, CARD_ON_BG);
  txt(s, [
    { text: "Matching loss", options: { bold: true, color: NAVY, breakLine: true } },
    { text: "Stays at 0.70–0.82 for all 40,000 steps. Chance is 4.85.", options: { breakLine: true, paraSpaceAfter: 12 } },
    { text: "Fill-in-the-blanks loss", options: { bold: true, color: NAVY, breakLine: true } },
    { text: "3.60 → 0.16 in phase 1; 2.28 → about 0.58 in phase 2.", options: { breakLine: true, paraSpaceAfter: 12 } },
    { text: "Embedding spread", options: { bold: true, color: NAVY, breakLine: true } },
    { text: "z_std 0.088, far from collapse." },
  ], { x: 10.25, y: 2.15, w: 2.15, h: 4.2, fontSize: 13 });
  footer(s, 10, "Losses logged every 1,000 steps");
  s.addNotes("A healthy matching loss stays well above zero: the game is still hard, so the model keeps learning.");
}

// 11 · Gain chart
{
  const s = pres.addSlide(); s.background = { color: BG };
  header(s, "Results · gain from pretraining", "Pretraining helps in every setting");
  s.addChart(pres.charts.BAR, [{ name: "Gain from pretraining", labels: ["ADL · 5 %", "ADL · 30 %", "Next-30 · 5 %", "Next-30 · 30 %"], values: [0.076, 0.059, 0.053, 0.030] }], {
    x: M, y: 1.75, w: 8.6, h: 4.9, barDir: "col", barGapWidthPct: 70, chartColors: [BLUE],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.000", dataLabelFontSize: 14, dataLabelColor: NAVY, dataLabelFontBold: true,
    valAxisMinVal: 0, valAxisMaxVal: 0.1, valAxisMajorUnit: 0.025, valAxisLabelFormatCode: "+0.000;-0.000;0",
    valAxisLabelColor: MUTED, catAxisLabelColor: BODY, valAxisLabelFontSize: 11, catAxisLabelFontSize: 13,
    valGridLine: { color: "E1E5EA", size: 0.75 }, catGridLine: { style: "none" }, showLegend: false,
    showTitle: true, title: "Mean F1 gain over 7 held-out homes (pretrained − from scratch)", titleFontSize: 12, titleColor: BODY,
  });
  card(s, 9.7, 1.9, 2.93, 3.3, CARD_ON_BG);
  txt(s, "Homes where pretraining wins", { x: 9.95, y: 2.1, w: 2.45, h: 0.7, fontSize: 16, bold: true, color: NAVY });
  txt(s, [["ADL 5 %", "7 of 7"], ["ADL 30 %", "6 of 7"], ["Next-30 5 %", "7 of 7"], ["Next-30 30 %", "6 of 7"]].map(([a, b], i, arr) => (
    [{ text: a + ": ", options: {} }, { text: b, options: { bold: true, color: NAVY, breakLine: i < arr.length - 1 } }]
  )).flat(), { x: 9.95, y: 2.85, w: 2.45, h: 2.2, fontSize: 15, paraSpaceAfter: 8 });
  txt(s, "Total: 26 of 28 settings. The two losses are about 0.01, within noise.", { x: 9.7, y: 5.4, w: 2.93, h: 0.9, fontSize: 13 });
  footer(s, 11, "Gain = pretrained F1 − from-scratch F1, mean over 7 held-out homes");
  s.addNotes("The gain shrinks as labels grow (ADL +0.076 to +0.059, Next-30 +0.053 to +0.030). That is the expected shape when pretraining is useful: it matters most when labels are scarce.");
}

// 12 · Scores
{
  const s = pres.addSlide(); s.background = { color: BG };
  header(s, "Results · absolute scores", "Scores across the 7 test homes");
  const R = { align: "right" }, G = { align: "right", color: BLUE_D, bold: true };
  const rows = [
    [HDR("Mean F1, 7 homes"), HDR("Pretrained", "right"), HDR("From scratch", "right"), HDR("Gain", "right")],
    [C("ADL · 5 %"), C("0.517", R), C("0.441", R), C("+0.076", G)],
    [C("ADL · 30 %"), C("0.529", R), C("0.470", R), C("+0.059", G)],
    [C("Next-30 · 5 %"), C("0.649", R), C("0.597", R), C("+0.053", G)],
    [C("Next-30 · 30 %"), C("0.613", R), C("0.583", R), C("+0.030", G)],
  ];
  table(s, rows, { y: 1.85, w: CW, colW: [3.93, 2.67, 2.67, 2.66], fontSize: 16, rowH: 0.48 });
  const cw = (CW - 0.3) / 2;
  card(s, M, 4.6, cw, 1.75, TINT, null);
  txt(s, "The gain shrinks as labels grow", { x: M + 0.3, y: 4.8, w: cw - 0.6, h: 0.4, fontSize: 17, bold: true, color: NAVY });
  txt(s, "ADL +0.076 → +0.059, Next-30 +0.053 → +0.030. Pretraining matters most when labels are scarce.", { x: M + 0.3, y: 5.25, w: cw - 0.6, h: 1.0, fontSize: 14 });
  card(s, M + cw + 0.3, 4.6, cw, 1.75, CARD_ON_BG);
  txt(s, "Without UCI Home B", { x: M + cw + 0.6, y: 4.8, w: cw - 0.6, h: 0.4, fontSize: 17, bold: true, color: NAVY });
  txt(s, "ADL gain +0.055 / +0.022, Next-30 gain +0.061 / +0.037 (5 % / 30 % labels).", { x: M + cw + 0.6, y: 5.25, w: cw - 0.6, h: 1.0, fontSize: 14 });
  footer(s, 12, "From scratch = same model trained from random weights, same fine-tuning");
}

// 13, 14 · Per-home tables
function perHome(n, title, rowsData, mean, metric, note) {
  const s = pres.addSlide(); s.background = { color: BG2 };
  header(s, "Results per home", title);
  const R = { align: "right" };
  const gain = g => C(g, { align: "right", bold: true, color: g.startsWith("−") ? RUST : BLUE_D });
  const rows = [[HDR("Home"), HDR("5 % pretrained", "right"), HDR("5 % scratch", "right"), HDR("Gain", "right"), HDR("30 % pretrained", "right"), HDR("30 % scratch", "right"), HDR("Gain", "right")]];
  rowsData.forEach(r => rows.push([C(r[0]), C(r[1], R), C(r[2], R), gain(r[3]), C(r[4], R), C(r[5], R), gain(r[6])]));
  const MF = { fill: { color: "DDE3EC" }, bold: true };
  rows.push([C("Mean", MF), C(mean[0], Object.assign({}, R, MF)), C(mean[1], Object.assign({}, R, MF)), C(mean[2], { align: "right", bold: true, color: BLUE_D, fill: { color: "DDE3EC" } }),
    C(mean[3], Object.assign({}, R, MF)), C(mean[4], Object.assign({}, R, MF)), C(mean[5], { align: "right", bold: true, color: BLUE_D, fill: { color: "DDE3EC" } })]);
  table(s, rows, { y: 1.85, w: CW, colW: [1.53, 1.95, 1.95, 1.3, 1.95, 1.95, 1.3], fontSize: 14, fill: { color: "FFFFFF" }, rowH: 0.47 });
  footer(s, n, `${metric}, mean ± std over 3 contiguous folds · Red = pretraining lost`);
  s.addNotes(note);
}
perHome(13, "Activity recognition (ADL)", [
  ["uci_b", "0.452 ± 0.021", "0.252 ± 0.041", "+0.200", "0.568 ± 0.030", "0.285 ± 0.036", "+0.283"],
  ["hh101", "0.581 ± 0.020", "0.550 ± 0.023", "+0.031", "0.555 ± 0.025", "0.567 ± 0.001", "−0.012"],
  ["hh103", "0.744 ± 0.023", "0.683 ± 0.006", "+0.061", "0.758 ± 0.020", "0.729 ± 0.013", "+0.029"],
  ["hh105", "0.453 ± 0.027", "0.397 ± 0.021", "+0.056", "0.443 ± 0.032", "0.433 ± 0.028", "+0.010"],
  ["hh110", "0.433 ± 0.006", "0.341 ± 0.027", "+0.092", "0.427 ± 0.009", "0.369 ± 0.027", "+0.058"],
  ["hh119", "0.472 ± 0.047", "0.411 ± 0.048", "+0.061", "0.463 ± 0.065", "0.441 ± 0.037", "+0.022"],
  ["hh122", "0.484 ± 0.009", "0.453 ± 0.024", "+0.031", "0.491 ± 0.020", "0.465 ± 0.013", "+0.026"],
], ["0.517", "0.441", "+0.076", "0.529", "0.470", "+0.059"], "Weighted F1",
  "Pretraining wins on every home with 5 % labels. The largest gain is on UCI B, the smallest and most unusual home. Without UCI B the mean gain is +0.055 (5 %) and +0.022 (30 %). The hh101 loss at 30 % and the hh119 win at 30 % are both within the fold spread.");
perHome(14, "Next-30 event prediction", [
  ["uci_b", "0.705 ± 0.022", "0.703 ± 0.008", "+0.002", "0.695 ± 0.002", "0.708 ± 0.012", "−0.013"],
  ["hh101", "0.714 ± 0.015", "0.635 ± 0.014", "+0.079", "0.649 ± 0.016", "0.616 ± 0.021", "+0.033"],
  ["hh103", "0.670 ± 0.005", "0.610 ± 0.009", "+0.060", "0.680 ± 0.007", "0.621 ± 0.003", "+0.059"],
  ["hh105", "0.581 ± 0.005", "0.534 ± 0.003", "+0.047", "0.530 ± 0.003", "0.498 ± 0.004", "+0.032"],
  ["hh110", "0.650 ± 0.015", "0.576 ± 0.010", "+0.074", "0.607 ± 0.019", "0.554 ± 0.016", "+0.053"],
  ["hh119", "0.573 ± 0.026", "0.509 ± 0.022", "+0.064", "0.527 ± 0.016", "0.499 ± 0.022", "+0.028"],
  ["hh122", "0.653 ± 0.003", "0.608 ± 0.007", "+0.045", "0.600 ± 0.008", "0.585 ± 0.006", "+0.015"],
], ["0.649", "0.597", "+0.053", "0.613", "0.583", "+0.030"], "Multiset F1",
  "Pretraining helps on 6 of 7 homes at both label amounts. Here UCI B is the exception, not the driver: without it the gain is +0.061 and +0.037.");

// 15 · UCI Home B vs paper
{
  const s = pres.addSlide(); s.background = { color: BG };
  header(s, "Paper vs this work · the one shared home", "UCI Home B against the paper");
  txt(s, "Each cell: pretrained / from scratch", { x: M, y: 1.8, w: 7, h: 0.3, fontSize: 13, color: MUTED });
  const R = { align: "right" };
  const rows = [
    [HDR("Setting"), HDR("Paper", "right"), HDR("Ours", "right")],
    [C("ADL · 5 %"), C("0.38 / 0.30", R), C("0.45 / 0.25", R)],
    [C("ADL · 30 %"), C("0.60 / 0.36", R), C("0.57 / 0.29", R)],
    [C("Next-30 · 5 %"), C("0.76 / 0.53", R), C("0.71 / 0.70", R)],
    [C("Next-30 · 30 %"), C("0.90 / 0.65", R), C("0.70 / 0.71", R)],
  ];
  table(s, rows, { y: 2.2, w: 7.0, colW: [2.6, 2.2, 2.2], fontSize: 16, rowH: 0.5 });
  const notes = [
    ["ADL follows the paper's pattern.", "Gain +0.20 / +0.28, against the paper's +0.08 / +0.24."],
    ["The Next-30 gain does not reproduce.", "The paper gains about +0.24; our from-scratch model already scores 0.70."],
    ["Folds differ.", "Random folds leak near-copies of test windows; on our from-scratch ADL they add about 0.12."],
  ];
  notes.forEach(([b, t], i) => {
    const y = 2.2 + i * 1.4;
    numCircle(s, 8.1, y, i + 1);
    txt(s, [{ text: b + " ", options: { bold: true, color: NAVY } }, { text: t }], { x: 8.7, y: y - 0.02, w: 3.93, h: 1.2, fontSize: 14 });
  });
  footer(s, 15, "Paper: 5-fold CV · Ours: 3 contiguous folds · Two decimals for comparison");
  s.addNotes("UCI Home B is the only test home in both the paper and this work, so it is the only direct comparison. Even here, fold protocol and pretraining data differ. Random folds raised our from-scratch ADL at 30 % from 0.28 to 0.40, close to the paper's 0.36.");
}

// 16 · Limitations
{
  const s = pres.addSlide(); s.background = { color: BG2 };
  header(s, "Limitations", "What these results do not show", RUST);
  const cols = [
    ["Setup", [
      "Not directly comparable to the paper: 6 of 7 test homes differ, and the 2025 CASAS release merges sensors per room.",
      "Independent code: 28.6M vs 36.1M parameters, and many settings are our guesses.",
      "Our pretraining game adds six changes, so this is a modified DomusFM, not an exact copy of the paper's recipe.",
      "Only ADL and Next-30, at 5 % and 30 % labels. No Next-10, clustering or other baselines.",
    ]],
    ["Findings", [
      "A single pretraining run and 3 folds per home. Differences under about 0.03 on one home are noise.",
      "UCI Home B drives much of the ADL gain. Without it: +0.055 and +0.022.",
      "The six changes were applied together; we don't yet know which ones matter.",
      "Embedding spread (z_std) sat at exactly 0.088 from step 1,000 on; check it before trusting it.",
      "ADL is still only about 0.5 weighted F1.",
    ]],
  ];
  const cw = (CW - 0.3) / 2;
  cols.forEach(([h, items], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 1.85, cw, 4.75, CARD_ON_BG2);
    txt(s, h, { x: x + 0.3, y: 2.05, w: cw - 0.6, h: 0.4, fontSize: 18, bold: true, color: NAVY });
    txt(s, items.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < items.length - 1, paraSpaceAfter: 8 } })), { x: x + 0.3, y: 2.55, w: cw - 0.6, h: 3.9, fontSize: 14 });
  });
  footer(s, 16);
  s.addNotes("Test sets are also subsampled to at most 20,000 windows per fold. Pretraining timing is not representative: a game shared the GPU for most of the run.");
}

// 17–19 · DomusFM limitations → HomeFM remedies
function limSlide(n, part, title, rows, extra) {
  const s = pres.addSlide(); s.background = { color: BG };
  header(s, `DomusFM limitations → HomeFM remedies · ${part} of 3`, title);
  const t = [[HDR("Limitation"), HDR("What goes wrong"), HDR("HomeFM remedy"), HDR("Status")]];
  rows.forEach(r => t.push([C(r[0], { bold: true }), C(r[1]), C(r[2]), C(r[3], { color: MUTED })]));
  table(s, t, { y: 1.85, w: CW, colW: [2.55, 3.45, 3.7, 2.23], fontSize: 14, rowH: 0.62 });
  if (extra) extra(s);
  footer(s, n, "Source: docs/LIMITATIONS_AND_REMEDIES.md");
  return s;
}
limSlide(17, 1, "Limits on what it can answer", [
  ["L1 · Closed label set", "Can only name activities it was shown examples of", "Link sensor activity to words, so any activity can be described", "Designed; text alignment built"],
  ["L5 · No start or end", "Cannot count \"cooked 3 times\" or see two things at once", "Several tags at once, start and end detection, counting rules", "Designed"],
  ["L6 · No anomaly output", "Cannot say \"this is unusual\" or \"this sensor is broken\"", "Surprise score, anomaly and device-health engines", "Surprise score built; engines designed"],
  ["L7 · No timing", "Knows what comes next, not when", "Predicts each next event with a range of likely times", "Head built; overdue checks designed"],
  ["L8 · Not searchable", "Its knowledge cannot be searched with a question", "A \"map of meanings\" shared by minutes and sentences", "Designed"],
]).addNotes("L1 and L8 share one remedy: linking sensor activity to words. L6 and L7 share another: predicting what happens next and when. Built means implemented and tested on small data; designed means specified in DESIGN.md but not yet built.");
limSlide(18, 2, "Limits on what it can see", [
  ["L2 · On/off only", "Sees \"fridge ON\", not \"fridge draws 180 W\"", "Keep real numbers as input", "Input built; device health designed"],
  ["L3 · 30-event window", "Memory is sometimes 2 minutes, sometimes 6 hours, never weeks", "1-minute steps, hours of context, daily summaries", "Minutes built; days designed"],
  ["L4 · One person assumed", "Cannot tell people apart or count them", "Person cards, identity clues, per-room counts", "Designed"],
  ["L9 · No sound or camera", "Cannot hear a baby cry or see a parcel", "On-hub sound and camera detectors send short tags", "Designed; event format built"],
]).addNotes("DomusFM turns every sensor into ON/OFF events and remembers the last 30 events, however long they took. HomeFM keeps real values, reads in 1-minute steps, and adds daily summaries for long-range habits.");
limSlide(19, 3, "Limits on how it learns", [
  ["L10 · Small, old data", "Never saw smart locks, cameras or robot vacuums", "Many more homes, a simulator, pilot and donated homes", "84 homes built; rest designed"],
  ["L11 · Game too easy", "Learns sensor tricks instead of behaviour", "Three harder, more useful practice games", "Games built on small data; corpus run next"],
], s => {
  const cw = (CW - 0.3) / 2;
  card(s, M, 4.4, cw, 1.8, TINT, null);
  txt(s, "L10 and L11 are linked", { x: M + 0.3, y: 4.6, w: cw - 0.6, h: 0.4, fontSize: 17, bold: true, color: NAVY });
  txt(s, "More data does not help while the game is too easy. With a harder game, the same 77 homes do help.", { x: M + 0.3, y: 5.05, w: cw - 0.6, h: 1.0, fontSize: 14 });
  card(s, M + cw + 0.3, 4.4, cw, 1.8, CARD_ON_BG);
  txt(s, "We tested L11 on DomusFM", { x: M + cw + 0.6, y: 4.6, w: cw - 0.6, h: 0.4, fontSize: 17, bold: true, color: NAVY });
  txt(s, "The paper's game and a harder game, same model and data. Results on the next slide.", { x: M + cw + 0.6, y: 5.05, w: cw - 0.6, h: 1.0, fontSize: 14 });
});

// 20 · L11 evidence
{
  const s = pres.addSlide(); s.background = { color: BG2 };
  header(s, "L11 in our own runs · same model, same 77 homes", "An easy game hurts; a harder one helps");
  const P = { align: "right", color: RUST, bold: true }, H = { align: "right", color: BLUE_D, bold: true };
  const rows = [
    [HDR("Measure"), HDR("Paper's game", "right"), HDR("Harder game", "right")],
    [C("Matching loss, end of training"), C("about 0.0005", P), C("0.70–0.82", H)],
    [C("Settings where pretraining wins"), C("6 of 28", P), C("26 of 28", H)],
    [C("ADL gain · 5 % / 30 % labels"), C("−0.050 / −0.068", P), C("+0.076 / +0.059", H)],
    [C("Next-30 gain · 5 % / 30 % labels"), C("−0.009 / +0.002", P), C("+0.053 / +0.030", H)],
  ];
  table(s, rows, { y: 1.85, w: CW, colW: [5.53, 3.2, 3.2], fontSize: 16, fill: { color: "FFFFFF" }, rowH: 0.5 });
  txt(s, [
    { text: "Tricks the paper's game allowed: ", options: { bold: true, color: NAVY } },
    { text: "candidates from other homes, exact-second timestamps, one copy never hidden, no separate head for the game, nothing to predict.", options: { breakLine: true, paraSpaceAfter: 10 } },
    { text: "Remedies: ", options: { bold: true, color: NAVY } },
    { text: "the six changes on slide 9. The from-scratch baseline barely moved, so the gain comes from the harder game." },
  ], { x: M, y: 4.65, w: CW, h: 1.6, fontSize: 15 });
  footer(s, 20, "Gain = pretrained F1 − from-scratch F1, mean over 7 held-out homes · Red = paper's game, blue = harder game");
  s.addNotes("Paper's game run finished 24 September 2026; harder game run finished 25 September 2026. With the paper's game, pretraining was worse than training from scratch on 6 of 7 homes for activity recognition. Settings where pretraining wins, paper's game: ADL 1 + 1, Next-30 1 + 3 = 6 of 28.");
}

// 21 · HomeFM games
{
  const s = pres.addSlide(); s.background = { color: BG2 };
  header(s, "HomeFM's remedy for L11", "Three harder practice games");
  const games = [
    ["What happens next, and when?", "Predict the next event and the time until it. The future is unseen, so the model must learn routines. Also gives the surprise score (L6) and timing (L7)."],
    ["Hide a big chunk, guess its meaning", "Hide 10 minutes, a room or a device type, and predict what it meant. No ON/OFF partner or timestamp is left to copy."],
    ["Match minutes with sentences", "Pair minutes with descriptions, allowing many right answers, so two nights of sleep end up together (L1, L8)."],
  ];
  const cw = (CW - 0.6) / 3;
  games.forEach(([t, d], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 1.9, cw, 3.45, CARD_ON_BG2);
    numCircle(s, x + 0.3, 2.15, i + 1);
    txt(s, `GAME ${i + 1}`, { x: x + 0.85, y: 2.21, w: cw - 1.1, h: 0.3, fontSize: 12, bold: true, color: BLUE_D, charSpacing: 2 });
    txt(s, t, { x: x + 0.3, y: 2.75, w: cw - 0.6, h: 0.75, fontSize: 17, bold: true, color: NAVY });
    txt(s, d, { x: x + 0.3, y: 3.55, w: cw - 0.6, h: 1.5, fontSize: 14, lineSpacingMultiple: 1.1 });
  });
  card(s, M, 5.6, CW, 0.95, TINT, null);
  txt(s, "Built and tested on small data. Next: HomeFM against the fixed DomusFM on the same 77 homes. The DomusFM \"fill in the blanks\" change is a small version of game 2.", { x: M + 0.3, y: 5.675, w: CW - 0.6, h: 0.8, fontSize: 15, color: NAVY, valign: "middle" });
  footer(s, 21, "Details: docs/LIMITATIONS_AND_REMEDIES.md (L11), docs/DESIGN.md §8");
  s.addNotes("Also carried over from the DomusFM fix into HomeFM training: batches drawn from a few homes at a time, gentler fine-tuning, and loss health checks. HomeFM must now beat the fixed DomusFM: mean activity F1 0.517 / 0.529 and Next-30 F1 0.649 / 0.613 at 5 % / 30 % labels.");
}

// 22 · Next steps
{
  const s = pres.addSlide(); s.background = { color: BG };
  header(s, "Next steps", "What comes next");
  const cards = [
    ["PLANNED", "Compare against HomeFM", "Run strong-masking DomusFM, and HomeFM at 8.0M and at a size-matched 28.6M, on this same protocol.", TINT, BLUE_D],
    ["PLANNED", "Ablate the six changes", "Turn each change off in turn to see which ones carry the gain and which can go.", TINT, BLUE_D],
    ["SUGGESTED", "Harden the numbers", "Repeat runs with new seeds, look into the flat z_std, and add the paper's missing tasks and baselines.", CARD_ON_BG, MUTED],
  ];
  const cw = (CW - 0.6) / 3;
  cards.forEach(([tag, t, d, f, tc], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 2.0, cw, 3.4, f, f === TINT ? null : LINE);
    numCircle(s, x + 0.3, 2.3, i + 1, f === TINT ? BLUE : "9AA1AD");
    txt(s, tag, { x: x + 0.85, y: 2.36, w: cw - 1.1, h: 0.3, fontSize: 12, bold: true, color: tc, charSpacing: 2 });
    txt(s, t, { x: x + 0.3, y: 2.9, w: cw - 0.6, h: 0.75, fontSize: 18, bold: true, color: NAVY });
    txt(s, d, { x: x + 0.3, y: 3.75, w: cw - 0.6, h: 1.5, fontSize: 15, lineSpacingMultiple: 1.12 });
  });
  footer(s, 22, "Configs: homefm_corpus.yaml (8.0M), homefm_corpus_384.yaml (28.6M)");
}

// 23 · Takeaways
{
  const s = pres.addSlide(); s.background = { color: NAVY };
  txt(s, "TAKEAWAYS", { x: M, y: 0.5, w: CW, h: 0.3, fontSize: 12, bold: true, color: "8FB4E0", charSpacing: 2 });
  txt(s, "Three things to remember", { x: M, y: 0.82, w: CW, h: 0.75, fontFace: HEAD, fontSize: 36, bold: true, color: "F7F6F2" });
  [
    "Pretrained on 77 homes, DomusFM beats training from scratch in 26 of 28 settings.",
    "The gain is largest when labels are scarce: +0.076 activity F1 with 5 % labels.",
    "HomeFM targets DomusFM's 11 limitations; its first test is to beat this fixed DomusFM.",
  ].forEach((t, i) => {
    const y = 2.2 + i * 1.35;
    numCircle(s, M, y, i + 1, "8FB4E0", NAVY, 0.6);
    txt(s, t, { x: M + 0.9, y: y - 0.05, w: 10.8, h: 0.8, fontSize: 22, color: "E6ECF5", valign: "middle" });
  });
  footer(s, 23, "Full results: results/domusfm_corpus_fixed/", "9FB0C8");
}

pres.writeFile({ fileName: out }).then(f => console.log("wrote", f));
