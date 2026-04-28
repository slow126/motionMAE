const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { imageSize } = require("image-size");

let pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.author = 'Spencer Low';
pres.title = 'Bridging Domain Gaps in Visual Correspondence';

// ── Path configuration ─────────────────────────────────────────
// Run from the `presentation/` directory:  node build_pptx.js
//
// Expected layout:
//   presentation/
//     build_pptx.js           ← this file
//     figures/                ← ALL image assets go here, including:
//                                umap_all_datasets.png, umap_ft3d_sweep.png
//                                ch1_*, ch2_*, ch3_*, table_*, benchmark_*, train_*
//                                fig_density_mismatch.png, fig_gist_illustration.png
//     node_modules/           ← npm deps (pptxgenjs etc.)
//     dissertation_proposal_defense.pptx   ← generated output (written here)
const FIG_DIR = path.join(__dirname, "figures");
const PPTX_PATH = path.join(__dirname, "dissertation_proposal_defense.pptx");

function fig(name) {
  return path.join(FIG_DIR, name);
}

// ── Color Palette ──────────────────────────────────────────────
// Deep navy + electric teal + warm gold accent
const C = {
  navy:    "0D1B2A",   // dominant dark bg
  navyMid: "132237",   // slightly lighter panels
  teal:    "00B4D8",   // primary accent (coverage, arrows, highlights)
  tealDim: "0077A8",   // secondary teal
  gold:    "F4A261",   // warm accent for "key insight" callouts
  white:   "FFFFFF",
  offWhite:"E8EFF7",
  muted:   "8BA3BF",
  red:     "E76F51",   // for "missing support" / warning elements
  green:   "52B788",   // for "covered" / positive elements
  lightBg: "F0F4F8",   // light content bg
  darkText:"0D1B2A",
  midGray: "4A6785",
};

const makeShadow = () => ({ type: "outer", blur: 8, offset: 3, angle: 135, color: "000000", opacity: 0.18 });

// ── Helper: dark slide background ──────────────────────────────
function darkBg(slide) {
  slide.background = { color: C.navy };
}
function lightBg(slide) {
  slide.background = { color: C.lightBg };
}

// ── Helper: section header bar ─────────────────────────────────
function sectionBar(slide, text) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 10, h: 0.72,
    fill: { color: C.teal }, line: { color: C.teal }
  });
  slide.addText(text, {
    x: 0.4, y: 0, w: 9.2, h: 0.72, margin: 0,
    fontSize: 22, bold: true, color: C.navy, valign: "middle"
  });
}

// ── Helper: light slide title ───────────────────────────────────
function lightTitle(slide, text) {
  slide.addText(text, {
    x: 0.5, y: 0.15, w: 9, h: 0.7,
    fontSize: 26, bold: true, color: C.navy
  });
  // thin teal rule
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.82, w: 9, h: 0.04,
    fill: { color: C.teal }, line: { color: C.teal }
  });
}

// ── Helper: bullet block ────────────────────────────────────────
function bullets(slide, items, opts = {}) {
  const { x=0.5, y=1.1, w=9, h=4, fs=15, color=C.darkText } = opts;
  const runs = items.map((item, i) => {
    const isLast = i === items.length - 1;
    if (typeof item === "string") {
      return { text: item, options: { bullet: true, breakLine: !isLast, fontSize: fs, color, paraSpaceAfter: 8 } };
    }
    // sub-item: { text, sub: true }
    return { text: item.text, options: { bullet: true, indentLevel: 1, breakLine: !isLast, fontSize: fs-1, color: C.midGray, paraSpaceAfter: 4 } };
  });
  slide.addText(runs, { x, y, w, h, valign: "top" });
}

// ── Helper: callout box ─────────────────────────────────────────
function callout(slide, text, opts = {}) {
  const { x=0.5, y=4.5, w=9, h=0.8, bg=C.gold, fc=C.navy, fs=14 } = opts;
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: bg }, line: { color: bg }
  });
  slide.addText(text, {
    x: x+0.15, y, w: w-0.3, h, margin: 0,
    fontSize: fs, bold: true, color: fc, valign: "middle", align: "left"
  });
}

// ── Helper: image placeholder box ──────────────────────────────
function imgPlaceholder(slide, label, x, y, w, h, note="") {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h,
    fill: { color: C.navyMid }, line: { color: C.tealDim, width: 1.5 },
    shadow: makeShadow()
  });
  slide.addText(label, {
    x: x+0.1, y: y+(h/2)-0.25, w: w-0.2, h: 0.5,
    fontSize: 11, bold: true, color: C.teal, align: "center"
  });
  if (note) {
    slide.addText(note, {
      x: x+0.1, y: y+(h/2)+0.05, w: w-0.2, h: 0.35,
      fontSize: 9, italic: true, color: C.muted, align: "center"
    });
  }
}

function addImageOrPlaceholder(slide, imagePath, x, y, w, h, note="") {
  if (fs.existsSync(imagePath)) {
    slide.addImage({ path: imagePath, x, y, w, h });
    return;
  }
  imgPlaceholder(slide, "Missing Figure", x, y, w, h, note || path.basename(imagePath));
}

function addImageContainOrPlaceholder(slide, imagePath, x, y, w, h, note="") {
  if (fs.existsSync(imagePath)) {
    const dims = imageSize(imagePath);
    if (dims.width && dims.height) {
      const scale = Math.min(w / dims.width, h / dims.height);
      const fitW = dims.width * scale;
      const fitH = dims.height * scale;
      const fitX = x + (w - fitW) / 2;
      const fitY = y + (h - fitH) / 2;
      slide.addImage({ path: imagePath, x: fitX, y: fitY, w: fitW, h: fitH });
      return;
    }
  }
  imgPlaceholder(slide, "Missing Figure", x, y, w, h, note || path.basename(imagePath));
}

// ── Helper: two-col layout scaffold ────────────────────────────
function twoColDivider(slide, xPos=5.05) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: xPos, y: 0.85, w: 0.04, h: 4.6,
    fill: { color: C.teal }, line: { color: C.teal }
  });
}


// ═══════════════════════════════════════════════════════════════
// SLIDE 1: TITLE
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  darkBg(s);

  // Background accent rectangle
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 3.5, w: 10, h: 2.125,
    fill: { color: C.navyMid }, line: { color: C.navyMid }
  });
  // teal left accent
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.18, h: 5.625,
    fill: { color: C.teal }, line: { color: C.teal }
  });

  s.addText("Bridging Domain Gaps in Visual Correspondence", {
    x: 0.5, y: 0.55, w: 9.2, h: 1.8,
    fontSize: 36, bold: true, color: C.white,
    wrap: true, valign: "top"
  });
  s.addText("From Multi-Modal Fusion to Data-Curated Motion Latents", {
    x: 0.5, y: 2.25, w: 9.2, h: 0.7,
    fontSize: 20, italic: true, color: C.teal
  });
  s.addText("Spencer Low  ·  PhD Dissertation Proposal  ·  BYU Computer Science  ·  April 2026", {
    x: 0.5, y: 3.65, w: 9, h: 0.45,
    fontSize: 13, color: C.muted
  });

  // Three chapter pills
  const pills = ["Chapter 1: Cross-Modal Aerial Sensing", "Chapter 2: Directed Coverage Metrics", "Chapter 3: Coverage-Based Data Curation"];
  const pilColors = [C.tealDim, C.teal, C.gold];
  const pilTextColors = [C.white, C.navy, C.navy];
  pills.forEach((p, i) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.45 + i*3.18, y: 4.25, w: 3.0, h: 0.55, rectRadius: 0.1,
      fill: { color: pilColors[i] }, line: { color: pilColors[i] }
    });
    s.addText(p, {
      x: 0.45 + i*3.18, y: 4.25, w: 3.0, h: 0.55,
      fontSize: 10, bold: true, color: pilTextColors[i], align: "center", valign: "middle"
    });
  });

  s.addText("Advisor: Ryan Farrell  ·  Committee: Bryan Morse, David Grimsman, Porter Jenkins", {
    x: 0.5, y: 5.1, w: 9, h: 0.35,
    fontSize: 11, color: C.muted
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 2: MOTIVATION — What is visual correspondence?
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  lightBg(s);
  lightTitle(s, "Visual Correspondence: One Problem, Many Domains");

  // Left column: definition + tasks
  s.addText("Dense Correspondence", {
    x: 0.5, y: 1.05, w: 4.5, h: 0.4,
    fontSize: 16, bold: true, color: C.navy
  });
  s.addText("Given a point in image A, find its counterpart in image B.\nUnderlies: optical flow · semantic matching · 3D reconstruction · tracking · navigation", {
    x: 0.5, y: 1.45, w: 4.5, h: 1.0,
    fontSize: 13, color: C.darkText, wrap: true
  });

  // Task grid
  const tasks = [
    { label: "Optical Flow", color: C.teal },
    { label: "Semantic Match", color: C.tealDim },
    { label: "3D Reconstruct", color: C.gold },
    { label: "Cross-Modal", color: C.red },
  ];
  tasks.forEach((t, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const tx = 0.5 + col * 2.3, ty = 2.55 + row * 0.72;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: tx, y: ty, w: 2.1, h: 0.55, rectRadius: 0.08,
      fill: { color: t.color }, line: { color: t.color }
    });
    s.addText(t.label, {
      x: tx, y: ty, w: 2.1, h: 0.55,
      fontSize: 11, bold: true, color: C.white, align: "center", valign: "middle"
    });
  });

  // Right column: domain shift is the problem
  twoColDivider(s);
  s.addText("The Core Challenge: Domain Shift", {
    x: 5.2, y: 1.05, w: 4.5, h: 0.4,
    fontSize: 16, bold: true, color: C.navy
  });

  const shiftItems = [
    "Training data ≠ deployment data",
    "Sensing modality gaps (EO → SAR → IR)",
    "Motion statistic mismatches",
    "Supervision format heterogeneity",
    "Dense flow vs. sparse keypoints",
  ];
  bullets(s, shiftItems, { x: 5.2, y: 1.5, w: 4.5, h: 2.5, fs: 13 });

  // Domain shift examples — red = cross-domain mismatch, green = within-domain (transfers)
  s.addText("Train on", { x: 5.2, y: 3.42, w: 2.05, h: 0.25,
    fontSize: 9, color: C.muted, italic: true, align: "center" });
  s.addText("Deploy to", { x: 7.5, y: 3.42, w: 2.05, h: 0.25,
    fontSize: 9, color: C.muted, italic: true, align: "center" });

  const shiftExamples = [
    { src: "FlyingThings (synthetic)", tgt: "KITTI-2015 (real driving)", color: C.red,   match: false },
    { src: "EO (visible light)",       tgt: "SAR (radar backscatter)",   color: C.red,   match: false },
    { src: "PointOdyssey (tracking)",  tgt: "SPair-71k (semantic)",      color: C.red,   match: false },
    { src: "SPair-71k (semantic)",     tgt: "PF-PASCAL (semantic)",      color: C.green, match: true  },
  ];
  shiftExamples.forEach((ex, i) => {
    const y = 3.68 + i * 0.36;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.2, y, w: 2.05, h: 0.30,
      fill: { color: C.navyMid }, line: { color: ex.color, width: 1.2 }
    });
    s.addText(ex.src, { x: 5.2, y, w: 2.05, h: 0.30,
      fontSize: 8.5, color: C.offWhite, align: "center", valign: "middle", wrap: true });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 7.5, y, w: 2.05, h: 0.30,
      fill: { color: C.navyMid }, line: { color: ex.color, width: 1.2 }
    });
    s.addText(ex.tgt, { x: 7.5, y, w: 2.05, h: 0.30,
      fontSize: 8.5, color: ex.color, bold: true, align: "center", valign: "middle", wrap: true });
  });

}

// ═══════════════════════════════════════════════════════════════
// SLIDE 2b: Correspondence Task Gallery — What do these tasks look like?
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  lightBg(s);
  lightTitle(s, "Correspondence Tasks Across Vision Domains");

  s.addText("All three dissertation chapters address variants of the same fundamental challenge: find matching points across mismatched images.", {
    x: 0.5, y: 0.95, w: 9, h: 0.38,
    fontSize: 12, italic: true, color: C.midGray, wrap: true
  });

  // 2 rows × 3 cols of task example placeholders
  const tasks = [
    { label: "Optical Flow", sub: "KITTI-2015\nper-pixel motion vectors", color: C.teal, file: "task_optical_flow_kitti.png" },
    { label: "Semantic Matching", sub: "SPair-71k / PF-PASCAL\nkeypoint transfer across poses", color: C.gold, file: "task_semantic_matching_spair.png" },
    { label: "Dense Tracking", sub: "PointOdyssey\nlong-range point trajectories", color: C.tealDim, file: "task_dense_tracking_pointodyssey.png" },
    { label: "Cross-Modal", sub: "MAGIC (SAR ↔ EO)\nmatching across sensor physics", color: C.red, file: "task_cross_modal_magic.png" },
    { label: "3D Scene Flow", sub: "SDF-Fractal3D / FlyingThings\nsynthetic correspondence pairs", color: C.green, file: "task_scene_flow_flyingthings.png" },
    { label: "Template Matching", sub: "TSS / PF-WILLOW\ninstances across categories", color: C.midGray, file: "task_template_matching_tss.png" },
  ];

  const cols = 3, cellW = 2.82, cellH = 1.72, gapX = 0.18, gapY = 0.18;
  const startX = 0.47, startY = 1.46;

  tasks.forEach((task, i) => {
    const col = i % cols, row = Math.floor(i / cols);
    const x = startX + col * (cellW + gapX);
    const y = startY + row * (cellH + gapY);
    const imageH = cellH - 0.32;
    if (task.file) {
      addImageOrPlaceholder(s, fig(task.file), x, y, cellW, imageH, task.sub);
    } else {
      imgPlaceholder(s, task.label, x, y, cellW, imageH, task.sub);
    }
    // Color label band
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: y + cellH - 0.32, w: cellW, h: 0.3, rectRadius: 0.04,
      fill: { color: task.color }, line: { color: task.color }
    });
    s.addText(task.label, {
      x, y: y + cellH - 0.32, w: cellW, h: 0.3,
      fontSize: 10, bold: true, color: C.navy, align: "center", valign: "middle"
    });
  });

  callout(s, "Ch. 1: cross-modal (SAR↔EO) · Ch. 2: metric & synthetic data design · Ch. 3: training data curation for all task families", {
    x: 0.5, y: 5.28, w: 9, h: 0.3, bg: C.navy, fc: C.teal, fs: 11
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 3: THESIS + CHAPTER OVERVIEW
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  darkBg(s);
  sectionBar(s, "Dissertation Overview");

  // Thesis statement — Option A
  s.addText("Appearance alignment between training and deployment data is a weaker predictor of correspondence transfer than is commonly assumed. The geometric structure of motion — how displacements are distributed across directions, magnitudes, and scene locations — is the more reliable and actionable axis: it explains transfer variance that appearance metrics cannot, predicts zero-shot performance across diverse benchmarks, and provides principled selection signals from whole datasets down to individual pairs.", {
    x: 0.5, y: 0.88, w: 9, h: 1.08,
    fontSize: 11.5, italic: true, color: C.offWhite, wrap: true
  });

  const chapters = [
    {
      num: "Ch. 1", title: "Cross-Modal Aerial Sensing",
      desc: "EO↔SAR features transfer despite extreme visual disparities — appearance-domain match is not required. Progress is bottlenecked by training data scale, motivating synthetic augmentation.\nPublished (CVPRW 2022–2024)",
      color: C.tealDim, tc: C.white
    },
    {
      num: "Ch. 2", title: "Directed Coverage Metrics",
      desc: "Non-photorealistic fractals match photorealistic baselines zero-shot — motion geometry, not appearance, is the key predictor. Directed BFV coverage separates failure modes that symmetric metrics conflate.\nUnder review (ECCV 2026)",
      color: C.teal, tc: C.navy
    },
    {
      num: "Ch. 3", title: "Coverage-Based Data Curation",
      desc: "BFV motion coverage as utility function curates heterogeneous training pools — outperforms random sampling, recovers expert source routing without source labels.\nIn progress",
      color: C.gold, tc: C.navy
    },
  ];

  const cardY = 2.1, cardH = 3.1;
  chapters.forEach((ch, i) => {
    const x = 0.35 + i * 3.12;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: cardY, w: 2.95, h: cardH,
      fill: { color: C.navyMid }, line: { color: ch.color, width: 2 },
      shadow: makeShadow()
    });
    // top color bar
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: cardY, w: 2.95, h: 0.5,
      fill: { color: ch.color }, line: { color: ch.color }
    });
    s.addText(ch.num, {
      x: x+0.1, y: cardY, w: 2.75, h: 0.5,
      fontSize: 14, bold: true, color: ch.tc, valign: "middle"
    });
    s.addText(ch.title, {
      x: x+0.12, y: cardY+0.55, w: 2.7, h: 0.55,
      fontSize: 13, bold: true, color: C.white, wrap: true
    });
    s.addText(ch.desc, {
      x: x+0.12, y: cardY+1.15, w: 2.7, h: 1.8,
      fontSize: 10, color: C.muted, wrap: true
    });
  });

  // Arrow connectors
  [3.3, 6.42].forEach(ax => {
    s.addShape(pres.shapes.RECTANGLE, {
      x: ax, y: cardY + cardH/2, w: 0.12, h: 0.08,
      fill: { color: C.teal }, line: { color: C.teal }
    });
    s.addText("→", { x: ax - 0.1, y: cardY + cardH/2 - 0.2, w: 0.4, h: 0.4, fontSize: 18, bold: true, color: C.teal, align: "center" });
  });

  callout(s, "Appearance alignment is increasingly saturated by pre-trained features. Motion geometry is the underutilized axis — and each chapter adds evidence.", {
    x: 0.5, y: cardY + cardH + 0.08, w: 9, h: 0.38, bg: C.navy, fc: C.teal, fs: 11
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 3b: THESIS STATEMENT
// ═══════════════════════════════════════════════════════════════
if (false) {
  let s = pres.addSlide();
  darkBg(s);
  sectionBar(s, "Thesis Statement");

  // Problem sentence
  s.addText("Domain shift between training and deployment data is a central obstacle to robust dense visual correspondence.", {
    x: 0.6, y: 0.95, w: 8.8, h: 1.0,
    fontSize: 18, color: C.offWhite, wrap: true, italic: true
  });

  // Thesis sentence — broken into three colored spans highlighting the three contributions
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 2.05, w: 9, h: 2.55,
    fill: { color: C.navyMid }, line: { color: C.teal, width: 2 }
  });
  s.addText("This dissertation takes a data-centric approach: correspondence transfer failures are training data composition problems — diagnosable and reducible through rigorous motion-aware analysis and selection:", {
    x: 0.72, y: 2.15, w: 8.56, h: 0.65,
    fontSize: 14, color: C.offWhite, wrap: true
  });

  const pillars = [
    { label: "Cross-modal representation learning", sub: "for aerial sensing", color: C.tealDim },
    { label: "Directed coverage metrics", sub: "that identify motion support as a practical predictor of transfer", color: C.teal },
    { label: "Coverage-and-diversity-based curation", sub: "that turns diagnostics into targeted training-data selection", color: C.gold },
  ];
  pillars.forEach((p, i) => {
    const y = 2.86 + i * 0.54;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.72, y, w: 0.32, h: 0.38, rectRadius: 0.04,
      fill: { color: p.color }, line: { color: p.color }
    });
    s.addText(`${i + 1}`, {
      x: 0.72, y, w: 0.32, h: 0.38,
      fontSize: 13, bold: true, color: C.navy, align: "center", valign: "middle"
    });
    s.addText(p.label + "  ", {
      x: 1.14, y: y + 0.02, w: 3.5, h: 0.34,
      fontSize: 13, bold: true, color: p.color
    });
    s.addText(p.sub, {
      x: 4.62, y: y + 0.04, w: 4.6, h: 0.3,
      fontSize: 11, italic: true, color: C.muted
    });
  });

  callout(s, "One problem. Three interventions. Each chapter adds one lever.", {
    x: 0.5, y: 4.72, w: 9, h: 0.38, bg: C.teal, fc: C.navy, fs: 14
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 4: CHAPTER 1 — Aerial Domain Shift
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  lightBg(s);
  sectionBar(s, "Chapter 1: Cross-Modal Representation Learning for Aerial Imagery");

  s.addText("Motivation: the most severe form of domain shift — different sensing physics", {
    x: 0.5, y: 0.85, w: 9, h: 0.35,
    fontSize: 13, italic: true, color: C.midGray
  });

  // UNICORN EO/SAR chip pairs figure
  s.addText("UNICORN: Co-registered EO + SAR chips (10 vehicle classes)", {
    x: 0.4, y: 1.28, w: 4.2, h: 0.3,
    fontSize: 10, bold: true, color: C.navy
  });
  addImageOrPlaceholder(s, fig("ch1_unicorn_chips.png"), 0.4, 1.62, 4.2, 2.1);

  // SAR→EO results
  s.addText("SAR→EO Translation (MAVIC-T)", {
    x: 0.4, y: 3.82, w: 4.2, h: 0.28,
    fontSize: 10, bold: true, color: C.navy
  });
  addImageOrPlaceholder(s, fig("ch1_sar2eo_comparison.png"), 0.4, 4.12, 4.2, 0.85);

  // Right: contributions
  s.addText("Key Contributions & Findings", {
    x: 4.65, y: 1.3, w: 5.0, h: 0.35,
    fontSize: 14, bold: true, color: C.navy
  });

  const contribs = [
    "MAVOC/MAVIC-C: SAR+EO classification benchmark series (CVPRW 2022–2024), geographic held-out splits exposed severe location overfitting",
    "MAVIC-T: Multi-directional translation (SAR→EO, SAR→RGB, SAR→IR, RGB→IR); SAR→RGB substantially harder than SAR→EO",
    "MAGIC dataset: Aligned SAR · RGB · IR stacks for cross-modal training & evaluation",
    "MINNIMAN: Magnetic anomaly map prediction from EO+hyperspectral via VQ-GAN; proof-of-concept for GPS-denied navigation",
  ];
  bullets(s, contribs, { x: 4.65, y: 1.75, w: 5.0, h: 3.15, fs: 12 });

  callout(s, "🔑  Appearance similarity did not predict transfer as strongly as expected; cross-modal supervision and shared structure mattered more than looking visually closer.", {
    x: 0.5, y: 5.02, w: 9, h: 0.36, bg: C.tealDim, fc: C.white, fs: 11
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 4b+4c (combined): MAVIC-T benchmark + MINNIMAN inference
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  lightBg(s);
  sectionBar(s, "Chapter 1: MAVIC-T (Benchmark) + MINNIMAN (Cross-Domain Inference)");

  // ── Left: MAVIC-T ─────────────────────────────────────────────
  s.addText("MAVIC-T — Open-Source Translation Benchmark", {
    x: 0.4, y: 0.85, w: 4.45, h: 0.3,
    fontSize: 11, bold: true, color: C.navy
  });
  addImageContainOrPlaceholder(s, fig("ch1_mavict_overview.png"), 0.4, 1.2, 4.45, 2.35);
  bullets(s, [
    "4 directions: SAR→EO, SAR→RGB, SAR→IR, RGB→IR  ·  MAGIC dataset",
    "SAR→RGB substantially harder — larger sensing physics gap",
    "Metric: L1 + LPIPS + FID  ·  Geographic disjoint splits exposed location overfitting",
  ], { x: 0.4, y: 3.6, w: 4.45, h: 1.65, fs: 11 });

  // ── Divider ───────────────────────────────────────────────────
  twoColDivider(s, 5.0);

  // ── Right: MINNIMAN ───────────────────────────────────────────
  s.addText("MINNIMAN — Magnetic Map Inference for GPS-Denied Nav", {
    x: 5.15, y: 0.85, w: 4.5, h: 0.3,
    fontSize: 11, bold: true, color: C.navy
  });
  addImageOrPlaceholder(s, fig("ch1_magnav_concept.png"), 5.15, 1.2, 4.5, 1.75);
  addImageOrPlaceholder(s, fig("ch1_minniman_arch.png"), 5.15, 3.0, 4.5, 0.9);
  bullets(s, [
    "Predicts magnetic anomaly maps from EO + hyperspectral inputs",
    "VQ-GAN codebook discretizes map space  ·  teacher-student for modality changes",
    "Proof-of-concept: same cross-modal logic, applied to navigation signals",
  ], { x: 5.15, y: 3.97, w: 4.5, h: 1.28, fs: 11 });

  callout(s, "MAVIC-T = open benchmark (CVPRW 2022–2024)  ·  MINNIMAN = our system  ·  Both published & complete.", {
    x: 0.4, y: 5.38, w: 9.2, h: 0.28, bg: C.tealDim, fc: C.white, fs: 11
  });
}
// ═══════════════════════════════════════════════════════════════
// SLIDE Ch1→Ch2 BRIDGE: The Labeled Data Bottleneck
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  darkBg(s);
  sectionBar(s, "From Chapter 1 to Chapter 2: Why Labeled Data Is Not Enough");

  // Left column: aerial sensing data challenge
  s.addText("The Aerial Sensing Challenge", {
    x: 0.5, y: 1.05, w: 4.5, h: 0.35,
    fontSize: 14, bold: true, color: C.teal
  });
  const aerialItems = [
    "Sensors capture at different resolutions, projection models, and ground footprints — spatial alignment is non-trivial",
    "Temporal co-registration requires the scene to be stable across passes — rare in practice",
    "MAGIC is one of the few open-source datasets spatially and temporally co-registered across SAR · RGB · IR · hyperspectral",
    "Despite this, scale and geographic diversity remain limited — and dense correspondence labels do not exist",
  ];
  bullets(s, aerialItems, { x: 0.5, y: 1.45, w: 4.45, h: 3.2, fs: 11, color: C.offWhite });

  twoColDivider(s);

  // Right column: broader correspondence labeling problem
  s.addText("Correspondence Labels Are Expensive Everywhere", {
    x: 5.2, y: 1.05, w: 4.5, h: 0.35,
    fontSize: 14, bold: true, color: C.gold
  });
  const broadItems = [
    "Dense optical flow: requires calibrated stereo rigs or LiDAR depth — hardware-constrained",
    "Semantic keypoints: human annotators per pair — slow, expensive, domain-specific",
    "Labels for driving ≠ labels for aerial ≠ labels for medical — no single dataset generalizes",
    "The bottleneck is not modeling — it is labeled data",
  ];
  bullets(s, broadItems, { x: 5.2, y: 1.45, w: 4.45, h: 3.2, fs: 11, color: C.offWhite });

  callout(s, "Chapter 2 Approach: generate unlimited labeled correspondence pairs synthetically — and ask whether controllable motion structure predicts real-world transfer", {
    x: 0.5, y: 4.9, w: 9, h: 0.5, bg: C.teal, fc: C.navy, fs: 12
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 8 (Ch2 opener): SDF-Fractal3D
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  lightBg(s);
  lightTitle(s, "Chapter 2: SDF-Fractal3D — A Controllable Synthetic Pipeline");

  // Key idea banner
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.95, w: 9, h: 0.48,
    fill: { color: C.navy }, line: { color: C.navy }
  });
  s.addText("Intentionally non-photorealistic — yet transfers strongly zero-shot across diverse real benchmarks", {
    x: 0.6, y: 0.95, w: 8.8, h: 0.48,
    fontSize: 13, bold: true, color: C.teal, valign: "middle"
  });

  // Real SDF fractal figures — 2694×892 (3.02:1); at w=4.3 correct h=1.42
  addImageOrPlaceholder(s, fig("ch2_sdf_fractal_grid.png"), 0.5, 1.55, 4.3, 1.42);

  // Right column properties
  s.addText("Design Principles", {
    x: 5.0, y: 1.55, w: 4.7, h: 0.38,
    fontSize: 14, bold: true, color: C.navy
  });
  bullets(s, [
    "Asset-free, procedurally scalable — no 3D assets or rendering pipelines",
    "Factorized: motion sampler × appearance stage are independent",
    "Motion sampler: camera viewpoints + controllable zoom distribution",
    "Appearance: procedural textures, independent lighting & backgrounds",
    "Allows motion-only interventions while holding appearance fixed",
    "2D Warp variant: same 3D correspondence field, warped photometry",
  ], { x: 5.0, y: 2.0, w: 4.7, h: 2.1, fs: 11 });

  // Bottom: contrast with existing synthetic datasets
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 4.2, w: 9, h: 0.05,
    fill: { color: C.teal }, line: { color: C.teal }
  });
  s.addText("Existing synthetic datasets require:", {
    x: 0.5, y: 4.28, w: 5.3, h: 0.26,
    fontSize: 11, bold: true, color: C.navy
  });

  const existingDatasets = [
    {
      label: "FlyingThings3D",
      desc: "Pre-rendered offline\n3D asset library download required\nPhoto-real textures & lighting",
    },
    {
      label: "Kubric",
      desc: "Pre-rendered offline\nAsset pipeline download required\nRealistic physics, reflections & shadows",
    },
  ];
  existingDatasets.forEach((ds, i) => {
    const bx = 0.52 + i * 2.72;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: bx, y: 4.58, w: 2.52, h: 1.05, rectRadius: 0.08,
      fill: { color: C.navyMid }, line: { color: C.red, width: 1.5 }
    });
    s.addText(ds.label, {
      x: bx + 0.1, y: 4.62, w: 2.32, h: 0.28,
      fontSize: 11, bold: true, color: C.red, valign: "middle"
    });
    s.addText(ds.desc, {
      x: bx + 0.1, y: 4.93, w: 2.32, h: 0.66,
      fontSize: 9.5, color: C.offWhite, wrap: true
    });
  });

  callout(s, "No real imagery, no 3D assets, no photorealistic textures. Does it transfer? Let's find out.", {
    x: 5.9, y: 4.28, w: 3.55, h: 1.35, bg: C.gold, fc: C.navy, fs: 12
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 11b: Zero-Shot Transfer Grid — surprising SDF result
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  darkBg(s);
  sectionBar(s, "Zero-Shot Transfer Grid: SDF-Fractal3D Competitive Despite Non-Photorealism");

  s.addText("PCK @ α=5%: a keypoint prediction is correct if within 5% of the image diagonal from ground-truth. RAFT + CATs++ averaged — zero-shot, no fine-tuning on any target.", {
    x: 0.5, y: 0.82, w: 9, h: 0.3,
    fontSize: 11, italic: true, color: C.muted
  });

  addImageOrPlaceholder(s, fig("table_transfer_grid.png"), 0.35, 1.18, 9.3, 2.5);

  // Four bold key-point annotations
  const kps = [
    { text: "SDF-Fractal3D (Zoom): 94.6 KITTI-15 — matches FlyingThings without real imagery", color: C.teal, x: 0.35, y: 3.78, w: 5.8 },
    { text: "SPair-only: strong semantic (PF-PASCAL) but weak optical flow — no single source dominates", color: C.gold, x: 0.35, y: 4.22, w: 5.8 },
    { text: "SDF-Fractal3D (Zoom): 46.3 TSS — non-photorealism doesn't prevent motion transfer", color: C.teal, x: 6.3, y: 3.78, w: 3.35 },
    { text: "FlyingThings wins some columns, SDF-Zoom wins others — realistic rendering alone is not the determining factor", color: C.muted, x: 6.3, y: 4.22, w: 3.35 },
  ];
  kps.forEach(kp => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: kp.x, y: kp.y, w: kp.w, h: 0.36, rectRadius: 0.05,
      fill: { color: C.navyMid }, line: { color: kp.color, width: 1.5 }
    });
    s.addText(kp.text, {
      x: kp.x + 0.1, y: kp.y, w: kp.w - 0.15, h: 0.36,
      fontSize: 9.5, color: C.offWhite, valign: "middle", wrap: true
    });
  });

  callout(s, "🔑  Fractals match photorealistic baselines on multiple benchmarks. Appearance is not the explanation — something else is driving transfer.", {
    x: 0.35, y: 4.67, w: 9.3, h: 0.38, bg: C.gold, fc: C.navy, fs: 11
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 8b: Training & Evaluation Datasets — the visual gap
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  lightBg(s);
  lightTitle(s, "What the Training & Evaluation Datasets Look Like");

  s.addText("SDF-Fractal3D has no real imagery — yet it just matched photorealistic baselines. Look at what the benchmarks actually look like.", {
    x: 0.5, y: 0.95, w: 9, h: 0.32,
    fontSize: 11.5, italic: true, color: C.midGray, wrap: true
  });

  // Row 1: Training sources
  s.addText("Training Sources", {
    x: 0.5, y: 1.35, w: 9, h: 0.26,
    fontSize: 11, bold: true, color: C.tealDim
  });

  const trainSrcs = [
    { label: "SPair-71k", sub: "20 object categories\nsparse semantic kpts", color: C.gold, file: "thumb_train_spair.png" },
    { label: "PointOdyssey", sub: "Synthetic video\ndense long-range flow", color: C.teal, file: "thumb_train_pointodyssey.png" },
    { label: "SDF-Fractal3D", sub: "Procedural 3D\nasset-free, no texture", color: C.tealDim, file: "thumb_train_sdf_fractal3d.png" },
    { label: "FlyingThings3D", sub: "Synthetic rendered\nscenes with 3D objects", color: C.muted, file: "thumb_train_flyingthings3d.png" },
  ];

  const cellW = 2.1, trainH = 1.3, evalH = 1.3;
  const gap = 0.1, startX = 0.45;

  trainSrcs.forEach((src, i) => {
    const x = startX + i * (cellW + gap);
    addImageOrPlaceholder(s, fig(src.file), x, 1.65, cellW, trainH - 0.28, src.sub);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 1.65 + trainH - 0.28, w: cellW, h: 0.26, rectRadius: 0.04,
      fill: { color: src.color }, line: { color: src.color }
    });
    s.addText(src.label, {
      x, y: 1.65 + trainH - 0.28, w: cellW, h: 0.26,
      fontSize: 8.5, bold: true, color: C.navy, align: "center", valign: "middle"
    });
  });

  // Red highlight box around SDF-Fractal3D (index 2 in trainSrcs)
  s.addShape(pres.shapes.RECTANGLE, {
    x: startX + 2*(cellW+gap) - 0.06, y: 1.59,
    w: cellW + 0.12, h: (trainH - 0.28 + 0.26) + 0.12,
    fill: { color: "FFFFFF", transparency: 100 },
    line: { color: C.red, width: 2.5 }
  });

  // Row 2: Evaluation targets
  s.addText("Evaluation Targets", {
    x: 0.5, y: 3.12, w: 9, h: 0.26,
    fontSize: 11, bold: true, color: C.red
  });

  const evalSrcs = [
    { label: "KITTI-2015", sub: "Real driving sequences\nLiDAR ground-truth flow", color: C.teal, file: "thumb_eval_kitti2015.png" },
    { label: "PF-PASCAL", sub: "PASCAL-VOC objects\nkeypoint annotations", color: C.gold, file: "thumb_eval_pfpascal.png" },
    { label: "PF-WILLOW", sub: "Willow-ObjectClass\nwildlife / sports kpts", color: C.gold, file: "thumb_eval_pfwillow.png" },
    { label: "TSS (3 subsets)", sub: "JODS + PASCAL + FG3DCar\ntemplate match sequences", color: C.tealDim, file: "thumb_eval_tss.png" },
  ];

  evalSrcs.forEach((ev, i) => {
    const x = startX + i * (cellW + gap);
    addImageOrPlaceholder(s, fig(ev.file), x, 3.42, cellW, evalH - 0.28, ev.sub);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 3.42 + evalH - 0.28, w: cellW, h: 0.26, rectRadius: 0.04,
      fill: { color: ev.color }, line: { color: ev.color }
    });
    s.addText(ev.label, {
      x, y: 3.42 + evalH - 0.28, w: cellW, h: 0.26,
      fontSize: 8.5, bold: true, color: C.navy, align: "center", valign: "middle"
    });
  });

  callout(s, "🔑  Fractals look nothing like any of these — no roads, no cars, no animals. If appearance isn't the signal, what is?", {
    x: 0.5, y: 4.87, w: 9, h: 0.38, bg: C.teal, fc: C.navy, fs: 12
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 6: BFV Descriptor  [moved before symmetric metrics]
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  lightBg(s);
  lightTitle(s, "Representing Datasets: Bag-of-Flow-Vectors (BFV)");

  // Left: formula and intuition — compact
  s.addText("Motion Descriptor", {
    x: 0.5, y: 1.0, w: 4.5, h: 0.35,
    fontSize: 14, bold: true, color: C.navy
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.38, w: 4.5, h: 0.72,
    fill: { color: C.navy }, line: { color: C.teal, width: 1.5 },
    shadow: makeShadow()
  });
  s.addText([
    { text: "f",    options: { bold: true, fontSize: 17, color: C.teal,     fontFace: "Courier New" } },
    { text: "flow", options: { bold: true, fontSize: 11, color: C.teal,     fontFace: "Courier New", subscript: true } },
    { text: " = [ ", options: { bold: true, fontSize: 17, color: C.offWhite, fontFace: "Courier New" } },
    { text: "x",    options: { bold: true, fontSize: 17, color: C.gold,     fontFace: "Courier New" } },
    { text: "src",  options: { fontSize: 11,             color: C.gold,     fontFace: "Courier New", subscript: true } },
    { text: ",  y", options: { bold: true, fontSize: 17, color: C.gold,     fontFace: "Courier New" } },
    { text: "src",  options: { fontSize: 11,             color: C.gold,     fontFace: "Courier New", subscript: true } },
    { text: ",  \u0394x,  \u0394y ]", options: { bold: true, fontSize: 17, color: C.teal, fontFace: "Courier New" } },
  ], { x: 0.6, y: 1.4, w: 4.3, h: 0.50, valign: "middle" });
  s.addText("All components normalized by image dimensions  ·  One 4-D sample per correspondence", {
    x: 0.5, y: 2.15, w: 4.5, h: 0.45,
    fontSize: 10, color: C.muted, wrap: true
  });

  bullets(s, [
    "Resolution-agnostic — works across datasets with different image sizes",
    "Captures where motion occurs AND how objects move",
    "No rigid spatial grid → avoids discretization artifacts",
    "Physically meaningful Euclidean scale → enables ε-coverage radius",
  ], { x: 0.5, y: 2.65, w: 4.5, h: 1.8, fs: 12 });

  s.addText("Appearance: patchwise DINOv2 embeddings projected onto unit hypersphere", {
    x: 0.5, y: 4.5, w: 4.5, h: 0.32,
    fontSize: 10, italic: true, color: C.midGray, wrap: true
  });

  // Right: semantic splat layout — training corpora vs benchmark/eval sets
  twoColDivider(s);
  s.addText("Representative BFV Splats", {
    x: 5.2, y: 1.0, w: 4.5, h: 0.28,
    fontSize: 12, bold: true, color: C.navy
  });

  const ps = 0.97, gp = 0.08, gx = 5.54;
  const trainY = 1.7, evalY = 3.18;

  const splatPanels = [
    {
      header: "Training corpora",
      y: trainY,
      cells: [
        { label: "SDF-Fractal3D", file: "train__sdf-fractal3d__directional_splat.png" },
        { label: "PointOdyssey",  file: "train__pointodyssey__directional_splat.png"   },
        { label: "SPair",         file: "train__spair__directional_splat.png"           },
        { label: "FlyingThings",  file: "train__flyingthings__directional_splat.png"    },
      ],
    },
    {
      header: "Benchmark / eval sets",
      y: evalY,
      cells: [
        { label: "KITTI-2012", file: "benchmark__kitti2012__directional_splat.png" },
        { label: "KITTI-2015", file: "benchmark__kitti2015__directional_splat.png" },
        { label: "PF-PASCAL",  file: "benchmark__pfpascal__directional_splat.png"  },
        { label: "PF-WILLOW",  file: "benchmark__pfwillow__directional_splat.png"  },
      ],
    },
  ];

  splatPanels.forEach((panel) => {
    s.addText(panel.header, {
      x: 5.42, y: panel.y - 0.22, w: 4.25, h: 0.16,
      fontSize: 8, bold: true, color: C.tealDeep
    });
    panel.cells.forEach((cell, idx) => {
      const px = gx + idx * (ps + gp);
      const py = panel.y;
      s.addShape(pres.shapes.RECTANGLE, {
        x: px, y: py, w: ps, h: ps,
        fill: { color: C.white }, line: { color: C.tealDim, width: 0.75 },
        shadow: makeShadow()
      });
      const dispSize = ps - 0.22;
      addImageOrPlaceholder(s, fig(cell.file),
        px + (ps - dispSize) / 2, py + 0.03, dispSize, dispSize);
      s.addText(cell.label, {
        x: px, y: py + ps - 0.19, w: ps, h: 0.17,
        fontSize: 6.6, bold: true, color: C.midGray, align: "center",
        fit: "shrink"
      });
    });
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 8.93, y: 1.12, w: 0.96, h: 0.47,
    fill: { color: C.navyMid }, line: { color: C.tealDim, width: 0.75 }
  });
  addImageOrPlaceholder(s, fig("legend__direction_colorwheel.png"), 8.99, 1.15, 0.31, 0.31);
  s.addText("Direction legend", {
    x: 9.32, y: 1.18, w: 0.5, h: 0.11,
    fontSize: 6.3, bold: true, color: C.muted
  });
  s.addText("Hue encodes dominant motion angle", {
    x: 9.32, y: 1.31, w: 0.5, h: 0.17,
    fontSize: 5.4, color: C.muted, valign: "mid", fit: "shrink"
  });

  callout(s, "Each dataset becomes a point cloud in BFV space — transferability becomes a geometric coverage problem.", {
    x: 0.5, y: 4.86, w: 4.45, h: 0.5, bg: C.tealDim, fc: C.white, fs: 11
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE Ch2-4: What explains transfer? Symmetric metrics fall short.
// ═══════════════════════════════════════════════════════════════
if (false) {
  let s = pres.addSlide();
  darkBg(s);
  sectionBar(s, "What Explains the Transfer? The Limits of Symmetric Metrics");

  s.addText("The standard approach: compute a symmetric distribution distance (MMD, FID) between training and target datasets.", {
    x: 0.5, y: 0.85, w: 9, h: 0.5,
    fontSize: 14, color: C.offWhite, wrap: true
  });

  s.addText("The problem:", {
    x: 0.5, y: 1.38, w: 9, h: 0.28,
    fontSize: 13, bold: true, color: C.gold
  });

  // Compact failure mode cards — shorter, side by side
  const failModes = [
    { title: "Under-coverage (E ⊆ε T)", body: "Target has motion modes absent from training.\nModel never encounters them → fails at test time.", color: C.red },
    { title: "Excess Mass (T ⊆ε E)", body: "Training contains irrelevant motion patterns.\nCapacity wasted; symmetric MMD can't distinguish this from case A.", color: C.gold },
  ];
  failModes.forEach((fm, i) => {
    const x = 0.4 + i * 4.8;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.72, w: 4.5, h: 1.35,
      fill: { color: C.navyMid }, line: { color: fm.color, width: 1.5 },
      shadow: makeShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.72, w: 4.5, h: 0.32,
      fill: { color: fm.color }, line: { color: fm.color }
    });
    s.addText(fm.title, {
      x: x+0.1, y: 1.72, w: 4.3, h: 0.32,
      fontSize: 11, bold: true, color: C.navy, valign: "middle"
    });
    s.addText(fm.body, {
      x: x+0.12, y: 2.08, w: 4.26, h: 0.9,
      fontSize: 11, color: C.offWhite, wrap: true
    });
  });

  // Real figure: directional vs symmetric concept — larger and better proportioned
  addImageOrPlaceholder(s, fig("ch2_directional_vs_symmetric_concept.png"), 0.35, 3.0, 9.3, 2.15);

  callout(s, "🔑  MMD² stays nearly constant across both cases — the directional terms swap, cleanly separating the two failure modes.", {
    x: 0.35, y: 5.22, w: 9.3, h: 0.3, bg: C.gold, fc: C.navy, fs: 11
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 7: Directed Coverage Framework
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  darkBg(s);
  sectionBar(s, "Directed Coverage: An Asymmetric Diagnostic Framework");

  // ── Central formula box (more prominent) ──────────────────────
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 0.82, w: 9.3, h: 0.72,
    fill: { color: C.navyMid }, line: { color: C.teal, width: 2 }
  });
  addImageOrPlaceholder(s, fig("equation_support_inclusion.png"), 0.78, 0.86, 5.9, 0.44);
  s.addText("HOF = Histogram of\nOptical Flow (angle bins)", {
    x: 6.8, y: 0.84, w: 2.7, h: 0.38,
    fontSize: 9, italic: true, color: C.muted, valign: "middle", wrap: true
  });
  s.addText("4 predictors: BFV Sε(Eval | Train) · BFV Sε(Train | Eval) · DINOv2 Sε(Eval | Train) · DINOv2 Sε(Train | Eval)", {
    x: 0.5, y: 1.22, w: 9, h: 0.28,
    fontSize: 10, italic: true, color: C.muted, valign: "middle"
  });

  // ── Two compact direction definitions ─────────────────────────
  const dirs = [
    {
      arrow: "Eval ⊆ε Train",
      title: "Target-Support View",
      body: "What fraction of eval BFV points have a training neighbor within ε?\nLow = model never encounters those motion modes → transfer fails.",
      color: C.red,
    },
    {
      arrow: "Train ⊆ε Eval",
      title: "Off-Target-Mass View",
      body: "What fraction of training BFV points have an eval neighbor within ε?\nLow = training wastes capacity on off-target motion patterns.",
      color: C.gold,
    },
  ];
  dirs.forEach((d, i) => {
    const x = 0.35 + i * 4.75;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.6, w: 4.5, h: 1.3,
      fill: { color: C.navyMid }, line: { color: d.color, width: 1.5 }
    });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: x+0.95, y: 1.67, w: 2.6, h: 0.45, rectRadius: 0.08,
      fill: { color: d.color }, line: { color: d.color }
    });
    s.addText(d.arrow, {
      x: x+0.95, y: 1.67, w: 2.6, h: 0.45,
      fontSize: 16, bold: true, color: C.navy, align: "center", valign: "middle"
    });
    s.addText(d.title, {
      x: x+0.12, y: 2.18, w: 4.25, h: 0.28,
      fontSize: 12, bold: true, color: C.white
    });
    s.addText(d.body, {
      x: x+0.12, y: 2.48, w: 4.25, h: 0.38,
      fontSize: 10, color: C.muted, wrap: true
    });
  });

  // ── "The 4 Coverage Regimes" heading ──────────────────────────
  s.addText("The 4 Coverage Regimes:", {
    x: 0.35, y: 3.02, w: 9.3, h: 0.28,
    fontSize: 13, bold: true, color: C.gold
  });

  const cases = [
    {
      et: "High", te: "High", title: "Well-Aligned",
      desc: "Training covers eval and is efficiently used. Best transfer outcome.",
      color: C.green, file: "fig_coverage_regime__well_aligned.png"
    },
    {
      et: "High", te: "Low", title: "Excess Train Mass",
      desc: "Eval is covered but training has many irrelevant motions. Wasteful but OK transfer.",
      color: C.gold, file: "fig_coverage_regime__excess_train_mass.png"
    },
    {
      et: "Low", te: "High", title: "Eval Under-Covered",
      desc: "Eval requires motion modes absent from training. Transfer will fail.",
      color: C.red, file: "fig_coverage_regime__eval_under_covered.png"
    },
    {
      et: "Low", te: "Low", title: "Complete Mismatch",
      desc: "Neither direction covered. Severe domain gap.",
      color: "993333", file: "fig_coverage_regime__complete_mismatch.png"
    },
  ];
  cases.forEach((c, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.35 + col * 4.75, y = 3.30 + row * 1.08;
    addImageOrPlaceholder(s, fig(c.file), x, y, 2.2, 0.94);
    s.addShape(pres.shapes.RECTANGLE, {
      x: x + 2.3, y, w: 2.2, h: 0.94,
      fill: { color: C.navyMid }, line: { color: c.color, width: 1.3 }
    });
    s.addText(`Eval ⊆ε Train: ${c.et}  ·  Train ⊆ε Eval: ${c.te}`, {
      x: x + 2.42, y: y + 0.08, w: 1.95, h: 0.18,
      fontSize: 7.2, color: c.color, bold: true
    });
    s.addText(c.title, {
      x: x + 2.42, y: y + 0.30, w: 1.95, h: 0.18,
      fontSize: 11, bold: true, color: C.white
    });
    s.addText(c.desc, {
      x: x + 2.42, y: y + 0.50, w: 1.95, h: 0.30,
      fontSize: 8.4, color: C.muted, wrap: true
    });
  });

  callout(s, "🔑  Directed support inclusion separates failure modes that KL-div and MMD conflate. KL-div is also numerically unstable (requires density estimation); ε-coverage operates directly on sample clouds. Eval ⊆ε Train is the primary predictor of transfer failure.", {
    x: 0.35, y: 5.38, w: 9.3, h: 0.38, bg: C.teal, fc: C.navy, fs: 10
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 6: BFV Descriptor
// ═══════════════════════════════════════════════════════════════
if (false) {
  let s = pres.addSlide();
  lightBg(s);
  lightTitle(s, "Representing Datasets: Bag-of-Flow-Vectors (BFV)");

  // Left: formula and intuition — compact
  s.addText("Motion Descriptor", {
    x: 0.5, y: 1.0, w: 4.5, h: 0.35,
    fontSize: 14, bold: true, color: C.navy
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.38, w: 4.5, h: 0.72,
    fill: { color: C.navy }, line: { color: C.teal, width: 1.5 },
    shadow: makeShadow()
  });
  s.addText("f_flow = [ x̂,  ŷ,  Δx̂,  Δŷ ]", {
    x: 0.6, y: 1.4, w: 4.3, h: 0.46,
    fontSize: 17, bold: true, color: C.teal, fontFace: "Courier New",
    shrinkText: true
  });
  s.addText("Normalized source position  +  normalized displacement  ·  One 4-D sample per correspondence", {
    x: 0.5, y: 2.15, w: 4.5, h: 0.45,
    fontSize: 10, color: C.muted, wrap: true
  });

  bullets(s, [
    "Resolution-agnostic — works across datasets with different image sizes",
    "Captures where motion occurs AND how objects move",
    "No rigid spatial grid → avoids discretization artifacts",
    "Physically meaningful Euclidean scale → enables ε-coverage radius",
  ], { x: 0.5, y: 2.65, w: 4.5, h: 2.15, fs: 12 });

  // Right: semantic splat layout — training corpora vs benchmark/eval sets
  // All images are 560×560 (1:1 square)
  twoColDivider(s);
  s.addText("Representative BFV Splats", {
    x: 5.2, y: 1.0, w: 4.5, h: 0.28,
    fontSize: 12, bold: true, color: C.navy
  });

  const ps = 0.97, gp = 0.08, gx = 5.54;
  const trainY = 1.7, evalY = 3.18;

  const splatPanels = [
    {
      header: "Training corpora",
      y: trainY,
      cells: [
        { label: "SDF-Fractal3D", file: "train__sdf-fractal3d__directional_splat.png" },
        { label: "PointOdyssey",  file: "train__pointodyssey__directional_splat.png"   },
        { label: "SPair",         file: "train__spair__directional_splat.png"           },
        { label: "FlyingThings",  file: "train__flyingthings__directional_splat.png"    },
      ],
    },
    {
      header: "Benchmark / eval sets",
      y: evalY,
      cells: [
        { label: "KITTI-2012", file: "benchmark__kitti2012__directional_splat.png" },
        { label: "KITTI-2015", file: "benchmark__kitti2015__directional_splat.png" },
        { label: "PF-PASCAL",  file: "benchmark__pfpascal__directional_splat.png"  },
        { label: "PF-WILLOW",  file: "benchmark__pfwillow__directional_splat.png"  },
      ],
    },
  ];

  splatPanels.forEach((panel) => {
    s.addText(panel.header, {
      x: 5.42, y: panel.y - 0.22, w: 4.25, h: 0.16,
      fontSize: 8, bold: true, color: C.tealDeep
    });

    panel.cells.forEach((cell, idx) => {
      const px = gx + idx * (ps + gp);
      const py = panel.y;
      s.addShape(pres.shapes.RECTANGLE, {
        x: px, y: py, w: ps, h: ps,
        fill: { color: C.white }, line: { color: C.tealDim, width: 0.75 },
        shadow: makeShadow()
      });
      const dispSize = ps - 0.22;
      addImageOrPlaceholder(s, fig(cell.file),
        px + (ps - dispSize) / 2, py + 0.03, dispSize, dispSize);
      s.addText(cell.label, {
        x: px, y: py + ps - 0.19, w: ps, h: 0.17,
        fontSize: 6.6, bold: true, color: C.midGray, align: "center",
        fit: "shrink"
      });
    });
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 8.93, y: 1.12, w: 0.96, h: 0.47,
    fill: { color: C.navyMid }, line: { color: C.tealDim, width: 0.75 }
  });
  addImageOrPlaceholder(s, fig("legend__direction_colorwheel.png"), 8.99, 1.15, 0.31, 0.31);
  s.addText("Direction legend", {
    x: 9.32, y: 1.18, w: 0.5, h: 0.11,
    fontSize: 6.3, bold: true, color: C.muted
  });
  s.addText("Hue encodes dominant motion angle", {
    x: 9.32, y: 1.31, w: 0.5, h: 0.17,
    fontSize: 5.4, color: C.muted, valign: "mid", fit: "shrink"
  });

  callout(s, "Each dataset becomes a point cloud in BFV space — transferability becomes a geometric coverage problem.", {
    x: 0.5, y: 4.86, w: 4.45, h: 0.5, bg: C.tealDim, fc: C.white, fs: 11
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 10b: Motion Tuning Table
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  lightBg(s);
  lightTitle(s, "Utility-Guided Motion Interventions: Coverage Predicts Transfer");

  s.addText("Appearance held fixed (‖ΔA‖ ≈ 0) — only the motion sampler varies. Each intervention is designed to target specific benchmark motion distributions.", {
    x: 0.5, y: 0.95, w: 9, h: 0.32,
    fontSize: 12, italic: true, color: C.midGray
  });

  // Left: 3 benchmark BFV splats stacked vertically — one per table section
  const benchSplats = [
    { label: "KITTI-2015", file: "benchmark__kitti2015__directional_splat.png" },
    { label: "TSS",        file: "benchmark__tss__directional_splat.png"        },
    { label: "PF-PASCAL",  file: "benchmark__pfpascal__directional_splat.png"   },
  ];
  const sW = 2.5, sH = 1.0, sLabelH = 0.2, sGapY = 0.04, sX0 = 0.35, sY0 = 1.3;
  benchSplats.forEach((bs, i) => {
    const by = sY0 + i * (sH + sLabelH + sGapY);
    addImageOrPlaceholder(s, fig(bs.file), sX0, by, sW, sH, "");
    s.addText(bs.label, {
      x: sX0, y: by + sH + 0.03, w: sW, h: sLabelH - 0.03,
      fontSize: 9, bold: true, color: C.teal, align: "center"
    });
  });

  // Divider — narrower left column, table gets most of the width
  twoColDivider(s, 2.97);

  // Right: table panel — wider
  s.addText("Base / Zoom / Flip — PCK @ α=5%", {
    x: 3.1, y: 1.3, w: 6.55, h: 0.22,
    fontSize: 10, bold: true, color: C.teal
  });
  addImageOrPlaceholder(s, fig("table_motion_tuning.png"), 3.1, 1.57, 6.55, 3.41);

  callout(s, "🔑  Motion is the causal lever: zoom into target motion space → PCK rises; flip mismatch → PCK drops sharply. Appearance is constant throughout — BFV coverage is an actionable intervention signal.", {
    x: 0.35, y: 5.1, w: 9.3, h: 0.42, bg: C.navy, fc: C.teal, fs: 11
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE PCK: Evaluation Primer — What is PCK?
// ═══════════════════════════════════════════════════════════════
if (false) {
  let s = pres.addSlide();
  darkBg(s);
  sectionBar(s, "Evaluation Metric: Percentage of Correct Keypoints (PCK)");

  // Central formula box
  s.addShape(pres.shapes.RECTANGLE, {
    x: 1.0, y: 0.9, w: 8, h: 1.05,
    fill: { color: C.navyMid }, line: { color: C.teal, width: 2 }
  });
  s.addText("PCK @ α = 5%", {
    x: 1.1, y: 0.93, w: 7.8, h: 0.42,
    fontSize: 24, bold: true, color: C.teal, align: "center"
  });
  s.addText("A predicted keypoint is correct if it falls within α × max(H, W) pixels of the ground-truth location", {
    x: 1.1, y: 1.37, w: 7.8, h: 0.5,
    fontSize: 12, color: C.offWhite, align: "center", wrap: true
  });

  // Benchmark overview table
  s.addText("Benchmarks used in this dissertation:", {
    x: 0.5, y: 2.1, w: 9, h: 0.3,
    fontSize: 13, bold: true, color: C.gold
  });

  const benchmarks = [
    { name: "KITTI-2012 / 2015",  type: "Optical Flow",     domain: "Real driving, car-mounted camera",  color: C.teal },
    { name: "PF-PASCAL",          type: "Semantic Matching", domain: "PASCAL-VOC objects, keypoint pairs", color: C.gold },
    { name: "PF-WILLOW",          type: "Semantic Matching", domain: "Wildlife / sports categories",       color: C.gold },
    { name: "TSS",                type: "Template Matching", domain: "JODS + PASCAL + FG3DCar sequences",  color: C.tealDim },
    { name: "PointOdyssey (eval)","type": "Dense Tracking",  domain: "Long-range synthetic trajectories",  color: C.muted },
  ];

  benchmarks.forEach((b, i) => {
    const y = 2.5 + i * 0.54;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: y+0.04, w: 0.12, h: 0.38,
      fill: { color: b.color }, line: { color: b.color }
    });
    s.addText(b.name, {
      x: 0.75, y, w: 2.65, h: 0.46,
      fontSize: 12, bold: true, color: C.white, valign: "middle"
    });
    s.addText(b.type, {
      x: 3.5, y, w: 2.0, h: 0.46,
      fontSize: 10, italic: true, color: b.color, valign: "middle"
    });
    s.addText(b.domain, {
      x: 5.6, y, w: 4.0, h: 0.46,
      fontSize: 10, color: C.muted, valign: "middle"
    });
  });

  callout(s, "Higher PCK = better. All results at α=5%: a prediction within 5% of image diagonal counts as correct. No fine-tuning on any target benchmark.", {
    x: 0.5, y: 5.2, w: 9, h: 0.38, bg: C.tealDim, fc: C.white, fs: 11
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 9: Transferability Estimator Results
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  darkBg(s);
  sectionBar(s, "Transferability Estimator: Predicting Transfer Without Retraining");

  s.addText("Ridge-regularized linear model · predicts within-context residual performance · evaluated under 3 leakage-safe protocols", {
    x: 0.5, y: 0.82, w: 9, h: 0.4,
    fontSize: 12, italic: true, color: C.muted
  });

  // Key result: big number callouts
  const stats = [
    { val: "0.70", label: "Pairwise Ranking\nAccuracy (c-index)", sub: "vs. 0.50 chance baseline" },
    { val: "E ⊆ε T", label: "Most Consistent\nPredictor", sub: "Eval subset of training — best across all 3 held-out protocols" },
    { val: "3", label: "Strict Held-out\nProtocols", sub: "Train, Eval, Both" },
  ];
  const statColors = [C.teal, C.gold, C.midGray];
  stats.forEach((st, i) => {
    const x = 0.5 + i * 3.05;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y: 1.3, w: 2.8, h: 1.7,
      fill: { color: C.navyMid }, line: { color: statColors[i], width: 2 },
      shadow: makeShadow()
    });
    s.addText(st.val, {
      x, y: 1.35, w: 2.8, h: 0.85,
      fontSize: 52, bold: true, color: statColors[i], align: "center"
    });
    s.addText(st.label, {
      x, y: 2.15, w: 2.8, h: 0.55,
      fontSize: 11, color: C.white, align: "center", wrap: true
    });
    s.addText(st.sub, {
      x, y: 2.7, w: 2.8, h: 0.25,
      fontSize: 9, italic: true, color: C.muted, align: "center"
    });
  });

  // Real transferability scatter plot
  addImageOrPlaceholder(s, fig("ch2_transferability_scatter.png"), 0.5, 3.05, 4.5, 2.45);

  // Ranking table — replaces Key Findings bullets
  addImageOrPlaceholder(s, fig("table_ranking.png"), 5.2, 3.05, 4.5, 2.5);
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 10: Chapter 2 Summary
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  lightBg(s);
  lightTitle(s, "Chapter 2 Summary & Bridge to Chapter 3");

  // Left: contributions
  s.addText("Contributions (ECCV 2026, under review)", {
    x: 0.5, y: 1.05, w: 4.5, h: 0.35,
    fontSize: 13, bold: true, color: C.navy
  });
  bullets(s, [
    "BFV motion descriptor: resolution-agnostic 4D representation in Euclidean space",
    "Directed coverage (Eval ⊆ε Train, Train ⊆ε Eval): separates missing support from off-target mass — symmetric MMD cannot",
    "SDF-Fractal3D: non-photorealistic synthetic pipeline; motion structure alone drives transfer",
    "Transferability estimator: 0.70 pairwise ranking accuracy across 3 strict held-out protocols",
    "Diagnostics support intervention — same signal drives Chapter 3 curation",
  ], { x: 0.5, y: 1.48, w: 4.5, h: 3.5, fs: 12 });

  twoColDivider(s);

  // Right: bridge
  s.addText("Bridge to Chapter 3", {
    x: 5.2, y: 1.05, w: 4.5, h: 0.35,
    fontSize: 13, bold: true, color: C.navy
  });
  s.addText("Ch. 2 asks at the dataset level:\n\"Does dataset A transfer better than B to target T?\"", {
    x: 5.2, y: 1.48, w: 4.5, h: 0.85,
    fontSize: 12, color: C.darkText, wrap: true
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 2.42, w: 4.5, h: 1.0,
    fill: { color: C.navy }, line: { color: C.teal, width: 1.5 }
  });
  s.addText("Ch. 3 asks at the pair level:\n\"Which specific training pairs from a mixed pool best support transfer to multiple targets simultaneously?\"", {
    x: 5.35, y: 2.47, w: 4.2, h: 0.9,
    fontSize: 12, italic: true, color: C.teal, wrap: true
  });
  bullets(s, [
    "ClusterCov is the first probe: does BFV coverage-based selection help at all?",
    "GIST is the full framework: BFV ε-coverage utility + diversity + approximation guarantee",
    "Eval ⊆ε Train is the reward signal — measures missing target support",
    "λ sweep cached offline — no retraining to explore Pareto frontier",
  ], { x: 5.2, y: 3.55, w: 4.5, h: 1.85, fs: 12 });

  callout(s, "🔑  The diagnostic from Ch. 2 becomes the objective function in Ch. 3.", {
    x: 0.5, y: 5.1, w: 9, h: 0.38, bg: C.teal, fc: C.navy, fs: 12
  });
}
// ═══════════════════════════════════════════════════════════════
// SLIDE Ch3: Chapter 3 Opening — Data Curation
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  darkBg(s);
  sectionBar(s, "Chapter 3: Data Curation for Dense Correspondence");

  s.addText("Which training pairs — from a single source or across many — best support transfer to target benchmarks?", {
    x: 0.5, y: 0.85, w: 9, h: 0.45,
    fontSize: 15, color: C.offWhite, wrap: true
  });

  // Two-column framing: homogeneous first, then heterogeneous
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.35, y: 1.38, w: 4.25, h: 2.55,
    fill: { color: C.navyMid }, line: { color: C.teal, width: 1.5 }
  });
  s.addText("Even within a single source", {
    x: 0.5, y: 1.45, w: 4.0, h: 0.32,
    fontSize: 12, bold: true, color: C.teal
  });
  bullets(s, [
    "PointOdyssey: 3 clips × ~1,800 frames → ~4M candidate pairs",
    "Consecutive frames share nearly identical flow statistics",
    "Exhaustive training is impractical — massive within-source redundancy",
    "ClusterCov is the first probe: does motion-space selection beat random?",
  ], { x: 0.5, y: 1.82, w: 4.0, h: 2.0, fs: 11, color: C.offWhite });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 4.8, y: 1.38, w: 4.85, h: 2.55,
    fill: { color: C.navyMid }, line: { color: C.gold, width: 1.5 }
  });
  s.addText("Even harder across heterogeneous sources", {
    x: 4.95, y: 1.45, w: 4.55, h: 0.32,
    fontSize: 12, bold: true, color: C.gold
  });
  bullets(s, [
    "Pool: PointOdyssey (98.7%) · SPair-71k (1.2%) · PF-PASCAL (0.07%)",
    "Dense flow vs. sparse keypoints: incomparable supervision density",
    "Naive pooling over-rewards densest source by volume not usefulness",
    "No principled cross-source budget allocation without explicit criteria",
  ], { x: 4.95, y: 1.82, w: 4.55, h: 2.0, fs: 11, color: C.offWhite });

  // Pool composition summary cards
  s.addText("Candidate pool composition:", {
    x: 0.5, y: 4.0, w: 9.0, h: 0.22,
    fontSize: 10, color: C.muted, align: "center"
  });
  const poolCards = [
    { x: 1.1, w: 3.3, color: C.teal, title: "PointOdyssey", pct: "98.7%", total: "~4.3M pairs" },
    { x: 4.55, w: 2.2, color: C.gold, title: "SPair-71k", pct: "1.2%", total: "~52k pairs" },
    { x: 6.95, w: 2.0, color: C.red, title: "PF-PASCAL", pct: "0.07%", total: "~3k pairs" },
  ];
  poolCards.forEach(card => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: card.x, y: 4.24, w: card.w, h: 0.68, rectRadius: 0.05,
      fill: { color: C.navyMid }, line: { color: card.color, width: 1.5 }
    });
    s.addText(card.title, {
      x: card.x, y: 4.3, w: card.w, h: 0.18,
      fontSize: 9.5, bold: true, color: card.color, align: "center"
    });
    s.addText(`${card.pct}  |  ${card.total}`, {
      x: card.x + 0.05, y: 4.52, w: card.w - 0.1, h: 0.18,
      fontSize: 8.5, color: C.offWhite, align: "center"
    });
  });

  // Why it matters
  bullets(s, [
    "ClusterCov establishes that coverage-based curation is worth doing",
    "GIST is needed to add guarantees, explicit diversity control, and principled heterogeneous allocation",
  ], { x: 0.5, y: 5.0, w: 9.1, h: 0.58, fs: 11, color: C.offWhite });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 12: Cluster-Coverage Selection (ClusterCov)
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  lightBg(s);
  lightTitle(s, "ClusterCov: Preliminary Coverage Probe");

  s.addText("Preliminary selector: test whether BFV coverage alone is already enough to beat random curation.", {
    x: 0.5, y: 0.95, w: 9, h: 0.38,
    fontSize: 13, italic: true, color: C.midGray
  });

  // Left: original workflow
  const steps = [
    { n: "1", t: "Cluster target BFV space", d: "K-means on target benchmark correspondences → motion cluster centroids" },
    { n: "2", t: "Score candidate pairs", d: "Each training pair scores by how many uncovered target clusters it reaches within radius ε" },
    { n: "3", t: "Greedy selection", d: "Iteratively select highest-scoring pair, mark covered clusters, repeat to budget" },
    { n: "4", t: "Normalize by supervision density", d: "Divide coverage gain by # valid correspondences in pair → pragmatic probe, not a formal guarantee" },
  ];
  steps.forEach((st, i) => {
    const y = 1.5 + i * 0.86;
    s.addShape(pres.shapes.OVAL, {
      x: 0.28, y: y + 0.03, w: 0.54, h: 0.54,
      fill: { color: C.teal }, line: { color: C.teal }
    });
    s.addText(st.n, {
      x: 0.28, y: y + 0.03, w: 0.54, h: 0.54,
      fontSize: 16, bold: true, color: C.navy, align: "center", valign: "middle"
    });
    s.addText(st.t, {
      x: 0.98, y: y - 0.01, w: 2.45, h: 0.28,
      fontSize: 12.2, bold: true, color: C.navy
    });
    s.addText(st.d, {
      x: 0.98, y: y + 0.27, w: 2.45, h: 0.4,
      fontSize: 9.5, color: C.midGray
    });
    if (i < steps.length - 1) {
      s.addShape(pres.shapes.RECTANGLE, {
        x: 0.505, y: y + 0.55, w: 0.07, h: 0.27,
        fill: { color: C.teal }, line: { color: C.teal }
      });
    }
  });

  // Right: compact visual probe
  s.addText("Visual intuition", {
    x: 3.6, y: 1.42, w: 5.6, h: 0.25,
    fontSize: 11.5, bold: true, color: C.midGray
  });
  addImageOrPlaceholder(s, fig("fig_clustercov_probe.png"), 3.55, 1.7, 6.05, 3.62);
  s.addText("Cluster target motion modes, score new coverage, then grow support across modes greedily.", {
    x: 3.7, y: 5.35, w: 5.75, h: 0.28,
    fontSize: 10.5, italic: true, color: C.midGray, align: "center"
  });

  callout(s, "🔑  ClusterCov answers the first question: yes, coverage-based selection beats random. But clustering, heuristic normalization, and cross-source allocation are still unresolved.", {
    x: 0.5, y: 5.62, w: 9, h: 0.3, bg: C.navy, fc: C.teal, fs: 11.2
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 13: Homogeneous Source Efficiency Ablations
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  darkBg(s);
  sectionBar(s, "Source Ablations: ClusterCov vs. Random Within Each Source");

  s.addText("ClusterCov outperforms random at matched budgets even within a single source — motion-aware selection is sample-efficient · PCK @ α=10%", {
    x: 0.5, y: 0.82, w: 9, h: 0.38,
    fontSize: 12, italic: true, color: C.muted
  });

  // Side-by-side PointOdyssey (blue) and SPair (orange)
  s.addText("PointOdyssey pool", {
    x: 0.35, y: 1.28, w: 4.7, h: 0.28,
    fontSize: 11, bold: true, color: C.teal
  });
  addImageOrPlaceholder(s, fig("homogeneous_ablation_efficiency_pointodyssey.png"), 0.35, 1.6, 4.7, 3.5);

  s.addText("SPair pool", {
    x: 5.25, y: 1.28, w: 4.4, h: 0.28,
    fontSize: 11, bold: true, color: C.gold
  });
  addImageOrPlaceholder(s, fig("homogeneous_ablation_efficiency_spair.png"), 5.25, 1.6, 4.4, 3.5);

  callout(s, "Counterintuitive: BFV coverage beats random on SPair semantic benchmarks — motion geometry encodes cross-domain compatibility even for keypoint data.", {
    x: 0.35, y: 5.18, w: 9.3, h: 0.38, bg: C.gold, fc: C.navy, fs: 10
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 13b: Preliminary Results — Converged PCK
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  darkBg(s);
  sectionBar(s, "Preliminary Results: Converged PCK @ α=10% Across All Benchmarks");

  // Real converged summary bar chart
  addImageOrPlaceholder(s, fig("validation_converged_summary_last8_auto.png"), 0.35, 0.82, 9.3, 3.1);

  // Key findings
  s.addText("What the results show:", {
    x: 0.5, y: 4.05, w: 9, h: 0.32,
    fontSize: 13, bold: true, color: C.gold
  });
  const findings = [
    "Joint ClusterCov + n_valid leads on TSS and is competitive across all 5 benchmarks",
    "Single-source baselines (PF-PASCAL-only, PointOdyssey-only) confirm strong Pareto tradeoff",
    "Hand-tuned mixed baseline competitive on semantic — but requires manual source proportions",
    "Pooled random substantially below all ClusterCov variants — selection matters",
  ];
  const fRuns = findings.map((f, i) => ({
    text: f,
    options: { bullet: true, breakLine: i < findings.length-1, fontSize: 11.5, color: C.offWhite, paraSpaceAfter: 5 }
  }));
  s.addText(fRuns, { x: 0.5, y: 4.42, w: 9, h: 1.05, valign: "top" });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 13c: Validation PCK @ α=10% Convergence Curves  [HIDDEN — backup only]
// ═══════════════════════════════════════════════════════════════
if (false) {
  let s = pres.addSlide();
  darkBg(s);
  sectionBar(s, "Validation PCK @ α=10% Convergence Curves");

  addImageOrPlaceholder(s, fig("ch3_clustercov_pck_comparison.png"), 0.35, 0.82, 9.3, 4.5);

  callout(s, "Joint ClusterCov + n_valid shows consistently strong TSS/PF-WILLOW; PointOdyssey-only never recovers on semantic targets.", {
    x: 0.35, y: 5.38, w: 9.3, h: 0.27, bg: C.teal, fc: C.navy, fs: 11
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 14: The Cross-Source Budget Allocation Problem
// ═══════════════════════════════════════════════════════════════
if (false) {
  let s = pres.addSlide();
  lightBg(s);
  lightTitle(s, "Cross-Source Budget Allocation");

  s.addText("ClusterCov works as a probe, but it does not provide principled heterogeneous allocation across sparse and dense sources.", {
    x: 0.5, y: 0.95, w: 9, h: 0.38,
    fontSize: 13, italic: true, color: C.midGray
  });

  // Density mismatch figure — replaces text diagnosis, tells the story visually
  addImageOrPlaceholder(s, fig("fig_density_mismatch.png"), 0.5, 1.42, 9, 3.45);

  callout(s, "Hand-balanced run (2K PF-PASCAL + 9K SPair + 9K PointOdyssey) achieves superior semantic perf — confirming that allocation is the missing ingredient the full GIST formulation must solve.", {
    x: 0.5, y: 5.1, w: 9, h: 0.4, bg: C.teal, fc: C.navy, fs: 11
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 15: Source Routing Results
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  lightBg(s);
  lightTitle(s, "Multi-Target Selector: Automatic Source Routing");

  s.addText("Deduplicated union of per-target ClusterCov selections. Routing is learned from motion descriptors alone — no class labels, no appearance features, no source-identity flags.", {
    x: 0.5, y: 0.95, w: 9, h: 0.38,
    fontSize: 12, italic: true, color: C.midGray
  });

  // Table with real per-target routing counts from analysis dumps
  const tableData = [
    [
      { text: "Target", options: { bold: true, color: "FFFFFF", fill: { color: C.navy } } },
      { text: "PointOdyssey", options: { bold: true, color: "FFFFFF", fill: { color: C.navy } } },
      { text: "SPair-71k", options: { bold: true, color: "FFFFFF", fill: { color: C.navy } } },
      { text: "PF-PASCAL", options: { bold: true, color: "FFFFFF", fill: { color: C.navy } } },
    ],
    ["Candidate pool", "98.7%", "1.2%", "0.07%"],
    ["Dedup union (22,495)", "8,386 (37.3%)", "13,129 (58.4%)", "980 (4.4%)"],
    [{ text:"KITTI-2012 (4,499)", options:{bold:true}}, { text:"2,546 (56.6%)", options:{color:C.tealDim,bold:true}}, "1,598 (35.5%)", "355 (7.9%)"],
    [{ text:"KITTI-2015 (4,499)", options:{bold:true}}, { text:"2,376 (52.8%)", options:{color:C.tealDim,bold:true}}, "1,924 (42.8%)", "199 (4.4%)"],
    [{ text:"PF-PASCAL (4,499)", options:{bold:true}}, "1,867 (41.5%)", { text:"2,413 (53.6%)", options:{color:"B8860B",bold:true}}, "219 (4.9%)"],
    [{ text:"PF-WILLOW (4,499)", options:{bold:true}}, "911 (20.2%)", { text:"3,527 (78.4%)", options:{color:"B8860B",bold:true}}, "61 (1.4%)"],
    [{ text:"TSS (4,499)", options:{bold:true}}, "686 (15.2%)", { text:"3,667 (81.5%)", options:{color:"B8860B",bold:true}}, "146 (3.2%)"],
  ];
  s.addTable(tableData, {
    x: 0.5, y: 1.38, w: 9, h: 2.95,
    fontSize: 11,
    border: { pt: 0.5, color: "C5D3E0" },
    colW: [2.25, 2.25, 2.25, 2.25],
    rowH: 0.34,
  });

  s.addText("Comparison: per-target routing is against a highly imbalanced candidate pool, but the selected union becomes majority semantic.", {
    x: 0.5, y: 4.46, w: 9, h: 0.24,
    fontSize: 9.5, italic: true, color: C.midGray
  });

  callout(s, "🔑  Despite a 98.7% PointOdyssey candidate pool, the deduplicated selector shifts to 58.4% SPair. Semantic targets pull semantic sources; geometric targets pull synthetic video — emergent from motion geometry alone.", {
    x: 0.5, y: 4.82, w: 9, h: 0.5, bg: C.navy, fc: C.teal, fs: 11.5
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 16: GIST Framework — Formal Setup
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  darkBg(s);
  sectionBar(s, "GIST: Formal Coverage + Diversity Selection for Correspondence Training Sets");

  s.addText("ClusterCov proved coverage-based selection helps — but required k-means clustering and heuristic normalization. GIST replaces both with a formal objective: directed BFV ε-coverage + source-partitioned diversity, with a (½−η) approximation guarantee that extends cleanly to heterogeneous pools.", {
    x: 0.5, y: 0.85, w: 9, h: 0.32,
    fontSize: 12, italic: true, color: C.muted, wrap: true
  });

  // GIST illustration
  addImageOrPlaceholder(s, fig("fig_gist_illustration.png"), 0.35, 1.22, 9.3, 2.55);

  // Pool composition bar — concrete numbers
  s.addText("Heterogeneous pool composition:", {
    x: 0.5, y: 3.87, w: 3.2, h: 0.25,
    fontSize: 10, bold: true, color: C.gold
  });
  const poolBars = [
    { label: "PointOdyssey 98.7%  (~4.3M pairs)", frac: 8.87, color: C.teal },
    { label: "SPair-71k 1.2%  (~52k pairs)",       frac: 0.108, color: C.gold },
    { label: "PF-PASCAL 0.07%  (~3k pairs)",        frac: 0.006, color: C.red },
  ];
  const barTotalW = 6.8, barX0 = 3.0, barY = 3.88, barH = 0.28;
  poolBars.forEach((b, i) => {
    const w = Math.max((b.frac / 9.0) * barTotalW, 0.14);
    const x = i === 0 ? barX0 :
              i === 1 ? barX0 + (poolBars[0].frac / 9.0) * barTotalW :
                        barX0 + (poolBars[0].frac / 9.0) * barTotalW + (poolBars[1].frac / 9.0) * barTotalW;
    s.addShape(pres.shapes.RECTANGLE, { x, y: barY, w, h: barH,
      fill: { color: b.color }, line: { color: b.color } });
  });
  s.addText("~4.3M pairs (98.7%)", { x: barX0+0.08, y: barY, w: 3.5, h: barH,
    fontSize: 8.5, bold: true, color: C.navy, valign: "middle" });
  s.addText("1.2% SPair", { x: 9.43, y: barY-0.24, w: 0.5, h: 0.22, fontSize: 7.5, color: C.gold });
  s.addText("0.07% PF", { x: 9.43, y: barY-0.44, w: 0.5, h: 0.22, fontSize: 7, color: C.red });

  // Three correspondence-specific adaptations
  s.addText("3 Correspondence-Specific Adaptations:", {
    x: 0.5, y: 4.22, w: 5, h: 0.28,
    fontSize: 12, bold: true, color: C.gold
  });
  const adaptations = [
    "Coverage objective F(S) = BFV E ⊆ε T coverage: fraction of target motion space covered by S within radius ε. Monotone submodular → greedy yields (½−η) guarantee.",
    "Source-partitioned diversity: penalizes within-source redundancy independently — prevents PointOdyssey (98.7% of pool) from monopolizing budget; lets minority sources (SPair, PF-PASCAL) contribute at appropriate scale.",
    "Supervision-density-derived ε: scales ε with local correspondence density in target space — dense targets (KITTI flow) get small ε; sparse targets (PF-PASCAL keypoints) get large ε. No manual radius tuning.",
  ];
  const aRuns = adaptations.map((a, i) => ({
    text: a,
    options: { bullet: true, breakLine: i < adaptations.length-1, fontSize: 10.5, color: C.offWhite, paraSpaceAfter: 4 }
  }));
  s.addText(aRuns, { x: 0.5, y: 4.54, w: 9, h: 0.98, valign: "top" });

  // Heterogeneous challenge callout
  callout(s, "Unlike ClusterCov: no target-space clustering, automatic heterogeneous budget allocation, (½−η) guarantee. Same BFV signal that diagnoses transfer now drives principled selection.", {
    x: 5.1, y: 4.22, w: 4.55, h: 0.28, bg: C.tealDim, fc: C.white, fs: 9
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 17: Masked Correspondence Autoencoder (MAE) — with UMAP evidence
// ═══════════════════════════════════════════════════════════════
if (false) {
  let s = pres.addSlide();
  darkBg(s);
  sectionBar(s, "Stretch Goal: Masked Correspondence Autoencoder");

  s.addText("Can a model learn observation-invariant motion latents across heterogeneous supervision regimes?", {
    x: 0.5, y: 0.82, w: 9, h: 0.36,
    fontSize: 13.5, italic: true, color: C.teal
  });

  // ── Left column: architecture + perceiver ─────────────────────
  // Architecture card
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 1.24, w: 4.55, h: 2.5,
    fill: { color: C.navyMid }, line: { color: C.tealDim, width: 1 }
  });
  s.addText("Architecture", {
    x: 0.45, y: 1.29, w: 4.2, h: 0.28,
    fontSize: 12, bold: true, color: C.gold, margin: 0
  });
  bullets(s, [
    "Input: src RGB (3ch) + tgt RGB (3ch) + observed flow (2ch) + validity mask (1ch)",
    "ViT encoder (dim 512, depth 10, heads 8) · decoder (dim 384, depth 4)",
    "75% patch masking + 20% speckle keep → ~5% flow visibility",
    "SmoothL1 loss on masked valid pixels",
  ], { x: 0.45, y: 1.59, w: 4.2, h: 1.1, fs: 10, color: C.offWhite });

  // Perceiver IO card
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.3, y: 3.82, w: 4.55, h: 1.62,
    fill: { color: C.navyMid }, line: { color: C.teal, width: 1 }
  });
  s.addText("Proposed Fix: Perceiver IO Patch Embedding", {
    x: 0.45, y: 3.87, w: 4.2, h: 0.28,
    fontSize: 11, bold: true, color: C.teal, margin: 0
  });
  bullets(s, [
    "Treat each keypoint in a patch as a set element (density-agnostic)",
    "Cross-attention → fixed-dim token regardless of keypoint count",
    "Open Q: does downstream ViT still distinguish sparse/dense from Perceiver output statistics?",
  ], { x: 0.45, y: 4.17, w: 4.2, h: 1.2, fs: 10, color: C.offWhite });

  // ── Right column: UMAP evidence ───────────────────────────────
  twoColDivider(s, 5.0);

  s.addText("Evidence: Latent Space Clusters by Dataset Identity", {
    x: 5.15, y: 1.24, w: 4.6, h: 0.28,
    fontSize: 11.5, bold: true, color: C.gold, margin: 0
  });

  // UMAP 1 — all datasets (aspect 1545x1027 ≈ 1.504)
  addImageOrPlaceholder(s, fig("umap_all_datasets.png"), 5.15, 1.55, 4.6, 2.05);
  s.addText("All datasets: same color = same dataset. Clusters by identity — should overlap ideally.", {
    x: 5.15, y: 3.62, w: 4.6, h: 0.22,
    fontSize: 8.5, italic: true, color: C.muted, align: "center"
  });

  // UMAP 2 — FT3D mask sweep (aspect 1286x897 ≈ 1.434)
  addImageOrPlaceholder(s, fig("umap_ft3d_sweep.png"), 5.15, 3.88, 4.6, 1.55);
  s.addText("FlyingThings3D sweep: color = mask ratio. Ideal: intermixed. Actual: stratified.", {
    x: 5.15, y: 5.45, w: 4.6, h: 0.2,
    fontSize: 8.5, italic: true, color: C.muted, align: "center"
  });

  callout(s, "⚠ Latent space encodes dataset statistics, not motion. Density mismatch is the root cause. Perceiver IO is the proposed fix.", {
    x: 0.3, y: 5.3, w: 4.55, h: 0.27, bg: C.gold, fc: C.navy, fs: 9.5
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 18: Validation Plan
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  lightBg(s);
  lightTitle(s, "Validation Plan Across Chapters");

  s.addText("Each chapter makes a concrete, testable claim with chapter-specific metrics and success criteria.", {
    x: 0.5, y: 0.95, w: 9, h: 0.32,
    fontSize: 12, italic: true, color: C.midGray
  });

  const chapters = [
    {
      tag: "Ch. 1",
      name: "Cross-Modal Aerial Learning",
      desc: "Validate MAVOC/MAVIC classification on held-out geography, MAVIC-T translation with L1 + LPIPS + FID, and MINNIMAN via map-prediction ablations.",
      metric: "Success: improve on strong baselines and show modality contributions are measurable.",
      status: "Published / completed",
      sc: C.tealDim,
    },
    {
      tag: "Ch. 2",
      name: "Directed Coverage Metrics",
      desc: "Validate under held-out target, held-out train set, and joint held-out protocols; compare against symmetric baselines and test controlled SDF motion interventions.",
      metric: "Success: ranking accuracy stays meaningfully above 0.50 and directional metrics beat symmetric ones.",
      status: "Under review",
      sc: C.teal,
    },
    {
      tag: "Ch. 3",
      name: "Coverage-Based Data Curation",
      desc: "Validate pooled selection, homogeneous source ablations, joint-vs-merged allocation, source routing, GIST ablations, and lambda transfer/sensitivity.",
      metric: "Success: joint or normalized selector gives the strongest macro transfer profile at matched budget.",
      status: "In progress",
      sc: C.gold,
    },
  ];

  chapters.forEach((exp, i) => {
    const x = 0.35 + i * 3.15, y = 1.35;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 2.95, h: 3.75,
      fill: { color: "FFFFFF" }, line: { color: exp.sc, width: 1.5 },
      shadow: makeShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 2.95, h: 0.34,
      fill: { color: exp.sc }, line: { color: exp.sc }
    });
    s.addText(exp.tag, {
      x: x+0.1, y: y, w: 0.6, h: 0.34,
      fontSize: 11, bold: true, color: C.navy, valign: "middle"
    });
    s.addText(exp.name, {
      x: x+0.12, y: y+0.45, w: 2.72, h: 0.55,
      fontSize: 12, bold: true, color: C.navy, wrap: true
    });
    s.addText(exp.desc, {
      x: x+0.12, y: y+1.08, w: 2.72, h: 1.45,
      fontSize: 10, color: C.darkText, wrap: true
    });
    s.addText(exp.metric, {
      x: x+0.12, y: y+2.62, w: 2.72, h: 0.72,
      fontSize: 9, italic: true, color: exp.sc, wrap: true
    });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: x+0.12, y: y+3.36, w: 1.55, h: 0.24, rectRadius: 0.06,
      fill: { color: exp.sc }, line: { color: exp.sc }
    });
    s.addText(exp.status, {
      x: x+0.12, y: y+3.36, w: 1.55, h: 0.24,
      fontSize: 7.5, bold: true, color: C.navy, align: "center", valign: "middle"
    });
  });

  callout(s, "Dissertation-level validation claim: robust correspondence improves when training data is explicitly designed, diagnosed, and curated rather than treated as fixed background infrastructure.", {
    x: 0.35, y: 5.28, w: 9.3, h: 0.28, bg: C.navy, fc: C.teal, fs: 10
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 19: Timeline
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  darkBg(s);
  sectionBar(s, "Timeline & Status");

  // Timeline bar
  const months = ["Apr", "May", "Jun", "Jul", "Aug"];
  const barY = 1.3, barH = 0.45;
  const barX = 0.5, barW = 9.0;
  const timelineColors = {
    active: C.teal,
    draft: C.offWhite,
    review: C.tealDim,
    defense: C.gold,
  };
  const timelineStart = new Date(Date.UTC(2026, 3, 15)); // Apr 15
  const timelineEnd = new Date(Date.UTC(2026, 7, 21));   // Aug 21
  const timelineSpan = timelineEnd.getTime() - timelineStart.getTime();
  const posForDate = (month, day) => {
    const date = new Date(Date.UTC(2026, month - 1, day));
    return barX + ((date.getTime() - timelineStart.getTime()) / timelineSpan) * barW;
  };

  // Background bar
  s.addShape(pres.shapes.RECTANGLE, {
    x: barX, y: barY, w: barW, h: barH,
    fill: { color: C.navyMid }, line: { color: C.tealDim }
  });

  months.forEach((m, i) => {
    const x = i === 0 ? barX : posForDate(4 + i, 1);
    s.addShape(pres.shapes.RECTANGLE, {
      x: x-0.01, y: barY, w: 0.03, h: barH,
      fill: { color: C.teal }, line: { color: C.teal }
    });
    s.addText(m, {
      x: x-0.35, y: 2.24, w: 0.7, h: 0.24,
      fontSize: 11, color: C.muted, align: "center"
    });
  });

  // Current position marker
  s.addShape(pres.shapes.OVAL, {
    x: barX - 0.15, y: barY - 0.08, w: 0.3, h: barH + 0.16,
    fill: { color: C.teal }, line: { color: C.teal }
  });
  s.addText("NOW", {
    x: barX - 0.25, y: barY - 0.28, w: 0.7, h: 0.25,
    fontSize: 9, bold: true, color: C.teal, align: "center"
  });

  // Milestones — alternate above/below the bar
  const milestones = [
    { label: "Ch.3 experiments +\nablations complete\nMay 30", month: 5, day: 30, above: true,  color: timelineColors.active,  labelDx: -0.92, labelW: 1.84 },
    { label: "Latent model\nexperiments\nJun 27",              month: 6, day: 27, above: false, color: timelineColors.active,  labelDx: -0.78, labelW: 1.56 },
    { label: "Advisor draft\nJul 25",                          month: 7, day: 25, above: true,  color: timelineColors.draft,   labelDx: -0.68, labelW: 1.36 },
    { label: "Committee review\nsubmit Aug 1",                 month: 8, day: 1,  above: false, color: timelineColors.review,  labelDx: -0.9,  labelW: 1.8 },
    { label: "Defense\nAug 21",                                month: 8, day: 21, above: true,  color: timelineColors.defense, labelDx: -0.64, labelW: 1.28 },
  ];

  const aboveLabelY = 0.8;
  const aboveLabelH = 0.42;
  const belowLabelY = 1.8;
  const belowLabelH = 0.34;

  milestones.forEach(ms => {
    const mx = posForDate(ms.month, ms.day);
    const markerY = barY + barH / 2;   // centre of bar

    // dot on the bar
    s.addShape(pres.shapes.OVAL, {
      x: mx - 0.1, y: markerY - 0.1, w: 0.2, h: 0.2,
      fill: { color: ms.color }, line: { color: ms.color }
    });

    if (ms.above) {
      // vertical stem from label bottom up to bar top
      const stemTop  = aboveLabelY + aboveLabelH + 0.04;
      const stemBot  = barY - 0.04;
      s.addShape(pres.shapes.RECTANGLE, {
        x: mx - 0.015, y: stemTop, w: 0.03, h: Math.max(stemBot - stemTop, 0.05),
        fill: { color: ms.color }, line: { color: ms.color }
      });
      s.addText(ms.label, {
        x: mx + (ms.labelDx ?? -0.9), y: aboveLabelY, w: ms.labelW ?? 1.8, h: aboveLabelH,
        fontSize: 10, color: ms.color, align: "center", wrap: true
      });
    } else {
      // vertical stem from bar bottom down to label top
      const stemTop  = barY + barH + 0.04;
      const stemBot  = belowLabelY - 0.04;
      s.addShape(pres.shapes.RECTANGLE, {
        x: mx - 0.015, y: stemTop, w: 0.03, h: Math.max(stemBot - stemTop, 0.05),
        fill: { color: ms.color }, line: { color: ms.color }
      });
      s.addText(ms.label, {
        x: mx + (ms.labelDx ?? -0.9), y: belowLabelY, w: ms.labelW ?? 1.8, h: belowLabelH,
        fontSize: 10, color: ms.color, align: "center", wrap: true
      });
    }
  });

  // Status table
  const rows = [
    ["Chapter 1", "MAVOC/MAVIC series + MINNIMAN", "Published (CVPRW 2022–2024)"],
    ["Chapter 2", "Directed coverage metrics", "Under review (ECCV 2026)"],
    ["Chapter 3", "Coverage-based data curation", "In progress (Defense Aug 21)"],
    ["", "— Source ablations (PointOdyssey, SPair, PF-PASCAL)", "In progress (May 30)"],
    ["", "— GIST full framework + λ sweep", "Committee review (Aug 1)"],
    ["", "— Masked correspondence autoencoder", "Planned (stretch)"],
  ];
  const tData = rows.map(r => [
    { text: r[0], options: { bold: true, color: r[0] ? "FFFFFF" : "8BA3BF" } },
    { text: r[1], options: { color: r[0] ? "E8EFF7" : "8BA3BF" } },
    { text: r[2], options: {
      color: r[2].includes("Published") ? C.green : r[2].includes("review") ? C.tealDim : r[2].includes("progress") ? C.teal : C.muted,
      bold: r[2].includes("Published") || r[2].includes("review")
    }},
  ]);
  s.addTable(tData, {
    x: 0.4, y: 2.62, w: 9.2, h: 2.5,
    fontSize: 11,
    border: { pt: 0.5, color: "1A3050" },
    colW: [1.5, 5.5, 2.2],
    fill: { color: C.navyMid },
    rowH: 0.45,
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 20: Contributions Summary  [HIDDEN — probably won't reach in 25 min]
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  darkBg(s);
  sectionBar(s, "Dissertation Contributions at a Glance");

  const contribs = [
    {
      ch: "Ch. 1",
      items: [
        "MAVOC/MAVIC benchmark series + MAGIC aligned dataset",
        "SAR↔EO translation across 4 modality directions",
        "MINNIMAN: cross-modal magnetic map inference",
      ],
      color: C.tealDim,
    },
    {
      ch: "Ch. 2",
      items: [
        "BFV: resolution-agnostic 4D motion descriptor",
        "Directed coverage (E ⊆ε T, T ⊆ε E): separates failure modes symmetric metrics conflate",
        "SDF-Fractal3D: non-photorealistic pipeline outperforming photorealistic baselines",
        "Transferability estimator: 0.70 pairwise ranking accuracy, 3 held-out protocols",
      ],
      color: C.teal,
    },
    {
      ch: "Ch. 3",
      items: [
        "Coverage + diversity curation: GIST with BFV submodular utility + (½−η) guarantee",
        "Source-partitioned diversity: handles heterogeneous pool without source labels",
        "Supervision-density-derived ε: removes per-target manual radius tuning",
        "Empirical: motion-only selector recovers expert source routing without labels",
        "[Stretch] Masked correspondence autoencoder for observation-invariant latents",
      ],
      color: C.gold,
    },
  ];

  contribs.forEach((c, i) => {
    const y = 0.88 + i * 1.52;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.35, y, w: 0.65, h: 1.35,
      fill: { color: c.color }, line: { color: c.color }
    });
    s.addText(c.ch, {
      x: 0.35, y, w: 0.65, h: 1.35,
      fontSize: 13, bold: true, color: C.navy, align: "center", valign: "middle"
    });
    const runs = c.items.map((it, j) => ({
      text: it,
      options: { bullet: true, breakLine: j < c.items.length-1, fontSize: 11, color: C.offWhite, paraSpaceAfter: 4 }
    }));
    s.addShape(pres.shapes.RECTANGLE, {
      x: 1.1, y, w: 8.55, h: 1.35,
      fill: { color: C.navyMid }, line: { color: c.color }
    });
    s.addText(runs, { x: 1.25, y: y+0.05, w: 8.3, h: 1.25, valign: "top" });
  });

  s.addText("Unifying theme: motion structure is an underused, actionable signal for diagnosing and reducing domain shift in dense visual correspondence.", {
    x: 0.35, y: 5.45, w: 9.3, h: 0.18,
    fontSize: 10, italic: true, color: C.muted
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 21: Anticipated Questions / Discussion  [HIDDEN — probably won't reach in 25 min]
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  lightBg(s);
  lightTitle(s, "Anticipated Questions");

  const questions = [
    {
      q: "How sensitive is the method to the normalization exponent β?",
      a: "β derives from target supervision density — dense targets (KITTI) → low β, sparse targets (PF-PASCAL) → high β. Principled, not hand-tuned. Sensitivity ablation planned.",
      color: C.teal,
    },
    {
      q: "Does λ generalize across target benchmarks?",
      a: "Open question. We plan calibration/deployment split experiment. If yes → offline curation. If no → report as Pareto frontier indexed by λ. Either way is a valid result.",
      color: C.gold,
    },
    {
      q: "Why not use the learned MAE latent directly for selection instead of BFV?",
      a: "MAE latent clusters by dataset identity (appearance), not motion structure — the very problem we're trying to solve. BFV is the motion-aware alternative; MAE is a downstream representation goal.",
      color: C.red,
    },
    {
      q: "Is ClusterCov contribution sufficient if GIST ablations don't show big gains?",
      a: "ClusterCov itself already shows the core claim: subset selection substantially affects downstream transfer. GIST adds formal guarantees and handles heterogeneous pools — important even if marginal empirical gains are modest.",
      color: C.green,
    },
  ];

  questions.forEach((qa, i) => {
    const row = Math.floor(i / 2), col = i % 2;
    const x = 0.45 + col * 4.75, y = 1.1 + row * 2.1;
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 4.5, h: 1.9,
      fill: { color: "FFFFFF" }, line: { color: qa.color, width: 1.5 },
      shadow: makeShadow()
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x, y, w: 4.5, h: 0.06,
      fill: { color: qa.color }, line: { color: qa.color }
    });
    s.addText("Q: " + qa.q, {
      x: x+0.12, y: y+0.12, w: 4.26, h: 0.6,
      fontSize: 11, bold: true, color: C.navy, wrap: true
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: x+0.12, y: y+0.72, w: 4.2, h: 0.02,
      fill: { color: qa.color }, line: { color: qa.color }
    });
    s.addText("A: " + qa.a, {
      x: x+0.12, y: y+0.8, w: 4.26, h: 1.0,
      fontSize: 10, color: C.midGray, wrap: true
    });
  });

  callout(s, "Core framing if pushed: the dissertation's claim is Pareto dominance across benchmarks, not beating every single-target oracle.", {
    x: 0.45, y: 5.27, w: 9.1, h: 0.28, bg: C.navy, fc: C.teal, fs: 11
  });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 22: CLOSING
// ═══════════════════════════════════════════════════════════════
{
  let s = pres.addSlide();
  darkBg(s);

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.18, h: 5.625,
    fill: { color: C.teal }, line: { color: C.teal }
  });

  s.addText("Thank You", {
    x: 0.5, y: 0.7, w: 9, h: 0.85,
    fontSize: 44, bold: true, color: C.white, align: "center"
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 2.5, y: 1.6, w: 5, h: 0.05,
    fill: { color: C.teal }, line: { color: C.teal }
  });

  s.addText("Bridging Domain Gaps in Visual Correspondence:\nFrom Multi-Modal Fusion to Data-Curated Motion Latents", {
    x: 0.5, y: 1.75, w: 9, h: 0.95,
    fontSize: 17, italic: true, color: C.muted, align: "center", wrap: true
  });

  // Summary pills
  const pills2 = [
    "Ch. 1 ✓ Published", "Ch. 2 ✓ Under Review (ECCV)", "Ch. 3  In Progress"
  ];
  const pColors2 = [C.tealDim, C.teal, C.gold];
  const pText2 = [C.white, C.navy, C.navy];
  pills2.forEach((p, i) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.5 + i * 3.05, y: 2.85, w: 2.85, h: 0.55, rectRadius: 0.1,
      fill: { color: pColors2[i] }, line: { color: pColors2[i] }
    });
    s.addText(p, {
      x: 0.5 + i * 3.05, y: 2.85, w: 2.85, h: 0.55,
      fontSize: 11, bold: true, color: pText2[i], align: "center", valign: "middle"
    });
  });

  s.addText("Defense: August 21, 2026", {
    x: 0.5, y: 3.58, w: 9, h: 0.4,
    fontSize: 16, bold: true, color: C.teal, align: "center"
  });

  s.addText("Questions?", {
    x: 0.5, y: 4.1, w: 9, h: 0.55,
    fontSize: 28, color: C.white, align: "center"
  });

  s.addText("Spencer Low · BYU Computer Science · slowl@byu.edu", {
    x: 0.5, y: 4.85, w: 9, h: 0.35,
    fontSize: 12, color: C.muted, align: "center"
  });
}

// ── Write ────────────────────────────────────────────────────────
pres.writeFile({ fileName: PPTX_PATH })
  .then(() => console.log(`Written to ${PPTX_PATH}`))
  .catch(e => { console.error(e); process.exit(1); });
