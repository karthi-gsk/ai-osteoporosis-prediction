/**
 * PDF Report Generator for AI Osteoporosis Prediction System.
 * Uses jsPDF for programmatic PDF construction with clean vector drawing,
 * aspect-ratio-preserving image placement, and reliable text wrapping.
 *
 * All terminology adheres to academic research standards:
 * - "MODEL-PREDICTED CATEGORY" instead of clinical risk badges
 * - Softmax scores labeled as mathematical outputs, not clinical probabilities
 * - Separate educational reference for WHO DXA criteria (not associated with model scores)
 * - Standard ASCII / WinAnsi compatible characters for 100% cross-platform PDF fidelity
 */
import { jsPDF } from 'jspdf';

// ── Dark-theme color palette matching application tokens ─────────────────────
const C = {
  bgPage:          [10, 14, 26],       // #0A0E1A near-black page background
  bgCard:          [18, 26, 43],       // #121A2B dark card surface
  bgCardLt:        [24, 35, 56],       // #182338 slightly lighter card surface
  border:          [38, 55, 80],       // #263750 subtle border
  borderSubtle:    [28, 40, 60],       // #1C283C inner cell border
  accentCyan:      [56, 189, 248],     // #38BDF8 cyan-400
  accentTeal:      [45, 212, 191],     // #2DD4BF teal-400
  accentBlue:      [37, 99, 235],      // #2563EB blue-600
  colorNormal:     [16, 185, 129],     // #10B981 green
  colorOsteopenia: [245, 158, 11],     // #F59E0B amber
  colorOsteoporosis:[239, 68, 68],     // #EF4444 red
  textPrimary:     [248, 250, 252],    // #F8FAFC near-white
  textSecondary:   [148, 163, 184],    // #94A3B8 slate-400
  textMuted:       [100, 116, 139],    // #64748B slate-500
  white:           [255, 255, 255],
};

// ── Text sanitizer to prevent Unicode encoding defects in jsPDF standard fonts ─
function sanitizeText(str) {
  if (!str) return '';
  return String(str)
    .replace(/≥/g, '>=')
    .replace(/≤/g, '<=')
    .replace(/[−–—]/g, '-')
    .replace(/×/g, 'x')
    .replace(/μ/g, 'mean')
    .replace(/σ/g, 'std')
    .replace(/[•●■]/g, '-')
    .replace(/⚠/g, '[!]');
}

// ── Low-level drawing helpers ──────────────────────────────────────────────────
const rgb = (doc, [r, g, b]) => doc.setTextColor(r, g, b);
const fill = (doc, [r, g, b]) => doc.setFillColor(r, g, b);
const stroke = (doc, [r, g, b]) => doc.setDrawColor(r, g, b);

function rect(doc, x, y, w, h, color, radius = 0) {
  fill(doc, color);
  if (radius > 0) {
    doc.roundedRect(x, y, w, h, radius, radius, 'F');
  } else {
    doc.rect(x, y, w, h, 'F');
  }
}

function strokedRect(doc, x, y, w, h, fillColor, strokeColor, radius = 0, lineWidth = 0.2) {
  fill(doc, fillColor);
  stroke(doc, strokeColor);
  doc.setLineWidth(lineWidth);
  if (radius > 0) {
    doc.roundedRect(x, y, w, h, radius, radius, 'FD');
  } else {
    doc.rect(x, y, w, h, 'FD');
  }
}

function text(doc, rawStr, x, y, opts = {}) {
  const {
    size = 10,
    bold = false,
    color = C.textPrimary,
    align = 'left',
    maxWidth,
  } = opts;
  const str = sanitizeText(rawStr);
  doc.setFontSize(size);
  doc.setFont('helvetica', bold ? 'bold' : 'normal');
  rgb(doc, color);
  if (maxWidth) {
    doc.text(str, x, y, { align, maxWidth });
  } else {
    doc.text(str, x, y, { align });
  }
}

function horizontalRule(doc, y, color = C.border) {
  stroke(doc, color);
  doc.setLineWidth(0.3);
  doc.line(15, y, 195, y);
}

function getClassColor(className) {
  switch (className?.toLowerCase()) {
    case 'normal':       return C.colorNormal;
    case 'osteopenia':   return C.colorOsteopenia;
    case 'osteoporosis': default: return C.colorOsteoporosis;
  }
}

function drawProgressBar(doc, x, y, w, h, value, barColor) {
  // Track
  fill(doc, C.bgCardLt);
  doc.roundedRect(x, y, w, h, h / 2, h / 2, 'F');
  // Fill
  const fillW = Math.max(2, Math.min(w, w * Math.max(0, Math.min(1, value))));
  fill(doc, barColor);
  doc.roundedRect(x, y, fillW, h, h / 2, h / 2, 'F');
}

// ── Image dimension inspector to guarantee natural aspect ratio ───────────────
async function getImageNaturalSize(url) {
  return new Promise((resolve) => {
    if (!url || typeof Image === 'undefined') {
      resolve({ width: 512, height: 512 });
      return;
    }
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      resolve({
        width: img.naturalWidth || 512,
        height: img.naturalHeight || 512,
      });
    };
    img.onerror = () => {
      resolve({ width: 512, height: 512 });
    };
    img.src = url;
  });
}

// ── Helper to convert any URL (including blob: object URLs) to base64 ─────────
async function urlToBase64(url) {
  if (!url) return null;
  if (url.startsWith('data:')) return url;
  const response = await fetch(url);
  const blob = await response.blob();
  if (typeof FileReader !== 'undefined') {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  } else {
    const arrayBuffer = await blob.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    const mime = blob.type || 'image/jpeg';
    return `data:${mime};base64,${buffer.toString('base64')}`;
  }
}

// ── Main PDF Generation Function ──────────────────────────────────────────────
export async function generatePdfReport(result, imagePreviewUrl, uploadedFileName) {
  if (!result) throw new Error('No inference result available to export.');

  const {
    prediction            = 'Unknown',
    predicted_class_index = 0,
    confidence            = 0,
    model_identifier      = 'ResNet18',
    model_status          = 'operational',
    inference_latency_ms  = null,
    analysis_date         = new Date().toISOString(),
    class_probabilities   = { Normal: 0, Osteopenia: 0, Osteoporosis: 0 },
    explainability        = {},
    file_info             = {},
  } = result;

  // Resolve source image filename from arguments, result payload, or safe fallback
  const sourceImageName = (
    uploadedFileName ||
    file_info?.filename ||
    file_info?.name ||
    'knee-xray'
  ).trim();

  const overlayDataUri = explainability?.overlay_image || null;
  const gradcamStatus  = explainability?.status || 'unavailable';
  const gradcamLayer   = explainability?.target_layer || 'layer4[-1] (BasicBlock 2, 512ch, 7x7)';
  const gradcamTarget  = explainability?.target_class || prediction;

  const analysisTs = new Date(analysis_date).toLocaleString('en-US', {
    dateStyle: 'medium', timeStyle: 'medium',
  });
  const exportTs = new Date().toLocaleString('en-US', {
    dateStyle: 'medium', timeStyle: 'medium',
  });

  // ── Document Configuration ──────────────────────────────────────────────────
  const doc = new jsPDF({ unit: 'mm', format: 'a4', orientation: 'portrait' });
  const W = 210, H = 297;
  const PAD = 15;
  const CONTENT_W = W - 2 * PAD; // 180mm

  // ============================================================================
  // PAGE 1 — INFERENCE RESULTS & METADATA
  // ============================================================================

  // Background
  rect(doc, 0, 0, W, H, C.bgPage);

  // Top cyan decorative stripe
  fill(doc, C.accentCyan);
  doc.rect(0, 0, W, 1.5, 'F');

  // Header card
  strokedRect(doc, PAD, 5, CONTENT_W, 30, C.bgCard, C.border, 2);

  // Brand and title
  text(doc, 'AI OSTEOPOROSIS PREDICTION SYSTEM', PAD + 6, 13, {
    size: 12.5, bold: true, color: C.accentCyan,
  });
  text(doc, 'Explainable Deep Learning Knee Radiograph Research Report', PAD + 6, 19, {
    size: 8, color: C.textSecondary,
  });
  text(doc, 'RESEARCH USE ONLY - NOT A MEDICAL DIAGNOSTIC TOOL', PAD + 6, 26, {
    size: 7, bold: true, color: C.colorOsteopenia,
  });

  // Right-side timestamps
  text(doc, 'Analysis Timestamp', W - PAD - 6, 12, { size: 6.5, color: C.textMuted, align: 'right' });
  text(doc, analysisTs, W - PAD - 6, 16.5, { size: 7, bold: true, color: C.textPrimary, align: 'right' });
  text(doc, 'Report Exported', W - PAD - 6, 22, { size: 6.5, color: C.textMuted, align: 'right' });
  text(doc, exportTs, W - PAD - 6, 26.5, { size: 7, color: C.textSecondary, align: 'right' });

  let y = 39;

  // ── SECTION 1: INFERENCE RESULT ─────────────────────────────────────────────
  text(doc, 'INFERENCE CLASSIFICATION RESULT', PAD, y, { size: 7.5, bold: true, color: C.accentTeal });
  y += 2.5;
  horizontalRule(doc, y, C.accentTeal);
  y += 4;

  const predColor = getClassColor(prediction);
  const cardH = 26;
  strokedRect(doc, PAD, y, CONTENT_W, cardH, C.bgCard, C.border, 2.5);

  // Left accent stripe
  fill(doc, predColor);
  doc.roundedRect(PAD, y, 3, cardH, 1.5, 1.5, 'F');

  // Correct terminology: "MODEL-PREDICTED CATEGORY" instead of misleading risk badge
  text(doc, 'MODEL-PREDICTED CATEGORY', PAD + 8, y + 6, {
    size: 7, color: C.accentCyan, bold: true,
  });

  // Actual prediction text
  text(doc, `Predicted class: ${prediction}`, PAD + 8, y + 14.5, {
    size: 13, bold: true, color: predColor,
  });

  // Actual softmax score & class index
  const confidencePct = (confidence * 100).toFixed(2);
  text(doc, `Model softmax score: ${confidencePct}%`, PAD + 8, y + 21, {
    size: 8, color: C.textPrimary, bold: true,
  });
  text(doc, `(Softmax logit probability distribution output)`, PAD + 62, y + 21, {
    size: 6.8, color: C.textMuted,
  });

  // Right pill: Academic Research Notice (removes clinical risk badge)
  const pillW = 48, pillH = 14;
  const pillX = W - PAD - pillW - 6, pillY = y + 6;
  strokedRect(doc, pillX, pillY, pillW, pillH, C.bgCardLt, C.border, 2);
  text(doc, 'RESEARCH PROTOTYPE', pillX + pillW / 2, pillY + 5.5, {
    size: 6.5, bold: true, color: C.colorOsteopenia, align: 'center',
  });
  text(doc, 'Non-Diagnostic Evaluation', pillX + pillW / 2, pillY + 10.5, {
    size: 6, color: C.textMuted, align: 'center',
  });

  y += cardH + 7;

  // ── SECTION 2: SOFTMAX SCORE DISTRIBUTION (PROBABILITY CARDS) ───────────────
  // Rebuilt layout:
  // Line 1: Class name and softmax score
  // Line 2: Horizontal probability bar
  // No text behind the bar; DXA reference moved to separate section
  text(doc, 'SOFTMAX SCORE DISTRIBUTION ACROSS CANDIDATE CLASSES', PAD, y, { size: 7.5, bold: true, color: C.accentTeal });
  y += 2.5;
  horizontalRule(doc, y);
  y += 4;

  const classList = [
    { name: 'Normal',       prob: Number(class_probabilities.Normal || 0),       color: C.colorNormal },
    { name: 'Osteopenia',   prob: Number(class_probabilities.Osteopenia || 0),   color: C.colorOsteopenia },
    { name: 'Osteoporosis', prob: Number(class_probabilities.Osteoporosis || 0), color: C.colorOsteoporosis },
  ];

  for (const cls of classList) {
    const isPredicted = cls.name.toLowerCase() === prediction.toLowerCase();
    const rowCardH = 13.5;
    const bgCol = isPredicted ? C.bgCardLt : C.bgCard;
    const borderCol = isPredicted ? cls.color : C.border;

    strokedRect(doc, PAD, y, CONTENT_W, rowCardH, bgCol, borderCol, 2, isPredicted ? 0.35 : 0.2);

    if (isPredicted) {
      fill(doc, cls.color);
      doc.roundedRect(PAD, y, 2.5, rowCardH, 1, 1, 'F');
    }

    // Line 1: Class name on left, softmax percentage on right
    const pct = (cls.prob * 100).toFixed(2);
    text(doc, cls.name, PAD + 7, y + 4.8, {
      size: 8, bold: isPredicted, color: isPredicted ? cls.color : C.textPrimary,
    });

    if (isPredicted) {
      text(doc, '[Model Selected Class]', PAD + 32, y + 4.8, {
        size: 6.5, color: C.textMuted,
      });
    }

    text(doc, `${pct}%`, W - PAD - 7, y + 4.8, {
      size: 8.5, bold: isPredicted, color: isPredicted ? cls.color : C.textPrimary, align: 'right',
    });

    // Line 2: Horizontal probability bar with clean full-width spacing
    const barX = PAD + 7;
    const barW = CONTENT_W - 14;
    const barY = y + 7.8;
    drawProgressBar(doc, barX, barY, barW, 3.2, cls.prob, cls.color);

    y += rowCardH + 2.5;
  }

  y += 2.5;

  // ── SECTION 3: EDUCATIONAL REFERENCE SECTION (DXA T-SCORE CRITERIA) ─────────
  // Completely separate from model probability cards to avoid confusion
  text(doc, 'EDUCATIONAL REFERENCE - WHO DXA DIAGNOSTIC CRITERIA (BACKGROUND ONLY)', PAD, y, {
    size: 7.2, bold: true, color: C.accentTeal,
  });
  y += 2.5;
  horizontalRule(doc, y);
  y += 3.5;

  const refCardH = 17;
  strokedRect(doc, PAD, y, CONTENT_W, refCardH, C.bgCard, C.border, 2);

  // Reference ranges (using standard ASCII symbols: >=, <=, -)
  text(doc, 'WHO T-Score Diagnostic Ranges:', PAD + 6, y + 5, {
    size: 6.8, bold: true, color: C.accentCyan,
  });
  text(doc, 'Normal: T-score >= -1.0   |   Osteopenia: -2.5 < T-score < -1.0   |   Osteoporosis: T-score <= -2.5', PAD + 46, y + 5, {
    size: 6.8, bold: true, color: C.textPrimary,
  });

  // Explicit non-association disclaimer
  text(doc,
    'Note: WHO T-score thresholds apply exclusively to dual-energy X-ray absorptiometry (DXA) BMD examinations. ' +
    'The ResNet18 model operates solely on plain knee radiographs and does NOT calculate bone mineral density or DXA T-scores. ' +
    'Do not equate model classification outputs with clinical DXA measurements.',
    PAD + 6, y + 10,
    { size: 6, color: C.textMuted, maxWidth: CONTENT_W - 12 }
  );

  y += refCardH + 6;

  // ── SECTION 4: MODEL & INFERENCE METADATA (DYNAMIC TEXT WRAPPING) ────────────
  text(doc, 'MODEL ARCHITECTURE & INFERENCE METADATA', PAD, y, { size: 7.5, bold: true, color: C.accentTeal });
  y += 2.5;
  horizontalRule(doc, y);
  y += 4;

  const latencyStr = inference_latency_ms != null ? `${inference_latency_ms.toFixed(1)} ms` : 'N/A';
  const filenameStr = sourceImageName || 'knee-xray';

  // Structured metadata rows: dynamic height based on text wrapping
  const metadataRows = [
    {
      type: 'full',
      label: 'Model Architecture',
      value: `${model_identifier} (PyTorch Torchvision, frozen weights, trained on knee radiographs)`,
    },
    {
      type: 'dual',
      col1: { label: 'Best Checkpoint', value: 'Epoch 21 (Early Stopping on Val F1)' },
      col2: { label: 'Val Macro F1 (Best)', value: '0.7913' },
    },
    {
      type: 'dual',
      col1: { label: 'Test Accuracy', value: '80.74% (109 / 135 images)' },
      col2: { label: 'Test Macro F1', value: '80.72% (Macro-averaged)' },
    },
    {
      type: 'dual',
      col1: { label: 'Test ROC-AUC', value: '91.74% (Multi-class Macro)' },
      col2: { label: 'Inference Device', value: 'CPU (PyTorch eval mode, deterministic)' },
    },
    {
      type: 'dual',
      col1: { label: 'Inference Latency', value: latencyStr },
      col2: { label: 'Backend Status', value: model_status || 'operational' },
    },
    {
      type: 'full',
      label: 'Input Preprocessing',
      value: '224 x 224 px, RGB, ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])',
    },
    {
      type: 'full',
      label: 'Radiograph Source',
      value: filenameStr,
    },
  ];

  for (let r = 0; r < metadataRows.length; r++) {
    const row = metadataRows[r];
    const isEven = r % 2 === 0;
    const bgCol = isEven ? C.bgCard : C.bgCardLt;

    if (row.type === 'full') {
      const labelW = 38;
      const valW = CONTENT_W - labelW - 6;
      doc.setFontSize(6.8);
      const valLines = doc.splitTextToSize(sanitizeText(row.value), valW);
      const rowH = Math.max(6.8, valLines.length * 3.4 + 3);

      strokedRect(doc, PAD, y, CONTENT_W, rowH, bgCol, C.borderSubtle, 1);
      text(doc, row.label, PAD + 3.5, y + 4.2, { size: 6.5, color: C.textMuted });
      text(doc, row.value, PAD + labelW + 2, y + 4.2, {
        size: 6.8, bold: true, color: C.textPrimary, maxWidth: valW,
      });

      y += rowH + 1.2;
    } else if (row.type === 'dual') {
      const colW = (CONTENT_W - 3) / 2; // ~88.5mm
      const labelW = 32;
      const valW = colW - labelW - 5;

      doc.setFontSize(6.8);
      const lines1 = doc.splitTextToSize(sanitizeText(row.col1.value), valW);
      const lines2 = doc.splitTextToSize(sanitizeText(row.col2.value), valW);
      const maxLines = Math.max(lines1.length, lines2.length);
      const rowH = Math.max(6.8, maxLines * 3.4 + 3);

      // Col 1
      strokedRect(doc, PAD, y, colW, rowH, bgCol, C.borderSubtle, 1);
      text(doc, row.col1.label, PAD + 3.5, y + 4.2, { size: 6.5, color: C.textMuted });
      text(doc, row.col1.value, PAD + labelW + 1, y + 4.2, {
        size: 6.8, bold: true, color: C.textPrimary, maxWidth: valW,
      });

      // Col 2
      const col2X = PAD + colW + 3;
      strokedRect(doc, col2X, y, colW, rowH, bgCol, C.borderSubtle, 1);
      text(doc, row.col2.label, col2X + 3.5, y + 4.2, { size: 6.5, color: C.textMuted });
      text(doc, row.col2.value, col2X + labelW + 1, y + 4.2, {
        size: 6.8, bold: true, color: C.textPrimary, maxWidth: valW,
      });

      y += rowH + 1.2;
    }
  }

  // ── PAGE 1 FOOTER ──────────────────────────────────────────────────────────
  const footerH = 18;
  const footerY = H - footerH;
  rect(doc, 0, footerY, W, footerH, C.bgCard);
  fill(doc, C.colorOsteopenia);
  doc.rect(0, footerY, W, 0.8, 'F');

  text(doc, 'IMPORTANT RESEARCH & ACADEMIC DISCLAIMER', PAD, footerY + 5, {
    size: 6.8, bold: true, color: C.colorOsteopenia,
  });
  text(doc,
    'This report is generated by an academic research prototype for educational and investigative purposes only. ' +
    'Outputs must NOT be used as clinical diagnoses, treatment recommendations, or medical advice. ' +
    'Always consult a certified radiologist, orthopedic specialist, or physician for clinical evaluation.',
    PAD, footerY + 9.5,
    { size: 5.6, color: C.textMuted, maxWidth: CONTENT_W - 25 }
  );
  text(doc, 'Page 1 of 2', W - PAD, footerY + 11, { size: 7, color: C.textSecondary, align: 'right' });

  // ============================================================================
  // PAGE 2 — GRAD-CAM EXPLAINABILITY MAPS & VISUALIZATIONS
  // ============================================================================
  doc.addPage();
  rect(doc, 0, 0, W, H, C.bgPage);

  // Top cyan decorative stripe
  fill(doc, C.accentCyan);
  doc.rect(0, 0, W, 1.5, 'F');

  // Page 2 header
  strokedRect(doc, PAD, 5, CONTENT_W, 20, C.bgCard, C.border, 2);
  text(doc, 'GRAD-CAM EXPLAINABILITY MAPS', PAD + 6, 12.5, {
    size: 12, bold: true, color: C.accentCyan,
  });
  text(doc, 'Gradient-Weighted Class Activation Mapping - ResNet18 Final Convolutional Layer', PAD + 6, 18.5, {
    size: 7.8, color: C.textSecondary,
  });
  text(doc, 'Live Overlay Generation', W - PAD - 6, 15, {
    size: 7.5, bold: true, color: C.colorNormal, align: 'right',
  });

  y = 29;

  // ── Grad-CAM Status & Parameter Bar ─────────────────────────────────────────
  const gcOk = gradcamStatus === 'success' && overlayDataUri;
  const gcColor = gcOk ? C.colorNormal : C.colorOsteopenia;
  const statusCardH = 11;
  strokedRect(doc, PAD, y, CONTENT_W, statusCardH, C.bgCard, C.border, 2);

  fill(doc, gcColor);
  doc.roundedRect(PAD, y, 2.5, statusCardH, 1, 1, 'F');

  text(doc, gcOk ? 'GRAD-CAM STATUS: GENUINE GRADIENT-BASED ACTIVATION MAP' : 'GRAD-CAM STATUS: UNAVAILABLE', PAD + 6, y + 4.8, {
    size: 7.2, bold: true, color: gcColor,
  });
  text(doc, `Target Layer: ${gradcamLayer}   |   Target Class: ${gradcamTarget}   |   Method: Selvaraju et al. (ICCV 2017)`, PAD + 6, y + 9, {
    size: 6.4, color: C.textMuted,
  });

  y += statusCardH + 5;

  // ── Side-by-side Image Cards (Original X-Ray vs Grad-CAM Overlay) ───────────
  const imgCardW = (CONTENT_W - 6) / 2; // ~87mm
  const imgCardH = 100;
  const maxImgW = imgCardW - 8; // ~79mm
  const maxImgH = 76; // 76mm

  // Card 1: Original Knee Radiograph
  strokedRect(doc, PAD, y, imgCardW, imgCardH, C.bgCard, C.border, 2.5);
  text(doc, 'ORIGINAL RADIOGRAPH', PAD + imgCardW / 2, y + 6.5, {
    size: 7.2, bold: true, color: C.textSecondary, align: 'center',
  });

  // Prepare and draw original image with strict aspect-ratio preservation
  if (imagePreviewUrl) {
    try {
      const origBase64 = await urlToBase64(imagePreviewUrl);
      const imgSize = await getImageNaturalSize(origBase64);
      const aspect = imgSize.width / imgSize.height;

      let drawW = maxImgW;
      let drawH = maxImgW / aspect;
      if (drawH > maxImgH) {
        drawH = maxImgH;
        drawW = maxImgH * aspect;
      }
      const imgX = PAD + 4 + (maxImgW - drawW) / 2;
      const imgY = y + 9 + (maxImgH - drawH) / 2;

      const format = origBase64.startsWith('data:image/png') ? 'PNG' : 'JPEG';
      doc.addImage(origBase64, format, imgX, imgY, drawW, drawH, '', 'FAST');
    } catch {
      text(doc, '[Original radiograph unavailable]', PAD + imgCardW / 2, y + 45, {
        size: 6.5, color: C.textMuted, align: 'center',
      });
    }
  } else {
    text(doc, '[No image uploaded]', PAD + imgCardW / 2, y + 45, {
      size: 6.5, color: C.textMuted, align: 'center',
    });
  }

  text(doc, 'Uploaded knee radiograph (unmodified input)', PAD + imgCardW / 2, y + imgCardH - 4.5, {
    size: 6, color: C.textMuted, align: 'center',
  });

  // Card 2: Grad-CAM Overlay
  const rx = PAD + imgCardW + 6;
  strokedRect(doc, rx, y, imgCardW, imgCardH, C.bgCard, C.border, 2.5);
  text(doc, 'GRAD-CAM OVERLAY (JET COLORMAP)', rx + imgCardW / 2, y + 6.5, {
    size: 7.2, bold: true, color: C.accentCyan, align: 'center',
  });

  // Prepare and draw Grad-CAM overlay with strict aspect-ratio preservation
  if (gcOk && overlayDataUri) {
    try {
      const overlaySize = await getImageNaturalSize(overlayDataUri);
      const aspect = overlaySize.width / overlaySize.height;

      let drawW = maxImgW;
      let drawH = maxImgW / aspect;
      if (drawH > maxImgH) {
        drawH = maxImgH;
        drawW = maxImgH * aspect;
      }
      const imgX = rx + 4 + (maxImgW - drawW) / 2;
      const imgY = y + 9 + (maxImgH - drawH) / 2;

      doc.addImage(overlayDataUri, 'PNG', imgX, imgY, drawW, drawH, '', 'FAST');
    } catch {
      text(doc, '[Grad-CAM overlay render failed]', rx + imgCardW / 2, y + 45, {
        size: 6.5, color: C.textMuted, align: 'center',
      });
    }
  } else {
    text(doc, '[Grad-CAM overlay not generated]', rx + imgCardW / 2, y + 45, {
      size: 6.5, color: C.textMuted, align: 'center',
    });
  }

  text(doc, `ResNet18 layer4[-1] gradient feature attribution`, rx + imgCardW / 2, y + imgCardH - 4.5, {
    size: 6, color: C.textMuted, align: 'center',
  });

  y += imgCardH + 5;

  // ── Colour Scale Legend ───────────────────────────────────────────────────
  const legendCardH = 14;
  strokedRect(doc, PAD, y, CONTENT_W, legendCardH, C.bgCard, C.border, 2);

  text(doc, 'Activation Scale:', PAD + 6, y + 7, { size: 7, bold: true, color: C.textSecondary });

  // Centered Jet Colormap gradient bar
  const swatchX = PAD + 42;
  const swatchW = 92;
  const swatchH = 5;
  const swatchY = y + 4;
  const swatchColors = [
    [0, 0, 128],    // Dark Blue
    [0, 0, 255],    // Blue
    [0, 128, 255],  // Cyan
    [0, 255, 255],  // Light Cyan
    [0, 255, 0],    // Green
    [255, 255, 0],  // Yellow
    [255, 128, 0],  // Orange
    [255, 0, 0],    // Red
  ];
  const segW = swatchW / swatchColors.length;
  swatchColors.forEach(([r, g, b], i) => {
    doc.setFillColor(r, g, b);
    doc.rect(swatchX + i * segW, swatchY, segW + 0.1, swatchH, 'F');
  });

  stroke(doc, C.border);
  doc.setLineWidth(0.2);
  doc.rect(swatchX, swatchY, swatchW, swatchH);

  // Markers
  text(doc, 'Low (0.0)', swatchX, y + 12, { size: 5.8, color: C.textMuted, align: 'center' });
  text(doc, 'Medium (0.5)', swatchX + swatchW / 2, y + 12, { size: 5.8, color: C.textMuted, align: 'center' });
  text(doc, 'High (1.0)', swatchX + swatchW, y + 12, { size: 5.8, color: C.textMuted, align: 'center' });

  text(doc, 'Feature Influence', W - PAD - 6, y + 7, { size: 6.5, color: C.textMuted, align: 'right' });

  y += legendCardH + 5;

  // ── Methodology & Academic Limitations Section ─────────────────────────────
  text(doc, 'EXPLAINABILITY METHODOLOGY & SCIENTIFIC LIMITATIONS', PAD, y, {
    size: 7.5, bold: true, color: C.accentTeal,
  });
  y += 2.5;
  horizontalRule(doc, y);
  y += 3.5;

  const methodPoints = [
    {
      title: 'Mathematical Formulation:',
      desc: 'Grad-CAM computes gradients of the target class score with respect to feature activation maps at ResNet18 layer4[-1] (512 channels, 7x7 spatial resolution). Gradients are global-average-pooled and passed through a ReLU activation to capture positive contributing features.',
    },
    {
      title: 'Feature Attribution vs Medical Pathology:',
      desc: 'Highlighted regions indicate convolutional pattern attribution that influenced the neural network prediction. They do NOT represent anatomical segmentations, bone mineral density, trabecular micro-architecture loss, or clinically confirmed osteoporotic lesions.',
    },
    {
      title: 'Anatomical Scope & Dataset Constraints:',
      desc: 'The model was trained, tuned, and evaluated exclusively on plain knee radiographs. It has not been validated for hip, femoral neck, lumbar spine, or other skeletal anatomical sites.',
    },
    {
      title: 'Non-Calibrated Probabilities:',
      desc: 'Softmax scores represent normalized output logits across the three target classes. They do NOT represent calibrated Bayesian disease probabilities or clinical certainty.',
    },
  ];

  const methodCardH = 75;
  strokedRect(doc, PAD, y, CONTENT_W, methodCardH, C.bgCard, C.border, 2.5);

  let noteY = y + 5;
  for (const pt of methodPoints) {
    // Bullet marker
    fill(doc, C.accentCyan);
    doc.rect(PAD + 5, noteY + 0.8, 1.8, 1.8, 'F');

    // Title
    text(doc, pt.title, PAD + 9, noteY + 2.4, {
      size: 6.8, bold: true, color: C.accentCyan,
    });

    // Wrapped body text
    const descLines = doc.splitTextToSize(sanitizeText(pt.desc), CONTENT_W - 15);
    text(doc, pt.desc, PAD + 9, noteY + 6.2, {
      size: 6.2, color: C.textSecondary, maxWidth: CONTENT_W - 15,
    });

    noteY += descLines.length * 3.3 + 5;
  }

  // ── PAGE 2 FOOTER ──────────────────────────────────────────────────────────
  const footer2H = 19;
  const footer2Y = H - footer2H;
  rect(doc, 0, footer2Y, W, footer2H, C.bgCard);
  fill(doc, C.colorOsteopenia);
  doc.rect(0, footer2Y, W, 0.8, 'F');

  text(doc, 'IMPORTANT RESEARCH & ACADEMIC DISCLAIMER', PAD, footer2Y + 5, {
    size: 6.8, bold: true, color: C.colorOsteopenia,
  });
  text(doc,
    'This report is generated by an academic research prototype for educational and investigative purposes only. ' +
    'Outputs must NOT be used as clinical diagnoses, treatment recommendations, or medical advice. ' +
    'Consult a certified radiologist, orthopedic specialist, or physician for professional medical evaluation. ' +
    'Frozen Checkpoint: ResNet18 (Epoch 21, Val F1: 0.7913) | SHA-256: 6ccc18b8...',
    PAD, footer2Y + 9.5,
    { size: 5.5, color: C.textMuted, maxWidth: CONTENT_W - 25 }
  );
  text(doc, 'Page 2 of 2', W - PAD, footer2Y + 11.5, { size: 7, color: C.textSecondary, align: 'right' });

  // Expose data URI on window for headless automated testing and verification
  if (typeof window !== 'undefined') {
    try {
      window.__latestPdfBase64 = doc.output('datauristring');
    } catch (e) {
      // ignore in non-browser context
    }
  }

  // ── File Name & Download ──────────────────────────────────────────────────
  const safeDate = new Date(analysis_date).toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const safePred = (prediction || 'Prediction').replace(/[^a-zA-Z0-9_-]/g, '_');
  const baseNameWithoutExt = (sourceImageName || 'knee-xray')
    .replace(/\.[^/.]+$/, '')
    .replace(/[^a-zA-Z0-9_-]/g, '_') || 'knee-xray';
  const reportPdfFilename = `OsteoCNN_Report_${safePred}_${baseNameWithoutExt}_${safeDate}.pdf`;

  if (typeof window !== 'undefined' && doc.save) {
    doc.save(reportPdfFilename);
  }

  return doc;
}
