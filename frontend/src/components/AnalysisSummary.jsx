import React from 'react';
import { ClipboardList, Activity, Info, FileDown } from 'lucide-react';
import PdfExportButton from './PdfExportButton';

export default function AnalysisSummary({ result, imagePreviewUrl, uploadedFileName }) {
  if (!result) return null;

  const {
    prediction = '',
    predicted_class_index,
    confidence = 0,
    risk_level = '',
    model_status = '',
    model_identifier = 'ResNet18',
    inference_latency_ms = null,
    analysis_date = new Date().toISOString(),
    file_info = {},
    class_probabilities = {},
    explainability = {},
  } = result;

  const formattedDate = new Date(analysis_date).toLocaleString('en-US', {
    dateStyle: 'long',
    timeStyle: 'long',
  });

  const probSum = (
    (class_probabilities.Normal || 0) +
    (class_probabilities.Osteopenia || 0) +
    (class_probabilities.Osteoporosis || 0)
  ).toFixed(4);

  return (
    <div className="analysis-summary-card">
      <div className="summary-card-header">
        <div className="summary-title-wrapper">
          <ClipboardList size={20} className="text-cyan" />
          <h3>Analysis Summary Record</h3>
        </div>
        <span className="summary-status-pill live-pill">
          <Activity size={12} />
          Live Inference Log
        </span>
      </div>

      <div className="summary-table-grid">
        {/* Prediction */}
        <div className="summary-item">
          <span className="summary-key">CNN Predicted Category</span>
          <span className="summary-val font-semibold text-primary">
            {prediction}
            {predicted_class_index !== undefined && (
              <span className="summary-index-tag"> (Class {predicted_class_index})</span>
            )}
          </span>
        </div>

        {/* Confidence */}
        <div className="summary-item">
          <span className="summary-key">Top-Class Softmax Score</span>
          <span className="summary-val font-semibold text-cyan">
            {(confidence * 100).toFixed(2)}%
          </span>
        </div>

        {/* Risk label */}
        <div className="summary-item">
          <span className="summary-key">Risk Label</span>
          <span className={`summary-val-badge ${risk_level === 'High' ? 'badge-high' : risk_level === 'Moderate' ? 'badge-mod' : 'badge-low'}`}>
            {risk_level}
          </span>
        </div>

        {/* Inference Time */}
        <div className="summary-item">
          <span className="summary-key">Measured Inference Time</span>
          <span className="summary-val-mono text-cyan">
            {inference_latency_ms != null ? `${inference_latency_ms.toFixed(1)} ms` : 'N/A'}
          </span>
        </div>

        {/* Model Identifier */}
        <div className="summary-item item-span-2">
          <span className="summary-key">Model Identifier</span>
          <span className="summary-val-mono">{model_identifier}</span>
        </div>

        {/* Model Status */}
        <div className="summary-item item-span-2">
          <span className="summary-key">Backend Model Status</span>
          <span className="summary-val-mono text-cyan">{model_status}</span>
        </div>

        {/* Class Softmax Scores */}
        <div className="summary-item item-span-2">
          <span className="summary-key">Softmax Score Distribution (sum ≈ {probSum})</span>
          <span className="summary-val-mono">
            Normal: {((class_probabilities.Normal || 0) * 100).toFixed(2)}% &nbsp;|&nbsp;
            Osteopenia: {((class_probabilities.Osteopenia || 0) * 100).toFixed(2)}% &nbsp;|&nbsp;
            Osteoporosis: {((class_probabilities.Osteoporosis || 0) * 100).toFixed(2)}%
          </span>
        </div>

        {/* Grad-CAM */}
        <div className="summary-item item-span-2">
          <span className="summary-key">Grad-CAM Status</span>
          <span className="summary-val-mono">
            {explainability.status === 'success'
              ? `✓ ${explainability.target_layer || 'layer4[-1]'} → Target: ${explainability.target_class}`
              : explainability.status || 'N/A'}
          </span>
        </div>

        {/* Analysis Date */}
        <div className="summary-item item-span-2">
          <span className="summary-key">Analysis Timestamp</span>
          <span className="summary-val-mono">{formattedDate}</span>
        </div>

        {/* File Info */}
        {file_info.filename && (
          <div className="summary-item item-span-2">
            <span className="summary-key">Uploaded Image</span>
            <span className="summary-val-mono">
              {file_info.filename} · {file_info.dimensions || 'Unknown'}{file_info.size_bytes ? ` · ${(file_info.size_bytes / 1024).toFixed(1)} KB` : ''}
            </span>
          </div>
        )}
      </div>

      <div className="summary-disclaimer-row">
        <Info size={13} className="text-muted" />
        <span>
          All values above are direct outputs of the trained ResNet18 model.
          Softmax scores are not calibrated clinical probabilities. Risk labels are research classifications only.
        </span>
      </div>

      {/* PDF Export */}
      <div className="summary-pdf-row">
        <PdfExportButton
          result={result}
          imagePreviewUrl={imagePreviewUrl}
          uploadedFileName={uploadedFileName || file_info?.filename}
        />
        <span className="pdf-export-note">
          2-page research report — original X-ray, Grad-CAM overlay, inference scores.
          Contains mandatory research disclaimer. Not a clinical document.
        </span>
      </div>
    </div>
  );
}
