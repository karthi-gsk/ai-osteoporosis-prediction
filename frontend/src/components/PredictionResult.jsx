import React from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Activity,
  BarChart3,
  Info,
  Clock,
  Cpu,
  Gauge
} from 'lucide-react';

export default function PredictionResult({ result }) {
  if (!result) return null;

  const {
    prediction = '',
    predicted_class_index = 0,
    confidence = 0,
    risk_level = '',
    model_status = '',
    model_identifier = 'ResNet18',
    inference_latency_ms = null,
    analysis_date = new Date().toISOString(),
    class_probabilities = { Normal: 0, Osteopenia: 0, Osteoporosis: 0 },
  } = result;

  const confidencePct = (confidence * 100).toFixed(1);

  const getRiskBadgeClass = (risk) => {
    switch (risk?.toLowerCase()) {
      case 'low': return 'risk-badge-low';
      case 'moderate': return 'risk-badge-mod';
      case 'high': default: return 'risk-badge-high';
    }
  };

  const getPredictionClass = (pred) => {
    switch (pred?.toLowerCase()) {
      case 'normal': return 'pred-normal';
      case 'osteopenia': return 'pred-osteopenia';
      case 'osteoporosis': default: return 'pred-osteoporosis';
    }
  };

  const formattedDate = new Date(analysis_date).toLocaleString('en-US', {
    dateStyle: 'medium',
    timeStyle: 'medium',
  });

  return (
    <div id="prediction-section" className="prediction-result-wrapper">
      {/* Real Model Status Banner */}
      <div className="live-status-banner">
        <CheckCircle2 size={18} className="live-banner-icon" />
        <div className="live-banner-text">
          <strong>LIVE INFERENCE — {model_identifier}</strong>
          <span>Genuine ResNet18 model inference. No mock data. Scores are mathematical softmax outputs, not clinical probabilities.</span>
        </div>
      </div>

      <div className="result-main-card">
        <div className="result-header-row">
          <div>
            <div className="prediction-label-row">
              <span className="card-label">CNN Classification</span>
              <span className="live-badge-inline">LIVE MODEL</span>
            </div>
            <div className="prediction-headline-group">
              <h3 className={`prediction-category ${getPredictionClass(prediction)}`}>
                {prediction}
              </h3>
              <span className={`risk-level-badge ${getRiskBadgeClass(risk_level)}`}>
                Model-Predicted Category
              </span>
            </div>
          </div>

          <div className="confidence-radial-pill">
            <Gauge size={22} className="text-cyan" />
            <div className="confidence-pill-text">
              <span className="confidence-label">Model Score</span>
              <span className="confidence-number">{confidencePct}%</span>
            </div>
          </div>
        </div>

        {/* Primary Confidence Progress Bar */}
        <div className="confidence-bar-container">
          <div className="bar-label-row">
            <span>Top Class Softmax Score</span>
            <span className="bar-value-text">{confidencePct}%</span>
          </div>
          <div className="progress-track">
            <div
              className={`progress-fill ${getPredictionClass(prediction)}`}
              style={{ width: `${Math.min(confidencePct, 100)}%` }}
            />
          </div>
        </div>

        {/* 3-Class Score Breakdown */}
        <div className="classes-breakdown-section">
          <div className="breakdown-header">
            <BarChart3 size={16} />
            <h4>Multi-Class Softmax Score Distribution</h4>
          </div>

          <div className="classes-grid">
            {/* Normal */}
            <div className={`class-item-card ${prediction === 'Normal' ? 'class-active' : ''}`}>
              <div className="class-item-top">
                <span className="class-title">Normal</span>
                <span className="class-pct">{((class_probabilities.Normal || 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="mini-progress-track">
                <div
                  className="mini-progress-fill fill-normal"
                  style={{ width: `${(class_probabilities.Normal || 0) * 100}%` }}
                />
              </div>
              <span className="class-clinical-note">Reference: DXA T-Score ≥ −1.0</span>
            </div>

            {/* Osteopenia */}
            <div className={`class-item-card ${prediction === 'Osteopenia' ? 'class-active' : ''}`}>
              <div className="class-item-top">
                <span className="class-title">Osteopenia</span>
                <span className="class-pct">{((class_probabilities.Osteopenia || 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="mini-progress-track">
                <div
                  className="mini-progress-fill fill-osteopenia"
                  style={{ width: `${(class_probabilities.Osteopenia || 0) * 100}%` }}
                />
              </div>
              <span className="class-clinical-note">Reference: −2.5 &lt; DXA T-Score &lt; −1.0</span>
            </div>

            {/* Osteoporosis */}
            <div className={`class-item-card ${prediction === 'Osteoporosis' ? 'class-active' : ''}`}>
              <div className="class-item-top">
                <span className="class-title">Osteoporosis</span>
                <span className="class-pct">{((class_probabilities.Osteoporosis || 0) * 100).toFixed(1)}%</span>
              </div>
              <div className="mini-progress-track">
                <div
                  className="mini-progress-fill fill-osteoporosis"
                  style={{ width: `${(class_probabilities.Osteoporosis || 0) * 100}%` }}
                />
              </div>
              <span className="class-clinical-note">Reference: DXA T-Score ≤ −2.5</span>
            </div>
          </div>

          <div className="reference-standard-note">
            <Info size={14} className="note-icon" />
            <span>
              <strong>Educational Reference Notice:</strong> DXA T-score ranges are standard clinical references shown for educational information only.
              This application does <em>not</em> measure Bone Mineral Density or DXA T-scores from X-ray images.
              Scores shown above are raw CNN softmax outputs.
            </span>
          </div>
        </div>

        {/* Inference Metadata */}
        <div className="inference-meta-row">
          <div className="meta-chip">
            <Clock size={13} />
            <span>Inference: {inference_latency_ms != null ? `${inference_latency_ms.toFixed(0)} ms` : '—'}</span>
          </div>
          <div className="meta-chip">
            <Cpu size={13} />
            <span>{model_identifier}</span>
          </div>
          <div className="meta-chip">
            <Activity size={13} />
            <span>{model_status}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
