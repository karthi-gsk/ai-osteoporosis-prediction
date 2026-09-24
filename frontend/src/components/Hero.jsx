import React from 'react';
import { ArrowDown, Cpu, Eye, Microscope, ShieldCheck, Activity } from 'lucide-react';

export default function Hero({ onStartClick }) {
  const handleScrollToUpload = (e) => {
    e.preventDefault();
    const uploadElement = document.getElementById('upload-section');
    if (uploadElement) {
      uploadElement.scrollIntoView({ behavior: 'smooth' });
    }
    if (onStartClick) onStartClick();
  };

  return (
    <section id="hero" className="hero-section">
      <div className="hero-content">
        <div className="hero-pill-badge">
          <Microscope size={15} className="hero-badge-icon" />
          <span>Academic &amp; Research Platform • Knee X-Ray Analysis • ResNet18 CNN</span>
        </div>

        <h1 className="hero-title">
          AI-Powered Osteoporosis <br />
          <span className="gradient-text">Risk Analysis</span>
        </h1>

        <h2 className="hero-subtitle">
          Explainable Deep Learning Based Knee Bone Pattern Recognition
        </h2>

        <p className="hero-description">
          Upload a knee X-ray image to analyze it using our trained ResNet18 model with
          genuine Grad-CAM explainability. Developed exclusively for academic and research purposes.
        </p>

        <div className="hero-actions">
          <button className="btn-primary" onClick={handleScrollToUpload}>
            <span>Start Analysis</span>
            <ArrowDown size={18} />
          </button>
          <a href="#about-section" className="btn-secondary">
            <span>Research Overview</span>
          </a>
        </div>

        <div className="hero-features-strip">
          <div className="feature-item">
            <Cpu size={18} className="feature-icon" />
            <div className="feature-text">
              <strong>ResNet18 CNN</strong>
              <span>Trained on Knee Radiographs</span>
            </div>
          </div>
          <div className="feature-item">
            <Eye size={18} className="feature-icon" />
            <div className="feature-text">
              <strong>Genuine Grad-CAM</strong>
              <span>Real Feature Attribution Maps</span>
            </div>
          </div>
          <div className="feature-item">
            <ShieldCheck size={18} className="feature-icon" />
            <div className="feature-text">
              <strong>Research Safeguard</strong>
              <span>Non-Diagnostic Boundary</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
