import React from 'react';
import { BookOpen, Network, Shield, Cpu, Eye, FlaskConical } from 'lucide-react';

export default function AboutProject() {
  return (
    <section id="about-section" className="about-section">
      <div className="section-header">
        <span className="section-eyebrow">Academic Scope &amp; Objectives</span>
        <h2 className="section-title">About The Research Project</h2>
        <p className="section-subtitle">
          Investigating CNN-based knee radiograph pattern recognition with explainable AI explainability.
        </p>
      </div>

      <div className="about-grid">
        <div className="about-card main-objective-card">
          <div className="about-icon-pill">
            <BookOpen size={20} className="text-teal" />
          </div>
          <h3>Research Purpose &amp; Scope</h3>
          <p>
            The <strong>AI Osteoporosis Prediction System</strong> is an academic research initiative
            studying how convolutional neural networks (CNNs) can identify structural patterns
            relevant to bone mineral density loss in conventional knee radiographic imaging.
          </p>
          <p className="academic-highlight">
            This project focuses on algorithmic feasibility, feature extraction, and explainability
            through Grad-CAM. <strong>It does not make claims that osteoporosis can be definitively
            diagnosed using this application.</strong>
          </p>
        </div>

        <div className="about-card">
          <div className="about-icon-pill">
            <Cpu size={20} className="text-cyan" />
          </div>
          <h3>Trained CNN Architecture</h3>
          <p>This deployment uses a <strong>ResNet18</strong> model trained on knee X-ray radiographs:</p>
          <ul className="methodology-list">
            <li>
              <strong>Architecture:</strong> ResNet18 (11.2M parameters, torchvision pretrained base)
            </li>
            <li>
              <strong>Training Data:</strong> Anteroposterior (AP) knee X-ray dataset — Normal, Osteopenia, Osteoporosis (Mendeley Data DOI: 10.17632/fxjm8fb6mw.2)
            </li>
            <li>
              <strong>Best Checkpoint:</strong> Epoch 21, Validation Macro F1: 0.7913
            </li>
            <li>
              <strong>Final Test Performance:</strong> Accuracy 80.74%, Macro F1 80.72%, ROC-AUC 91.74%
            </li>
            <li>
              <strong>Inference:</strong> CPU-only, frozen weights, no parameter updates
            </li>
          </ul>
        </div>

        <div className="about-card">
          <div className="about-icon-pill">
            <Eye size={20} className="text-cyan" />
          </div>
          <h3>Grad-CAM Explainability</h3>
          <p>
            Genuine Gradient-weighted Class Activation Mapping (Grad-CAM) is used for model transparency:
          </p>
          <ul className="methodology-list">
            <li>
              <strong>Target Layer:</strong> ResNet18 <code>layer4[-1]</code> (512 channels, 7×7 spatial map)
            </li>
            <li>
              <strong>Method:</strong> Selvaraju et al. ICCV 2017 — class-discriminative localization via gradient global average pooling
            </li>
            <li>
              <strong>Output:</strong> JET colormap heatmap overlaid on original radiograph at α=0.45
            </li>
            <li>
              <strong>Note:</strong> Heatmaps indicate CNN feature attribution, not clinically verified bone mineral loss regions
            </li>
          </ul>
        </div>

        <div className="about-card">
          <div className="about-icon-pill">
            <Shield size={20} className="text-amber" />
          </div>
          <h3>Ethical &amp; Regulatory Boundaries</h3>
          <p>
            Gold-standard clinical evaluation of bone mineral density relies on Dual-Energy X-ray Absorptiometry
            (DXA) scans measuring BMD T-scores. <strong>This application does not measure BMD
            or calculate DXA T-scores from X-ray images.</strong>
          </p>
          <p>
            The model was trained and validated exclusively on Anteroposterior (AP) knee radiographs.
            It has <em>not</em> been validated for lateral knee projections, hip, lumbar spine, femoral neck, or other anatomical regions.
            All predictions are research demonstrations and must never be used as clinical diagnoses or medical advice.
          </p>
        </div>
      </div>
    </section>
  );
}
