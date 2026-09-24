import React from 'react';
import { Activity, ShieldAlert } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="footer-container">
      <div className="footer-inner">
        <div className="footer-top">
          <div className="footer-brand">
            <div className="brand-icon-wrapper-small">
              <Activity size={18} className="brand-icon" />
            </div>
            <span className="footer-title">AI Osteoporosis Prediction System</span>
          </div>
          <p className="footer-subtext">
            Knee X-Ray Analysis via ResNet18 CNN + Genuine Grad-CAM &bull; Academic &amp; Research Prototype
          </p>
        </div>

        <div className="footer-disclaimer-box">
          <ShieldAlert size={16} className="text-amber" />
          <p>
            Academic Research Project. Trained exclusively on knee radiographs.
            Not validated for hip, lumbar spine, or other anatomies.
            Not certified for clinical medical diagnostics or patient care.
          </p>
        </div>

        <div className="footer-bottom">
          <span>&copy; {new Date().getFullYear()} AI Healthcare Research Initiative. All rights reserved.</span>
          <span className="footer-tech-tag">
            React + Vite &bull; FastAPI &bull; PyTorch ResNet18 &bull; Grad-CAM
          </span>
        </div>
      </div>
    </footer>
  );
}
