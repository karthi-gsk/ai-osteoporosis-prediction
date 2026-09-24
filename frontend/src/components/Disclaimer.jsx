import React from 'react';
import { AlertTriangle, ShieldCheck } from 'lucide-react';

export default function Disclaimer({ compact = false }) {
  if (compact) {
    return (
      <div className="disclaimer-compact">
        <AlertTriangle size={16} className="disclaimer-icon-amber" />
        <p>
          <strong>Academic Research Notice:</strong> This system is developed for academic and research purposes only.
          It is not a medical diagnostic tool and should not replace evaluation by qualified healthcare professionals.
        </p>
      </div>
    );
  }

  return (
    <div id="disclaimer-banner" className="disclaimer-banner">
      <div className="disclaimer-inner">
        <div className="disclaimer-icon-col">
          <div className="disclaimer-icon-bg">
            <AlertTriangle size={24} className="disclaimer-icon" />
          </div>
        </div>
        <div className="disclaimer-text-col">
          <div className="disclaimer-header">
            <h4>IMPORTANT RESEARCH & ACADEMIC DISCLAIMER</h4>
            <span className="disclaimer-tag">Non-Diagnostic Use Only</span>
          </div>
          <p className="disclaimer-body">
            This system is developed for academic and research purposes only.
            It is not a medical diagnostic tool and should not replace evaluation
            by qualified healthcare professionals.
          </p>
          <div className="disclaimer-footer-note">
            <span>Always consult a certified radiologist, orthopedic specialist, or healthcare provider for clinical bone mineral density (BMD / DXA) testing and medical diagnosis.</span>
          </div>
        </div>
      </div>
    </div>
  );
}
