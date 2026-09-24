import React, { useEffect, useState } from 'react';
import { Activity, ShieldAlert, CheckCircle2, AlertCircle, Loader, Menu, X } from 'lucide-react';
import { checkBackendHealth } from '../services/api';

export default function Navbar({ hasPrediction, onNavigateGradCAM, onNavigateSection }) {
  const [healthData, setHealthData] = useState(null);
  const [checking, setChecking] = useState(true);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    let isMounted = true;

    async function verifyHealth() {
      const data = await checkBackendHealth();
      if (isMounted) {
        setHealthData(data);
        setChecking(false);
      }
    }

    verifyHealth();
    const interval = setInterval(verifyHealth, 20000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const isOnline = healthData?.status === 'operational';
  const modelLoaded = healthData?.model_loaded === true;
  const modelReady = isOnline && modelLoaded;

  const handleNavClick = (e, sectionId) => {
    e.preventDefault();
    setMobileMenuOpen(false);
    if (onNavigateSection) {
      onNavigateSection(sectionId);
    } else {
      const el = document.getElementById(sectionId);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        window.history.pushState(null, '', `#${sectionId}`);
      }
    }
  };

  const handleGradCAMClick = (e) => {
    e.preventDefault();
    setMobileMenuOpen(false);
    if (onNavigateGradCAM) {
      onNavigateGradCAM();
    } else {
      const el = document.getElementById('explainability-section');
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        window.history.pushState(null, '', '#explainability-section');
      }
    }
  };

  return (
    <header className="navbar-container">
      <div className="navbar-inner">
        <a href="#hero" className="navbar-brand" onClick={(e) => handleNavClick(e, 'hero')}>
          <div className="brand-icon-wrapper">
            <Activity className="brand-icon" size={24} />
          </div>
          <div className="brand-text">
            <span className="brand-title">AI Osteoporosis Prediction System</span>
            <span className="brand-badge">Academic Research</span>
          </div>
        </a>

        <nav className="navbar-nav">
          <a href="#hero" className="nav-link" onClick={(e) => handleNavClick(e, 'hero')}>Home</a>
          <a href="#upload-section" className="nav-link" onClick={(e) => handleNavClick(e, 'upload-section')}>Analysis</a>
          <a
            href="#explainability-section"
            className="nav-link"
            onClick={handleGradCAMClick}
            title={hasPrediction ? "View genuine Grad-CAM activation maps" : "Upload an image to view Grad-CAM"}
          >
            Grad-CAM
          </a>
          <a href="#about-section" className="nav-link" onClick={(e) => handleNavClick(e, 'about-section')}>About Project</a>
        </nav>

        <div className="navbar-actions">
          {checking ? (
            <div className="status-pill status-pending" title="Checking backend connection...">
              <Loader size={12} className="status-dot-icon spinning" />
              <span className="status-text">Connecting...</span>
            </div>
          ) : modelReady ? (
            <div className="status-pill status-online" title={`ResNet18 loaded — Best Epoch: ${healthData.checkpoint_best_epoch}, Val F1: ${healthData.checkpoint_best_val_f1}`}>
              <span className="status-dot" />
              <span className="status-text">ResNet18 Ready</span>
            </div>
          ) : isOnline ? (
            <div className="status-pill status-warn" title="Backend online but model not loaded">
              <span className="status-dot status-dot-warn" />
              <span className="status-text">Model Loading...</span>
            </div>
          ) : (
            <div className="status-pill status-offline" title="FastAPI backend is offline">
              <span className="status-dot status-dot-offline" />
              <span className="status-text">Backend Offline</span>
            </div>
          )}

          <a href="#disclaimer-banner" className="nav-disclaimer-btn">
            <ShieldAlert size={14} />
            <span>Research Only</span>
          </a>

          <button
            type="button"
            className="mobile-nav-toggle"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label={mobileMenuOpen ? "Close navigation menu" : "Open navigation menu"}
          >
            {mobileMenuOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </div>

      {mobileMenuOpen && (
        <div className="mobile-nav-dropdown">
          <a href="#hero" className="mobile-nav-link" onClick={(e) => handleNavClick(e, 'hero')}>
            Home
          </a>
          <a href="#upload-section" className="mobile-nav-link" onClick={(e) => handleNavClick(e, 'upload-section')}>
            Analysis
          </a>
          <a href="#explainability-section" className="mobile-nav-link" onClick={handleGradCAMClick}>
            Grad-CAM
          </a>
          <a href="#about-section" className="mobile-nav-link" onClick={(e) => handleNavClick(e, 'about-section')}>
            About Project
          </a>
        </div>
      )}
    </header>
  );
}
