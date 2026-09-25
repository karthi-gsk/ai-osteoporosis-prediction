import React, { useState, useRef } from 'react';
import { 
  UploadCloud, 
  FileImage, 
  X, 
  RefreshCw, 
  CheckCircle, 
  AlertCircle, 
  Sparkles,
  FileCheck,
  Search,
  Eye
} from 'lucide-react';

const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png'];
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

export default function UploadSection({
  onAnalyze,
  onImageChange,
  isLoading,
  error,
  clearError,
  sectionRef,
  gradcamNotice,
  clearNotice,
}) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [clientError, setClientError] = useState('');
  const fileInputRef = useRef(null);

  // Helper to format bytes to human readable size
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const validateAndSetFile = (file) => {
    setClientError('');
    if (clearError) clearError();

    if (!file) return;

    // Check extension
    const extension = '.' + file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(extension)) {
      setClientError(`Unsupported file format "${extension}". Please upload a JPG, JPEG, or PNG X-ray image.`);
      return;
    }

    // Check size
    if (file.size > MAX_FILE_SIZE) {
      setClientError(`File is too large (${formatFileSize(file.size)}). Maximum supported file size is 10MB.`);
      return;
    }

    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleRemoveImage = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setClientError('');
    if (clearError) clearError();
    if (clearNotice) clearNotice();
    if (onImageChange) onImageChange();
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleTriggerFileSelect = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleAnalyzeClick = () => {
    if (!selectedFile) {
      setClientError('Please select or upload a bone X-ray image before running analysis.');
      return;
    }
    if (clearNotice) clearNotice();
    onAnalyze(selectedFile, previewUrl);
  };

  // Helper to load a synthetic clinical demo bone X-ray for instant testing
  const handleLoadSample = (sampleType = 'femur') => {
    // Generate a simulated high-contrast bone radiograph using Canvas
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');

    // Deep dark background mimicking radiographic negative
    ctx.fillStyle = '#060a12';
    ctx.fillRect(0, 0, 512, 512);

    // Radiographic noise texture
    for (let i = 0; i < 6000; i++) {
      const x = Math.random() * 512;
      const y = Math.random() * 512;
      const alpha = Math.random() * 0.12;
      ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
      ctx.fillRect(x, y, 1, 1);
    }

    if (sampleType === 'femur') {
      // Draw femoral head and neck contours
      ctx.strokeStyle = '#c8d6e5';
      ctx.lineWidth = 14;
      ctx.beginPath();
      // Femoral shaft
      ctx.moveTo(256, 480);
      ctx.lineTo(256, 260);
      // Femoral neck & head
      ctx.quadraticCurveTo(240, 180, 160, 160);
      ctx.stroke();

      // Femoral head glow
      const grad = ctx.createRadialGradient(160, 160, 10, 160, 160, 55);
      grad.addColorStop(0, 'rgba(230, 240, 255, 0.85)');
      grad.addColorStop(0.6, 'rgba(180, 200, 225, 0.45)');
      grad.addColorStop(1, 'rgba(100, 120, 150, 0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(160, 160, 55, 0, Math.PI * 2);
      ctx.fill();

      // Greater trochanter
      ctx.fillStyle = 'rgba(200, 215, 235, 0.6)';
      ctx.beginPath();
      ctx.ellipse(300, 240, 28, 45, Math.PI / 6, 0, Math.PI * 2);
      ctx.fill();

      // Trabecular micro-texture
      ctx.fillStyle = 'rgba(15, 23, 42, 0.4)';
      for (let i = 0; i < 40; i++) {
        const tx = 140 + Math.random() * 40;
        const ty = 140 + Math.random() * 40;
        ctx.fillRect(tx, ty, 2, 2);
      }
    } else {
      // Lumbar spine vertebrae outline
      ctx.fillStyle = 'rgba(210, 225, 245, 0.7)';
      for (let v = 0; v < 4; v++) {
        const vy = 100 + v * 90;
        ctx.beginPath();
        ctx.roundRect(176, vy, 160, 65, 12);
        ctx.fill();

        ctx.fillStyle = 'rgba(20, 30, 50, 0.4)';
        ctx.fillRect(196, vy + 15, 120, 35);
        ctx.fillStyle = 'rgba(210, 225, 245, 0.7)';
      }
    }

    // Add synthetic UI-test watermark & orientation
    ctx.fillStyle = '#94a3b8';
    ctx.font = '13px monospace';
    ctx.fillText('SYNTHETIC UI-TEST PATTERN (NOT A MEDICAL X-RAY)', 20, 35);
    ctx.fillText('512x512 • Procedural Canvas Simulation', 20, 55);

    canvas.toBlob((blob) => {
      const file = new File(
        [blob],
        `synthetic_ui_test_artifact.jpg`,
        { type: 'image/jpeg' }
      );
      validateAndSetFile(file);
    }, 'image/jpeg', 0.95);
  };

  const currentError = clientError || error;

  return (
    <section id="upload-section" ref={sectionRef} className="upload-section">
      <div className="section-header">
        <span className="section-eyebrow">Input Pipeline</span>
        <h2 className="section-title">Knee Radiograph Upload</h2>
        <p className="section-subtitle">
          Upload an Anteroposterior (AP) knee radiograph for ResNet18 deep learning inference.
          This model was trained and validated exclusively on AP knee radiographs; lateral views and other anatomical regions are unsupported.
        </p>
      </div>

      <div className="upload-container">
        {/* Grad-CAM Navigation Guidance Notice (Case A) */}
        {gradcamNotice && (
          <div className="gradcam-nav-notice" role="status" aria-live="polite">
            <div className="gradcam-notice-icon-box">
              <Eye size={20} className="gradcam-notice-icon" />
            </div>
            <div className="gradcam-notice-body">
              <div className="gradcam-notice-title">Grad-CAM Notice</div>
              <p className="gradcam-notice-text">{gradcamNotice}</p>
            </div>
            {clearNotice && (
              <button
                type="button"
                className="gradcam-notice-dismiss"
                onClick={clearNotice}
                aria-label="Dismiss notice"
                title="Dismiss notice"
              >
                <X size={16} />
              </button>
            )}
          </div>
        )}

        {/* Hidden File Input */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".jpg,.jpeg,.png,image/jpeg,image/png"
          onChange={handleFileChange}
          style={{ display: 'none' }}
          id="xray-file-input"
        />

        {/* Drag and Drop Zone */}
        {!selectedFile ? (
          <div
            className={`dropzone ${dragActive ? 'dropzone-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={handleTriggerFileSelect}
          >
            <div className="dropzone-inner">
              <div className="dropzone-icon-glow">
                <UploadCloud size={48} className="dropzone-icon" />
              </div>
              <h3 className="dropzone-title">Drag and drop your bone X-ray image here</h3>
              <p className="dropzone-formats">
                Supports: <strong>JPG</strong>, <strong>JPEG</strong>, <strong>PNG</strong> (Up to 10MB)
              </p>
              <button
                type="button"
                className="btn-browse"
                onClick={(e) => {
                  e.stopPropagation();
                  handleTriggerFileSelect();
                }}
              >
                <FileImage size={18} />
                <span>Browse Local Files</span>
              </button>
            </div>

            {/* Quick Demo Preloader Strip — Synthetic canvas pattern for UI testing only */}
            <div className="sample-loader-strip" onClick={(e) => e.stopPropagation()}>
              <span className="sample-strip-label">Or load a synthetic canvas test pattern for interface demonstration:</span>
              <button 
                type="button" 
                className="btn-sample" 
                onClick={() => handleLoadSample('femur')}
              >
                <Sparkles size={14} />
                <span>Load Synthetic Test Pattern</span>
              </button>
              <p className="sample-strip-note">
                Note: Synthetic test patterns are procedurally generated for UI testing and are not real radiographs. For meaningful model evaluation, upload a real Anteroposterior (AP) knee radiograph.
              </p>
            </div>
          </div>
        ) : (
          /* Image Selected & Preview Card */
          <div className="preview-card">
            <div className="preview-grid">
              <div className="preview-image-wrapper">
                <img
                  src={previewUrl}
                  alt="Uploaded X-ray scan preview"
                  className="preview-image"
                />
                <div className="preview-badge">
                  <FileCheck size={14} />
                  <span>Valid Image Upload</span>
                </div>
              </div>

              <div className="preview-metadata-col">
                <div className="preview-header">
                  <div>
                    <span className="file-status-tag">Ready For Analysis</span>
                    <h3 className="preview-filename" title={selectedFile.name}>
                      {selectedFile.name}
                    </h3>
                  </div>
                </div>

                <div className="file-spec-grid">
                  <div className="spec-card">
                    <span className="spec-label">File Size</span>
                    <span className="spec-value">{formatFileSize(selectedFile.size)}</span>
                  </div>
                  <div className="spec-card">
                    <span className="spec-label">Format</span>
                    <span className="spec-value">{selectedFile.type || 'image/jpeg'}</span>
                  </div>
                  <div className="spec-card">
                    <span className="spec-label">Target Architecture</span>
                    <span className="spec-value">ResNet18 (Frozen)</span>
                  </div>
                  <div className="spec-card">
                    <span className="spec-label">Status</span>
                    <span className="spec-value text-teal">Awaiting Inference</span>
                  </div>
                </div>

                <div className="preview-actions">
                  <button
                    type="button"
                    className="btn-analyze"
                    onClick={handleAnalyzeClick}
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <>
                        <RefreshCw size={18} className="spinner" />
                        <span>Processing with AI Service...</span>
                      </>
                    ) : (
                      <>
                        <Search size={18} />
                        <span>Analyze X-ray</span>
                      </>
                    )}
                  </button>

                  <div className="secondary-preview-buttons">
                    <button
                      type="button"
                      className="btn-replace"
                      onClick={handleTriggerFileSelect}
                      disabled={isLoading}
                    >
                      <RefreshCw size={15} />
                      <span>Replace Image</span>
                    </button>
                    <button
                      type="button"
                      className="btn-remove"
                      onClick={handleRemoveImage}
                      disabled={isLoading}
                    >
                      <X size={15} />
                      <span>Remove</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Error Feedback Display */}
        {currentError && (
          <div className="error-alert">
            <AlertCircle size={20} className="error-icon" />
            <div className="error-content">
              <strong>Validation Error:</strong> {currentError}
            </div>
            <button
              type="button"
              className="error-dismiss"
              onClick={() => {
                setClientError('');
                if (clearError) clearError();
              }}
            >
              <X size={16} />
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
