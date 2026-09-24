import React, { useState } from 'react';
import { Eye, Info, Loader, RefreshCw, ZoomIn, ZoomOut, AlertCircle } from 'lucide-react';
import { uploadAndPredict } from '../services/api';

const CLASS_TARGETS = [
  { label: 'Normal', index: 0, colorClass: 'target-normal' },
  { label: 'Osteopenia', index: 1, colorClass: 'target-osteopenia' },
  { label: 'Osteoporosis', index: 2, colorClass: 'target-osteoporosis' },
];

const VIEW_MODES = [
  { key: 'overlay', label: 'CAM Overlay' },
  { key: 'heatmap', label: 'Heatmap Only' },
  { key: 'original', label: 'Original' },
];

export default function GradCAMSection({ imageFile, imagePreviewUrl, result, onGradCAMUpdate, sectionRef }) {
  const [viewMode, setViewMode] = useState('overlay');
  const [selectedTarget, setSelectedTarget] = useState(null); // null = predicted class
  const [isRefetching, setIsRefetching] = useState(false);
  const [refetchError, setRefetchError] = useState('');
  const [localResult, setLocalResult] = useState(null);

  // Use local result (from target-class refetch) if available, else prop result
  const activeResult = localResult || result;

  const overlayDataUri = activeResult?.explainability?.overlay_image || null;
  const heatmapDataUri = activeResult?.explainability?.heatmap_image || null;
  const targetClass = activeResult?.explainability?.target_class || activeResult?.prediction || null;
  const targetLayer = activeResult?.explainability?.target_layer || 'model.layer4[-1] (BasicBlock 2, 512 channels, 7×7)';
  const gradcamStatus = activeResult?.explainability?.status;
  const gradcamMessage = activeResult?.explainability?.message || '';

  const handleTargetClassChange = async (targetIndex) => {
    if (!imageFile) return;

    // If already selected, deselect (revert to predicted class)
    if (selectedTarget === targetIndex) {
      setSelectedTarget(null);
      setLocalResult(null);
      setRefetchError('');
      return;
    }

    setSelectedTarget(targetIndex);
    setIsRefetching(true);
    setRefetchError('');

    try {
      const data = await uploadAndPredict(imageFile, targetIndex);
      setLocalResult(data);
      if (onGradCAMUpdate) onGradCAMUpdate(data);
    } catch (err) {
      setRefetchError(err.message || 'Failed to generate class-specific Grad-CAM.');
      setSelectedTarget(null);
      setLocalResult(null);
    } finally {
      setIsRefetching(false);
    }
  };

  // Display image for current view mode
  const getDisplayImage = () => {
    if (viewMode === 'overlay') return overlayDataUri || imagePreviewUrl;
    if (viewMode === 'heatmap') return heatmapDataUri;
    return imagePreviewUrl;
  };

  const displayImage = getDisplayImage();
  const hasRealGradCAM = gradcamStatus === 'success' && overlayDataUri;

  return (
    <div id="explainability-section" ref={sectionRef} className="gradcam-card">
      <div className="gradcam-header">
        <div className="gradcam-title-group">
          <div className="gradcam-icon-box">
            <Eye size={20} className="text-cyan" />
          </div>
          <div>
            <h3 className="gradcam-heading">Grad-CAM Explainability</h3>
            <span className="gradcam-tag">
              {hasRealGradCAM ? 'Genuine Gradient-Weighted Class Activation Mapping' : 'Gradient-Weighted Class Activation Mapping'}
            </span>
          </div>
        </div>

        {hasRealGradCAM && (
          <div className="gradcam-live-badge">
            <span className="pulse-dot" />
            <span>LIVE GRAD-CAM</span>
          </div>
        )}
      </div>

      {/* Description */}
      <p className="gradcam-description">
        {hasRealGradCAM
          ? `Heatmap computed via backward gradients at ResNet18 ${targetLayer}. Highlighted regions drove the model's "${targetClass}" score — not medically verified anatomical findings.`
          : result
          ? 'Awaiting Grad-CAM data from backend...'
          : 'Upload a knee X-ray and run analysis to view genuine Grad-CAM activation maps.'}
      </p>

      {/* View Mode Selector */}
      {hasRealGradCAM && (
        <div className="view-mode-tabs">
          {VIEW_MODES.map((mode) => (
            <button
              key={mode.key}
              type="button"
              className={`view-tab-btn ${viewMode === mode.key ? 'view-tab-active' : ''}`}
              onClick={() => setViewMode(mode.key)}
            >
              {mode.label}
            </button>
          ))}
        </div>
      )}

      {/* Main Viewport */}
      <div className="heatmap-viewport">
        {displayImage ? (
          <div className="heatmap-relative-container">
            <img
              src={displayImage}
              alt={
                viewMode === 'heatmap'
                  ? 'Grad-CAM standalone heatmap'
                  : viewMode === 'overlay'
                  ? 'Grad-CAM overlay on X-ray'
                  : 'Original uploaded X-ray'
              }
              className="heatmap-base-image"
            />
            {hasRealGradCAM && (
              <div className="overlay-live-badge">
                <span>
                  {viewMode === 'overlay' && `CAM Overlay — Target: ${targetClass}`}
                  {viewMode === 'heatmap' && `Heatmap — Target: ${targetClass}`}
                  {viewMode === 'original' && 'Original X-ray'}
                </span>
              </div>
            )}
            {isRefetching && (
              <div className="heatmap-loading-overlay">
                <Loader size={28} className="spinning" />
                <span>Generating Grad-CAM...</span>
              </div>
            )}
          </div>
        ) : (
          <div className="heatmap-empty-state">
            <div className="empty-radar-circle">
              <Eye size={36} className="text-muted" />
            </div>
            <h4>Empty Grad-CAM Viewport</h4>
            <p>Upload a knee radiograph and run analysis to view activation maps.</p>
          </div>
        )}
      </div>

      {/* Target Class Controls */}
      {result && imageFile && (
        <div className="target-class-controls">
          <div className="target-class-label">
            <Info size={14} />
            <span>Re-generate Grad-CAM for a specific class:</span>
          </div>
          <div className="target-class-buttons">
            {CLASS_TARGETS.map((cls) => (
              <button
                key={cls.index}
                type="button"
                className={`target-btn ${cls.colorClass} ${selectedTarget === cls.index ? 'target-btn-active' : ''}`}
                onClick={() => handleTargetClassChange(cls.index)}
                disabled={isRefetching}
                title={`Generate Grad-CAM targeting the "${cls.label}" class logit`}
              >
                {isRefetching && selectedTarget === cls.index ? (
                  <Loader size={12} className="spinning" />
                ) : null}
                {cls.label}
              </button>
            ))}
            {selectedTarget !== null && (
              <button
                type="button"
                className="target-btn target-btn-reset"
                onClick={() => { setSelectedTarget(null); setLocalResult(null); setRefetchError(''); }}
                disabled={isRefetching}
              >
                <RefreshCw size={12} />
                Reset to Predicted
              </button>
            )}
          </div>
          {refetchError && (
            <div className="gradcam-error-inline">
              <AlertCircle size={14} />
              <span>{refetchError}</span>
            </div>
          )}
        </div>
      )}

      {/* Color Scale Legend */}
      <div className="cam-legend-bar">
        <span className="legend-label">Gradient Attribution:</span>
        <div className="legend-gradient" />
        <div className="legend-markers">
          <span>Low</span>
          <span>Medium</span>
          <span>High</span>
        </div>
      </div>

      {/* Methodology Note */}
      <div className="architecture-note">
        <Info size={15} className="note-icon" />
        <p>
          <strong>Methodology:</strong> Grad-CAM computes gradients of the target class logit with respect
          to the final convolutional feature maps at <code>layer4[-1]</code> of ResNet18 (512 channels, 7×7 spatial map).
          Heatmaps highlight CNN feature attribution regions — <em>not</em> medically verified bone mineral loss,
          DXA T-scores, or anatomical segmentations.
        </p>
      </div>
    </div>
  );
}
