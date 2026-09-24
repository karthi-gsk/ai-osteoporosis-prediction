import React, { useState, useRef, useEffect } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Disclaimer from './components/Disclaimer';
import UploadSection from './components/UploadSection';
import PredictionResult from './components/PredictionResult';
import GradCAMSection from './components/GradCAMSection';
import AnalysisSummary from './components/AnalysisSummary';
import AboutProject from './components/AboutProject';
import Footer from './components/Footer';
import { uploadAndPredict } from './services/api';
import './App.css';

export default function App() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [analysisError, setAnalysisError] = useState('');
  const [activePreviewUrl, setActivePreviewUrl] = useState(null);
  const [activeImageFile, setActiveImageFile] = useState(null);
  const [gradcamNotice, setGradcamNotice] = useState('');

  // Element refs for reliable, non-hardcoded scrolling
  const uploadSectionRef = useRef(null);
  const gradcamSectionRef = useRef(null);
  const predictionSectionRef = useRef(null);

  // Reliable scroll helper utilizing element refs with ID fallback
  const scrollToTarget = (targetRef, fallbackId, options = { behavior: 'smooth', block: 'start' }) => {
    const element = targetRef?.current || (fallbackId ? document.getElementById(fallbackId) : null);
    if (element) {
      element.scrollIntoView(options);
      return true;
    }
    return false;
  };

  // Handle URL hash navigation on direct load or browser refresh
  useEffect(() => {
    const hash = window.location.hash;
    if (hash === '#explainability-section') {
      if (!predictionResult) {
        // Case A on refresh/direct load: No prediction exists yet
        setGradcamNotice('Upload and analyze a knee X-ray to view its Grad-CAM explanation.');
        setTimeout(() => {
          scrollToTarget(uploadSectionRef, 'upload-section');
        }, 120);
      } else {
        // Case B on refresh: prediction is available
        setTimeout(() => {
          scrollToTarget(gradcamSectionRef, 'explainability-section');
        }, 120);
      }
    } else if (hash === '#upload-section') {
      setTimeout(() => {
        scrollToTarget(uploadSectionRef, 'upload-section');
      }, 120);
    } else if (hash === '#about-section') {
      setTimeout(() => {
        scrollToTarget(null, 'about-section');
      }, 120);
    }
  }, []);

  // Handle Grad-CAM navigation from Navbar (Desktop & Mobile)
  const handleNavGradCAM = () => {
    if (predictionResult) {
      // CASE B — Prediction available: smoothly scroll to genuine Grad-CAM section
      setGradcamNotice('');
      window.history.pushState(null, '', '#explainability-section');

      const scrolled = scrollToTarget(gradcamSectionRef, 'explainability-section');
      if (!scrolled) {
        setTimeout(() => {
          scrollToTarget(gradcamSectionRef, 'explainability-section');
        }, 50);
      }
    } else {
      // CASE A — No image analyzed: smoothly scroll to upload section with clear guidance
      setGradcamNotice('Upload and analyze a knee X-ray to view its Grad-CAM explanation.');
      window.history.pushState(null, '', '#upload-section');

      scrollToTarget(uploadSectionRef, 'upload-section');
    }
  };

  // Handle standard section navigation
  const handleNavSection = (sectionId) => {
    window.history.pushState(null, '', `#${sectionId}`);
    if (sectionId === 'upload-section') {
      scrollToTarget(uploadSectionRef, 'upload-section');
    } else {
      scrollToTarget(null, sectionId);
    }
  };

  const handleAnalyzeXray = async (file, previewUrl) => {
    // Prevent duplicate requests while one is in progress
    if (isAnalyzing) return;

    setIsAnalyzing(true);
    setAnalysisError('');
    setGradcamNotice('');
    setActivePreviewUrl(previewUrl);
    setActiveImageFile(file);
    setPredictionResult(null);

    try {
      const resultData = await uploadAndPredict(file);
      setPredictionResult(resultData);

      // Handle navigation after asynchronous API response mounts result container
      setTimeout(() => {
        if (window.location.hash === '#explainability-section') {
          scrollToTarget(gradcamSectionRef, 'explainability-section');
        } else {
          scrollToTarget(predictionSectionRef, 'prediction-section');
        }
      }, 150);
    } catch (err) {
      setAnalysisError(err.message || 'An unexpected error occurred during radiograph analysis.');
      setActiveImageFile(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleClearError = () => {
    setAnalysisError('');
  };

  // When the user uploads a new image, clear old results
  const handleImageChange = () => {
    setPredictionResult(null);
    setAnalysisError('');
    setGradcamNotice('');
    setActivePreviewUrl(null);
    setActiveImageFile(null);
  };

  return (
    <div className="app-container">
      <Navbar
        hasPrediction={Boolean(predictionResult)}
        onNavigateGradCAM={handleNavGradCAM}
        onNavigateSection={handleNavSection}
      />

      <main className="main-content">
        {/* 1. Hero Section */}
        <Hero onStartClick={() => scrollToTarget(uploadSectionRef, 'upload-section')} />

        {/* 2. Research Disclaimer Banner */}
        <Disclaimer />

        {/* 3. Knee X-Ray Upload Section */}
        <UploadSection
          sectionRef={uploadSectionRef}
          onAnalyze={handleAnalyzeXray}
          onImageChange={handleImageChange}
          isLoading={isAnalyzing}
          error={analysisError}
          clearError={handleClearError}
          gradcamNotice={gradcamNotice}
          clearNotice={() => setGradcamNotice('')}
        />

        {/* 4. Results Section (only shown after real inference) */}
        {predictionResult && (
          <section className="results-container">
            {/* 4a. Prediction Scores */}
            <PredictionResult result={predictionResult} />

            {/* 4b. Genuine Grad-CAM Explainability */}
            <GradCAMSection
              sectionRef={gradcamSectionRef}
              imageFile={activeImageFile}
              imagePreviewUrl={activePreviewUrl}
              result={predictionResult}
            />

            {/* 4c. Full Analysis Summary */}
            <AnalysisSummary
              result={predictionResult}
              imagePreviewUrl={activePreviewUrl}
              uploadedFileName={activeImageFile?.name}
            />
          </section>
        )}

        {/* 5. About the Research Project */}
        <AboutProject />
      </main>

      <Footer />
    </div>
  );
}
