import React, { useState } from 'react';
import { FileDown, Loader, CheckCircle2, AlertCircle } from 'lucide-react';
import { generatePdfReport } from '../services/pdfReport';

/**
 * PDF Export Button component.
 * Renders a single CTA button that generates and downloads the PDF report.
 *
 * @param {Object}  result          - Full inference result from FastAPI
 * @param {string}  imagePreviewUrl - The original uploaded image object URL
 */
export default function PdfExportButton({ result, imagePreviewUrl, uploadedFileName }) {
  const [status, setStatus] = useState('idle'); // idle | generating | done | error
  const [errorMsg, setErrorMsg] = useState('');

  const handleExport = async () => {
    if (!result || status === 'generating') return;

    setStatus('generating');
    setErrorMsg('');

    try {
      await generatePdfReport(result, imagePreviewUrl, uploadedFileName);
      setStatus('done');
      // Reset back to idle after 3 seconds
      setTimeout(() => setStatus('idle'), 3000);
    } catch (err) {
      setErrorMsg(err.message || 'PDF generation failed.');
      setStatus('error');
      setTimeout(() => setStatus('idle'), 5000);
    }
  };

  const isGenerating = status === 'generating';
  const isDone       = status === 'done';
  const isError      = status === 'error';

  return (
    <div className="pdf-export-wrapper">
      <button
        type="button"
        id="pdf-export-btn"
        className={`pdf-export-btn ${isDone ? 'pdf-btn-done' : ''} ${isError ? 'pdf-btn-error' : ''}`}
        onClick={handleExport}
        disabled={!result || isGenerating}
        title={!result ? 'Run analysis first to export report' : 'Download PDF research report'}
      >
        {isGenerating ? (
          <>
            <Loader size={16} className="spinning" />
            <span>Generating PDF...</span>
          </>
        ) : isDone ? (
          <>
            <CheckCircle2 size={16} />
            <span>Report Downloaded</span>
          </>
        ) : isError ? (
          <>
            <AlertCircle size={16} />
            <span>Export Failed</span>
          </>
        ) : (
          <>
            <FileDown size={16} />
            <span>Export PDF Report</span>
          </>
        )}
      </button>

      {isError && errorMsg && (
        <p className="pdf-error-text">{errorMsg}</p>
      )}
    </div>
  );
}
