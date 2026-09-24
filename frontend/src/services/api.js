/**
 * API Service for communicating with the FastAPI Osteoporosis Prediction Backend.
 * Connects to genuine ResNet18 inference and Grad-CAM generation.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const REQUEST_TIMEOUT_MS = 60000; // 60 s — model loading on first request can be slow

/**
 * Wraps fetch() with a configurable timeout.
 */
async function fetchWithTimeout(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timerId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    return response;
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new Error(
        `Request timed out after ${timeoutMs / 1000}s. The backend may be under heavy load or unreachable.`
      );
    }
    throw err;
  } finally {
    clearTimeout(timerId);
  }
}

/**
 * Uploads an X-ray image to the FastAPI backend for ResNet18 osteoporosis inference.
 *
 * @param {File} imageFile - The knee X-ray image file (JPG, JPEG, PNG)
 * @param {number|null} targetClass - Optional Grad-CAM target class override (0=Normal, 1=Osteopenia, 2=Osteoporosis)
 * @returns {Promise<Object>} The real prediction results from FastAPI
 */
export async function uploadAndPredict(imageFile, targetClass = null) {
  if (!imageFile) {
    throw new Error('No image file selected for analysis.');
  }

  // Client-side format check (authoritative validation is on the backend)
  const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
  if (!validTypes.includes(imageFile.type) && !imageFile.name.match(/\.(jpe?g|png)$/i)) {
    throw new Error('Invalid file format. Only JPG, JPEG, and PNG images are supported.');
  }

  const formData = new FormData();
  formData.append('file', imageFile);

  // Build URL with optional target_class query parameter
  let url = `${API_BASE_URL}/api/predict`;
  if (targetClass !== null && targetClass !== undefined) {
    url += `?target_class=${encodeURIComponent(targetClass)}`;
  }

  try {
    const response = await fetchWithTimeout(
      url,
      {
        method: 'POST',
        body: formData,
        // Do NOT manually set Content-Type — browser must set it with the multipart boundary
      },
      REQUEST_TIMEOUT_MS
    );

    if (!response.ok) {
      let errorMessage = `Server error: ${response.status} ${response.statusText}`;
      try {
        const errorData = await response.json();
        if (errorData?.detail) {
          errorMessage = Array.isArray(errorData.detail)
            ? errorData.detail.map((e) => e.msg || e).join('; ')
            : errorData.detail;
        }
      } catch {
        /* fallback to status */
      }

      // Specific error messages for known status codes
      if (response.status === 503) {
        throw new Error(
          `Trained AI model is not available: ${errorMessage}. Please ensure the FastAPI backend started correctly with the checkpoint loaded.`
        );
      }
      if (response.status === 413) {
        throw new Error('Image file exceeds maximum allowed size of 10 MB. Please upload a smaller file.');
      }
      throw new Error(errorMessage);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    if (error.message.includes('timed out') || error.message.includes('AbortError')) {
      throw error;
    }
    if (error.name === 'TypeError' && error.message.toLowerCase().includes('fetch')) {
      throw new Error(
        `Could not connect to FastAPI backend (${API_BASE_URL}). Please ensure the backend server is running.`
      );
    }
    throw error;
  }
}

/**
 * Checks backend health status and returns full health payload.
 * @returns {Promise<Object|null>} Health data or null if unavailable
 */
export async function checkBackendHealth() {
  try {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/health`,
      { method: 'GET' },
      5000 // short timeout for health checks
    );
    if (!response.ok) return null;
    const data = await response.json();
    return data;
  } catch {
    return null;
  }
}
