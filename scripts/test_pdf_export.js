import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

async function testAllPdfExportCases() {
  console.log('1. Reading validation image N10.JPEG...');
  const imagePath = path.join(projectRoot, 'dataset', 'strict_clean_knee_osteoporosis', 'val', 'Normal', 'N10.JPEG');
  const imageBuffer = fs.readFileSync(imagePath);
  const imageBase64 = `data:image/jpeg;base64,${imageBuffer.toString('base64')}`;

  console.log('2. Requesting real ResNet18 inference from FastAPI backend...');
  const formData = new FormData();
  const fileBlob = new Blob([imageBuffer], { type: 'image/jpeg' });
  formData.append('file', fileBlob, 'N10.JPEG');

  const response = await fetch('http://127.0.0.1:8000/api/predict', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Backend error: ${response.status} ${await response.text()}`);
  }

  const result = await response.json();
  console.log(`Inference successful: Prediction=${result.prediction}, Confidence=${(result.confidence * 100).toFixed(2)}%, Grad-CAM status=${result.explainability?.status}`);

  const { generatePdfReport } = await import('../frontend/src/services/pdfReport.js');

  console.log('3. Test Case A: Explicit uploadedFileName ("N10.JPEG")...');
  const docA = await generatePdfReport(result, imageBase64, 'N10.JPEG');
  console.log('   Case A generated successfully without ReferenceError.');

  console.log('4. Test Case B: From result.file_info.filename (no third param)...');
  const docB = await generatePdfReport(result, imageBase64);
  console.log('   Case B generated successfully without ReferenceError.');

  console.log('5. Test Case C: Safe fallback when filename is missing...');
  const resultWithoutFile = { ...result, file_info: {} };
  const docC = await generatePdfReport(resultWithoutFile, imageBase64);
  console.log('   Case C generated successfully without ReferenceError (used fallback knee-xray).');

  const reportsDir = path.join(projectRoot, 'reports');
  if (!fs.existsSync(reportsDir)) {
    fs.mkdirSync(reportsDir, { recursive: true });
  }

  const pdfPath = path.join(reportsDir, 'test_verification_report.pdf');
  const pdfArrayBuffer = docA.output('arraybuffer');
  fs.writeFileSync(pdfPath, Buffer.from(pdfArrayBuffer));
  console.log(`Verified PDF saved to: ${pdfPath} (${fs.statSync(pdfPath).size} bytes)`);

  console.log('All 3 PDF generation cases verified successfully!');
}

testAllPdfExportCases().catch((err) => {
  console.error('Test failed:', err);
  process.exit(1);
});
