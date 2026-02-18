import React, { useState } from "react";
import axios from "axios";

function LungPredictPage() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError("");
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a NIfTI (.nii) file.");
      return;
    }

    const formData = new FormData();
    formData.append("image_file", file);

    try {
      setLoading(true);
      setError("");
      const response = await axios.post("http://localhost:5000/predict", formData);
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError("Prediction failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: "2rem", maxWidth: "600px", margin: "auto" }}>
      <h2>Lung Segmentation Prediction</h2>
      <input type="file" accept=".nii,.nii.gz" onChange={handleFileChange} />
      <br />
      <button onClick={handleUpload} style={{ marginTop: "1rem" }}>
        {loading ? "Predicting..." : "Upload & Predict"}
      </button>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {result && result.success && (
        <div style={{ marginTop: "1rem" }}>
          <p><strong>Original File:</strong> {result.original_path}</p>
          <p><strong>Predicted Mask:</strong> {result.mask_path}</p>
          <a href={`http://localhost:5000/view_nifti`} target="_blank" rel="noopener noreferrer">
            View in NIfTI Viewer
          </a>
        </div>
      )}
    </div>
  );
}

export default LungPredictPage;
