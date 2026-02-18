import React, { useState } from 'react';
import axios from 'axios';

const BASE_URL = 'http://localhost:5001/api';

// Upload image function
const uploadImage = async (imageFile, modelType, onUploadProgress) => {
  const formData = new FormData();
  formData.append("image", imageFile);
  formData.append("modelType", modelType);

  try {
    const response = await axios.post(`${BASE_URL}/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress,
    });
    return response.data;
  } catch (error) {
    return { success: false, error: error.response?.data?.message || error.message };
  }
};

// Upload .nii.gz file function
const uploadNiiFile = async (file, onUploadProgress) => {
  const formData = new FormData();
  formData.append("image_file", file);

  try {
    const response = await axios.post(`http://localhost:8000/predict/segmentation/nii`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress,
    });
    return response.data;
  } catch (error) {
    return { success: false, error: error.response?.data?.message || error.message };
  }
};

const ThreeDVis = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [progress, setProgress] = useState(0);

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setUploadError(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadError('No file selected.');
      return;
    }

    setUploading(true);
    setUploadError(null);
    setProgress(0);

    try {
      let response;
      if (selectedFile.name.endsWith('.nii.gz')) {
        response = await uploadNiiFile(selectedFile, (e) => {
          if (e.lengthComputable) {
            setProgress(Math.round((e.loaded * 100) / e.total));
          }
        });
      } else {
        response = await uploadImage(selectedFile, 'lung', (e) => {
          if (e.lengthComputable) {
            setProgress(Math.round((e.loaded * 100) / e.total));
          }
        });
      }

      if (response && response.success) {
        const originalPath = encodeURIComponent(response.originalImage);
        const predictedPath = encodeURIComponent(response.predictedImage || '');
        const papayaViewerUrl = `http://localhost:8000/papaya?originalPath=${originalPath}&predictedPath=${predictedPath}`;
        window.location.href = papayaViewerUrl;
      } else {
        setUploadError(response?.error || 'Upload failed');
      }
    } catch (error) {
      setUploadError(error.message || 'An unexpected error occurred');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="visualization-container">
      <header className="visualization-header">
        <h1>3D Diagnosis Viewer</h1>
        <p>Visualize and interact with 3D medical imaging data.</p>
      </header>

      <main className="visualization-main">
        <input
          type="file"
          accept=".nii.gz,image/*"
          hidden
          id="fileUpload"
          onChange={handleFileChange}
        />
        <label htmlFor="fileUpload" className="upload-label">
          {selectedFile ? selectedFile.name : 'Choose File to Upload'}
        </label>

        <button
          className="predict-button"
          onClick={handleUpload}
          disabled={uploading || !selectedFile}
        >
          {selectedFile?.name.endsWith('.nii.gz') ? 'Predict & Visualize' : 'View'}
        </button>

        {uploading && (
          <div className="progress-section">
            <p>Uploading and Processing...</p>
            <progress value={progress} max="100" />
          </div>
        )}
        {uploadError && <p className="error-message">{uploadError}</p>}
      </main>

      <style jsx>{`
        .visualization-container {
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          background: linear-gradient(135deg, #4c669f, #2a3a5e);
          display: flex;
          justify-content: center;
          align-items: center;
          min-height: 100vh;
          margin: 0;
        }

        .visualization-header {
          text-align: center;
          margin-bottom: 30px;
        }

        .visualization-header h1 {
          color: #fff;
          margin-bottom: 10px;
          font-size: 2.2em;
          font-weight: 500;
        }

        .visualization-header p {
          color: #eee;
          font-size: 1.1em;
        }

        .visualization-main {
          background-color: rgba(255, 255, 255, 0.08);
          backdrop-filter: blur(10px);
          padding: 40px;
          border-radius: 15px;
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
          text-align: center;
          width: 90%;
          max-width: 550px;
        }

        .upload-label {
          display: inline-block;
          background: linear-gradient(135deg, #8e2de2, #4a00e0);
          color: #fff;
          padding: 14px 28px;
          border-radius: 7px;
          cursor: pointer;
          transition: background 0.3s ease;
          font-weight: bold;
          margin-bottom: 25px;
        }

        .upload-label:hover {
          background: linear-gradient(135deg, #7624c5, #3d00b3);
        }

        .predict-button {
          background: linear-gradient(135deg, #00b09b, #96c93d);
          color: #fff;
          border: none;
          padding: 14px 30px;
          border-radius: 7px;
          cursor: pointer;
          font-size: 1.1em;
          transition: background 0.3s ease;
          font-weight: bold;
        }

        .predict-button:disabled {
          background: #6c757d;
          cursor: not-allowed;
        }

        .predict-button:hover:not(:disabled) {
          background: linear-gradient(135deg, #008c7e, #7ab833);
        }

        .progress-section {
          margin-top: 25px;
        }

        .progress-section p {
          color: #ddd;
          margin-bottom: 8px;
        }

        .progress-section progress {
          width: 100%;
          height: 12px;
          border-radius: 6px;
          background-color: #333;
        }

        .error-message {
          color: #ff6b6b;
          margin-top: 20px;
          font-size: 1.1em;
        }
      `}</style>
    </div>
  );
};

export default ThreeDVis;