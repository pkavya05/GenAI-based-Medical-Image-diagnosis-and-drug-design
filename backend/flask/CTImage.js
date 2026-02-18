import React, { useState } from 'react';
import { Brain, Settings as Lungs, Activity, ArrowLeft, Loader2, Heart, Clock, Users } from 'lucide-react';
import ImageUploader from './ImageUploader';
import axios from 'axios';
import './CTImage.css'; // Import the custom CSS file

function App() {
  const [selectedType, setSelectedType] = useState(null);
  const [inputType, setInputType] = useState(null);
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const [segmentedVideo, setSegmentedVideo] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);

  const handleImageUpload = async (file) => {
    if (!file) return;

    const formData = new FormData();
    formData.append('image', file);

    setIsProcessing(true);
    setUploadedVideo(URL.createObjectURL(file));
    setError(null);

    try {
      const response = await axios.post('http://127.0.0.1:5001/api/segment', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob',
      });

      const imageBlob = new Blob([response.data], { type: 'image/png' });
      const imageUrl = URL.createObjectURL(imageBlob);

      setSegmentedVideo(imageUrl);
    } catch (error) {
      console.error('Error processing image:', error);
      setError('Failed to process the image. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleNpyUpload = async (file) => {
    if (!file) return;

    const formData = new FormData();
    formData.append('npy', file);

    setIsProcessing(true);
    setError(null);

    try {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const arrayBuffer = e.target?.result;
        if (arrayBuffer) {
          const response = await axios.post('http://127.0.0.1:5001/api/process_npy', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
            responseType: 'blob',
          });

          const imageBlob = new Blob([response.data], { type: 'image/png' });
          const imageUrl = URL.createObjectURL(imageBlob);
          setUploadedVideo(imageUrl);
        }
      };
      reader.readAsArrayBuffer(file);

      const segmentationResponse = await axios.post('http://127.0.0.1:5001/api/segment_lung_npy', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob',
      });

      const segmentedImageBlob = new Blob([segmentationResponse.data], { type: 'image/png' });
      const segmentedImageUrl = URL.createObjectURL(segmentedImageBlob);
      setSegmentedVideo(segmentedImageUrl);
    } catch (error) {
      console.error('Error processing .npy file:', error);
      if (error.response && error.response.data) {
        const reader = new FileReader();
        reader.onload = () => {
          setError(reader.result.toString()); // Convert to string explicitly
        };
        reader.readAsText(error.response.data);
      } else {
        setError('Failed to process the .npy file. Please try again.');
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handleVideoUpload = async (file) => {
    if (!file) return;

    const formData = new FormData();
    formData.append('video', file);

    setIsProcessing(true);
    setError(null);

    try {
      const videoUrl = URL.createObjectURL(file);
      setUploadedVideo(videoUrl);

      const response = await axios.post('http://127.0.0.1:5001/api/segment_lung_video', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob',
      });

      const videoBlob = new Blob([response.data], { type: 'video/mp4' });
      const segmentedVideoUrl = URL.createObjectURL(videoBlob);
      setSegmentedVideo(segmentedVideoUrl);
    } catch (error) {
      console.error('Error processing video:', error);
      setError('Failed to process the video. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const resetState = () => {
    setSelectedType(null);
    setInputType(null);
    setUploadedVideo(null);
    setSegmentedVideo(null);
    setIsProcessing(false);
    setError(null);
  };

  return (
    <div className="ct-image-container">
      <header className="ct-image-header">
        <div className="header-content">
          <div className="logo-container">
            <Activity className="logo-icon" />
            <h1 className="logo-text">Medical Image Segmentation</h1>
          </div>
          {selectedType && (
            <button onClick={resetState} className="back-button">
              <ArrowLeft className="back-icon" />
              Back to selection
            </button>
          )}
        </div>
      </header>

      <main className="ct-image-main">
        {!selectedType && (
          <div className="stats-container">
            <div className="stat-item">
              <Heart className="stat-icon" />
              <span className="stat-label">Accuracy Rate</span>
              <span className="stat-value">98.5%</span>
            </div>
            <div className="stat-item">
              <Clock className="stat-icon" />
              <span className="stat-label">Processing Time</span>
              <span className="stat-value">&lt; 2s</span>
            </div>
            <div className="stat-item">
              <Users className="stat-icon" />
              <span className="stat-label">Cases Analyzed</span>
              <span className="stat-value">10,000+</span>
            </div>
          </div>
        )}

        {!selectedType ? (
          <>
            <h2 className="selection-title">Select Segmentation Type</h2>
            <div className="selection-grid">
              <button onClick={() => setSelectedType('brain')} className="selection-card">
                <Brain className="card-icon" />
                <h2 className="card-title">Brain Tumor Segmentation</h2>
                <p className="card-description">
                  Advanced MRI analysis for precise tumor detection and boundary mapping
                </p>
              </button>

              <button onClick={() => setSelectedType('lung')} className="selection-card">
                <Lungs className="card-icon" />
                <h2 className="card-title">Lung Tumor Segmentation</h2>
                <p className="card-description">
                  State-of-the-art CT scan analysis for accurate pulmonary tumor identification
                </p>
              </button>
            </div>
          </>
        ) : selectedType === 'lung' ? (
          <div className="lung-segmentation-container">
            <h2 className="segmentation-title">Lung Tumor Segmentation</h2>

            {!inputType ? (
              <div className="input-type-grid">
                <button onClick={() => setInputType('npy')} className="input-type-card">
                  <h2 className="input-type-title">Upload .npy File</h2>
                  <p className="input-type-description">
                    Upload a single .npy file for segmentation
                  </p>
                </button>

                <button onClick={() => setInputType('video')} className="input-type-card">
                  <h2 className="input-type-title">Upload Video</h2>
                  <p className="input-type-description">
                    Upload a video for segmentation
                  </p>
                </button>
              </div>
            ) : inputType === 'npy' ? (
              <>
                {!uploadedVideo ? (
                  <ImageUploader onUpload={handleNpyUpload} accept=".npy" />
                ) : (
                  <div className="result-container">
                    <div className="result-grid">
                      <div className="result-item">
                        <h3 className="result-label">Input CT Scan</h3>
                        <div className="result-image-container">
                          <img src={uploadedVideo} alt="Input CT Scan" className="result-image" />
                        </div>
                      </div>
                      <div className="result-item">
                        <h3 className="result-label">Segmentation Result</h3>
                        <div className="result-image-container">
                          {isProcessing ? (
                            <div className="processing-overlay">
                              <Loader2 className="processing-icon" />
                              <p className="processing-text">Processing scan...</p>
                            </div>
                          ) : segmentedVideo && (
                            <img src={segmentedVideo} alt="Segmented result" className="result-image" />
                          )}
                        </div>
                      </div>
                    </div>
                    {error && <div className="error-message">{error}</div>}
                    <button onClick={() => { setUploadedVideo(null); setSegmentedVideo(null); }} className="upload-another-button">
                      Upload Another File
                    </button>
                  </div>
                )}
              </>
            ) : (
              <>
                {!uploadedVideo ? (
                  <ImageUploader onUpload={handleVideoUpload} accept="video/*" />
                ) : (
                  <div className="result-container">
                    <div className="result-grid">
                      <div className="result-item">
                        <h3 className="result-label">Input Video</h3>
                        <div className="result-video-container">
                          <video controls className="result-video">
                            <source src={uploadedVideo} type="video/mp4" />
                            Your browser does not support the video tag.
                          </video>
                        </div>
                      </div>
                      <div className="result-item">
                        <h3 className="result-label">Segmentation Result</h3>
                        <div className="result-video-container">
                          {isProcessing ? (
                            <div className="processing-overlay">
                              <Loader2 className="processing-icon" />
                              <p className="processing-text">Processing video...</p>
                            </div>
                          ) : segmentedVideo && (
                            <video controls className="result-video">
                              <source src={segmentedVideo} type="video/mp4" />
                              Your browser does not support the video tag.
                            </video>
                          )}
                        </div>
                      </div>
                    </div>
                    {error && <div className="error-message">{error}</div>}
                    <button onClick={() => { setUploadedVideo(null); setSegmentedVideo(null); }} className="upload-another-button">
                      Upload Another Video
                    </button>
                  </div>
                )}
              </>
            )}
          </div>
        ) : (
          <div className="brain-segmentation-container">
            <h2 className="segmentation-title">Brain Tumor Segmentation</h2>

            {!uploadedVideo ? (
              <ImageUploader onUpload={handleImageUpload} />
            ) : (
              <div className="result-container">
                <div className="result-grid">
                  <div className="result-item">
                    <h3 className="result-label">Original Scan</h3>
                    <div className="result-image-container">
                      <img src={uploadedVideo} alt="Original scan" className="result-image" />
                    </div>
                  </div>
                  <div className="result-item">
                    <h3 className="result-label">Segmentation Result</h3>
                    <div className="result-image-container">
                      {isProcessing ? (
                        <div className="processing-overlay">
                          <Loader2 className="processing-icon" />
                          <p className="processing-text">Processing scan...</p>
                        </div>
                      ) : segmentedVideo && (
                        <img src={segmentedVideo} alt="Segmented result" className="result-image" />
                      )}
                    </div>
                  </div>
                </div>
                {error && <div className="error-message">{error}</div>}
                <button onClick={() => { setUploadedVideo(null); setSegmentedVideo(null); }} className="upload-another-button">
                  Upload New Scan
                </button>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;