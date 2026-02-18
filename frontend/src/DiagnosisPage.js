import React, { useState } from "react";
import { useNavigate } from 'react-router-dom'; // Corrected import

const DiagnosisPage = () => {
  const [selectedType, setSelectedType] = useState("");
  const [file, setFile] = useState(null);
  const [resultURL, setResultURL] = useState("");
  const [fileType, setFileType] = useState("");
  const navigate = useNavigate(); // Corrected initialization

  const handleTypeChange = (type) => {
    setSelectedType(type);
    setFile(null);
    setResultURL("");
    setFileType("");
  };

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    if (!f) return;

    setFile(f);
    if (f.name.endsWith(".npy")) setFileType("npy");
    else if (f.type.startsWith("video/")) setFileType("video");
    else if (f.type.startsWith("image/")) setFileType("image");
    else setFileType("");
  };

  const getEndpoint = () => {
    if (selectedType === "brain") return "/api/segment";
    if (selectedType === "lung") {
      if (fileType === "npy") return "/api/segment_lung_npy";
      if (fileType === "video") return "/api/diagnose_lung_video";
      return "/api/diagnose_lung";
    }
  };

  const handleSubmit = async () => {
    if (!file || !selectedType || !fileType) return;

    const formData = new FormData();
    formData.append(fileType, file);
    const endpoint = getEndpoint();

    try {
      const response = await fetch(`http://localhost:5000${endpoint}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Failed to get diagnosis");

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setResultURL(url);
    } catch (error) {
      console.error(error);
      alert("An error occurred while processing the request.");
    }
  };

  // Updated navigation handler
  const handle3DLung = () => {
    navigate("/threedvis"); // Corrected navigation to 3D Lung page
  };

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <h1 style={styles.title}>🧬 Tumor Diagnosis</h1>
        <p style={styles.subtitle}>Select the tumor type and upload a medical scan</p>

        <div style={styles.buttonGroup}>
          <button
            onClick={() => handleTypeChange("brain")}
            style={{
              ...styles.selectButton,
              background: selectedType === "brain" ? "#ae62f7" : "transparent",
              color: selectedType === "brain" ? "#fff" : "#ae62f7",
              borderColor: "#ae62f7",
            }}
          >
            Brain Tumor
          </button>
          <button
            onClick={() => handleTypeChange("lung")}
            style={{
              ...styles.selectButton,
              background: selectedType === "lung" ? "#34caff" : "transparent",
              color: selectedType === "lung" ? "#fff" : "#34caff",
              borderColor: "#34caff",
            }}
          >
            Lung Tumor
          </button>
          <button
            onClick={handle3DLung}
            style={{
              ...styles.selectButton,
              background: "transparent",
              color: "#34caff",
              borderColor: "#34caff",
            }}
          >
            3D Lung
          </button>
        </div>

        {selectedType && (
          <div style={styles.uploadSection}>
            <h3 style={styles.sectionHeader}>
              Upload {selectedType === "brain" ? "Image" : "Image / .npy / Video"}
            </h3>
            <input
              type="file"
              accept={selectedType === "brain" ? "image/*" : "image/*,.npy,video/*"}
              onChange={handleFileChange}
              style={styles.input}
            />
            <button onClick={handleSubmit} style={styles.submitButton}>
              🚀 Diagnose
            </button>
          </div>
        )}

        {resultURL && (
          <div style={styles.resultSection}>
            <h3 style={styles.sectionHeader}>Diagnosis Result:</h3>
            {fileType === "video" ? (
              <video controls width="300" src={resultURL} style={styles.resultMedia}></video>
            ) : (
              <img src={resultURL} alt="Segmentation Result" width="300" style={styles.resultMedia} />
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const styles = {
  page: {
    background: "radial-gradient(ellipse at center, #1f1c2c 0%, #928dab 100%)",
    minHeight: "100vh",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    padding: "2rem",
    fontFamily: "'Segoe UI', sans-serif",
    color: "#fff",
  },
  card: {
    background: "rgba(255, 255, 255, 0.05)",
    padding: "2.5rem 3rem",
    borderRadius: "20px",
    boxShadow: "0 15px 50px rgba(0, 0, 0, 0.5)",
    width: "90%",
    maxWidth: "600px",
    textAlign: "center",
    backdropFilter: "blur(10px)",
    border: "1px solid rgba(255, 255, 255, 0.1)",
  },
  title: {
    fontSize: "2.5rem",
    marginBottom: "0.5rem",
  },
  subtitle: {
    fontSize: "1.1rem",
    marginBottom: "2rem",
    color: "#cccccc",
  },
  buttonGroup: {
    display: "flex",
    justifyContent: "center",
    gap: "1rem",
    marginBottom: "2rem",
  },
  selectButton: {
    padding: "0.7rem 1.4rem",
    fontSize: "1rem",
    border: "2px solid",
    borderRadius: "50px",
    cursor: "pointer",
    transition: "all 0.3s ease",
    background: "transparent",
  },
  uploadSection: {
    marginBottom: "2rem",
  },
  sectionHeader: {
    fontSize: "1.2rem",
    marginBottom: "1rem",
    color: "#e0e0e0",
  },
  input: {
    margin: "1rem 0",
    padding: "0.6rem",
    borderRadius: "10px",
    border: "none",
    backgroundColor: "#ffffff",
    color: "#000",
  },
  submitButton: {
    padding: "0.7rem 1.5rem",
    background: "linear-gradient(to right, #00f2fe, #4facfe)",
    color: "#fff",
    fontSize: "1rem",
    border: "none",
    borderRadius: "40px",
    cursor: "pointer",
    marginTop: "1rem",
    boxShadow: "0 8px 20px rgba(79, 172, 254, 0.4)",
  },
  resultSection: {
    marginTop: "2rem",
  },
  resultMedia: {
    marginTop: "1rem",
    borderRadius: "12px",
    boxShadow: "0 5px 25px rgba(0,0,0,0.4)",
  },
};

export default DiagnosisPage;
