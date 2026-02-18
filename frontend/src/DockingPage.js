import React, { useState } from "react";
import axios from "axios";

const DockingPage = () => {
  const [ecNumber, setEcNumber] = useState("");
  const [ligandId, setLigandId] = useState("");
  const [message, setMessage] = useState("");
  const [visualizationLink, setVisualizationLink] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage("");
    setVisualizationLink("");

    try {
      const response = await axios.post("http://localhost:5000/api/autodocking", {
        ecNumber,
        ligandId,
      });

      if (response.data.error) {
        setMessage(`Error: ${response.data.error}`);
      } else {
        setMessage(response.data.message);
        setVisualizationLink(response.data.visualization_html);
      }
    } catch (error) {
      console.error(error);
      setMessage("Failed to perform docking. Please try again.");
    }
  };

  const backendUrl = "http://localhost:5000";

  return (
    <div style={{ 
      padding: "30px", 
      fontFamily: "Arial, sans-serif", 
      backgroundColor: "#0a0f1a", 
      minHeight: "100vh", 
      color: "#ffffff" 
    }}>
      <div style={{ 
        maxWidth: "800px", 
        margin: "0 auto", 
        backgroundColor: "#121826", 
        padding: "30px", 
        borderRadius: "12px", 
        boxShadow: "0 0 20px rgba(0, 123, 255, 0.4)" 
      }}>
        <h2 style={{ color: "#00BFFF", marginBottom: "20px" }}>AutoDocking Page</h2>
        
        <form onSubmit={handleSubmit} style={{ marginBottom: "30px" }}>
          <div style={{ marginBottom: "20px" }}>
            <label style={{ fontWeight: "bold", color: "#9bbcff" }}>EC Number:</label><br />
            <input
              type="text"
              value={ecNumber}
              onChange={(e) => setEcNumber(e.target.value)}
              placeholder="e.g. 2.7.11.1"
              style={{ 
                width: "100%", 
                padding: "10px", 
                borderRadius: "8px", 
                border: "1px solid #1e3a8a", 
                backgroundColor: "#0f172a", 
                color: "#ffffff", 
                marginTop: "8px" 
              }}
            />
          </div>
          <div style={{ marginBottom: "20px" }}>
            <label style={{ fontWeight: "bold", color: "#9bbcff" }}>Ligand ID:</label><br />
            <input
              type="text"
              value={ligandId}
              onChange={(e) => setLigandId(e.target.value)}
              placeholder="e.g. ATP"
              style={{ 
                width: "100%", 
                padding: "10px", 
                borderRadius: "8px", 
                border: "1px solid #1e3a8a", 
                backgroundColor: "#0f172a", 
                color: "#ffffff", 
                marginTop: "8px" 
              }}
            />
          </div>
          <button 
            type="submit" 
            style={{ 
              padding: "12px 24px", 
              backgroundColor: "#00BFFF", 
              border: "none", 
              borderRadius: "8px", 
              color: "#0a0f1a", 
              fontWeight: "bold", 
              fontSize: "16px", 
              cursor: "pointer",
              transition: "background-color 0.3s ease"
            }}
            onMouseOver={(e) => (e.target.style.backgroundColor = "#1e90ff")}
            onMouseOut={(e) => (e.target.style.backgroundColor = "#00BFFF")}
          >
            Start Docking
          </button>
        </form>

        {message && (
          <p style={{ color: "#9bbcff", fontWeight: "bold" }}>{message}</p>
        )}

        {visualizationLink && (
          <div style={{ marginTop: "30px" }}>
            <h3 style={{ color: "#00BFFF" }}>3D Visualization:</h3>

            {/* Embed inside page */}
            <iframe
              src={`${backendUrl}/${visualizationLink}`}
              title="3D Docking Visualization"
              width="100%"
              height="600"
              style={{ 
                border: "2px solid #1e3a8a", 
                borderRadius: "10px", 
                marginTop: "15px" 
              }}
            ></iframe>

            {/* Also give link to open in new tab */}
            <p style={{ marginTop: "15px" }}>
              <a
                href={`${backendUrl}/static/${visualizationLink}`}
                target="_blank"
                rel="noopener noreferrer"
                style={{ 
                  color: "#00BFFF", 
                  textDecoration: "none", 
                  fontWeight: "bold" 
                }}
              >
                Open in New Tab
              </a>
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default DockingPage;
