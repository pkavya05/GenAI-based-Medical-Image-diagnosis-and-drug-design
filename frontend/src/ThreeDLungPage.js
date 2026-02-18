// ThreeDLungPage.js
import React from 'react';

const ThreeDLungPage = () => {
  return (
    <div style={styles.page}>
      <h1 style={styles.title}>3D Lung Visualization</h1>
      <div style={styles.content}>
        {/* Add your 3D visualization here. This could be a 3D model viewer */}
        <p>Here you can display your 3D lung model or related content.</p>
        {/* If you are using a library to visualize the model, integrate it here */}
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
  title: {
    fontSize: "2.5rem",
    marginBottom: "0.5rem",
  },
  content: {
    textAlign: "center",
    marginTop: "2rem",
  },
};

export default ThreeDLungPage;
