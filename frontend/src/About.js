import React from 'react';

const About = () => {
  return (
    <div style={styles.container}>
      <h1 style={styles.title}>About DrugRin</h1>
      <p style={styles.text}>
        <strong>DrugRin</strong> is an advanced drug discovery dashboard designed to accelerate and simplify
        the early stages of drug design using modern AI technologies. This platform integrates various tools
        to help researchers and scientists explore molecular structures, predict drug properties, and simulate
        interactions between molecules and proteins.
      </p>
      <h3 style={styles.subtitle}>Key Features:</h3>
<ul style={styles.list}>
  <li>🧬 <strong>LIGAND IDENTIFIER</strong> – Predict masked tokens in a SMILES string using machine learning.</li>
  <li>⚖️ <strong>Predict Active/Inactive</strong> – Classify molecules as biologically active or inactive.</li>
  <li>🧫 <strong>Protein to SMILES</strong> – Generate SMILES representations from protein sequences.</li>
  <li>🧪 <strong>Denova Ligand Generator</strong> – Discover structurally and functionally similar drug compounds.</li>
  <li>🔬 <strong>3D Visualization of Protein</strong> – Visualize 3D protein structures from their sequences.</li>
  <li>⚗️ <strong>3D Visualization of Protein-Ligand Interaction</strong> – Simulate and analyze molecular docking between proteins and ligands.</li>
</ul>

      <p style={styles.footer}>
        DrugRin is built using React for the frontend and powered by machine learning models on the backend.
        It aims to empower drug discovery teams with intuitive tools and insightful predictions.
      </p>
    </div>
  );
};

const styles = {
  container: {
    backgroundColor: '#121212',
    color: '#f0f0f0',
    padding: '40px',
    minHeight: '100vh',
    fontFamily: 'Segoe UI, sans-serif',
  },
  title: {
    fontSize: '32px',
    fontWeight: 'bold',
    color: '#90caf9',
    marginBottom: '20px',
  },
  text: {
    fontSize: '16px',
    lineHeight: '1.7',
    marginBottom: '20px',
    color: '#ccc',
  },
  subtitle: {
    fontSize: '20px',
    marginBottom: '10px',
    color: '#ffffff',
  },
  list: {
    paddingLeft: '20px',
    marginBottom: '20px',
    color: '#bbb',
    lineHeight: '1.6',
  },
  footer: {
    fontSize: '15px',
    color: '#999',
    marginTop: '30px',
  },
};

export default About;
