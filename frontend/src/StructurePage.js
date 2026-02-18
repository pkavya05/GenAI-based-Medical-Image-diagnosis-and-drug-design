import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import * as $3Dmol from '3dmol';

const StructurePage = () => {
  const [sequence, setSequence] = useState('');
  const [loading, setLoading] = useState(false);
  const [structure, setStructure] = useState(null);
  const viewerRef = useRef(null);

  const handleInputChange = (e) => {
    setSequence(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (sequence.trim().length < 10) {
      alert('Please provide a valid protein sequence with at least 10 amino acids.');
      return;
    }

    setLoading(true);
    setStructure(null);

    try {
      const response = await axios.post('http://localhost:5000/api/fold', { sequence });
      const pdbString = response.data.pdb;

      if (pdbString) {
        setStructure(pdbString);
      } else {
        alert('No structure returned. Check the backend or the input sequence.');
      }
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Error predicting protein structure');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (structure) {
      const viewer = $3Dmol.createViewer(viewerRef.current, {
        backgroundColor: 'white',
      });
      viewer.resize(); // Let 3Dmol adapt to the container size
      viewer.addModel(structure, 'pdb');
      viewer.setStyle({}, { stick: {}, cartoon: { color: 'spectrum' } });
      viewer.zoomTo();
      viewer.render();
    }
  }, [structure]);

  return (
    <div style={styles.wrapper}>
      <h1 style={styles.title}>3D Visualization of Protein</h1>
      <div style={styles.container}>
        <form onSubmit={handleSubmit} style={styles.form}>
          <label htmlFor="sequence" style={styles.label}>Protein Sequence:</label>
          <textarea
            id="sequence"
            value={sequence}
            onChange={handleInputChange}
            rows="4"
            placeholder="Enter protein sequence (e.g., MALWMRLLPLLALLALWGPDPAAAF...)"
            style={styles.textarea}
            required
          />
          <button type="submit" style={styles.button} disabled={loading}>
            {loading ? 'Predicting...' : 'Predict Structure'}
          </button>
        </form>

        {loading && <p style={styles.loading}>Predicting structure... hang tight!</p>}

        {structure && (
          <div style={styles.viewerContainer}>
            <h2 style={styles.subTitle}>3D Structure Visualization</h2>
            <div ref={viewerRef} style={styles.viewer}></div>
          </div>
        )}
      </div>
    </div>
  );
};

const styles = {
  wrapper: {
    fontFamily: 'Poppins, sans-serif',
    padding: '20px',
    background: 'linear-gradient(to top, #0f172a, #1e293b)',
    minHeight: '100vh',
    color: '#e2e8f0',
  },
  title: {
    textAlign: 'center',
    fontSize: '2.2rem',
    marginBottom: '25px',
    fontWeight: 700,
    background: 'linear-gradient(to right, #3b82f6, #6366f1)',
    WebkitBackgroundClip: 'text',
    color: 'transparent',
  },
  container: {
    maxWidth: '800px',
    margin: '0 auto',
    backgroundColor: '#1e293b',
    padding: '30px',
    borderRadius: '16px',
    boxShadow: '0 10px 30px rgba(0, 0, 0, 0.3)',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
  },
  label: {
    fontSize: '1rem',
    color: '#cbd5e1',
    marginBottom: '8px',
    fontWeight: '500',
  },
  textarea: {
    padding: '12px',
    fontSize: '15px',
    borderRadius: '10px',
    backgroundColor: '#0f172a',
    border: '1px solid #334155',
    color: 'white',
    resize: 'vertical',
    fontFamily: 'monospace',
    marginBottom: '20px',
  },
  button: {
    padding: '12px',
    background: 'linear-gradient(to right, #3b82f6, #6366f1)',
    color: 'white',
    fontSize: '16px',
    borderRadius: '10px',
    border: 'none',
    cursor: 'pointer',
    fontWeight: 'bold',
  },
  loading: {
    color: '#60a5fa',
    fontSize: '16px',
    marginTop: '10px',
    textAlign: 'center',
  },
  viewerContainer: {
    marginTop: '30px',
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
  },
  subTitle: {
    marginBottom: '10px',
    fontSize: '1.3rem',
    fontWeight: '600',
    background: 'linear-gradient(to right, #38bdf8, #818cf8)',
    WebkitBackgroundClip: 'text',
    color: 'transparent',
    textAlign: 'center',
  },
  viewer: {
    width: '100%',
    maxWidth: '700px',
    height: '500px',
    border: '1px solid #334155',
    backgroundColor: '#0f172a',
    borderRadius: '10px',
    overflow: 'hidden',
    position: 'relative',
  },
};
export default StructurePage;
