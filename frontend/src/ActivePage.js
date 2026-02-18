import React, { useState } from 'react';
import axios from 'axios';

const ActivePage = () => {
  const [smiles, setSmiles] = useState('');
  const [prediction, setPrediction] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => setSmiles(e.target.value);

  const handleSubmit = async () => {
    if (!smiles) {
      setError('⚠ Please enter a SMILES string!');
      return;
    }

    setLoading(true);
    setError('');
    setPrediction('');

    try {
      const response = await axios.post(
        'http://localhost:5000/predict-vit',
        { smiles },
        { headers: { 'Content-Type': 'application/json' } }
      );
      setPrediction(response.data.predicted_class);
    } catch (err) {
      console.error(err);
      setError('🚫 Failed to fetch prediction. Is the Flask server running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.wrapper}>
      <div style={styles.card}>
        <h1 style={styles.title}>💊 SMILES Activity Predictor</h1>
        <p style={styles.subtitle}>
          Enter a SMILES string and predict whether it's <strong>Active</strong> or <strong>Inactive</strong>
        </p>

        <input
          type="text"
          value={smiles}
          onChange={handleChange}
          placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O"
          style={styles.input}
        />
        <button onClick={handleSubmit} style={styles.button}>
          {loading ? '⏳ Predicting...' : '🚀 Predict'}
        </button>

        {error && <div style={styles.error}>{error}</div>}

        {prediction && (
          <div style={styles.result}>
            <span>🧬 Predicted Class:</span>
            <strong style={styles.predictionText}>{prediction}</strong>
          </div>
        )}
      </div>
    </div>
  );
};

const styles = {
  wrapper: {
    minHeight: '100vh',
    backgroundColor: '#121212',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    padding: '40px',
    fontFamily: 'Arial, sans-serif',
    color: '#e0e0e0',
  },
  card: {
    background: '#1e1e1e',
    borderRadius: '16px',
    padding: '40px 30px',
    width: '100%',
    maxWidth: '600px',
    boxShadow: '0 8px 30px rgba(0, 0, 0, 0.4)',
    textAlign: 'center',
  },
  title: {
    fontSize: '2rem',
    marginBottom: '10px',
    color: '#90caf9',
  },
  subtitle: {
    fontSize: '1rem',
    marginBottom: '25px',
    color: '#b0bec5',
  },
  input: {
    width: '100%',
    padding: '12px 16px',
    borderRadius: '10px',
    border: '1px solid #444',
    backgroundColor: '#2a2a2a',
    color: '#fff',
    fontSize: '1rem',
    marginBottom: '20px',
    outline: 'none',
  },
  button: {
    padding: '12px 30px',
    background: 'linear-gradient(to right, #26c6da, #00acc1)',
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
    fontWeight: 'bold',
    cursor: 'pointer',
    fontSize: '1rem',
    transition: 'background 0.3s ease',
  },
  error: {
    marginTop: '15px',
    color: '#ef5350',
    fontWeight: 'bold',
  },
  result: {
    marginTop: '30px',
    backgroundColor: '#263238',
    padding: '20px',
    borderRadius: '10px',
    color: '#b2dfdb',
    fontSize: '1.2rem',
  },
  predictionText: {
    display: 'block',
    marginTop: '10px',
    fontSize: '1.5rem',
    color: '#ffab91',
    fontWeight: 'bold',
  },
};

export default ActivePage;
