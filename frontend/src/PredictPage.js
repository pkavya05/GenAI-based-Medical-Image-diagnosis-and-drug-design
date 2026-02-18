import React, { useState } from 'react';
import axios from 'axios';

const PredictPage = () => {
  const [selectedDisease, setSelectedDisease] = useState('');
  const [showInfo, setShowInfo] = useState(false);
  const [smiles, setSmiles] = useState('');
  const [prediction, setPrediction] = useState([]);
  const [error, setError] = useState(null);

  const diseaseInfo = {
    alzheimers: {
      name: 'Alzheimer’s Disease',
      description:
        'Alzheimer’s is a progressive neurological disorder that causes brain cells to waste away. One therapeutic strategy involves targeting JAK2, a tyrosine kinase involved in neuroinflammation and cell survival. Predicting suitable inhibitors using SMILES with masked tokens helps explore novel compounds for this target.',
    },
  };

  const handleDiseaseSelect = (e) => {
    setSelectedDisease(e.target.value);
    setShowInfo(true);
    setPrediction([]);
    setSmiles('');
    setError(null);
  };

  const handlePredict = async () => {
    if (!smiles.includes('<mask>')) {
      setError('SMILES must contain <mask>');
      return;
    }

    setError(null);
    try {
      const res = await axios.post('http://localhost:5000/predict', { smiles });
      setPrediction(res.data.prediction.split('\n'));
    } catch (err) {
      setError('Prediction failed');
    }
  };

  return (
    <div style={styles.container}>
      {/* Animation keyframes */}
      <style>
        {`
          @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
          }
        `}
      </style>

      <h2 style={styles.heading}>Ligand Identifier</h2>

      <div style={styles.centerBox}>
        <div style={styles.contentBox}>
          <label style={styles.label}>Select Disease</label>
          <select style={styles.dropdown} onChange={handleDiseaseSelect} value={selectedDisease}>
            <option value="">-- Select --</option>
            <option value="alzheimers">Alzheimer’s</option>
          </select>

          {showInfo && selectedDisease && (
            <div style={styles.infoBox}>
              <h3 style={styles.infoTitle}>{diseaseInfo[selectedDisease].name}</h3>
              <p style={styles.infoText}>{diseaseInfo[selectedDisease].description}</p>
            </div>
          )}

          {selectedDisease && (
            <>
              <label style={styles.label}>Enter SMILES (with &lt;mask&gt;)</label>
              <input
                type="text"
                value={smiles}
                onChange={(e) => setSmiles(e.target.value)}
                placeholder="Example: CC<mask>OC"
                style={styles.input}
              />
              <button onClick={handlePredict} style={styles.button}>Predict</button>
              {error && <p style={styles.error}>{error}</p>}

              {prediction.length > 0 && (
                <div style={styles.resultBox}>
                  <h4 style={styles.resultTitle}>Predictions:</h4>
                  <ul style={{ paddingLeft: '20px' }}>
                    {prediction.map((p, idx) => (
                      <li
                        key={idx}
                        style={{
                          ...styles.resultItem,
                          ':hover': styles.resultItemHover, // Not supported natively but included for reference
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = '#37474f';
                          e.currentTarget.style.transform = 'scale(1.02)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = '#263238';
                          e.currentTarget.style.transform = 'scale(1)';
                        }}
                      >
                        {p}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

const styles = {
  container: {
    padding: '30px',
    backgroundColor: '#121212',
    color: '#fff',
    minHeight: '100vh',
    fontFamily: 'Arial, sans-serif',
  },
  heading: {
    fontSize: '24px',
    color: '#90caf9',
    marginBottom: '20px',
    textAlign: 'center',
  },
  centerBox: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'flex-start',
  },
  contentBox: {
    width: '100%',
    maxWidth: '600px',
  },
  label: {
    display: 'block',
    marginBottom: '8px',
    fontSize: '16px',
  },
  dropdown: {
    padding: '10px',
    fontSize: '16px',
    width: '100%',
    marginBottom: '20px',
    borderRadius: '8px',
    border: '1px solid #333',
    backgroundColor: '#1e1e1e',
    color: '#fff',
  },
  infoBox: {
    backgroundColor: '#1e1e1e',
    padding: '20px',
    borderRadius: '12px',
    marginBottom: '20px',
    border: '1px solid #444',
  },
  infoTitle: {
    fontSize: '18px',
    color: '#90caf9',
    marginBottom: '10px',
  },
  infoText: {
    fontSize: '14px',
    color: '#ccc',
  },
  input: {
    width: '100%',
    padding: '10px',
    fontSize: '16px',
    borderRadius: '8px',
    border: '1px solid #333',
    backgroundColor: '#1e1e1e',
    color: '#fff',
    marginBottom: '10px',
  },
  button: {
    padding: '10px 20px',
    backgroundColor: '#90caf9',
    color: '#000',
    fontWeight: 'bold',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    marginBottom: '20px',
  },
  error: {
    color: 'red',
    marginTop: '10px',
  },
  resultBox: {
    marginTop: '20px',
    padding: '20px',
    backgroundColor: '#1a1a1a',
    borderRadius: '12px',
    border: '1px solid #555',
    boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
    animation: 'fadeIn 0.5s ease-in-out',
  },
  resultTitle: {
    fontSize: '16px',
    color: '#90caf9',
    marginBottom: '10px',
  },
  resultItem: {
    fontSize: '15px',
    color: '#bbdefb',
    backgroundColor: '#263238',
    padding: '8px 12px',
    borderRadius: '6px',
    marginBottom: '8px',
    listStyleType: 'none',
    boxShadow: 'inset 0 0 5px rgba(255, 255, 255, 0.05)',
    transition: 'background-color 0.3s, transform 0.2s',
    cursor: 'pointer',
  },
};

export default PredictPage;
