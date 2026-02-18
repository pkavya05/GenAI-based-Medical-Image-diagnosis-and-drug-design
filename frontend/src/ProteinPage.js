import { useState } from "react";

const Protein2Smiles = () => {
  const [protein, setProtein] = useState("");
  const [smiles, setSmiles] = useState("");
  const [smilesImage, setSmilesImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePredict = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setSmiles("");
    setSmilesImage(null);

    try {
      const response = await fetch("http://localhost:5000/api/predict-smiles", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ protein_sequence: protein }),
      });

      const data = await response.json();
      if (data.smiles) {
        setSmiles(data.smiles);
        setSmilesImage(`data:image/png;base64,${data.image}`);
      } else {
        setSmiles("N/A");
      }
    } catch (err) {
      console.error("Error predicting SMILES:", err);
      setSmiles("Error generating SMILES");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <h1 style={styles.heading}>
          🧪 Protein → <span style={{ color: "#3b82f6" }}>SMILES</span> Predictor
        </h1>
        <p style={styles.subtext}>
          Enter a protein sequence to generate the predicted <b>SMILES</b> representation.
        </p>

        <form onSubmit={handlePredict} style={styles.form}>
          <input
            type="text"
            placeholder="e.g. MVKVYAPASSANMS..."
            value={protein}
            onChange={(e) => setProtein(e.target.value)}
            required
            style={styles.input}
          />
          <button type="submit" style={styles.button}>
            🚀 {isLoading ? "Predicting..." : "Predict"}
          </button>
        </form>

        {(smiles || smilesImage) && (
          <div style={styles.resultContainer}>
            {smiles && (
              <div style={styles.resultBox}>
                <p style={styles.resultTitle}>Predicted SMILES:</p>
                <code style={styles.resultText}>{smiles}</code>
              </div>
            )}
            {smilesImage && (
              <div style={styles.resultBox}>
                <p style={styles.resultTitle}>Molecule Structure:</p>
                <img src={smilesImage} alt="Molecule" style={styles.image} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const styles = {
  page: {
    background: "#0d1117",
    minHeight: "100vh",
    padding: "40px 20px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontFamily: "'Poppins', sans-serif",
    color: "#e5e7eb",
  },
  card: {
    backgroundColor: "#161b22",
    padding: "40px 30px",
    borderRadius: "20px",
    boxShadow: "0 10px 25px rgba(59, 130, 246, 0.2)",
    maxWidth: "600px",
    width: "100%",
    textAlign: "center",
  },
  heading: {
    fontSize: "2.2rem",
    fontWeight: "700",
    background: "linear-gradient(to right, #60a5fa, #3b82f6)",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
    marginBottom: "10px",
  },
  subtext: {
    fontSize: "1rem",
    color: "#9ca3af",
    marginBottom: "30px",
  },
  form: {
    display: "flex",
    flexDirection: "column",
    gap: "20px",
  },
  input: {
    padding: "12px 18px",
    borderRadius: "12px",
    border: "none",
    background: "#0d1117",
    color: "#f3f4f6",
    fontSize: "1rem",
    outline: "none",
    borderBottom: "2px solid #3b82f6",
  },
  button: {
    background: "linear-gradient(to right, #3b82f6, #2563eb)",
    padding: "12px",
    borderRadius: "12px",
    fontSize: "1rem",
    fontWeight: "600",
    color: "white",
    border: "none",
    cursor: "pointer",
    transition: "all 0.3s ease",
  },
  resultContainer: {
    marginTop: "30px",
    display: "flex",
    flexDirection: "column",
    gap: "20px",
  },
  resultBox: {
    backgroundColor: "#0d1117",
    padding: "20px",
    borderRadius: "12px",
    border: "1px solid #1f2937",
    textAlign: "left",
  },
  resultTitle: {
    color: "#60a5fa",
    fontSize: "1rem",
    marginBottom: "8px",
  },
  resultText: {
    color: "#dbeafe",
    fontSize: "1rem",
    wordBreak: "break-word",
  },
  image: {
    width: "100%",
    maxHeight: "250px",
    borderRadius: "10px",
    marginTop: "10px",
    backgroundColor: "white",
  },
};

export default Protein2Smiles;
