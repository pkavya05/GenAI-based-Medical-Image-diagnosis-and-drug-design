import React, { useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { FlaskConical, FlaskRound, Loader, Flame } from "lucide-react";

const SimilarPage = () => {
  const [smiles, setSmiles] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!smiles.trim()) {
      setError("Please enter a SMILES string.");
      return;
    }
    setError("");
    setLoading(true);
    setResults([]);

    try {
      const response = await axios.post("http://localhost:5000/api/reinforcement", {
        SMILES: smiles.trim(),
      });
      setResults(response.data.top_results || []);
    } catch (err) {
      console.error("Reinforcement error:", err);
      setError("Failed to fetch reinforcement data.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={styles.container}>
        <h1 style={styles.title}>🧪 Denova Ligand Generator</h1>
        <form onSubmit={handleSubmit} style={styles.form}>
          <div style={styles.inputWrapper}>
            <FlaskConical style={styles.icon} size={18} />
            <input
              type="text"
              value={smiles}
              onChange={(e) => setSmiles(e.target.value)}
              placeholder="Enter SMILES (e.g. CCO)"
              style={styles.input}
            />
          </div>
          <button type="submit" style={styles.button} disabled={loading}>
            {loading ? <Loader className="animate-spin" size={20} /> : "Generate Molecules"}
          </button>
        </form>

        {error && <p style={styles.error}>{error}</p>}

        <div style={styles.results}>
          {results.map((item, idx) => (
            <motion.div key={idx} style={styles.card} whileHover={{ scale: 1.02 }}>
              <img src={`data:image/png;base64,${item.image}`} alt="Molecule" style={styles.image} />
              <div style={styles.info}>
                <p><FlaskRound size={16} /> <strong>SMILES:</strong> {item.smiles}</p>
                <p><strong>pIC50:</strong> {item.pIC50}</p>
                <p><strong>logP:</strong> {item.logP}</p>
                <p><Flame size={16} /> <strong>Reward:</strong> {item.reward}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </div>
  );
};

const styles = {
  page: {
    minHeight: "100vh",
    background: "radial-gradient(ellipse at center, #0f172a 0%, #1e293b 100%)",
    padding: "40px 20px",
    color: "#f1f5f9",
    fontFamily: "Poppins, sans-serif",
  },
  container: {
    maxWidth: "960px",
    margin: "0 auto",
  },
  title: {
    fontSize: "2rem",
    fontWeight: "600",
    marginBottom: "24px",
    textAlign: "center",
    background: "linear-gradient(to right, #38bdf8, #6366f1)",
    WebkitBackgroundClip: "text",
    color: "transparent",
  },
  form: {
    display: "flex",
    gap: "12px",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: "30px",
    flexWrap: "wrap",
  },
  inputWrapper: {
    position: "relative",
    display: "flex",
    alignItems: "center",
  },
  icon: {
    position: "absolute",
    left: "12px",
    color: "#60a5fa",
  },
  input: {
    padding: "10px 14px 10px 36px",
    borderRadius: "8px",
    fontSize: "1rem",
    border: "1px solid #334155",
    backgroundColor: "#1e293b",
    color: "white",
    width: "280px",
    outline: "none",
  },
  button: {
    padding: "10px 20px",
    background: "linear-gradient(to right, #3b82f6, #6366f1)",
    color: "white",
    fontWeight: "bold",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    transition: "0.3s",
  },
  error: {
    color: "#f87171",
    textAlign: "center",
    marginTop: "10px",
  },
  results: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
    gap: "20px",
    marginTop: "30px",
  },
  card: {
    background: "#0f172a",
    border: "1px solid #334155",
    borderRadius: "16px",
    padding: "16px",
    boxShadow: "0 4px 20px rgba(0,0,0,0.2)",
  },
  image: {
    width: "100%",
    borderRadius: "8px",
    border: "1px solid #334155",
    marginBottom: "12px",
    backgroundColor: "#fff",
  },
  info: {
    fontSize: "0.95rem",
    lineHeight: "1.6",
  },
};

export default SimilarPage;
