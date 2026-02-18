import React from 'react';
import { useNavigate } from 'react-router-dom';

const Dashboard = () => {
  const navigate = useNavigate();

  const tools = [
    {
      title: 'LIGAND IDENTIFIER',
      key: 'smiles',
      image: 'https://cdn-icons-png.flaticon.com/512/1055/1055646.png',
      route: '/predict',
    },
    {
      title: 'Predict Active/Inactive',
      key: 'active',
      image: 'https://cdn-icons-png.flaticon.com/512/103/103442.png',
      route: '/active',
    },
    {
      title: 'Protein to SMILES',
      key: 'protein',
      image: 'https://cdn-icons-png.flaticon.com/512/2942/2942840.png',
      route: '/protein',
    },
    {
      title: 'Denova Ligand Generator',
      key: 'similar',
      image: 'https://cdn-icons-png.flaticon.com/512/3039/3039431.png',
      route: '/similar',
    },
    {
      title: '3D Visualization of Protein ',
      key: 'structure',
      image: 'https://cdn-icons-png.flaticon.com/512/4298/4298542.png',
      route: '/structure'
    },
    {
      title: '3D visualization of protein ligand Interaction',
      key: 'docking',
      image: 'https://cdn-icons-png.flaticon.com/512/5948/5948085.png',
      route: '/docking'
    },
  ];

  const toolDescriptions = {
    smiles: 'Predict masked tokens in a SMILES string.',
    active: 'Determine if a molecule is likely to be active or inactive.',
    protein: 'Generate SMILES molecules from protein sequences.',
    similar: 'Find structurally and functionally similar drug compounds.',
    structure: 'Visualize 3D protein from its sequence.',
    docking: 'Simulate molecular docking between drug and protein.',
  };

  return (
    <div style={styles.page}>
      {/* NAVBAR */}
      <nav style={styles.navbar}>
        <div style={styles.navTitle}>DrugRin</div>
        <div style={styles.navLinks}>
          <span style={styles.navLink} onClick={() => navigate('/about')}>About</span>
          <span style={styles.navLink} onClick={() => navigate('/')}>Logout</span>
        </div>
      </nav>

      {/* PAGE CONTENT */}
      <h1 style={styles.heading}>DRUG DESIGN AND DISCOVERY</h1>
      <div style={styles.gridContainer}>
        {tools.map((tool, index) => (
          <div
            key={index}
            style={styles.toolCard}
            onClick={() => tool.route && navigate(tool.route)}
          >
            <img src={tool.image} alt={tool.title} style={styles.toolImage} />
            <h3 style={styles.toolTitle}>{tool.title}</h3>
            <p style={styles.toolDescription}>{toolDescriptions[tool.key]}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

const styles = {
  page: {
    minHeight: '100vh',
    backgroundColor: '#121212',
    color: '#fff',
    padding: '0px 40px 40px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  navbar: {
    width: '100%',
    backgroundColor: '#1e1e1e',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '16px 32px',
    borderBottom: '1px solid #333',
    position: 'sticky',
    top: 0,
    zIndex: 100,
  },
  navTitle: {
    fontSize: '20px',
    fontWeight: 'bold',
    color: '#90caf9',
  },
  navLinks: {
    display: 'flex',
    gap: '20px',
  },
  navLink: {
    cursor: 'pointer',
    color: '#ccc',
    fontSize: '14px',
    transition: 'color 0.2s',
  },
  heading: {
    fontSize: '28px',
    color: '#90caf9',
    margin: '30px 0',
    fontWeight: 'bold',
  },
  gridContainer: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '20px',
    width: '80%',
  },
  toolCard: {
    backgroundColor: '#1e1e1e',
    padding: '20px',
    borderRadius: '12px',
    textAlign: 'center',
    border: '1px solid #444',
    cursor: 'pointer',
    transition: 'transform 0.2s ease',
  },
  toolImage: {
    width: '50px',
    height: '50px',
    marginBottom: '10px',
  },
  toolTitle: {
    fontSize: '18px',
    color: '#90caf9',
    marginBottom: '8px',
  },
  toolDescription: {
    fontSize: '14px',
    color: '#ccc',
  },
};

export default Dashboard;
