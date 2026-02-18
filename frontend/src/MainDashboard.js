import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const MainDashboard = () => {
  const navigate = useNavigate();
  const [hovered, setHovered] = useState(null);

  const handleHover = (index) => setHovered(index);
  const handleLeave = () => setHovered(null);

  return (
    <div style={styles.container}>
      {/* NAVBAR */}
      <nav style={styles.navbar}>
        <div style={styles.navTitle}>DrugRin</div>
        <div style={styles.navLinks}>
          <span style={styles.navLink} onClick={() => navigate('/about')}>About</span>
          <span style={styles.navLink} onClick={() => navigate('/')}>Logout</span>
        </div>
      </nav>

      {/* MAIN CONTENT */}
      <h1 style={styles.heading}>Welcome to DrugRin</h1>

      <div style={styles.options}>
        {[{
          title: "DRUG DESIGN AND DISCOVERY",
          desc: "Explore the universe of molecular innovation.",
          icon: "🧬",
          route: "/dashboard"
        }, {
          title: "MED_DIAGNOSIS",
          desc: "Navigate the pathways of diagnostic excellence.",
          icon: "⚕️",
          route: "/diagnosis"
        }].map((item, index) => (
          <div
            key={index}
            onClick={() => navigate(item.route)}
            onMouseEnter={() => handleHover(index)}
            onMouseLeave={handleLeave}
            style={{
              ...styles.card,
              ...(hovered === index ? styles.cardHover : {}),
            }}
          >
            <div style={styles.icon}>{item.icon}</div>
            <h2 style={styles.cardHeading}>{item.title}</h2>
            <p style={styles.cardDescription}>{item.desc}</p>
          </div>
        ))}
      </div>

      <footer style={styles.footer}>
        <div style={styles.footerText}>
          Made with <span style={styles.heart}></span> by DrugRin Team
        </div>
      </footer>
    </div>
  );
};

const styles = {
  container: {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #1c1c1c, #000000)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    color: '#f5f5f5',
    paddingTop: '80px', // accommodate navbar height
    fontFamily: 'Segoe UI, sans-serif',
    position: 'relative',
  },
  navbar: {
    width: '100%',
    backgroundColor: '#1e1e1e',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '16px 32px',
    borderBottom: '1px solid #333',
    position: 'fixed',
    top: 0,
    zIndex: 1000,
  },
  navTitle: {
    fontSize: '20px',
    fontWeight: 'bold',
    color: '#90caf9',
    marginLeft: '10px'
  },
  navLinks: {
    display: 'flex',
    gap: '20px',
    marginRight: '8px'
  },
  navLink: {
    cursor: 'pointer',
    color: '#ccc',
    fontSize: '14px',
    transition: 'color 0.2s',
  },
  heading: {
    fontSize: '3.2em',
    marginBottom: '60px',
    color: '#ffffff',
    fontWeight: '300',
    letterSpacing: '1px',
  },
  options: {
    display: 'flex',
    gap: '40px',
    flexWrap: 'wrap',
    justifyContent: 'center',
  },
  card: {
    backgroundColor: 'rgba(33, 42, 50, 0.85)',
    padding: '35px',
    borderRadius: '14px',
    border: '1px solid #4f5b62',
    cursor: 'pointer',
    textAlign: 'center',
    width: '280px',
    boxShadow: '0 6px 15px rgba(0, 0, 0, 0.35)',
    transition: 'transform 0.3s ease, box-shadow 0.3s ease',
  },
  cardHover: {
    transform: 'scale(1.05)',
    boxShadow: '0 10px 25px rgba(0, 0, 0, 0.5)',
  },
  icon: {
    fontSize: '3em',
    marginBottom: '20px',
    color: '#81d4fa',
    textShadow: '1px 1px 1px rgba(0,0,0,0.3)',
  },
  cardHeading: {
    fontSize: '1.6em',
    color: '#f48fb1',
    marginBottom: '12px',
    fontWeight: '300',
    letterSpacing: '0.5px',
  },
  cardDescription: {
    fontSize: '1em',
    color: '#b0bec5',
    lineHeight: '1.5',
  },
  footer: {
    position: 'absolute',
    bottom: '20px',
    fontSize: '0.9em',
    color: '#90a4ae',
  },
  footerText: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  heart: {
    color: '#e91e63',
    marginLeft: '6px',
  },
};

export default MainDashboard;
