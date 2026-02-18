import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';

const LandingPage = () => {
  const navigate = useNavigate();

  return (
    <div style={styles.landingContainer}>
      <div style={styles.overlay}></div> {/* Dark overlay for better readability */}

      <motion.h1 
        style={styles.landingTitle}
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        Welcome to <span style={styles.highlight}>Drugrin</span>
      </motion.h1>
      
      <motion.p 
        style={styles.landingSubtitle}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3, duration: 0.6 }}
      >
        Discover and classify drugs with AI-powered predictions.
      </motion.p>
      
      <motion.div 
        style={styles.buttonContainer}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6, duration: 0.6 }}
      >
        <button style={styles.actionButton} onClick={() => navigate('/signup')}>Signup</button>
        <button style={styles.actionButton} onClick={() => navigate('/login')}>Login</button>
      </motion.div>
    </div>
  );
};

const styles = {
  landingContainer: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    textAlign: 'center',
    background: 'url("https://media.expert.ai/expertai/uploads/2024/04/AdobeStock_778241336.jpeg") center/cover no-repeat',
    position: 'relative',
    padding: '20px',
    color: 'white',
    fontFamily: '"Poppins", sans-serif',
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    backgroundColor: 'rgba(10, 25, 47, 0.85)', // Professional dark blue overlay
    zIndex: 1,
  },
  landingTitle: {
    fontSize: '2.8rem',
    fontWeight: 'bold',
    marginBottom: '20px',
    textShadow: '2px 2px 15px rgba(255, 255, 255, 0.3)',
    position: 'relative',
    zIndex: 2,
  },
  highlight: {
    color: '#00d4ff', // Cyan highlight for a professional look
  },
  landingSubtitle: {
    fontSize: '1.4rem',
    marginBottom: '30px',
    maxWidth: '600px',
    color: '#cfd8dc',
    position: 'relative',
    zIndex: 2,
  },
  buttonContainer: {
    display: 'flex',
    gap: '20px',
    position: 'relative',
    zIndex: 2,
  },
  actionButton: {
    padding: '12px 24px',
    fontSize: '1.1rem',
    fontWeight: 'bold',
    borderRadius: '30px',
    border: '2px solid #00d4ff',
    cursor: 'pointer',
    background: 'transparent',
    color: 'white',
    transition: 'all 0.3s ease',
    position: 'relative',
    zIndex: 2,
    overflow: 'hidden',
    boxShadow: '0px 4px 10px rgba(0, 212, 255, 0.3)',
  },
};

export default LandingPage;
