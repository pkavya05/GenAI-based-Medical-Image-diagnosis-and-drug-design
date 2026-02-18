import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const Login = () => {
  const [formData, setFormData] = useState({ email: '', password: '' });
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData((prevData) => ({ ...prevData, [e.target.name]: e.target.value }));
  };

  const handleLogin = async () => {
    try {
      const response = await axios.post('http://localhost:5001/login', formData);
      alert(response.data?.message || 'Login successful');
      navigate('/maindashboard');

      // navigate('/dashboard');
    } catch (error) {
      console.error('Login Error:', error.response?.data || error.message);
      alert('Error: ' + (error.response?.data?.message || 'Something went wrong'));
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.background}></div>

      <div style={styles.loginBox}>
        <h2 style={styles.title}>Welcome Back</h2>
        <p style={styles.subtitle}>Please log in to continue</p>

        <label style={styles.label}>Email</label>
        <input
          type="email"
          name="email"
          placeholder="Enter your email"
          onChange={handleChange}
          style={styles.input}
        />

        <label style={styles.label}>Password</label>
        <input
          type="password"
          name="password"
          placeholder="Enter your password"
          onChange={handleChange}
          style={styles.input}
        />

        <button onClick={handleLogin} style={styles.button}>Login</button>
      </div>
    </div>
  );
};

const styles = {
  container: {
    position: 'relative',
    height: '100vh',
    width: '100vw',
    overflow: 'hidden',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    fontFamily: 'Segoe UI, sans-serif',
    background: '#121212',
  },
  background: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    zIndex: -1,
    backgroundImage: `url('background-drug-icons.png')`, // Use a custom background image with drug discovery icons
    backgroundSize: 'contain',
    backgroundRepeat: 'repeat',
    opacity: 0.1,
  },
  loginBox: {
    width: '400px',
    padding: '40px 30px',
    borderRadius: '12px',
    background: 'rgba(30, 41, 59, 0.85)',
    backdropFilter: 'blur(10px)',
    WebkitBackdropFilter: 'blur(10px)',
    boxShadow: '0 8px 20px rgba(0, 0, 0, 0.6)',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    textAlign: 'center',
    color: '#e0e0e0',
  },
  title: {
    fontSize: '2.2rem',
    fontWeight: '600',
    marginBottom: '15px',
    color: '#ffffff',
  },
  subtitle: {
    fontSize: '1.1rem',
    color: '#a3a3a3',
    marginBottom: '25px',
  },
  label: {
    display: 'block',
    textAlign: 'left',
    marginBottom: '8px',
    fontSize: '1rem',
    color: '#d4d4d4',
    fontWeight: '500',
  },
  input: {
    width: '100%',
    padding: '12px 15px',
    marginBottom: '20px',
    borderRadius: '8px',
    border: '1px solid #4a5568',
    backgroundColor: '#1e293b',
    color: '#f0f0f0',
    fontSize: '1rem',
    outline: 'none',
    transition: '0.3s',
    backdropFilter: 'blur(2px)',
  },
  button: {
    width: '100%',
    padding: '14px',
    background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '1.1rem',
    fontWeight: '500',
    transition: '0.3s',
    ':hover': {
      opacity: 0.9,
    },
  },
};

export default Login;