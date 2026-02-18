import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const Signup = () => {
  const [formData, setFormData] = useState({ name: '', email: '', phone: '', password: '' });
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData((prevData) => ({ ...prevData, [e.target.name]: e.target.value }));
  };

  const handleSignup = async () => {
    try {
        const response = await axios.post('http://localhost:5001/signup', formData);
        localStorage.setItem('email', formData.email.trim().toLowerCase()); // Store email correctly
        alert(response.data.message);
        navigate('/verify-otp'); // Redirect to OTP page
    } catch (error) {
        alert('Error: ' + (error.response?.data?.message || 'Signup failed'));
    }
  };

  return (
    <div style={styles.container}>
      {/* Left Side - Full Screen Image */}
      <div style={styles.leftPanel}>
        <img src="https://news.mit.edu/sites/default/files/styles/news_article__image_gallery/public/images/202406/MIT-MOLECular-Design-01_0.jpg?itok=zw5JmYFH" alt="Healthcare" style={styles.image} />
      </div>

      {/* Right Side - Full Screen Signup Box */}
      <div style={styles.rightPanel}>
        <div style={styles.formContainer}>
          <h2 style={styles.title}>Create an Account</h2>
          <input type="text" name="name" placeholder="Name" onChange={handleChange} required style={styles.input} />
          <input type="email" name="email" placeholder="Email" onChange={handleChange} required style={styles.input} />
          <input type="text" name="phone" placeholder="Phone" onChange={handleChange} required style={styles.input} />
          <input type="password" name="password" placeholder="Password" onChange={handleChange} required style={styles.input} />
          
          <button onClick={handleSignup} disabled={loading} style={styles.button}>
            {loading ? "Signing Up..." : "Sign Up"}
          </button>

          {message && <p style={styles.errorMessage}>{message}</p>}
        </div>
      </div>
    </div>
  );
};

const styles = {
  container: {
    display: 'flex',
    height: '100vh',
    width: '100vw',
  },
  leftPanel: {
    flex: 1,
  },
  rightPanel: {
    flex: 1,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: 'linear-gradient(to right, rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.9))',
  },
  formContainer: {
    width: '70%',
    maxWidth: '400px',
    background: 'rgba(255, 255, 255, 0.1)', // Glassmorphism effect
    padding: '30px',
    borderRadius: '12px',
    boxShadow: '0px 4px 10px rgba(255, 255, 255, 0.2)',
    backdropFilter: 'blur(10px)',
    textAlign: 'center',
    color: 'white',
  },
  title: {
    fontSize: '2rem',
    marginBottom: '15px',
  },
  input: {
    width: '100%',
    padding: '12px',
    margin: '10px 0',
    borderRadius: '5px',
    border: 'none',
    outline: 'none',
    fontSize: '1rem',
    backgroundColor: 'rgba(255, 255, 255, 0.2)', // Light transparency
    color: 'white',
  },
  button: {
    width: '100%',
    padding: '12px 24px',
    fontSize: '1.1rem',
    fontWeight: 'bold',
    borderRadius: '25px',
    border: '2px solid #00d4ff',
    cursor: 'pointer',
    background: 'transparent',
    color: 'white',
    transition: 'all 0.3s ease',
    boxShadow: '0px 4px 10px rgba(0, 212, 255, 0.3)',
  },
  errorMessage: {
    color: 'red',
    marginTop: '10px',
  },
  image: {
    width: '100%',
    height: '100vh',
    objectFit: 'cover',
  },
};

export default Signup;
