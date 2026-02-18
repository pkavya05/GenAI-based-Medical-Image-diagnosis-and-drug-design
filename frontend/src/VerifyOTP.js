import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const VerifyOTP = () => {
  const [email, setEmail] = useState('');
  const [otp, setOtp] = useState('');
  const navigate = useNavigate();

  const handleVerifyOTP = async () => {
    try {
      const email = localStorage.getItem('email');
      if (!email) {
        alert("No email found. Please sign up again.");
        return;
      }

      const response = await axios.post('http://localhost:5001/verify-otp', { email, otp });
      alert(response.data.message);
      navigate('/dashboard');
    } catch (error) {
      alert('Error: ' + (error.response?.data?.message || 'Verification failed'));
    }
  };

  return (
    <div style={{
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      justifyContent: 'center', 
      height: '100vh',
      backgroundColor: '#f4f4f4'
    }}>
      <div style={{
        padding: '30px',
        borderRadius: '10px',
        background: '#fff',
        boxShadow: '0px 4px 10px rgba(0, 0, 0, 0.1)',
        textAlign: 'center'
      }}>
        <h2 style={{ marginBottom: '20px' }}>Verify OTP</h2>
        <input
          type="email"
          placeholder="Enter Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          style={{
            padding: '10px',
            width: '250px',
            marginBottom: '10px',
            borderRadius: '5px',
            border: '1px solid #ccc'
          }}
        /><br />
        <input
          type="text"
          placeholder="Enter OTP"
          value={otp}
          onChange={(e) => setOtp(e.target.value)}
          style={{
            padding: '10px',
            width: '250px',
            marginBottom: '15px',
            borderRadius: '5px',
            border: '1px solid #ccc'
          }}
        /><br />
        <button 
          onClick={handleVerifyOTP} 
          style={{
            padding: '10px 20px',
            border: 'none',
            background: '#007bff',
            color: 'white',
            borderRadius: '5px',
            cursor: 'pointer',
            fontSize: '16px'
          }}>
          Verify OTP
        </button>
      </div>
    </div>
  );
};

export default VerifyOTP;
