import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './LandingPage';
import Signup from './Signup';
import Login from './Login';
import VerifyOTP from './VerifyOTP';
import Dashboard from './Dashboard';
import PredictPage from './PredictPage';
import ProteinPage from './ProteinPage';
import ActivePage from './ActivePage';
import SimilarPage from './SimilarPage';
import StructurePage from './StructurePage';
import DockingPage from './DockingPage';
import About from './About';
import MainDashboard from './MainDashboard';
import DiagnosisPage from './DiagnosisPage';
import ThreeDVis from './ThreeDVis'; // ✅ Add this line

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/login" element={<Login />} />
        <Route path="/verify-otp" element={<VerifyOTP />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/predict" element={<PredictPage />} />
        <Route path="/protein" element={<ProteinPage />} />
        <Route path="/active" element={<ActivePage />} />
        <Route path="/similar" element={<SimilarPage />} />
        <Route path="/structure" element={<StructurePage />} />
        <Route path="/docking" element={<DockingPage />} />
        <Route path="/about" element={<About />} />
        <Route path="/maindashboard" element={<MainDashboard />} />
        <Route path="/diagnosis" element={<DiagnosisPage />} />
        <Route path="/threedvis" element={<ThreeDVis />} /> {/* ✅ Added Route */}
        
        {/* Fallback route for 404 */}
        <Route path="*" element={<div>404 - Page Not Found</div>} />
      </Routes>
    </Router>
  );
};

export default App;
