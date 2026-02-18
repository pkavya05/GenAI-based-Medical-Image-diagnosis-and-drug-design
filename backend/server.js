// Backend - server.js
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const bodyParser = require('body-parser');
const dotenv = require('dotenv');
const twilio = require('twilio');
const nodemailer = require('nodemailer');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const axios = require('axios');
const FormData = require('form-data');

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const uploadPath = 'uploads/';
        if (!fs.existsSync(uploadPath)) {
            fs.mkdirSync(uploadPath);
        }
        cb(null, uploadPath);
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + path.extname(file.originalname));
    }
});
const upload = multer({ storage });

dotenv.config();
const app = express();
app.use(cors());
app.use(bodyParser.json());

mongoose.connect(process.env.MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
    .then(() => console.log('MongoDB Connected'))
    .catch(err => console.log(err));

const User = mongoose.model('User', new mongoose.Schema({
    name: String,
    email: String,
    phone: String,
    password: String,
    otp: String,
    otpExpires: Date
}));

const generateOTP = () => Math.floor(100000 + Math.random() * 900000).toString();

const sendOTPBySMS = async (phone, otp) => {
    console.log(`📲 Sending OTP ${otp} to ${phone}`);
    const client = twilio(process.env.TWILIO_SID, process.env.TWILIO_AUTH_TOKEN);
    await client.messages.create({ body: `Your OTP is ${otp}`, from: process.env.TWILIO_PHONE, to: phone });
};

const sendOTPByEmail = async (email, otp) => {
    console.log(`📧 Sending OTP ${otp} to ${email}`);
    const transporter = nodemailer.createTransport({
        service: 'gmail',
        auth: { user: process.env.EMAIL, pass: process.env.EMAIL_PASS }
    });
    await transporter.sendMail({
        from: process.env.EMAIL,
        to: email,
        subject: 'Your OTP Code',
        text: `Your OTP is ${otp}`
    });
};
app.post('/signup', async (req, res) => {
    try {
        const { name, email, phone, password } = req.body;
        const existingUser = await User.findOne({ email });

        if (existingUser) {
            return res.status(400).json({ message: 'User already exists. Please log in.' });
        }

        const hashedPassword = await bcrypt.hash(password, 10);
        const otp = generateOTP();
        const otpExpires = new Date(Date.now() + 10 * 60000); // OTP valid for 10 mins

        const newUser = new User({
            name,
            email: email.toLowerCase().trim(), // Ensure email is stored in lowercase
            phone,
            password: hashedPassword,
            otp,
            otpExpires,
        });

        await newUser.save();
        console.log(`📩 Signup request received for: ${email}, Phone: ${phone}`);
        console.log(`📲 Sending OTP ${otp} to ${phone}`);
        console.log(`📧 Sending OTP ${otp} to ${email}`);

        await sendOTPBySMS(phone, otp);
        await sendOTPByEmail(email, otp);

        res.json({ message: 'OTP sent to email and phone' });
    } catch (error) {
        console.error('❌ Signup error:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
});
app.post('/verify-otp', async (req, res) => {
    try {
        const { email, otp } = req.body;
        console.log(`🛠 Received verify-otp request: { email: '${email}', otp: '${otp}' }`);

        const user = await User.findOne({ email: email.toLowerCase().trim() });

        if (!user) {
            console.log(`❌ User not found for email: ${email}`);
            return res.status(400).json({ message: 'User not found' });
        }

        console.log(`🔍 Stored OTP for ${email}: ${user.otp}, Expires at: ${user.otpExpires}`);

        if (!user.otp || user.otpExpires < new Date()) {
            return res.status(400).json({ message: 'Invalid or expired OTP' });
        }

        if (user.otp !== otp) {
            console.log(`❌ Invalid OTP entered: ${otp}`);
            return res.status(400).json({ message: 'Invalid OTP' });
        }

        // OTP verified, reset OTP fields
        user.otp = null;
        user.otpExpires = null;
        await user.save();

        const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET, { expiresIn: '1h' });
        res.json({ message: 'OTP verified', token });
    } catch (error) {
        console.error('❌ OTP verification error:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

app.post('/login', async (req, res) => {
    try {
        const { email, password } = req.body;
        console.log(`🔑 Login attempt for email: ${email}`);

        const user = await User.findOne({ email: email.toLowerCase().trim() });
        if (!user) {
            console.log(`❌ User not found for email: ${email}`);
            return res.status(400).json({ message: 'User not found' });
        }

        const isPasswordValid = await bcrypt.compare(password, user.password);
        if (!isPasswordValid) {
            console.log(`❌ Invalid password entered for ${email}`);
            return res.status(400).json({ message: 'Invalid credentials' });
        }

        const token = jwt.sign({ id: user._id }, process.env.JWT_SECRET, { expiresIn: '1h' });
        res.json({ message: 'Login successful', token });
    } catch (error) {
        console.error('❌ Login error:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

app.post("/api/upload", upload.single("image"), async (req, res) => {
    const { modelType } = req.body;

    if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
    }
    if (!modelType || !["brain", "lung"].includes(modelType)) {
        return res.status(400).json({ error: "Invalid model type" });
    }

    const flaskUrl = "http://localhost:8000/predict/segmentation";

    try {
        const formData = new FormData();
        formData.append("image_file", fs.createReadStream(req.file.path));
        formData.append("model_type", modelType);

        const response = await axios.post(flaskUrl, formData, {
            headers: formData.getHeaders(),
        });

        // Cleanup uploaded file
        fs.unlinkSync(req.file.path);

        if (response.data.success) {
            return res.json({
                success: true,
                predictedImage: response.data.mask_path,
            });
        } else {
            return res.status(500).json({
                error: "Model failed",
                details: response.data.message || "Unknown error",
            });
        }
    } catch (error) {
        console.error("Error forwarding to Flask:", error.message);
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }
        return res.status(500).json({
            error: "Failed to get prediction from Flask",
            details: error.message,
        });
    }
});

app.post("/api/uploadnii", upload.single("file"), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
    }

    const filePath = req.file.path;
    const originalName = req.file.originalname;

    try {
        const fastApiUrl = "http://localhost:8000/predict/segmentation/nii";

        const formData = new FormData();
        formData.append("image_file", fs.createReadStream(filePath), originalName);

        const response = await axios.post(fastApiUrl, formData, {
            headers: formData.getHeaders(),
            maxBodyLength: Infinity,
            maxContentLength: Infinity,
        });

        console.log("Response from FastAPI:", response.data);

        fs.unlinkSync(filePath); // Clean up uploaded temp file

        return res.json(response.data);
    } catch (error) {
        console.error("Error sending file to FastAPI:", error);

        if (fs.existsSync(filePath)) {
            fs.unlinkSync(filePath);
        }

        return res.status(500).json({
            error: "Failed to process .nii.gz file through FastAPI.",
            details: error.message,
        });
    }
});

app.use(express.json()); // Ensure JSON requests are parsed
app.listen(5001, '0.0.0.0', () => console.log('Node.js server running on port 5001'));

// app.listen(5001, () => console.log('Node.js server running on port 5001'));

