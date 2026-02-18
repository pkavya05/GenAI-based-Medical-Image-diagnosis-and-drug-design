# ================================================
# Imports (Combined)
# ================================================

# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
import torch
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import base64
from predict import predict_smiles

import numpy as np
import cv2
import os
import uuid
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# ================================================
# Flask App Initialization
# # ================================================
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
from flask import Flask, request, jsonify
from flask_cors import CORS

# app = Flask(__name__)
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

app = Flask(__name__)
CORS(app)

# ================================================
# SMILES MODEL SETUP
# ================================================

with open("C:\\Users\\kavya\\OneDrive\\Documents\\flasskkk[1]\\flask1\\flask\\updated_vocab.json", "r") as f:
    vocab = json.load(f)

bpe_model = BPE.from_file(
    "C:\\Users\\kavya\\OneDrive\\Documents\\flasskkk[1]\\flask1\\flask\\updated_vocab.json",
    "C:\\Users\\kavya\\OneDrive\\Documents\\flasskkk[1]\\flask1\\flask\\merges.txt"
)
tokenizer = Tokenizer(bpe_model)
tokenizer.add_special_tokens(['<mask>'])

class RoBERTaEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, max_len=128):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Embedding(max_len, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        positions = torch.arange(0, input_ids.size(1)).unsqueeze(0).to(input_ids.device)
        x = self.token_embed(input_ids) + self.position_embed(positions)
        return self.dropout(self.norm(x))

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)
        return self.dropout(self.out_proj(out))

class CustomTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden=512):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

class RoBERTaForMaskedLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_layers=4, max_len=128):
        super().__init__()
        self.embedding = RoBERTaEmbedding(vocab_size, embed_dim, max_len)
        self.encoder = nn.Sequential(*[
            CustomTransformerBlock(embed_dim, num_heads=8)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.encoder(x)
        return self.lm_head(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RoBERTaForMaskedLM(len(vocab))
model.load_state_dict(torch.load(
    "C:\\Users\\kavya\\OneDrive\\Documents\\flasskkk[1]\\flask1\\flask\\model.pth",
    map_location=device
))
model.to(device)
model.eval()

def encode_input(smiles, tokenizer, vocab):
    encoded = tokenizer.encode(smiles)
    tokens = []
    for token in encoded.tokens:
        if token == "<mask>":
            tokens.append(vocab["<mask>"])
        else:
            tokens.append(vocab.get(token, vocab["[UNK]"]))
    return torch.tensor(tokens).unsqueeze(0).to(device)

@app.route('/predict', methods=['POST'])
def predict_masked_smiles():
    data = request.get_json()
    smiles_input = data.get('smiles', '')

    if '<mask>' not in smiles_input:
        return jsonify({'error': 'Input must contain <mask> token.'}), 400

    input_ids = encode_input(smiles_input, tokenizer, vocab)
    mask_token_id = vocab["<mask>"]
    mask_indices = (input_ids == mask_token_id).nonzero(as_tuple=True)[1]

    if len(mask_indices) == 0:
        return jsonify({"error": "No <mask> token found."})

    mask_index = mask_indices[0].item()
    with torch.no_grad():
        logits = model(input_ids)
        mask_logits = logits[0, mask_index]
        probs = F.softmax(mask_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=5)
        topk_probs = topk_probs.cpu().numpy()
        topk_indices = topk_indices.cpu().numpy()

        results = []
        for prob, token_id in zip(topk_probs, topk_indices):
            token = list(vocab.keys())[list(vocab.values()).index(token_id)]
            results.append(f"{token} ({prob:.4f})")

    return jsonify({'prediction': "\n".join(results)})

def generate_smiles_image(smiles_string):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            raise ValueError("Invalid SMILES: RDKit parsing failed")
        img = Draw.MolToImage(mol, size=(300, 300))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"SMILES image generation error: {e}")
        return None

@app.route("/api/predict-smiles", methods=["POST"])
def predict_protein_to_smiles():
    try:
        data = request.get_json()
        protein_seq = data.get("protein_sequence")

        if not protein_seq:
            return jsonify({"error": "No protein sequence provided"}), 400

        smiles_str = predict_smiles(protein_seq)
        image_base64 = generate_smiles_image(smiles_str)

        if image_base64 is None:
            return jsonify({
                "smiles": smiles_str,
                "error": "Generated SMILES is invalid and cannot be visualized"
            }), 200

        return jsonify({
            "smiles": smiles_str,
            "image": image_base64
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500
from flask_cors import cross_origin
import requests
@app.route('/api/fold', methods=['POST'])
def fold_protein():
    try:
        sequence = request.json.get('sequence')

        if not sequence or len(sequence) < 10:
            return jsonify({'error': 'Please provide a valid protein sequence with at least 10 characters'}), 400

        response = requests.post(
            'https://api.esmatlas.com/foldSequence/v1/pdb/',
            data=sequence,
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )

        if response.status_code != 200:
            return jsonify({'error': 'Failed to get structure from ESMFold API'}), response.status_code

        pdb_data = response.text
        return jsonify({'pdb': pdb_data})

    except Exception as e:
        return jsonify({'error': str(e)}),500

# ================================================
# Segmentation Models Setup
# ================================================

# Load segmentation models

from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import tensorflow as tf
from io import BytesIO
import os
from keras.saving import register_keras_serializable
from patchify import patchify
from flask_cors import CORS
import tempfile

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow internal logs

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# Register custom loss and metric functions
@register_keras_serializable()
def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2.0 * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

@register_keras_serializable()
def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Load the brain tumor model
BRAIN_MODEL_PATH = "C:\\Users\\kavya\\Desktop\\MERNPS\\drugdiscovery\\backend\\flask\\model.keras"
try:
    print("Loading brain tumor model...")
    brain_model = tf.keras.models.load_model(BRAIN_MODEL_PATH, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})
    print("Brain tumor model loaded successfully.")
except Exception as e:
    print(f"Error loading brain tumor model: {e}")
    exit(1)

# Load the lung tumor model
LUNG_MODEL_PATH = "C:\\Users\\kavya\\Desktop\\MERNPS\\drugdiscovery\\backend\\flask\\model (4).keras"
try:
    print("Loading lung tumor model...")
    lung_model = tf.keras.models.load_model(LUNG_MODEL_PATH, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})
    print("Lung tumor model loaded successfully.")
except Exception as e:
    print(f"Error loading lung tumor model: {e}")
    exit(1)

# Configuration
cf = {
    "image_size": 256,
    "num_channels": 3,
    "patch_size": 16,
    "num_patches": (256 // 16) * (256 // 16),  # = 256 patches (16x16 grid)
    "flat_patches_shape": (256, 16 * 16 * 3)  # (256 patches, 768 features each)
}

def preprocess_image(image):
    """Preprocess the image for model input."""
    print("Preprocessing image...")
    # Resize the image to the expected input shape (256, 256)
    image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
    # Ensure 3 channels
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    # Normalize the image
    if image.max() > 1:
        image = image / 255.0
    return image

def patchify_image(image):
    """Create patches from the image."""
    print("Patchifying image...")
    patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
    patches = patchify(image, patch_shape, cf["patch_size"])
    patches = patches.reshape(cf["num_patches"], cf["patch_size"] * cf["patch_size"] * cf["num_channels"])
    patches = patches.astype(np.float32)
    patches = np.expand_dims(patches, axis=0)  # Add batch dimension (1, 256, 768)
    return patches

@app.route('/api/segment', methods=['POST'])
def segment():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        processed_image = preprocess_image(image)
        patches = patchify_image(processed_image)

        print("Predicting segmentation mask...")
        # Get raw model output (should be 256x256)
        prediction = brain_model.predict(patches, verbose=0)
        
        # Handle different output shapes:
        if prediction.shape[-1] == 1:  # Single-channel output
            prediction = prediction[0,:,:,0]  # Remove batch dim, keep 256x256
        else:  # Multi-channel output
            prediction = prediction[0,:,:,1]  # Assuming class 1 is tumor

        print(f"Prediction shape: {prediction.shape}")  # Debug
        
        # Threshold and convert to uint8
        prediction = (prediction > 0.5).astype(np.uint8)

        # Resize to original image dimensions (if needed)
        original_height, original_width = image.shape[:2]
        if prediction.shape != (original_height, original_width):
            prediction = cv2.resize(prediction, (original_width, original_height))

        # Create red mask and blend
        red_mask = np.zeros_like(image)
        red_mask[prediction == 1] = [0, 0, 255]  # BGR red
        
        merged_image = cv2.addWeighted(image, 0.7, red_mask, 0.3, 0)

        _, buffer = cv2.imencode('.png', merged_image)
        return send_file(BytesIO(buffer), mimetype='image/png')

    except Exception as e:
        print(f"Error during segmentation: {e}")
        return jsonify({"error": str(e)}), 500

# ... (Keep the rest of the code unchanged)

@app.route('/api/segment_lung_npy', methods=['POST'])
def segment_lung_npy():
    try:
        # Check if an .npy file is uploaded
        if 'npy' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        # Read the uploaded .npy file
        file = request.files['npy']
        data = np.load(file)

        # Log the shape of the data
        print(f"Input data shape: {data.shape}")

        # Preprocess the image
        image = preprocess_image(data)

        # Debug: Print shape and type after preprocessing
        print(f"Preprocessed image shape: {image.shape}, dtype: {image.dtype}")

        # Patchify the image
        patches = patchify_image(image)

        # Predict using the lung tumor model
        print("Predicting segmentation mask...")
        prediction = lung_model.predict(patches, verbose=0)[0]

        # Reshape the prediction to match the original image size
        prediction = np.squeeze(prediction)  # Remove extra dimensions if needed

        # Threshold the prediction to create a binary mask
        prediction = (prediction > 0.5).astype(np.uint8)  # Binary mask (0 or 1)

        # Debug: Print shape and type of prediction
        print(f"Prediction shape: {prediction.shape}, dtype: {prediction.dtype}")

        # Create a red mask with uint8 data type
        red_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        red_mask[prediction == 1] = [0, 0, 255]  # Set mask region to red (BGR format)

        # Ensure the image is in uint8 format for OpenCV
        image_uint8 = (image * 255).astype(np.uint8)

        # Debug: Print shape and type of image_uint8 and red_mask
        print(f"Image uint8 shape: {image_uint8.shape}, dtype: {image_uint8.dtype}")
        print(f"Red mask shape: {red_mask.shape}, dtype: {red_mask.dtype}")

        # Merge the input image with the red mask
        merged_image = cv2.addWeighted(image_uint8, 0.7, red_mask, 0.3, 0)

        # Save the output image to a bytes buffer
        _, buffer = cv2.imencode('.png', merged_image)
        io_buf = BytesIO(buffer)

        # Return the image as a response
        return send_file(io_buf, mimetype='image/png')
    except Exception as e:
        print(f"Error during lung tumor segmentation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/process_npy', methods=['POST'])
def process_npy():
    try:
        # Check if an .npy file is uploaded
        if 'npy' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        # Read the uploaded .npy file
        file = request.files['npy']
        data = np.load(file)

        # Log the shape and type of the data
        print(f"Input data shape: {data.shape}, dtype: {data.dtype}")

        # Check if the data is valid
        if data.size == 0:
            return jsonify({"error": "The .npy file is empty"}), 400
        if np.isnan(data).any() or np.isinf(data).any():
            return jsonify({"error": "The .npy file contains NaN or Inf values"}), 400

        # Convert float16 to float32 (if necessary)
        if data.dtype == np.float16:
            data = data.astype(np.float32)

        # Normalize the data to the range [0, 255] for image display
        normalized_data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
        normalized_data = normalized_data.astype(np.uint8)

        # Convert the data to a 3-channel image (if it's grayscale)
        if len(normalized_data.shape) == 2:
            normalized_data = cv2.cvtColor(normalized_data, cv2.COLOR_GRAY2BGR)

        # Save the image to a bytes buffer
        _, buffer = cv2.imencode('.png', normalized_data)
        io_buf = BytesIO(buffer)

        # Return the image as a response
        return send_file(io_buf, mimetype='image/png')
    except Exception as e:
        print(f"Error processing .npy file: {e}")
        return jsonify({"error": str(e)}), 500



@app.route('/api/segment_lung_video', methods=['POST'])
def segment_lung_video():
    try:
        # Check if a video file is uploaded
        if 'video' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        # Read the uploaded video file
        file = request.files['video']
        video_path = tempfile.mktemp(suffix=".mp4")
        file.save(video_path)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({"error": "Failed to open video file"}), 400

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # Use 'H264' codec for MP4 format

        # Create a temporary output video file
        output_video_path = tempfile.mktemp(suffix=".mp4")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # Batch processing configuration
        batch_size = 16  # Number of frames to process in a single batch
        frame_batch = []  # List to store frames in the current batch

        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess the frame
            processed_frame = preprocess_image(frame)
            frame_batch.append((frame, processed_frame))  # Store both original and processed frames

            # If the batch is full, process it
            if len(frame_batch) == batch_size:
                # Extract processed frames from the batch
                processed_frames = [processed_frame for _, processed_frame in frame_batch]

                # Patchify the batch of frames
                patches = np.array([patchify_image(frame) for frame in processed_frames])
                patches = np.concatenate(patches, axis=0)

                # Predict using the lung tumor model
                predictions = lung_model.predict(patches, verbose=0)

                # Process each prediction in the batch
                for i, (original_frame, _) in enumerate(frame_batch):
                    # Reshape the prediction to match the original frame size
                    prediction = np.squeeze(predictions[i])
                    prediction = (prediction > 0.5).astype(np.uint8)

                    # Create a red mask
                    red_mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                    red_mask[prediction == 1] = [0, 0, 255]  # Set mask region to red (BGR format)

                    # Merge the original frame with the red mask
                    merged_frame = cv2.addWeighted(original_frame, 0.7, red_mask, 0.3, 0)

                    # Write the processed frame to the output video
                    out.write(merged_frame)

                # Clear the batch
                frame_batch = []

        # Process any remaining frames in the batch
        if frame_batch:
            # Extract processed frames from the batch
            processed_frames = [processed_frame for _, processed_frame in frame_batch]

            # Patchify the batch of frames
            patches = np.array([patchify_image(frame) for frame in processed_frames])
            patches = np.concatenate(patches, axis=0)

            # Predict using the lung tumor model
            predictions = lung_model.predict(patches, verbose=0)

            # Process each prediction in the batch
            for i, (original_frame, _) in enumerate(frame_batch):
                # Reshape the prediction to match the original frame size
                prediction = np.squeeze(predictions[i])
                prediction = (prediction > 0.5).astype(np.uint8)

                # Create a red mask
                red_mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                red_mask[prediction == 1] = [0, 0, 255]  # Set mask region to red (BGR format)

                # Merge the original frame with the red mask
                merged_frame = cv2.addWeighted(original_frame, 0.7, red_mask, 0.3, 0)

                # Write the processed frame to the output video
                out.write(merged_frame)

        # Release video capture and writer
        cap.release()
        out.release()

        # Return the processed video as a response
        return send_file(output_video_path, mimetype='video/mp4')
    except Exception as e:
        print(f"Error during lung tumor video segmentation: {e}")
        return jsonify({"error": str(e)}), 500

# import os
# import subprocess
# import requests
# import json
# import MDAnalysis as mda
# import prolif as plf
# from rcsbsearchapi.search import TextQuery
# from rcsbsearchapi import rcsb_attributes as attrs
# from rdkit import Chem
# from flask import request, jsonify
import joblib 
from rdkit import Chem
from rdkit.Chem import AllChem  # ✅ MAKE SURE THIS IS INCLUDED
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, embedding_dim=48):
        super().__init__()
        self.patcher = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2, 3)

    def forward(self, x):
        x = self.patcher(x)
        x = self.flatten(x)
        return x.permute(0, 2, 1)

class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=48, num_heads=4, attn_dropout=0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads, 
            dropout=attn_dropout, 
            batch_first=True
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(x_norm, x_norm, x_norm, need_weights=False)
        return attn_output

class MLPBlock(nn.Module):
    def __init__(self, embedding_dim=48, mlp_size=3072, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(self.layer_norm(x))

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim=48, num_heads=4, mlp_size=3072, mlp_dropout=0.1, attn_dropout=0):
        super().__init__()
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim, num_heads, attn_dropout)
        self.mlp_block = MLPBlock(embedding_dim, mlp_size, mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embedding_dim=48, num_transformer_layers=12, num_heads=4,
                 mlp_size=3072, attn_dropout=0, mlp_dropout=0.1, embedding_dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.class_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embedding_dim)
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embedding_dim, num_heads, mlp_size, mlp_dropout, attn_dropout)
            for _ in range(num_transformer_layers)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.embedding_dropout(x + self.position_embedding)
        for block in self.encoder_blocks:
            x = block(x)
        return self.classifier(x[:, 0])

# ====================================
# Load Model and Helper Functions
# ====================================
def load_model():
    # Load model config from pkl file
    model_data = joblib.load("C:\\Users\\kavya\\Desktop\\MERNPS\\drugdiscovery\\backend\\flask\\vit_model.pkl")
    
    # Initialize model
    model = ViT(**model_data["model_config"])
    
    # Load state dict from pth file
    model.load_state_dict(torch.load(model_data["model_state_dict"], map_location=torch.device('cpu')))
    model.eval()
    
    return model, model_data["label_encoder"]

vit, le = load_model()

def morgan_to_image(x):
    flat = np.pad(x, (0, 3 * 32 * 32 - len(x)), constant_values=0)
    return flat.reshape(3, 32, 32)

# ====================================
# API Endpoints
# ====================================
@app.route('/predict-vit', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        smiles = data.get('smiles', '')
        
        # Convert SMILES to fingerprint
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return jsonify({"error": "Invalid SMILES string"}), 400
            
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fp_array = np.array(fingerprint, dtype=np.float32)
        fp_array = fp_array / (fp_array.max() + 1e-6)
        
        # Convert to image and predict
        image = morgan_to_image(fp_array)
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            output = vit(image_tensor)
            predicted_class_idx = output.argmax(1).item()
            predicted_label = le.inverse_transform([predicted_class_idx])[0]
            
        return jsonify({
            "smiles": smiles,
            "predicted_class": predicted_label
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})
# Function to download the PDB and Ligand files
from flask import request, jsonify
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw
import base64
from io import BytesIO
import numpy as np
import pandas as pd
from data import GeneratorData
from stackRNN import StackAugmentedRNN
#REINFORCEMENT LEARNING
from flask import request, jsonify
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw
import base64
from io import BytesIO
import numpy as np
import pandas as pd
from data import GeneratorData
from stackRNN import StackAugmentedRNN
from pathlib import Path
from pathlib import Path
import requests
# Initialize Flask app and CORS
# app = Flask(name)
# #CORS(app)
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)

# Global cache dictionaries for storing input-output mappings
property_prediction_cache = {}
reinforcement_generation_cache = {}

# Cache file paths (optional: change as needed)
property_cache_file = "property_prediction_cache.pkl"
reinforcement_cache_file = "reinforcement_generation_cache.pkl"

# Load cached data from files (if exists)
def load_cache():
    global property_prediction_cache, reinforcement_generation_cache
    if Path(property_cache_file).exists():
        property_prediction_cache = joblib.load(property_cache_file)
    if Path(reinforcement_cache_file).exists():
        reinforcement_generation_cache = joblib.load(reinforcement_cache_file)

# Save cached data to files
def save_cache():
    joblib.dump(property_prediction_cache, property_cache_file)
    joblib.dump(reinforcement_generation_cache, reinforcement_cache_file)

# Load cache when the app starts
load_cache()

# --- Load data ---
gen_data_path = 'C:\\Users\\kavya\\Desktop\\MERNPS\\drugdiscovery\\backend\\flask\\123.smi'  # Adjust path if needed
tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']
gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t', cols_to_read=[0], keep_header=True, tokens=tokens)

my_generator = StackAugmentedRNN(
    input_size=gen_data.n_characters,
    hidden_size=1500,
    output_size=gen_data.n_characters,
    layer_type='GRU',
    n_layers=1,
    is_bidirectional=False,
    has_stack=True,
    stack_width=1500,
    stack_depth=200,
    use_cuda=None,
    optimizer_instance=torch.optim.Adadelta,
    lr=0.001
)
my_generator.load_model('C:\\Users\\kavya\\Desktop\\MERNPS\\drugdiscovery\\backend\\flask\\latest')  # Make sure model is present

# --- Helper Functions ---
def smiles_to_fp_array(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(list(fp.ToBitString())).astype(int)

def calculate_molecular_properties(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    properties = {}
    if molecule is not None:
        properties['Molecular Weight'] = Descriptors.MolWt(molecule)
        properties['LogP'] = Descriptors.MolLogP(molecule)
        properties['H-Bond Donor Count'] = Descriptors.NumHDonors(molecule)
        properties['H-Bond Acceptor Count'] = Descriptors.NumHAcceptors(molecule)
        properties['pIC50'] = 5 + 0.1 * properties['LogP'] + 0.01 * properties['Molecular Weight']
    return properties

def smiles_to_base64(smiles):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=(300, 300))
    buf = BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def compute_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    return AllChem.DataStructs.TanimotoSimilarity(fp1, fp2)

# --- Flask Route ---
@app.route('/api/reinforcement', methods=['POST'])
def reinforcement_learning_generate():
    try:
        data = request.get_json()
        input_smiles = data.get('SMILES', '').strip()

        if not input_smiles:
            return jsonify({'error': 'SMILES input missing'}), 400
        
        if input_smiles in reinforcement_generation_cache:
            print("Returning cached result for:", input_smiles)
            cached_results = reinforcement_generation_cache[input_smiles]
            best = cached_results[0]
            return jsonify({
                'top_results': cached_results,
                'best_smile': best.get('smiles'),
                'best_logP': best.get('logP'),
                'best_pIC50': best.get('pIC50'),
                'best_reward': best.get('reward'),
                'best_image': best.get('image')
            })

        user_smile = '<' + input_smiles
        results = []
        seen = set()
        count = 0

        while count < 5:
            generated_smile = my_generator.evaluate(gen_data, prime_str=user_smile)
            try:
                mol = Chem.MolFromSmiles(generated_smile[1:-1])
                valid_smile = generated_smile[1:-1]

                if mol and valid_smile not in seen:
                    seen.add(valid_smile)

                    props = calculate_molecular_properties(valid_smile)
                    reward = compute_similarity(input_smiles, valid_smile)
                    img_b64 = smiles_to_base64(valid_smile)

                    results.append({
                        'smiles': valid_smile,
                        'pIC50': round(props['pIC50'], 3),
                        'logP': round(props['LogP'], 3),
                        'reward': round(reward, 3),
                        'image': img_b64
                    })
                    count += 1
            except Exception as e:
                print(f"Error parsing: {generated_smile} - {e}")
                continue

        sorted_results = sorted(results, key=lambda x: x['reward'], reverse=True)
        best = sorted_results[0]
        reinforcement_generation_cache[input_smiles] = sorted_results
        save_cache()

        return jsonify({
            'top_results': sorted_results,
            'best_smile': best.get('smiles'),
            'best_logP': best.get('logP'),
            'best_pIC50': best.get('pIC50'),
            'best_reward': best.get('reward'),
            'best_image': best.get('image')
        })
        
    except Exception as e:
        print(f"Reinforcement error: {e}")
        return jsonify({'error':str(e)}),500
#---------------------------------------------------------------------------------------docking\
import sys
import MDAnalysis as mda
import prolif as plf
from rcsbsearchapi.search import TextQuery
from rcsbsearchapi import rcsb_attributes as attrs
import requests
import subprocess
def download_pdb_and_ligand(ec_number, ligand_id):
    """Downloads the PDB and Ligand files."""
    q1 = attrs.rcsb_polymer_entity.rcsb_ec_lineage.id == ec_number
    q2 = TextQuery(ligand_id)
    query = q1 & q2

    results = list(query())
    if not results:
        print("No results found.")
        return None, None

    pdb_id = results[0].lower()
    ligand_id = ligand_id.lower()

    protein_directory = "C:\\Users\\kavya\\Desktop\\MERNPS\\drugdiscovery\\backend\\flask\\protein_structures_final"
    os.makedirs(protein_directory, exist_ok=True)

    pdb_request = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb")
    ligand_request = requests.get(f"https://files.rcsb.org/ligands/download/{ligand_id}_ideal.sdf")

    with open(f"{protein_directory}/{pdb_id}.pdb", "w+") as f:
        f.write(pdb_request.text)

    os.makedirs("ligands_to_dock_final", exist_ok=True)
    with open(f"ligands_to_dock_final/{ligand_id}_ideal.sdf", "w+") as file:
        file.write(ligand_request.text)

    return pdb_id, ligand_id


def convert_ligand_to_pdbqt(ligand_id):
    """Converts the ligand file to PDBQT format."""
    os.makedirs("pdbqt_final", exist_ok=True)
    try:
        subprocess.run(
            ['C:\\Users\\kavya\\Desktop\\MERNPS\\drugdiscovery\\backend\\flask\\OpenBabel-3.1.1\\OpenBabel-3.1.1\\obabel.exe',
             f"ligands_to_dock_final/{ligand_id}_ideal.sdf",
             "-O",
             f"pdbqt_final/{ligand_id}.pdbqt"],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Ligand {ligand_id} converted to pdbqt format.")
    except subprocess.CalledProcessError as e:
        print(f"Error converting ligand: {e.stderr}")
        raise


def run_plif_and_visualize(pdb_id, ligand_id):
    """Performs PLIF analysis and generates a 3D visualization."""
    pdb_file = f"C:\\Users\\kavya\\Desktop\\MERNPS\\drugdiscovery\\backend\\flask\\protein_structures_final/{pdb_id}.pdb"
    sdf_file = f"C:\\Users\\kavya\\Desktop\\MERNPS\\drugdiscovery\\backend\\flask\\ligands_to_dock_final/{ligand_id}_ideal.sdf"

    try:
        protein = mda.Universe(pdb_file)
    except Exception as e:
        print(f"Error loading protein: {e}")
        return None

    try:
        protein_plf = plf.Molecule.from_mda(protein, NoImplicit=False)
        print("Protein loaded successfully.")
    except Exception as e:
        print(f"Error creating protein PLIF object: {e}")
        return None

    try:
        poses_plf = list(plf.sdf_supplier(sdf_file))
    except Exception as e:
        print(f"Error loading ligand poses: {e}")
        return None
    num_poses = len(poses_plf)
    print(f"Total poses available: {num_poses}")

    pose_index = 0
    if pose_index >= num_poses:
        print(f"Invalid pose index: {pose_index}. Only {num_poses} poses available.")
        return None

    ligand_mol = poses_plf[pose_index]
    fp = plf.Fingerprint(count=True)
    print("Fingerprint object created.")

    try:
        fp.run_from_iterable(poses_plf, protein_plf,n_jobs=1)
        results = fp.ifp
        print("PLIF calculation completed successfully.")
    except Exception as e:
        print(f"Error running PLIF: {e}")
        return None

    if not results:
        print("PLIF returned no results.")
        return None
    else:
        try:
            view = fp.plot_3d(
                ligand_mol, protein_plf, frame=pose_index, display_all=False
            )
            os.makedirs("C:\\Users\\kavya\\Desktop\\MERNPS\\drugdiscovery\\backend\\flask\\static\\outputs", exist_ok=True)
            html_path = "static/outputs/3d_view.html"
            full_html_path = os.path.abspath(html_path)
            with open(html_path, "w") as f:
                f.write(view._make_html())
            return html_path
        except Exception as e:
            print(f"Error generating or saving 3D view: {e}")
            return None

def run_docking_workflow(ec_number, ligand_id):
    """Orchestrates the molecular docking workflow."""
    pdb_id, ligand_id = download_pdb_and_ligand(ec_number, ligand_id)
    if not pdb_id or not ligand_id:
        return {"error": "Failed to download PDB or Ligand."}

    try:
        convert_ligand_to_pdbqt(ligand_id)
    except Exception as e:
        return {"error": f"Failed to convert ligand: {e}"}

    html_file_path = run_plif_and_visualize(pdb_id, ligand_id)
    if not html_file_path:
        return {"error": "Failed to generate visualization."}

    return {
        "message": "Docking process completed!",
        "pdb_id": pdb_id,
        "ligand_id": ligand_id,
        "visualization_html": html_file_path,
    }

@app.route('/api/autodocking', methods=['POST'])
def docking():
    try:
        data = request.get_json()
        ec_number = data.get('ecNumber')
        ligand_id = data.get('ligandId')

        if not ec_number or not ligand_id:
            return jsonify({"error": "Missing EC number or Ligand ID in JSON payload"}), 400

        results = run_docking_workflow(ec_number, ligand_id)
        return jsonify(results)

    except Exception as e:
        print(f"Error in docking route: {e}")
        return jsonify({"error": str(e)}), 500

# --- Helper Function for Visualization ---
def generate_smiles_image(smiles_string):
    try:
        print("Generated SMILES:", smiles_string)
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            raise ValueError("Invalid SMILES: RDKit parsing failed")

        img = Draw.MolToImage(mol, size=(300, 300))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded
    except Exception as e:
        print(f"SMILES image generation error: {e}")
    return None



# ================================================
# Home
# ================================================

@app.route('/')
def home():
    return "Unified API for Drug Discovery and Tumor Segmentation"

# ================================================
# Run
# ================================================

if __name__ == '__main__':
    app.run(debug=True)
