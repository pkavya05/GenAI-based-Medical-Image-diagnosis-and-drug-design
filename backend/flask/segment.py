
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

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
BRAIN_MODEL_PATH = "D:\Gen_AI_Drug_Discovery-main\server\models\model.keras"
try:
    print("Loading brain tumor model...")
    brain_model = tf.keras.models.load_model(BRAIN_MODEL_PATH, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})
    print("Brain tumor model loaded successfully.")
except Exception as e:
    print(f"Error loading brain tumor model: {e}")
    exit(1)

# Load the lung tumor model
LUNG_MODEL_PATH = "D:\Gen_AI_Drug_Discovery-main\server\models\model (4).keras"
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
if __name__ == "__main__":
    # Run Flask in production mode (disable debug mode)
    app.run(host="127.0.0.1", port=5001, debug=False)