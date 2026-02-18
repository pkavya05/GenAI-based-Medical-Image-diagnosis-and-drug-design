import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Image as ImageIcon, X ,Activity} from 'lucide-react';
import './ImageUploader.css'; // Import the custom CSS file

function ImageUploader({ onUpload, accept = '.npy', multiple = false }) {
  const onDrop = useCallback(
    (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        if (multiple) {
          onUpload(new FileListWrapper(acceptedFiles));
        } else {
          onUpload(acceptedFiles[0]);
        }
      }
    },
    [onUpload, multiple]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.dicom'],
      'application/octet-stream': ['.npy'],
      'video/*': ['.mp4', '.avi', '.mov'], // Added video support
    },
    multiple,
    maxSize: 10485760, // 10MB
  });

  return (
    <div
      {...getRootProps()}
      className={`image-uploader-container ${isDragActive ? 'drag-active' : ''} ${
        isDragReject ? 'drag-reject' : ''
      }`}
    >
      <input {...getInputProps()} />
      <div className="image-uploader-content">
        {isDragReject ? (
          <>
            <X className="icon reject-icon" />
            <p className="error-message">Invalid file type or size</p>
          </>
        ) : isDragActive ? (
          <>
            <Upload className="icon upload-icon" />
            <p className="drag-message">Drop your files here</p>
          </>
        ) : (
          <>
            <ImageIcon className="icon image-icon" />
            <div>
              <p className="drag-message">Drag and drop your files here</p>
              <p className="browse-message">
                or <span className="browse-link">click to browse</span>
              </p>
            </div>
          </>
        )}
        <div className="file-info">
          <p>Supported formats: JPEG, PNG, DICOM, NPY, MP4, AVI, MOV</p>
          <p>Maximum file size: 10MB</p>
        </div>
      </div>
    </div>
  );
}

// Helper class to simulate FileList
class FileListWrapper {
  constructor(files) {
    this.files = files;
  }

  get length() {
    return this.files.length;
  }

  item(index) {
    return this.files[index] || null;
  }

  *[Symbol.iterator]() {
    for (const file of this.files) {
      yield file;
    }
  }

  entries() {
    return this.files.entries();
  }

  keys() {
    return this.files.keys();
  }

  values() {
    return this.files.values();
  }
}

export default ImageUploader;