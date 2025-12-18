// frontend/src/App.tsx

import React, { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<{ prediction: string } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file: File) => {
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setPrediction(null);
    setError(null);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      processFile(file);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://localhost:5000/api/upload', formData);
      setPrediction(response.data);
    } catch (err: any) {
      setError('Analysis failed. Please try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="background-glow"></div>
      
      <header className="navbar">
        <div className="logo">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
          </svg>
          <span>ChestVision</span>
        </div>
      </header>

      <main className="main-content">
        <div className="hero-section">
          <h1>AI-Powered Chest CT Analysis</h1>
          <p>Advanced deep learning algorithms to detect and classify lung conditions with high accuracy.</p>
        </div>

        <div className="upload-card">
          <div 
            className={`drop-zone ${preview ? 'has-image' : ''}`}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={triggerFileInput}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              hidden
            />
            
            {preview ? (
              <div className="image-preview-wrapper">
                 <img src={preview} alt="Scan Preview" className="preview-image" />
                 <div className="change-overlay">
                   <span>Click to Change</span>
                 </div>
              </div>
            ) : (
              <div className="upload-placeholder">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                <span className="upload-text">Drag & drop your scan here</span>
                <span className="upload-subtext">or click to browse</span>
              </div>
            )}
          </div>

          <div className="actions">
            <button 
              className={`analyze-btn ${isLoading ? 'loading' : ''}`} 
              onClick={handleUpload} 
              disabled={!selectedFile || isLoading}
            >
              {isLoading ? (
                <>
                  <span className="spinner"></span> Processing...
                </>
              ) : (
                <>Analyze Scan</>
              )}
            </button>
          </div>

          {(error || prediction) && (
            <div className={`result-panel ${error ? 'error' : 'success'}`}>
              {error ? (
                <div className="result-content error-content">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                  <span>{error}</span>
                </div>
              ) : (
                <div className="result-content success-content">
                  <span className="label">Diagnosis Prediction</span>
                  <div className="prediction-value">{prediction?.prediction}</div>
                  <div className="confidence-indicator">
                    <div className="confidence-bar"></div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
      
      <footer className="footer">
        <p>&copy; 2025 ChestVision Medical AI. For research purpose only.</p>
      </footer>
    </div>
  );
}

export default App;
