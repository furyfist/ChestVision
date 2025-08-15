// frontend/src/App.tsx

import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; // We'll create this file for styling

function App() {
  // State for the selected file and its preview URL
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  // State for the prediction result, loading status, and errors
  const [prediction, setPrediction] = useState<{ prediction: string } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Function to handle file selection from the input
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      // Reset previous results when a new file is selected
      setPrediction(null);
      setError(null);
    }
  };

  // Function to handle the upload and prediction request
  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Send the file to the Node.js backend endpoint
      const response = await axios.post('http://localhost:5000/api/upload', formData);
      setPrediction(response.data);
    } catch (err) {
      setError('An error occurred. Please try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Chest Vision</h1>
        <p>Upload a Chest CT scan image to classify the lung condition</p>
        <div className="card">
          <input
            id="fileInput"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
           
          />
          <br/>
          {preview && (
            <div className="preview-container">
              <img src={preview} alt="Selected Preview" className="preview-image" />
            </div>
          )}
          <button onClick={handleUpload} disabled={!selectedFile || isLoading}>
            {isLoading ? 'Classifying...' : 'Classify Image'}
          </button>
          
          {/* --- Display Results --- */}
          <div className="result-container">
            {error && <p className="error-message">{error}</p>}
            {prediction && (
              <p className="prediction-result">
                Prediction: <strong>{prediction.prediction}</strong>
              </p>
            )}
          </div>
        </div>
      </header>
    </div>
  );
}

export default App;
