import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import {
  Navbar,
  Hero,
  Features,
  TechStack,
  HowItWorks,
  Classifications,
  UploadSection,
  Footer
} from './components';

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<{ prediction: string } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Map technical prediction names to user-friendly labels
  const formatPrediction = (rawPrediction: string): string => {
    const predictionMap: Record<string, string> = {
      'normal': 'Normal (Healthy)',
      'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa': 'Squamous Cell Carcinoma',
      'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa': 'Large Cell Carcinoma',
      'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib': 'Adenocarcinoma'
    };
    return predictionMap[rawPrediction] || rawPrediction;
  };

  const handleFileChange = (file: File) => {
    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setPrediction(null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://localhost:8000/predict', formData);
      setPrediction(response.data);
    } catch (err: any) {
      setError('Analysis failed. Please check if all services are running.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <Navbar />
      <Hero />
      <Features />
      <TechStack />
      <HowItWorks />
      <Classifications />
      <UploadSection
        selectedFile={selectedFile}
        preview={preview}
        isLoading={isLoading}
        error={error}
        prediction={prediction}
        onFileChange={handleFileChange}
        onUpload={handleUpload}
        formatPrediction={formatPrediction}
      />
      <Footer />
    </div>
  );
}

export default App;
