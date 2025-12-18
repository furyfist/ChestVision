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
      setError('Analysis failed. Please check if all services are running.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const features = [
    {
      icon: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <polyline points="12 6 12 12 16 14" />
        </svg>
      ),
      title: 'Real-Time Analysis',
      description: 'Get instant classification results within seconds of uploading your CT scan.'
    },
    {
      icon: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
        </svg>
      ),
      title: 'Deep Learning',
      description: 'Powered by ResNet-50, a state-of-the-art convolutional neural network trained on medical imaging data.'
    },
    {
      icon: (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="2" y="3" width="20" height="14" rx="2" ry="2" />
          <line x1="8" y1="21" x2="16" y2="21" />
          <line x1="12" y1="17" x2="12" y2="21" />
        </svg>
      ),
      title: 'Microservice Architecture',
      description: 'Scalable design with separate frontend, backend, and AI services for reliability.'
    }
  ];

  const techStack = [
    { name: 'React', color: '#61DAFB' },
    { name: 'TypeScript', color: '#3178C6' },
    { name: 'Node.js', color: '#339933' },
    { name: 'Express', color: '#000000' },
    { name: 'Python', color: '#3776AB' },
    { name: 'PyTorch', color: '#EE4C2C' },
    { name: 'Flask', color: '#000000' }
  ];

  return (
    <div className="App">
      {/* Navigation */}
      <nav className="navbar">
        <div className="nav-content">
          <div className="logo">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
            </svg>
            <span>ChestVision</span>
          </div>
          <a href="https://github.com/furyfist/ChestVision" target="_blank" rel="noopener noreferrer" className="github-link">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
            </svg>
            GitHub
          </a>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero">
        <div className="hero-content animate-fade-in">
          <h1>AI-Powered Lung Condition Classifier</h1>
          <p className="hero-subtitle">
            ChestVision uses deep learning to analyze chest CT scans and classify lung conditions
            in real-time. Built with a modern microservice architecture for scalability and reliability.
          </p>
          <a href="#upload" className="cta-button">
            Try It Now
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="5" y1="12" x2="19" y2="12" />
              <polyline points="12 5 19 12 12 19" />
            </svg>
          </a>
        </div>
      </section>

      {/* Features Section */}
      <section className="features">
        <div className="section-container">
          <h2 className="section-title">Key Features</h2>
          <div className="features-grid">
            {features.map((feature, index) => (
              <div
                key={index}
                className="feature-card animate-slide-up"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className="feature-icon">{feature.icon}</div>
                <h3>{feature.title}</h3>
                <p>{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Tech Stack Section */}
      <section className="tech-stack">
        <div className="section-container">
          <h2 className="section-title">Tech Stack</h2>
          <div className="tech-badges">
            {techStack.map((tech, index) => (
              <span
                key={index}
                className="tech-badge animate-fade-in"
                style={{ animationDelay: `${index * 0.05}s` }}
              >
                <span className="tech-dot" style={{ backgroundColor: tech.color }}></span>
                {tech.name}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="how-it-works">
        <div className="section-container">
          <h2 className="section-title">How It Works</h2>
          <p className="section-subtitle">End-to-end pipeline from data collection to prediction</p>

          <div className="pipeline-stages">
            {/* Stage 1: Data Preparation */}
            <div className="pipeline-stage animate-slide-up" style={{ animationDelay: '0s' }}>
              <div className="stage-header">
                <span className="stage-number">1</span>
                <h3>Data Preparation</h3>
              </div>
              <div className="stage-content">
                <p><strong>Dataset:</strong> Hugging Face <code>lung-cancer</code> dataset</p>
                <p><strong>Preprocessing:</strong> Resize to 224×224, normalize with ImageNet stats</p>
                <p><strong>Output:</strong> PyTorch DataLoader ready for training</p>
              </div>
              <div className="stage-file">dataset.py</div>
            </div>

            {/* Stage 2: Model Training */}
            <div className="pipeline-stage animate-slide-up" style={{ animationDelay: '0.15s' }}>
              <div className="stage-header">
                <span className="stage-number">2</span>
                <h3>Model Training</h3>
              </div>
              <div className="stage-content">
                <p><strong>Architecture:</strong> ResNet18 (transfer learning)</p>
                <p><strong>Modified:</strong> Final FC layer → 4 classes</p>
                <p><strong>Training:</strong> CrossEntropy loss, Adam optimizer, 10 epochs</p>
              </div>
              <div className="stage-file">train.py → lung_cancer_classifier.pth</div>
            </div>

            {/* Stage 3: Prediction Service */}
            <div className="pipeline-stage animate-slide-up" style={{ animationDelay: '0.3s' }}>
              <div className="stage-header">
                <span className="stage-number">3</span>
                <h3>Prediction API</h3>
              </div>
              <div className="stage-content">
                <p><strong>Load:</strong> ResNet18 + saved weights</p>
                <p><strong>Process:</strong> PIL image → PyTorch transforms</p>
                <p><strong>Inference:</strong> Forward pass in eval mode</p>
              </div>
              <div className="stage-file">predict.py + app.py</div>
            </div>
          </div>

          {/* Service Architecture */}
          <h3 className="subsection-title">Service Architecture</h3>
          <div className="architecture-diagram animate-fade-in">
            <div className="arch-box">
              <span className="arch-label">React Frontend</span>
              <span className="arch-tech">:3000</span>
            </div>
            <div className="arch-arrow">→</div>
            <div className="arch-box">
              <span className="arch-label">Express Gateway</span>
              <span className="arch-tech">:5000</span>
            </div>
            <div className="arch-arrow">→</div>
            <div className="arch-box">
              <span className="arch-label">Flask AI Service</span>
              <span className="arch-tech">:8000</span>
            </div>
            <div className="arch-arrow">→</div>
            <div className="arch-box highlight">
              <span className="arch-label">Classification</span>
              <span className="arch-tech">JSON Response</span>
            </div>
          </div>
        </div>
      </section>

      {/* Classifications Section */}
      <section className="classifications">
        <div className="section-container">
          <h2 className="section-title">What We Detect</h2>
          <p className="section-subtitle">Our AI model classifies chest CT scans into 4 categories</p>
          <div className="classifications-grid">
            <div className="classification-card normal animate-slide-up" style={{ animationDelay: '0s' }}>
              <div className="classification-icon success">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                  <polyline points="22 4 12 14.01 9 11.01" />
                </svg>
              </div>
              <h3>Normal</h3>
              <p>Healthy lung tissue with no signs of malignancy</p>
            </div>
            <div className="classification-card animate-slide-up" style={{ animationDelay: '0.1s' }}>
              <div className="classification-icon warning">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
              </div>
              <h3>Adenocarcinoma</h3>
              <p>Most common type, starting in mucus-producing cells</p>
            </div>
            <div className="classification-card animate-slide-up" style={{ animationDelay: '0.2s' }}>
              <div className="classification-icon warning">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
              </div>
              <h3>Squamous Cell Carcinoma</h3>
              <p>Cancer arising from flat cells lining the airways</p>
            </div>
            <div className="classification-card animate-slide-up" style={{ animationDelay: '0.3s' }}>
              <div className="classification-icon warning">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
              </div>
              <h3>Large Cell Carcinoma</h3>
              <p>Fast-growing cancer in any part of the lung</p>
            </div>
          </div>
        </div>
      </section>

      {/* Upload Section */}
      <section id="upload" className="upload-section">
        <div className="section-container">
          <h2 className="section-title">Classify Your Scan</h2>
          <p className="section-subtitle">Upload a chest CT scan image to get a classification prediction</p>

          <div className="upload-card animate-slide-up">
            <div
              className={`drop-zone ${preview ? 'has-image' : ''} ${isLoading ? 'loading' : ''}`}
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
                    <span>Click to change</span>
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

            <button
              className={`analyze-btn ${isLoading ? 'loading' : ''}`}
              onClick={handleUpload}
              disabled={!selectedFile || isLoading}
            >
              {isLoading ? (
                <>
                  <span className="spinner"></span>
                  Analyzing...
                </>
              ) : (
                'Analyze Scan'
              )}
            </button>

            {(error || prediction) && (
              <div className={`result-panel ${error ? 'error' : 'success'}`}>
                {error ? (
                  <div className="result-content error-content">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="12" cy="12" r="10" />
                      <line x1="12" y1="8" x2="12" y2="12" />
                      <line x1="12" y1="16" x2="12.01" y2="16" />
                    </svg>
                    <span>{error}</span>
                  </div>
                ) : (
                  <div className="result-content success-content">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                      <polyline points="22 4 12 14.01 9 11.01" />
                    </svg>
                    <div>
                      <span className="result-label">Prediction</span>
                      <span className="prediction-value">{formatPrediction(prediction?.prediction || '')}</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <p>© 2025 ChestVision. For research and educational purposes only.</p>
          <p className="disclaimer">This tool is not intended for medical diagnosis. Always consult a healthcare professional.</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
