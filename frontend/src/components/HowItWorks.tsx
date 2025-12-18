import React from 'react';

const HowItWorks: React.FC = () => {
    return (
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
                            <p><strong>Dataset:</strong> Hugging Face <a href="https://huggingface.co/datasets/dorsar/lung-cancer" target="_blank" rel="noopener noreferrer" className="inline-link"><code>lung-cancer</code></a> dataset</p>
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
                            <p><strong>Architecture:</strong> ResNet50 (transfer learning)</p>
                            <p><strong>Modified:</strong> Final FC layer → 4 classes</p>
                            <p><strong>Training:</strong> CrossEntropy loss, Adam optimizer, 10 epochs</p>
                        </div>
                        <div className="stage-file">train.py → <a href="https://github.com/furyfist/ChestVision/blob/main/ai-service/models/lung_cancer_classifier.pth" target="_blank" rel="noopener noreferrer" className="inline-link">lung_cancer_classifier.pth</a></div>
                    </div>

                    {/* Stage 3: Prediction Service */}
                    <div className="pipeline-stage animate-slide-up" style={{ animationDelay: '0.3s' }}>
                        <div className="stage-header">
                            <span className="stage-number">3</span>
                            <h3>Prediction API</h3>
                        </div>
                        <div className="stage-content">
                            <p><strong>Load:</strong> ResNet50 + saved weights</p>
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
    );
};

export default HowItWorks;
