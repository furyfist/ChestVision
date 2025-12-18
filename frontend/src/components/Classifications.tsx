import React from 'react';

const Classifications: React.FC = () => {
    return (
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
    );
};

export default Classifications;
