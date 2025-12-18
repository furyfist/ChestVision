import React from 'react';

const Hero: React.FC = () => {
    return (
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
    );
};

export default Hero;
