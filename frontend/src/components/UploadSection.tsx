import React, { useRef } from 'react';

interface UploadSectionProps {
    selectedFile: File | null;
    preview: string | null;
    isLoading: boolean;
    error: string | null;
    prediction: {
        prediction: string | null;
        confidence?: number;
        invalid_image?: boolean;
        message?: string;
    } | null;
    onFileChange: (file: File) => void;
    onUpload: () => void;
    formatPrediction: (rawPrediction: string) => string;
}

const sampleImages = [
    { src: '/samples/test1.png', label: 'Sample 1' },
    { src: '/samples/test2.png', label: 'Sample 2' },
    { src: '/samples/test3.png', label: 'Sample 3' },
    { src: '/samples/test4.png', label: 'Sample 4' },
    { src: '/samples/test5.png', label: 'Sample 5' },
    { src: '/samples/test6.png', label: 'Sample 6' },
    { src: '/samples/test7.png', label: 'Sample 7' },
    { src: '/samples/test8.png', label: 'Sample 8' },
    { src: '/samples/test9.png', label: 'Sample 9' },
    { src: '/samples/test10.png', label: 'Sample 10' }
];

const UploadSection: React.FC<UploadSectionProps> = ({
    selectedFile,
    preview,
    isLoading,
    error,
    prediction,
    onFileChange,
    onUpload,
    formatPrediction
}) => {
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            onFileChange(file);
        }
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
            onFileChange(file);
        }
    };

    const triggerFileInput = () => {
        fileInputRef.current?.click();
    };

    const loadSampleImage = async (imageSrc: string) => {
        try {
            const response = await fetch(imageSrc);
            const blob = await response.blob();
            const file = new File([blob], imageSrc.split('/').pop() || 'sample.png', { type: blob.type });
            onFileChange(file);
        } catch (err) {
            console.error('Failed to load sample image:', err);
        }
    };

    const loadRandomSample = () => {
        const randomIndex = Math.floor(Math.random() * sampleImages.length);
        loadSampleImage(sampleImages[randomIndex].src);
    };

    return (
        <section id="upload" className="upload-section">
            <div className="section-container">
                <h2 className="section-title">Classify Your Scan</h2>
                <p className="section-subtitle">Upload a chest CT scan image or try a sample below</p>

                {/* Sample Images */}
                <div className="sample-section">
                    <div className="sample-header">
                        <span className="sample-label">Try a sample scan:</span>
                        <button className="random-btn" onClick={loadRandomSample}>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <polyline points="16 3 21 3 21 8" />
                                <line x1="4" y1="20" x2="21" y2="3" />
                                <polyline points="21 16 21 21 16 21" />
                                <line x1="15" y1="15" x2="21" y2="21" />
                                <line x1="4" y1="4" x2="9" y2="9" />
                            </svg>
                            Random
                        </button>
                    </div>
                    <div className="sample-images">
                        {sampleImages.map((sample, index) => (
                            <button
                                key={index}
                                className="sample-btn"
                                onClick={() => loadSampleImage(sample.src)}
                                title={sample.label}
                            >
                                <img src={sample.src} alt={sample.label} />
                                <span className="sample-number">{index + 1}</span>
                            </button>
                        ))}
                    </div>
                </div>

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
                        onClick={onUpload}
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

                {/* Disclaimer */}
                <p className="upload-disclaimer">
                    ⚠️ Results are only valid for chest CT scan images. Uploading other images may produce incorrect predictions.
                </p>
            </div>
        </section>
    );
};

export default UploadSection;
