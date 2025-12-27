import React from 'react';

const techStack = [
    { name: 'React', color: '#61DAFB' },
    { name: 'TypeScript', color: '#3178C6' },
    { name: 'Python', color: '#3776AB' },
    { name: 'PyTorch', color: '#EE4C2C' },
    { name: 'Flask', color: '#000000' }
];

const TechStack: React.FC = () => {
    return (
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
    );
};

export default TechStack;
