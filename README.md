# Answer Sequence Analysis for Online Examinations

> **Imperial College London Individual Project**  
> Exploring how learning analytics and educational data mining (EDM) can enhance online assessments through answer submission sequence analysis.

## 📌 Overview

This repository contains the implementation and findings of a comprehensive study on online examination analytics. The project focuses on analyzing student answer submission sequences to support invigilation dashboards, improve post-exam feedback, and explore automated marking capabilities.

### Key Challenges Addressed

- **Exam Integrity**: Maintaining assessment security under remote conditions
- **Workload Reduction**: Minimizing instructor burden in invigilation and marking
- **Pedagogical Insights**: Extracting meaningful learning analytics from exam data

## 🎯 Objectives

- [x] Model and cluster student answer sequences to identify behavioral patterns
- [x] Develop actionable visualizations and analytics for instructors
- [x] Explore automated marking approaches for multiple-choice and short-answer questions
- [x] Design real-time analytics for invigilation support

## 🏗️ System Architecture

### Platform: AnswerBook Online Examination System

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   Database      │
│                 │    │                 │    │                 │
│ • Bootstrap     │◄──►│ • Flask         │◄──►│ • PostgreSQL    │
│ • JavaScript    │    │ • Flask-REST    │    │ • Event Logs    │
│ • Server-side   │    │ • Gunicorn      │    │ • User Data     │
│   Rendering     │    │ • Docker        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Pipeline Features

- **Event-driven Architecture**: Captures every student submission in real-time
- **Incremental Saving**: Prevents data loss during exam interruptions
- **State Restoration**: Supports recovery of partially completed answers
- **Scalable Storage**: Handles large-scale concurrent examinations

## 🔬 Methodology

### 1. Answer Sequence Analysis

Two complementary approaches were developed and compared:

#### Vector-based Clustering
- **Strengths**: Simple, interpretable, computationally efficient
- **Limitations**: Limited capture of temporal dynamics and sequential patterns

#### Graph Embedding + LSTM Sequential Modeling
- **Advantages**: Captures complex temporal relationships
- **Results**: More coherent, pedagogically interpretable clusters
- **Identified Patterns**:
  - Sequential completion strategies
  - Backtracking behaviors
  - Systematic review patterns
  - Time-based submission patterns

### 2. Automated Marking Engine

#### Multiple Choice Questions (MCQ)
- Rubric-based comparison algorithms
- Partial credit assignment capabilities
- Confidence scoring for answers

#### Short Answer Questions
- NLP-based semantic analysis
- Transformer model exploration
- Similarity matching with reference answers

## 📊 Key Findings

### Behavioral Clustering Results
- Successfully identified distinct exam-taking strategies
- Clusters showed pedagogical relevance for instructor feedback
- Visualizations revealed insights invisible in raw log data

### Automated Marking Performance
- MCQ marking: High accuracy with partial credit support
- Short-answer grading: Promising results requiring further validation
- Identified need for fairness and bias mitigation

### Pedagogical Applications
- **Integrity Monitoring**: Detection of anomalous behavioral patterns
- **Learning Insights**: Understanding common student strategies
- **Feedback Enhancement**: Data-driven post-exam discussions

## 🚀 Implementation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/answer-sequence-analysis.git
cd answer-sequence-analysis

# Install dependencies
pip install -r requirements.txt

# Run data analysis pipeline
python src/analysis/main.py

# Generate visualizations
python src/visualization/generate_plots.py
```

### Project Structure

```
├── src/
│   ├── models/           # LSTM, clustering, embedding models
│   ├── utils/            # Data processing utilities
│   ├── visualization/    # Plot generation and dashboard components
│   └── analysis/         # Main analysis scripts
├── data/                 # Sample datasets and processed results
├── notebooks/            # Jupyter notebooks for exploration
├── visualization/        # Generated plots and HTML reports
└── docs/                 # Documentation and reports
```

## 📈 Results & Visualizations

### Student Behavior Clustering
![Clustering Results](visualization/clusters/cluster_overview.png)

### Temporal Sequence Analysis
![Sequence Analysis](visualization/sequences/temporal_patterns.png)

### Automated Marking Performance
![Marking Results](visualization/marking/performance_metrics.png)

## 🔮 Future Directions

### Short-term Goals
- [ ] Real-time invigilation dashboard integration
- [ ] Expanded dataset validation across different subjects
- [ ] Enhanced transformer-based NLP for short answers

### Long-term Vision
- [ ] Human-in-the-loop cluster refinement systems
- [ ] Advanced student feedback visualization tools
- [ ] Cross-institutional deployment and validation
- [ ] Integration with learning management systems

## 📚 Publications & Documentation

- [Technical Report](docs/technical_report.pdf)
- [Project Presentation](docs/presentation.pdf)
- [API Documentation](docs/api_docs.md)

## 🤝 Contributing

This project is part of an academic thesis. For questions or collaboration inquiries:

- **Author**: [Your Name]
- **Institution**: Imperial College London
- **Supervisor**: [Supervisor Name]
- **Email**: [your.email@imperial.ac.uk]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Imperial College London Department of Computing
- AnswerBook development team
- Academic supervisors and peers who provided valuable feedback

---

*This project demonstrates the potential of educational data mining in transforming online assessment practices while maintaining academic integrity and enhancing learning outcomes.*