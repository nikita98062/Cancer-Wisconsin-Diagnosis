# Breast Cancer Wisconsin Diagnosis Project

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset Information](#dataset-information)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project implements machine learning models to diagnose breast cancer using the Wisconsin Breast Cancer dataset. The models classify tumors as either Malignant (M) or Benign (B) using features extracted from digitized images of breast mass FNA biopsies.

## Features

### Data Processing
- Automated data cleaning and preprocessing
- Feature selection and dimensionality reduction
- Handling of missing values
- Feature scaling using StandardScaler
- Train-test split (70-30 ratio)

### Machine Learning Models
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes Classifier

### Analysis Tools
- Cross-validation scoring
- Model performance comparison
- Feature importance analysis
- Confusion matrix visualization
- ROC curve analysis

### Visualizations
- Correlation heatmaps
- Pair plots for feature relationships
- Distribution plots
- Model performance comparisons
- Feature importance charts

## Dataset Information
The dataset includes 569 samples with 30 features including:
- Radius (mean, SE, worst)
- Texture (mean, SE, worst)
- Perimeter (mean, SE, worst)
- Area (mean, SE, worst)
- Smoothness (mean, SE, worst)
- Compactness (mean, SE, worst)
- Concavity (mean, SE, worst)
- Symmetry (mean, SE, worst)
- Fractal dimension (mean, SE, worst)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Cancer-Wisconsin-Diagnosis.git

# Navigate to project directory
cd Cancer-Wisconsin-Diagnosis

# Install required packages
pip install -r requirements.txt
```

## Requirements

```python
pandas>=1.2.0
numpy>=1.19.2
matplotlib>=3.3.2
seaborn>=0.11.0
scikit-learn>=0.24.0
jupyter>=1.0.0
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook "cancer project (1).py"
```

2. Run all cells to:
   - Load and preprocess data
   - Train models
   - Generate visualizations
   - View performance metrics

## Model Performance

Current model accuracies:
- Logistic Regression: 95.3%
- Random Forest: 96.5%
- SVM: 97.1%
- KNN: 95.9%
- Decision Tree: 93.6%

## Results

### Classification Metrics
- Average Accuracy: 96.5%
- Precision: 0.97
- Recall: 0.96
- F1-Score: 0.965

### Key Findings
- SVM performs best with 97.1% accuracy
- Low false positive rate across all models
- High feature correlation between radius and area

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
Created by Your Name
Last Updated: March 2024