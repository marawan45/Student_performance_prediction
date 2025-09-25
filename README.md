# Student Performance Prediction Project

A comprehensive machine learning project that predicts student grades (A-F) and total scores using various algorithms and features interactive web applications for real-time predictions.

## 🎯 Project Overview

This project implements two main prediction tasks:
1. **Grade Classification**: Predicts student grades (A, B, C, D, F) using multiple features
2. **Total Score Regression**: Predicts numerical total scores (0-100) based on study hours

## 📊 Dataset Features

- **weekly_self_study_hours**: Hours spent studying per week
- **attendance_percentage**: Class attendance percentage
- **class_participation**: Class participation score
- **grade**: Target variable for classification (A, B, C, D, F)
- **total_score**: Target variable for regression (0-100)

## 🏗️ Project Structure

```
student_performance_project/
├── student_performace_project(Data loading ,Data cleaning,EDA).py
├── student_performace_project(Classification Modeling).py
├── student_performace_project(regression Modeling).py
├── grade_prediction_app.py
├── total_score_prediction_app.py
├── prepared_data/
│   ├── X_train.csv
│   ├── y_train.csv
│   ├── X_test.csv
│   ├── y_test.csv
│   ├── X_train_reg.csv
│   ├── y_train_reg.csv
│   ├── X_test_reg.csv
│   ├── y_test_reg.csv
│   ├── standard_scaler.pkl
│   └── standard_scaler_reg.pkl
└── models/
    ├── best_logistic_regression_model.pkl
    ├── best_random_forest_model.pkl
    ├── best_gradient_boosting_model.pkl
    ├── poly2_regression_model.pkl
    ├── poly3_regression_model.pkl
    └── transformers/
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit joblib imbalanced-learn scipy
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd student_performance_project
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Update file paths in the Python scripts to match your local directory structure.
## grade prediction app
<img width="1725" height="905" alt="لقطة شاشة 2025-09-25 173939" src="https://github.com/user-attachments/assets/1cdf5f13-877e-4b73-ad54-02010b519136" />
## total score prediction app
<img width="1772" height="914" alt="لقطة شاشة 2025-09-25 160258" src="https://github.com/user-attachments/assets/72aa3e6c-3951-4238-a442-40327de208cc" />

## 📈 Data Processing Pipeline

### 1. Data Loading & Cleaning (`student_performace_project(Data loading ,Data cleaning,EDA).py`)

- **Data Loading**: Reads CSV data and performs initial exploration
- **Data Quality Checks**: 
  - Identifies missing values and duplicates
  - Removes duplicates automatically
- **Outlier Detection**: Uses IQR method for outlier detection and removal
- **Data Sampling**: Limits dataset to 150,000 rows (100k train, 50k test)
- **Feature Engineering**: 
  - Separates numerical and categorical features
  - Applies appropriate imputation strategies
  - Label encodes categorical variables
  - Standardizes features using StandardScaler

### 2. Exploratory Data Analysis (EDA)

**Classification Task Visualizations:**
- Grade distribution (bar plots and pie charts)
- Feature correlation heatmap
- Box plots showing feature distributions by grade
- Pair plots for feature relationships
- Line plots showing average feature values by grade

**Regression Task Visualizations:**
- Total score distribution (binned)
- Correlation analysis between features and target
- Scatter plots showing study hours vs. total score relationship

## 🤖 Machine Learning Models

### Classification Models (`student_performace_project(Classification Modeling).py`)

**Algorithms Implemented:**
- **Random Forest Classifier**
- **Logistic Regression** 
- **Gradient Boosting Classifier**

**Model Optimization:**
- **SMOTE**: Handles class imbalance using Synthetic Minority Oversampling
- **GridSearchCV**: Hyperparameter tuning with 3-fold cross-validation
- **Class Balancing**: Uses `class_weight="balanced"` for applicable models
- **Evaluation Metric**: F1-macro score for multi-class classification

**Hyperparameter Grids:**
- Random Forest: n_estimators, max_depth, min_samples_split, min_samples_leaf
- Logistic Regression: C, solver, max_iter
- Gradient Boosting: n_estimators, learning_rate, max_depth

### Regression Models (`student_performace_project(regression Modeling).py`)

**Algorithms Implemented:**
- **Linear Regression** (baseline)
- **Polynomial Regression** (degree 2 and 3)
- **Ridge Regression** (L2 regularization)
- **Lasso Regression** (L1 regularization)

**Evaluation Metrics:**
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

**Model Analysis:**
- Residual analysis for model diagnostics
- Comparison plots showing actual vs predicted values
- Feature transformation using PolynomialFeatures

## 🌐 Web Applications

### Grade Prediction App (`grade_prediction_app.py`)

- **Framework**: Streamlit
- **Model**: Best performing classification model (Logistic Regression)
- **Features**: 
  - Interactive input fields for all three features
  - Real-time grade prediction
  - Grade mapping reference table
  - Automatic feature scaling

**Usage:**
```bash
streamlit run grade_prediction_app.py
```

### Total Score Prediction App (`total_score_prediction_app.py`)

- **Framework**: Streamlit
- **Model**: Polynomial Regression (degree 3)
- **Features**:
  - Input validation and constraints
  - Real-time score prediction
  - Automatic feature scaling and polynomial transformation

**Usage:**
```bash
streamlit run total_score_prediction_app.py
```

## 📊 Model Performance

### Classification Results
- **Random Forest**: Balanced performance with good interpretability
- **Logistic Regression**: Fast training, good baseline performance
- **Gradient Boosting**: Often achieves highest accuracy

### Regression Results
- **Linear Regression**: Simple baseline model
- **Polynomial (degree 3)**: Best balance of complexity and performance
- **Ridge/Lasso**: Regularized versions for better generalization

## 🔧 Key Features

- **Comprehensive Data Pipeline**: From raw data to deployment-ready models
- **Class Imbalance Handling**: SMOTE implementation for classification
- **Model Persistence**: All models and transformers saved using joblib
- **Interactive Visualization**: Extensive EDA with matplotlib, seaborn, and plotly
- **Production-Ready Apps**: Streamlit applications with proper input validation
- **Modular Design**: Separate scripts for different project phases

## 📝 Usage Examples

### Running the Complete Pipeline

1. **Data Processing & EDA:**
```bash
python "student_performace_project(Data loading ,Data cleaning,EDA).py"
```

2. **Train Classification Models:**
```bash
python "student_performace_project(Classification Modeling).py"
```

3. **Train Regression Models:**
```bash
python "student_performace_project(regression Modeling).py"
```

4. **Launch Web Applications:**
```bash
# Grade prediction
streamlit run grade_prediction_app.py

# Score prediction  
streamlit run total_score_prediction_app.py
```

## 🎯 Model Inputs and Outputs

### Grade Classification
**Inputs:**
- Weekly Self Study Hours (0-100)
- Attendance Percentage (0-100)
- Class Participation (0-100)

**Output:** 
- Predicted Grade (A, B, C, D, F)

### Score Regression
**Inputs:**
- Weekly Self Study Hours (0-24.10, capped for model stability)

**Output:**
- Predicted Total Score (0-100)

## 🔍 Future Improvements

- Add more features (homework scores, exam grades, demographic data)
- Implement ensemble methods combining multiple models
- Add model interpretability features (SHAP values, feature importance)
- Implement A/B testing for model comparison
- Add data drift monitoring for production deployment
- Create REST API endpoints for model serving

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
streamlit>=1.0.0
joblib>=1.1.0
imbalanced-learn>=0.8.0
scipy>=1.7.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

-Marwan Eslam Oda - Initial work

## 🙏 Acknowledgments

- Scikit-learn documentation and community
- Streamlit team for the excellent web framework
- Educational institutions for inspiring this student performance analysis
