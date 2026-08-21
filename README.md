# Electricity-Theft-Detection
A Data-Driven Framework for Non-Technical Loss Detection in Smart Distribution Networks
## Project Overview

Electricity theft is a major source of non-technical losses (NTLs) for power distribution companies, leading to financial losses, grid instability, and increased operational costs. Traditional detection methods based on manual inspections and fixed rules are difficult to scale and may fail to identify complex theft patterns.

This project develops a machine learning-based electricity theft detection system that analyzes historical smart-meter consumption patterns to classify consumers as **Normal** or **Potentially Fraudulent**.

### Objectives

- Analyze consumer electricity consumption behavior
- Extract meaningful temporal and statistical features
- Handle missing data and class imbalance
- Train and compare multiple machine learning models
- Optimize classification thresholds and model hyperparameters
- Interpret model predictions using SHAP and feature importance
- Deploy the final theft detection model

## Problem Statement

Detecting electricity theft is challenging because fraudulent consumption patterns can vary significantly and are often hidden within a large volume of normal customer records. Manual inspection is costly and inefficient, while rule-based systems may not generalize well to complex theft behavior.

The goal is to build an automated ML system capable of identifying suspicious electricity consumption patterns and prioritizing potentially fraudulent customers for further investigation.

## Electricity Theft

Electricity theft refers to the unauthorized consumption of electricity or manipulation of the metering system to avoid paying for actual consumption. Common forms include:

- **Meter Bypass** – Directly consuming electricity without passing through the meter
- **Meter Tampering** – Manipulating the meter to under-record consumption
- **Illegal Connections** – Unauthorized connections to distribution lines
- **Billing Fraud** – Manipulation of meter readings or billing information

Electricity theft contributes significantly to **Non-Technical Losses (NTLs)** and causes financial and operational challenges for utilities.

## Smart Meters and Smart Grids

Smart meters provide detailed, time-based electricity consumption data and enable automated communication between consumers and utility companies. Unlike traditional meters, they support remote monitoring and provide richer information for detecting abnormal consumption behavior.

Typical smart-meter information includes:

- Electricity consumption
- Voltage and current
- Power factor
- Date and time of consumption
- Peak and off-peak usage
- Meter status and tamper events

### Advanced Metering Infrastructure (AMI)

AMI connects smart meters with utility systems through communication networks, enabling:

- Automatic meter-data collection
- Remote monitoring
- Two-way communication
- Electricity theft and tamper detection
- Improved energy management and grid efficiency

## Project Workflow

1. Business Understanding
2. Dataset Overview
3. Exploratory Data Analysis
4. Data Preprocessing
5. Feature Engineering
6. Feature Selection
7. Baseline Model — Logistic Regression
8. Advanced ML Models & Evaluation
9. Hyperparameter Tuning & Threshold Optimization
10. Final Model Comparison
11. Model Interpretation — SHAP & Feature Importance
12. Conclusion
13. Future Work

## Exploratory Data Analysis

EDA was performed to understand electricity consumption behavior, data quality, and class distribution.

### Key Findings

- Significant missing values were present in daily consumption records.
- The target variable (`FLAG`) was highly imbalanced, with theft cases representing the minority class.
- Electricity consumption showed positive skewness and significant variability across consumers.
- Extreme consumption values were observed and retained carefully because they may represent genuine behavior.
- Temporal patterns such as sudden drops, spikes, and prolonged low/zero consumption were identified.
- Normal and theft consumers showed overlapping consumption distributions, indicating that individual consumption values alone are insufficient for reliable classification.

These findings motivated the use of time-series-based preprocessing and behavioral feature engineering.

## Data Preprocessing

The raw smart-meter data was preprocessed before model development.

### Missing Value Handling

- Missing consumption readings were handled using **time-series interpolation**.
- Linear interpolation was used to estimate missing values from neighboring observations.
- Remaining missing values were handled using forward and backward filling.

### Zero Consumption

Zero-consumption values were retained because they may represent genuine behavior or abnormal usage. Instead of removing them, zero-consumption patterns were converted into behavioral features.

### Outlier Handling

Extreme consumption values were analyzed carefully rather than removed blindly. Quantile-based clipping was used where appropriate to reduce the influence of extreme observations while preserving useful behavioral information.

## Feature Engineering

Since electricity theft cannot reliably be identified from a single consumption value, **37 consumer-level behavioral features** were engineered from the daily smart-meter readings.

### Feature Categories

| Category | Features | Purpose |
|---|---:|---|
| Statistical | 12 | Capture overall consumption behavior |
| Zero Consumption | 4 | Capture zero-usage patterns |
| Trend & Change | 8 | Detect sudden drops, jumps, and trends |
| Consumption Behavior | 4 | Capture high/low and seasonal usage |
| Distribution & Stability | 3 | Measure variability and distribution |
| Missing Value Behavior | 5 | Capture missing-data patterns |

### Key Engineered Features

Examples include:

- Mean, median, minimum and maximum consumption
- Standard deviation and coefficient of variation
- Consumption range, IQR and MAD
- Zero-consumption percentage and longest zero streak
- Average daily change and overall consumption trend
- Sudden drop/jump counts and maximum changes
- High- and low-consumption days
- Peak-to-average consumption ratio
- Seasonal consumption statistics
- Skewness and kurtosis
- Missing-value count and percentage

These features transform raw daily electricity readings into **consumer-level behavioral profiles**, providing richer information for machine learning-based electricity theft detection.

## Model Development

### Baseline Models

Logistic Regression and Decision Tree were initially trained as baseline models. Because the dataset is highly imbalanced, models were evaluated both with and without **SMOTE (Synthetic Minority Oversampling Technique)**.

A **stratified train-test split** was used to preserve the class distribution between training and testing data.

The baseline experiments showed that models trained on the original imbalanced data could achieve high accuracy while performing poorly on the minority theft class. SMOTE substantially improved theft detection recall.

### Advanced Models

The following ensemble models were evaluated using SMOTE:

- Random Forest
- Gradient Boosting
- XGBoost

These models were selected for their ability to capture non-linear relationships and interactions within structured smart-meter data.

### Model Evaluation

Because electricity theft is a highly imbalanced classification problem, accuracy alone was not considered sufficient. The primary evaluation metrics were:

| Metric | Purpose |
|---|---|
| Precision | Measures reliability of theft predictions |
| Recall | Measures the proportion of actual theft cases detected |
| F1-score | Balances precision and recall |
| ROC-AUC | Measures overall class discrimination |
| Accuracy | Measures overall classification correctness |

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 91.37% | 68.42% | 5.45% | 10.09% | 75.51% |
| Logistic Regression + SMOTE | 70.70% | 19.22% | 71.65% | 30.31% | 77.30% |
| Decision Tree | 88.09% | 35.02% | 39.66% | 37.20% | 66.24% |
| Decision Tree + SMOTE | 82.00% | 25.25% | 52.23% | 34.05% | 68.57% |
| Random Forest | 93.23% | 84.06% | 29.47% | 43.64% | 83.02% |
| Random Forest + SMOTE | 89.11% | 41.35% | 53.77% | 46.75% | **83.73%** |
| Gradient Boosting + SMOTE | 80.20% | 26.45% | **68.85%** | 38.22% | 82.36% |
| XGBoost + SMOTE | 85.94% | 33.49% | 58.94% | 42.71% | 82.44% |

The results demonstrated that SMOTE significantly improved minority-class detection, particularly recall, while ensemble models provided stronger overall discrimination than simpler baseline models.

## Cross-Validation

To evaluate model stability, **3-fold Stratified Cross-Validation** was performed on the ensemble models.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest + SMOTE | 88.99% | 40.66% | 51.55% | **45.44%** | 82.87% |
| Gradient Boosting + SMOTE | 80.52% | 26.51% | **67.23%** | 38.02% | 82.18% |
| XGBoost + SMOTE | 86.19% | 33.90% | 58.19% | 42.83% | **82.93%** |

The similarity between test-set and cross-validation ROC-AUC scores indicated stable model performance across different data splits.

## XGBoost Optimization

XGBoost + SMOTE was selected for further optimization because of its strong discrimination capability and ability to model complex feature interactions.

### Hyperparameter Tuning

The tuned XGBoost model achieved:

| Metric | Baseline | Tuned |
|---|---:|---:|
| Accuracy | 85.94% | 83.38% |
| Precision | 33.49% | 30.19% |
| Recall | 58.94% | **66.20%** |
| F1-score | 42.71% | 41.47% |
| ROC-AUC | 82.44% | **83.26%** |

Tuning improved recall and ROC-AUC, allowing the model to identify more theft cases.

### Threshold Optimization

The default classification threshold of 0.50 was optimized using the precision-recall trade-off.

The selected threshold was:

**Optimal Threshold = 0.55**

At this threshold, consumers with predicted theft probability ≥ 0.55 are classified as potential theft cases.

### Final Model Performance

| Metric | Final Score |
|---|---:|
| Accuracy | **86.18%** |
| Precision | **34.48%** |
| Recall | **61.59%** |
| F1-score | **44.21%** |
| ROC-AUC | **83.26%** |

Compared with the baseline XGBoost + SMOTE model, the optimized model improved:

- Accuracy: **85.94% → 86.18%**
- Precision: **33.49% → 34.48%**
- Recall: **58.94% → 61.59%**
- F1-score: **42.71% → 44.21%**
- ROC-AUC: **82.44% → 83.26%**

Threshold optimization reduced false positives while maintaining strong theft detection capability.

## Final Model

**Tuned XGBoost + SMOTE (Threshold = 0.55)** was selected as the final electricity theft detection model.

The final model achieved:

- **ROC-AUC:** 83.26%
- **Recall:** 61.59%
- **Precision:** 34.48%
- **F1-score:** 44.21%
- **Accuracy:** 86.18%

The model provides a practical balance between detecting electricity theft and limiting unnecessary inspections, making it suitable for prioritizing potentially fraudulent consumers for further investigation.

## Model Interpretation

To improve transparency and understand why the XGBoost model flags consumers, both **XGBoost Feature Importance** and **SHAP (SHapley Additive exPlanations)** were used.

### Feature Importance

XGBoost feature importance was used to identify the consumption characteristics that contributed most to the model's predictions, including:

- Consumption trends and variability
- Sudden changes in electricity usage
- Zero-consumption behavior
- Statistical consumption characteristics
- Missing-data patterns

### SHAP Analysis

SHAP provided both global and individual-level explanations.

- **Global explanations:** Identify the features that have the greatest overall influence on theft predictions.
- **Local explanations:** Explain why a specific consumer was classified as potentially fraudulent.
- **Feature direction:** Show whether a feature increases or decreases the predicted probability of theft.

SHAP waterfall and summary plots were used to understand individual predictions and overall model behavior.

Explainability is particularly important in electricity theft detection because utility operators need to understand **why a consumer was flagged** before conducting a physical inspection.

## Conclusion

This project developed an end-to-end **Machine Learning-based Electricity Theft Detection System** using the SGCC smart-meter dataset.

The workflow covered:

- Exploratory Data Analysis
- Data preprocessing
- Behavioral feature engineering
- Feature selection
- Class-imbalance handling using SMOTE
- Multiple ML model development and comparison
- Cross-validation
- XGBoost hyperparameter tuning
- Classification threshold optimization
- SHAP-based model interpretation

The final **Tuned XGBoost + SMOTE** model with a classification threshold of **0.55** achieved:

| Metric | Score |
|---|---:|
| Accuracy | **86.18%** |
| Precision | **34.48%** |
| Recall | **61.59%** |
| F1-score | **44.21%** |
| ROC-AUC | **83.26%** |

The results demonstrate that combining behavioral feature engineering, imbalance handling, ensemble learning, threshold optimization, and explainable AI can effectively identify suspicious electricity consumption patterns.

The system can serve as a **decision-support tool for prioritizing potentially fraudulent consumers for further investigation**, rather than automatically declaring a consumer guilty of theft.

## Future Work

Several extensions can improve the system's accuracy, scalability, and real-world applicability:

- **Real-Time Detection:** Process streaming smart-meter data and generate real-time risk scores.
- **Deep Learning:** Explore LSTM, GRU, TCN, and Transformer-based time-series models.
- **Cost-Sensitive Learning:** Incorporate different costs for false positives and false negatives.
- **Production Deployment:** Develop an automated prediction pipeline and utility-operator dashboard.
- **Enhanced Explainability:** Generate automated explanations for individual flagged consumers.
- **External Validation:** Evaluate the model on additional smart-meter and real-world utility datasets.

## Key Takeaways

- Electricity theft detection is a **highly imbalanced classification problem**.
- Accuracy alone is not a reliable metric for this problem.
- **SMOTE improved minority-class detection**, particularly recall.
- Ensemble models captured complex consumption behavior better than simple baseline models.
- **XGBoost + SMOTE** provided strong overall discrimination.
- **Threshold optimization** improved the precision-recall trade-off.
- **SHAP** provided transparency into both global model behavior and individual predictions.

