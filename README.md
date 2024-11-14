# Diabetes Prediction with MLP Classifier

**Description:**
This project uses machine learning techniques to predict diabetes outcomes based on medical data. It employs a multi-layer perceptron (MLP) classifier for analysis, focusing on data preprocessing, correlation visualization, and model performance evaluation.

---

## Objectives

1. **Data Preprocessing:**
   - Clean and preprocess the dataset by removing null values and unnecessary columns.
   - Normalize features for better model performance.

2. **Correlation Analysis:**
   - Visualize feature correlations using a heatmap to identify key predictors.

3. **MLP Model Implementation:**
   - Train and evaluate MLP classifiers with varying hidden layer configurations.
   - Analyze model errors and performance metrics.

---

## Features

1. **MLP Classifier Implementation:**
   - Build and train multi-layer perceptron models with customized hidden layers.
   - Evaluate performance using error percentage metrics.

2. **Data Visualization:**
   - Correlation heatmap for feature analysis.
   - Visual representation of prediction accuracy.

3. **Error Metrics:**
   - Calculate and display error rates for model predictions.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Diabetes-MLP-Analysis.git
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

---

## Usage

1. Import necessary libraries and load the dataset:
   ```python
   import pandas as pd
   import seaborn as sns
   from sklearn.neural_network import MLPClassifier

   df = pd.read_csv('diabetes-dataset.csv').drop(columns='Pregnancies', axis=1).dropna()
   ```
2. Visualize feature correlations:
   ```python
   sns.heatmap(df.corr(), annot=True, cmap='viridis')
   ```
3. Train an MLP classifier:
   ```python
   mlp = MLPClassifier(solver='sgd', hidden_layer_sizes=(20, 20), max_iter=1000)
   mlp.fit(x_train, y_train)
   ```
4. Evaluate the model:
   ```python
   error = error_percent(y_test, mlp.predict(x_test))
   print("Error Percentage:", error)
   ```

---

## Results

- Visualized feature correlations to identify important predictors.
- Trained MLP classifiers with varying configurations.
- Evaluated prediction accuracy and error rates.

---

## Author

- **Name:** Olha Nemkovych
- **Group:** FI-94
