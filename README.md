# Adult Census Income Prediction
## Course: Machine Learning and Data Mining 2

### 📌 Project Overview
The objective is to build a machine learning pipeline to predict whether an individual's annual income exceeds $50,000 based on census data.

We utilize the **Adult Census Income** dataset and implement classification models following the methodologies covered in our course labs (Data Preprocessing, PCA, Decision Trees, Random Forest, and SVM).

### 📂 Dataset Structure
The dataset (`adult.csv`) consists of 15 columns with a mix of categorical and numerical data. 
* **Target Variable**: `income` (Values: `<=50K`, `>50K`)
* **Feature Count**: 14 features (Age, Workclass, Education, Marital Status, Occupation, etc.)
* **Total Rows**: ~32,000+ entries.

**⚠️ Data Quality Note**: The dataset contains missing values represented as `"?"` which must be handled during preprocessing.

### 🛠️ Tasks & Workflow 

#### 1. Data Preprocessing (Module: `preprocessing.py`)
* **Cleaning**: Identify and replace/remove rows containing `"?"` values in columns like `workclass`, `occupation`, and `native.country`.
* **Encoding**:
    * Use **Label Encoding** for ordinal variables (e.g., `education` is already somewhat ordinal, but check `education.num`).
    * Use **One-Hot Encoding** for nominal variables (e.g., `race`, `sex`, `relationship`).
* **Scaling**: Apply `StandardScaler` or `MinMaxScaler` to numerical columns (`age`, `fnlwgt`, `capital.gain`, `hours.per.week`) to normalize distributions, especially important if we test SVM/KNN.

#### 2. Exploratory Data Analysis (Module: `eda.py`)
* [cite_start]Generate a **Correlation Matrix Heatmap** to identify highly correlated features[cite: 5].
* Visualize class distribution (Check for imbalance between `<=50K` and `>50K`).
* [cite_start]**Optional**: Apply **PCA (Principal Component Analysis)** to reduce dimensionality and visualize data separation in 2D space, as practiced in Lab 1[cite: 5].

#### 3. Model Implementation (Module: `models.py`)
We will implement and compare the following classifiers:
1.  [cite_start]**Decision Tree**: As a baseline model to visualize decision rules.
2.  **Random Forest Classifier**: **(Primary Model)**. [cite_start]According to Lab 4 results, this should be our main focus as it reduces variance and over-fitting compared to single Decision Trees.
3.  [cite_start]**Support Vector Machine (SVM)**: Test with both Linear and RBF kernels to see if the data is linearly separable (referencing Lab 3 insights)[cite: 3].

#### 4. Evaluation & Performance (Module: `evaluation.py`)
Measure performance using standard classification metrics:
* **Accuracy Score**
* **Confusion Matrix**: To visualize True Positives vs. False Negatives.
* **Precision, Recall, & F1-Score**: Critical due to potential class imbalance.

### 🚀 Getting Started

1.  **Clone the repo**:
    ```bash
    git clone [https://github.com/YourUsername/Adult-Census-Income-ML.git](https://github.com/YourUsername/Adult-Census-Income-ML.git)
    ```
2.  **Install dependencies**:
    ```bash
    pip install pandas numpy seaborn matplotlib scikit-learn
    ```
3.  **Run the pipeline**:
    ```bash
    python main.py
    ```

### 📊 Hypothesis
[cite_start] We expect the **Random Forest** model to outperform the Decision Tree by averaging predictions from multiple trees (Bagging), providing better generalization on this demographic data.
