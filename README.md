# Customer Churn Prediction using Deep Learning (ANN)

## 📌 Project Overview

This project aims to predict customer churn using a deep learning model (Artificial Neural Network - ANN). The model is trained on a dataset of customer behaviors to classify whether a customer is likely to leave or stay with a company.

## 🗂 Dataset

- The dataset contains customer demographic and account-related features.
- Key features include **Geography, Gender, Age, Balance, Tenure, Credit Score, and Active Status**.
- The target variable is **Exited (1 = Churned, 0 = Retained)**.

## 🔧 Data Preprocessing

1. **Handled Missing Values**: Checked and imputed missing data if needed.
2. **Feature Engineering**:
   - Removed unnecessary columns (`RowNumber`, `CustomerId`, `Surname`).
   - Encoded categorical variables (`Geography`, `Gender`) using One-Hot Encoding.
   - Scaled numerical features using **StandardScaler**.
3. **Handled Class Imbalance**:
   - Used **Random Over-Sampling (ROS)** to balance churned vs. non-churned customers.

## 🏗 Model Architecture

- **Input Layer**: 128 neurons, ReLU activation, L2 Regularization (`l2=0.002`)
- **Hidden Layer 1**: 128 neurons, ReLU activation, Dropout = 0.3
- **Hidden Layer 2**: 64 neurons, ReLU activation, Dropout = 0.4
- **Output Layer**: 1 neuron, Sigmoid activation (Binary Classification)
- **Loss Function**: `binary_crossentropy`
- **Optimizer**: `Adam` (learning rate = 0.0003)

## 🚀 Training Process

- **Batch Size**: 32
- **Epochs**: Used Early Stopping to prevent overfitting.
- **Validation Split**: 20% of training data
- **Performance Metrics**:
  - Accuracy
  - Precision, Recall, F1-Score
  - ROC-AUC Score

## 📊 Model Performance

| Metric              | Score   |
| ------------------- | ------- |
| Training Accuracy   | \~92.5% |
| Validation Accuracy | \~85.3% |
| ROC-AUC Score       | \~0.87  |

## 🔥 Key Improvements

- ✅ Implemented **Dropout Regularization** to prevent overfitting. 
- ✅ Fine-tuned **L2 Regularization (0.002)** to stabilize model weights. 
- ✅ Applied **Early Stopping** to capture the best performing model. 
- ✅ Tuned **learning rate (0.0003)** for stable convergence.

## 🛠  Dependencies
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Keras


## 📌 How to Run

### 1️⃣ Clone the Repository

```bash
git clone <repo-url>
cd customer-churn-prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model

```bash
python script.ipynb
```

---

💡 **Contributions Welcome!** Feel free to fork the repo and improve the model. 🚀
