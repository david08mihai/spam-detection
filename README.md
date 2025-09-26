# SMS Spam Detection 📱✉️

This project focuses on building a machine learning system that classifies SMS messages into two categories:  
- **Ham** (legitimate messages)  
- **Spam** (undisered messages)

---

## 📂 Data Understanding & Preprocessing

- The dataset contains thousands of SMS messages labeled as **Ham** or **Spam**.  
- The distribution is **imbalanced** (Ham >> Spam)( [Ham / Spam ] = 6, which means that the ham messages are 6 times more than spam ones.  
- To make the data usable for machine learning, the text was **cleaned and transformed** into numerical features using **TF-IDF Vectorization**. 
- Words were lowercased, special characters were removed, and stopwords were filtered out.  

---

## ⚙️ Machine Learning Pipeline

To ensure reproducibility and a clean workflow, a **Pipeline** was used that combines:  
1. **TF-IDF Vectorizer** → transforms raw text into numerical features.  
2. **Model (Naive Bayes or XGBoost)** → learns to classify the messages.  

This way, preprocessing and model training are linked together and can be reused consistently.

---

## 🔍 Model Selection with GridSearchCV

- Hyperparameter tuning was done using **GridSearchCV**.  
- For each model (Naive Bayes and XGBoost), a grid of parameters was tested.  
- GridSearch ensures that the best parameters are selected based on cross-validation performance, avoiding overfitting.  

---

## 📊 Results & Interpretation

### Naive Bayes (NB)
- **Precision (Spam):** 1.0 ✅  
- **Recall (Spam):** slightly lower than XGBoost (27 Spam messages were missed).  
- **Confusion Matrix:**  
  - No Ham misclassified as Spam (0 false positives).  
  - Safe model, user never loses important messages.  

### XGBoost (XGB)
- **Precision (Spam):** slightly lower than 1.0 (introduced 8 false positives).  
- **Recall (Spam):** higher than NB (detected more Spam, only 23 missed).  
- **Confusion Matrix:**  
  - Caught more Spam, but at the cost of flagging some Ham messages as Spam.  

---

## ✅ Final Choice

For this project, **Naive Bayes** was chosen because:  
- It achieved **perfect precision (1.0)** on Spam.  
- Even though XGBoost had better recall, NB ensures **no false alarms**.  
- In practice, it is more important not to lose legitimate messages than to catch every single Spam.  

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sms-spam-detector.git
   cd sms-spam-detector
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Open Jupyter Notebook:
   ```bash
   jupyter notebook spam-detection.ipynb
4. Run the UI interface:
   ```bash
   streamlit run app.py
5. Try your own SMS and see the results.
   
