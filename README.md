# 🌽 Maize Oil Fatty Acid Prediction

This project predicts **10 different fatty acids** in maize (corn) using **genotypic (SNPs)** and **phenotypic** data through machine learning models such as **Random Forest** and **Deep Neural Networks (DNN)**. It includes data preprocessing, feature selection, model building, evaluation, and interactive visualization.

---

## 🔍 Problem Statement

The goal is to build predictive models that can accurately estimate the levels of **fatty acids** like palmitic acid, stearic acid, oleic acid, linoleic acid, etc., using maize genotype and phenotype data. These predictions are critical for:

- Enhancing **nutritional breeding** strategies
- Improving **oil yield and quality** in maize varieties

---

## 📁 Project Structure

Maize_Oil_Fatty_Prediction/
├── Maize_Genotypic_Data.xlsx
├── Cleaned_Phenotypic_Data.csv
├── Data_Preprocessing.ipynb
├── ML_Oil_Fatty_Acid_Model.pkl
├── Fatty_Acid_Correlation_Heatmap.png
├── Feature_Importance_Plot.png
├── Streamlit_App/ ← Streamlit web interface
└── README.md

yaml
Copy
Edit

---

## 🧪 Machine Learning Models Used

- **Random Forest Regressor** (for multi-output regression)
- **MultiOutputRegressor** wrapper for multiple traits
- **Deep Neural Network (DNN)** (optional for comparison)
- Model evaluation using **R² Score**, **MAE**, **MSE**, etc.

---

## 📈 Visualizations

- 📊 **Feature Importance Plot**: Top SNPs influencing fatty acid traits
- 🔥 **Correlation Heatmap**: Relationships among the 10 fatty acids
- 🧬 Trait distribution graphs and prediction accuracy plots

---

## 💻 Streamlit Web App

An interactive dashboard built with **Streamlit** that allows:

- Uploading new genotype data
- Predicting all 10 fatty acids instantly
- Downloading results as `.csv` or `.xlsx`
- Visualizing distributions & feature importance

---

## 🧰 Tech Stack

| Tool           | Purpose                              |
|----------------|---------------------------------------|
| **Python**     | Core logic & ML model                 |
| **Pandas**     | Data preprocessing                    |
| **Scikit-learn** | ML modeling & evaluation             |
| **Matplotlib / Seaborn** | Visualizations              |
| **Streamlit**  | Web dashboard                         |
| **GitHub**     | Version control                       |

---

## 🧠 Key Learnings

- Multi-output regression for complex biological traits
- Handling large-scale genomic and phenotypic datasets
- Model interpretation (top SNPs) and correlation analysis
- Deployment using Streamlit and GitHub

---

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/sridharanrajagopal1/Maize_Oil_Fatty_Prediction.git
   cd Maize_Oil_Fatty_Prediction
Create virtual environment:

bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
📬 Contact
Sridharan Rajagopal
🔗 LinkedIn
📁 Portfolio
📧 Email: sridharanmettur@gmail.com

📌 License
This project is open-source under the MIT License.

yaml
Copy
Edit

---

## ✅ Next Step:

Would you like me to generate:

- A `.gitignore` (to avoid `.venv`, `.ipynb_checkpoints`, etc.)
- A `requirements.txt` (auto-detect your packages)?
- Add this `README.md` to your repo directly?

Let me know and I’ll provide the next command or file.
