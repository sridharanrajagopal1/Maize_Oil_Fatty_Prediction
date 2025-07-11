import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# Configure the app
st.set_page_config(
    page_title="🌽 Maize Fatty Acid Predictor",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        color: #2e8b57;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #2e8b57;
    }
    .footer {
        font-size: small;
        color: gray;
        text-align: center;
        margin-top: 2rem;
    }
    .warning {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("""
<div class="main-header">
    <h1>🌽 Maize Fatty Acid Composition Predictor</h1>
    <p>Predict fatty acid composition from genotypic data using machine learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This tool predicts 10 fatty acid components in maize:
    - **C16:0** (Palmitic acid)
    - **C16:1** (Palmitoleic acid)
    - **C18:0** (Stearic acid)
    - **C18:1** (Oleic acid)
    - **C18:2** (Linoleic acid)
    - **C18:3** (Linolenic acid)
    - **C20:0** (Arachidic acid)
    - **C20:1** (Gadoleic acid)
    - **C22:0** (Behenic acid)
    - **C24:0** (Lignoceric acid)
    """)
    
    st.markdown("---")
    st.header("Instructions")
    st.markdown("""
    1. For **batch prediction**, upload a CSV/Excel file with genotype data
    2. For **single prediction**, manually enter SNP values
    3. View and download results
    """)
    
    st.markdown("---")
    st.header("Model Information")
    st.markdown("""
    - **Algorithm**: Random Forest
    - **Input SNPs**: 62,077 markers
    - **Targets**: 10 fatty acids
    """)

# Function to load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    try:
        # Load phenotypic data
        lines_df = pd.read_excel("Maize_oli_fatty_Projects/Maize_Phenotypic_Data 1.xlsx", 
                               sheet_name="BGEM_Lines_Data")
        hybrids_df = pd.read_excel("Maize_oli_fatty_Projects/Maize_Phenotypic_Data 1.xlsx", 
                                 sheet_name="BGEM_F1_Hybrids_Data")

        # Standardize genotype column and merge
        lines_df.rename(columns={"Gen": "Genotype"}, inplace=True)
        hybrids_df.rename(columns={"Gen": "Genotype"}, inplace=True)
        phenotype = pd.concat([lines_df, hybrids_df], ignore_index=True)

        # Load genotypic data
        geno_raw = pd.read_excel("Maize_oli_fatty_Projects/Maize_Genotypic_Data_imputed_62077 1.xlsx")
        geno_only = geno_raw.drop(columns=geno_raw.columns[:11])
        geno_transposed = geno_only.T
        geno_transposed.reset_index(inplace=True)
        geno_transposed.rename(columns={"index": "Genotype"}, inplace=True)

        # Merge data
        phenotype["Genotype"] = phenotype["Genotype"].astype(str).str.strip()
        geno_transposed["Genotype"] = geno_transposed["Genotype"].astype(str).str.strip()
        full_data = pd.merge(phenotype, geno_transposed, on="Genotype", how="inner")

        # Prepare features and targets
        targets = ["C16:0", "C16:1", "C18:0", "C18:1", "C18:2", 
                  "C18:3", "C20:0", "C20:1", "C22:0", "C24:0"]
        full_data = full_data.dropna(subset=targets)
        X = full_data.drop(columns=["Env", "Rep", "Genotype", "Oil"] + targets, errors='ignore')
        y = full_data[targets]

        # Encode SNP values
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.factorize(X[col])[0]

        return X, y, full_data, targets

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

# Function to train and save model
@st.cache_resource
def train_model():
    X, y, _, _ = load_and_preprocess_data()
    if X is not None and y is not None:
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model = MultiOutputRegressor(rf)
        model.fit(X, y)
        joblib.dump(model, 'maize_fatty_acid_model.pkl')
        return model
    return None

# Load or train model
model_path = 'maize_fatty_acid_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    with st.spinner("Training model for the first time... This may take several minutes."):
        model = train_model()

# Generate all required SNP columns
REQUIRED_SNPS = [f"SNP_{i}" for i in range(1, 62078)]

# File upload section
st.header("📁 Batch Prediction")
uploaded_file = st.file_uploader("Upload genotype data (CSV or Excel)", type=['csv', 'xlsx'], 
                                help="Upload a file with genotype data for multiple samples")

if uploaded_file:
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            geno_data = pd.read_csv(uploaded_file)
        else:
            geno_data = pd.read_excel(uploaded_file)
        
        # Show uploaded data
        st.subheader("Uploaded Data Preview")
        st.dataframe(geno_data.head())
        
        # Check for required columns
        missing_columns = [col for col in REQUIRED_SNPS if col not in geno_data.columns]
        
        if missing_columns:
            st.markdown(f"""
            <div class="warning">
                <strong>Warning:</strong> {len(missing_columns)} SNP columns are missing from the uploaded file.
                Missing columns will be filled with zeros (major allele homozygote).
            </div>
            """, unsafe_allow_html=True)
            
            # Add missing columns with default value of 0
            for col in missing_columns:
                geno_data[col] = 0
            
            # Reorder columns to match model expectations
            geno_data = geno_data[REQUIRED_SNPS]
        
        # Encode categorical SNPs
        for col in geno_data.columns:
            if geno_data[col].dtype == 'object':
                # Convert letters to numbers (AA=0, AB=1, BB=2)
                geno_data[col] = geno_data[col].replace({'AA': 0, 'AB': 1, 'BB': 2})
                # For any remaining non-numeric values, convert to numeric
                geno_data[col] = pd.to_numeric(geno_data[col], errors='coerce').fillna(0)
        
        # Make predictions
        with st.spinner('Making predictions...'):
            progress_bar = st.progress(0)
            
            # Simulate progress
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.02)
            
            # Ensure we only use the required columns in the correct order
            prediction_data = geno_data[REQUIRED_SNPS]
            predictions = model.predict(prediction_data)
            pred_df = pd.DataFrame(predictions, columns=[
                "C16:0", "C16:1", "C18:0", "C18:1", "C18:2", 
                "C18:3", "C20:0", "C20:1", "C22:0", "C24:0"
            ])
            
            # Combine with original data
            results_df = pd.concat([geno_data, pred_df], axis=1)
            
            st.success("✅ Predictions completed!")
            st.subheader("Prediction Results")
            
            # Show interactive table
            st.dataframe(results_df.style.format({
                col: "{:.4f}" for col in [
                    "C16:0", "C16:1", "C18:0", "C18:1", "C18:2", 
                    "C18:3", "C20:0", "C20:1", "C22:0", "C24:0"
                ]
            }))
            
            # Visualization
            st.subheader("Results Visualization")
            selected_acid = st.selectbox("Select fatty acid to visualize", pred_df.columns)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(pred_df[selected_acid], kde=True, ax=ax)
            ax.set_title(f"Distribution of {selected_acid} Predictions")
            ax.set_xlabel("Predicted Value")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            
            # Download results
            st.markdown("### Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name='maize_fatty_acid_predictions.csv',
                    mime='text/csv',
                    help="Download full prediction results as CSV"
                )
            
            with col2:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    results_df.to_excel(writer, index=False)
                st.download_button(
                    label="Download as Excel",
                    data=excel_buffer,
                    file_name='maize_fatty_acid_predictions.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    help="Download full prediction results as Excel file"
                )
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.error("Please ensure your file contains genotype data with SNP markers.")
        st.error("If your columns are named differently, please rename them to SNP_1, SNP_2,... SNP_62077")

# Single sample prediction
st.header("🔢 Single Sample Prediction")
with st.expander("Enter genotype values manually", expanded=True):
    with st.form("single_prediction_form"):
        st.write("Enter values for key SNPs (example with 10 SNPs):")
        
        # Create input fields for some example SNPs
        cols = st.columns(5)
        snp_values = {}
        
        for i in range(10):  # Show 10 SNP inputs as example
            with cols[i % 5]:
                snp_values[f"SNP_{i+1}"] = st.number_input(
                    f"SNP_{i+1}",
                    min_value=0,
                    max_value=2,
                    value=1,
                    help="0=AA, 1=AB, 2=BB",
                    key=f"snp_{i}"
                )
        
        submitted = st.form_submit_button("Predict Composition")
        
        if submitted:
            try:
                # Create sample data with all SNPs (fill missing with 0)
                sample_data = np.zeros((1, 62077))  # All SNPs initialized to 0
                
                # Set values for the SNPs we collected input for
                for i in range(10):
                    sample_data[0, i] = snp_values[f"SNP_{i+1}"]
                
                # Make prediction
                with st.spinner('Predicting...'):
                    prediction = model.predict(sample_data)
                
                # Display results
                st.subheader("Predicted Fatty Acid Composition")
                
                acids = ["C16:0", "C16:1", "C18:0", "C18:1", "C18:2", 
                        "C18:3", "C20:0", "C20:1", "C22:0", "C24:0"]
                
                # Show metrics in cards
                cols = st.columns(2)
                for i, acid in enumerate(acids):
                    with cols[i % 2]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{acid}</h3>
                            <h2>{prediction[0][i]:.4f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show as radar chart
                st.subheader("Composition Profile")
                
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
                angles = np.linspace(0, 2 * np.pi, len(acids), endpoint=False).tolist()
                values = prediction[0].tolist()
                values += values[:1]  # Close the loop
                angles += angles[:1]  # Close the loop
                
                ax.fill(angles, values, color='green', alpha=0.25)
                ax.plot(angles, values, color='green', marker='o')
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(acids)
                ax.set_title("Fatty Acid Composition Profile", pad=20)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# Model information section
st.markdown("---")
st.header("📊 Model Information")

if st.button("Show Model Performance Metrics"):
    try:
        X, y, full_data, targets = load_and_preprocess_data()
        
        if X is not None and y is not None:
            # Calculate cross-validated metrics (simplified for demo)
            from sklearn.model_selection import cross_val_score
            
            st.subheader("Cross-Validated Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**R² Scores (10-fold CV):**")
                for target in targets[:5]:
                    scores = cross_val_score(model, X, y[target], cv=10, scoring='r2')
                    st.write(f"{target}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
            
            with col2:
                st.markdown("**R² Scores (10-fold CV):**")
                for target in targets[5:]:
                    scores = cross_val_score(model, X, y[target], cv=10, scoring='r2')
                    st.write(f"{target}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
            
            # Feature importance
            st.subheader("Feature Importance (Top 10 SNPs for C16:0)")
            importances = model.estimators_[0].feature_importances_
            indices = np.argsort(importances)[-10:][::-1]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_title("Top 10 Important SNPs for C16:0 Prediction")
            ax.barh(range(10), importances[indices], color="green", align="center")
            ax.set_yticks(range(10))
            ax.set_yticklabels([f"SNP_{i+1}" for i in indices])
            ax.invert_yaxis()
            ax.set_xlabel("Relative Importance")
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Could not calculate metrics: {str(e)}")

# Add footer
st.markdown("""
<div class="footer">
    <p>© 2023 Maize Fatty Acid Prediction System | For research use only</p>
    <p>Developed with ❤️ using Python, scikit-learn, and Streamlit</p>
</div>
""", unsafe_allow_html=True)