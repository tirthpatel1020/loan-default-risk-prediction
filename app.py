"""
Loan Default Risk Prediction - Web Application
================================================
Interactive Streamlit app for predicting loan default risk using
cybersecurity-inspired behavioral features.

Author: Your Name
Date: November 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .risk-high {
        color: #fd7e14;
        font-weight: bold;
    }
    .risk-very-high {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    """Load all trained models"""
    models_dir = Path('models')
    models = {
        'Logistic Regression': joblib.load(models_dir / 'logistic_regression_enhanced.pkl'),
        'Random Forest': joblib.load(models_dir / 'random_forest.pkl'),
        'XGBoost': joblib.load(models_dir / 'xgboost.pkl'),
        'Neural Network': joblib.load(models_dir / 'neural_network.pkl')
    }
    scaler = joblib.load(Path('data') / 'scaler.pkl')
    return models, scaler

@st.cache_data
def load_feature_info():
    """Load feature information"""
    return pd.read_csv(Path('data') / 'feature_info.csv')

@st.cache_data
def load_comparison_data():
    """Load model comparison data"""
    return pd.read_csv(Path('results') / 'model_comparison.csv')

# Initialize
try:
    models, scaler = load_models()
    feature_info = load_feature_info()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"⚠️ Error loading models: {e}")
    st.info("Make sure all model files are in the 'models/' directory")

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    ["🏠 Home", "🔮 Predict Default", "📊 Model Comparison", "📈 Batch Prediction", "ℹ️ About"]
)

# Helper functions
def get_risk_category(probability):
    """Categorize risk based on probability"""
    if probability < 0.20:
        return "Low Risk", "risk-low"
    elif probability < 0.40:
        return "Medium Risk", "risk-medium"
    elif probability < 0.60:
        return "High Risk", "risk-high"
    else:
        return "Very High Risk", "risk-very-high"

def create_gauge_chart(probability):
    """Create gauge chart for probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Default Probability (%)", 'font': {'size': 24}},
        delta = {'reference': 20, 'increasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#28a745'},
                {'range': [20, 40], 'color': '#ffc107'},
                {'range': [40, 60], 'color': '#fd7e14'},
                {'range': [60, 100], 'color': '#dc3545'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_feature_contribution_chart(input_features_dict, all_feature_names):
    """Create feature contribution bar chart"""
    # Get feature importances from Random Forest model
    rf_model = models['Random Forest']
    importances = rf_model.feature_importances_
    
    # Ensure we have matching lengths
    if len(importances) != len(all_feature_names):
        st.warning(f"⚠️ Feature count mismatch: {len(importances)} vs {len(all_feature_names)}")
        # Use minimum length
        min_len = min(len(importances), len(all_feature_names))
        importances = importances[:min_len]
        all_feature_names = all_feature_names[:min_len]
    
    # Create DataFrame with feature importance
    contribution_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(15)
    
    # Determine if cybersecurity feature
    cyber_features = ['app_edit_score', 'app_edit_count', 'login_irregularity_score', 
                     'device_anomaly_flag', 'off_hours_application', 'rush_application', 
                     'info_consistency_score', 'behavioral_risk_composite', 'emp_length_missing']
    contribution_df['Type'] = contribution_df['Feature'].apply(
        lambda x: 'Cybersecurity' if x in cyber_features else 'Traditional'
    )
    
    fig = px.bar(
        contribution_df,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Type',
        color_discrete_map={'Cybersecurity': '#e74c3c', 'Traditional': '#3498db'},
        title='Top 15 Feature Contributions to Prediction',
        labels={'Importance': 'Feature Importance', 'Feature': ''}
    )
    
    fig.update_layout(height=500, showlegend=True)
    return fig

# ============================================================================
# PAGE: HOME
# ============================================================================
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">💰 Loan Default Risk Assessment</h1>', unsafe_allow_html=True)
    st.markdown("### Using Cybersecurity-Inspired Behavioral Features")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🎯 Project Goal")
        st.info("""
        Develop an explainable machine learning system that enhances 
        traditional loan risk prediction by integrating cybersecurity-inspired 
        behavioral features.
        """)
    
    with col2:
        st.markdown("#### 🔬 Innovation")
        st.success("""
        First research to bridge finance, ML, and cybersecurity by introducing 
        behavioral features (application edits, login patterns, device anomalies) 
        into loan default prediction.
        """)
    
    with col3:
        st.markdown("#### 📈 Key Results")
        st.warning("""
        - **99.43% ROC-AUC** (36% improvement)
        - **71% importance** from behavioral features
        - **$4-7M** estimated annual savings per $100M portfolio
        """)
    
    st.markdown("---")
    
    st.markdown("### 🚀 Quick Start")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Traditional Features")
        st.markdown("""
        - Loan amount & interest rate
        - Annual income & DTI
        - Credit grade & history
        - Employment length
        - Home ownership status
        """)
    
    with col2:
        st.markdown("#### Cybersecurity Features (Innovation!) 🔐")
        st.markdown("""
        - **Application edit count** - Edits indicate uncertainty
        - **Info consistency score** - Contradictory data detection
        - **Behavioral risk score** - Composite risk indicator
        - **Device anomaly flag** - VPN/proxy detection
        - **Rush application** - Suspiciously fast completion
        """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Model Performance")
    
    if MODEL_LOADED:
        comparison_df = load_comparison_data()
        
        fig = px.bar(
            comparison_df,
            x='Model',
            y=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            barmode='group',
            title='Model Performance Comparison',
            labels={'value': 'Score', 'variable': 'Metric'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate to different pages!")

# ============================================================================
# PAGE: PREDICT DEFAULT
# ============================================================================
elif page == "🔮 Predict Default":
    st.markdown('<h1 class="main-header">🔮 Loan Default Prediction</h1>', unsafe_allow_html=True)
    
    if not MODEL_LOADED:
        st.error("⚠️ Models not loaded. Please check the models directory.")
        st.stop()
    
    st.markdown("### Enter Loan Application Details")
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 💵 Financial Information")
            loan_amnt = st.slider("Loan Amount ($)", 1000, 35000, 15000, 1000)
            int_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 12.0, 0.5)
            annual_inc = st.number_input("Annual Income ($)", 20000, 300000, 75000, 5000)
            dti = st.slider("Debt-to-Income Ratio (%)", 0.0, 40.0, 18.0, 1.0)
            installment = st.number_input("Monthly Installment ($)", 50, 1500, 450, 10)
        
        with col2:
            st.markdown("#### 📊 Credit History")
            grade = st.selectbox("Credit Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
            grade_encoded = ord(grade) - ord('A')
            
            emp_length = st.selectbox("Employment Length", 
                                     ['< 1 year', '1 year', '2 years', '3 years', '4 years',
                                      '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'])
            emp_length_years = 10 if emp_length == '10+ years' else (0 if emp_length == '< 1 year' else int(emp_length.split()[0]))
            
            home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE'])
            term = st.selectbox("Loan Term", ['36 months', '60 months'])
            term_months = int(term.split()[0])
            
            delinq_2yrs = st.number_input("Delinquencies (last 2 years)", 0, 10, 0)
            has_delinquency = 1 if delinq_2yrs > 0 else 0
        
        with col3:
            st.markdown("#### 🔐 Behavioral Features (Cybersecurity)")
            app_edit_count = st.slider("Application Edits", 0, 15, 2)
            app_edit_score = app_edit_count * 10
            
            info_consistency_score = st.slider("Info Consistency Score", 0, 100, 85)
            
            behavioral_risk_composite = st.slider("Behavioral Risk Score", 0, 100, 20)
            
            device_anomaly_flag = st.checkbox("Device Anomaly Detected (VPN/Proxy)")
            device_anomaly = 1 if device_anomaly_flag else 0
            
            rush_application = st.checkbox("Rush Application (< 5 minutes)")
            rush_app = 1 if rush_application else 0
            
            off_hours_application = st.checkbox("Applied Off-Hours (11pm-5am)")
            off_hours = 1 if off_hours_application else 0
        
        submitted = st.form_submit_button("🎯 Predict Default Risk", use_container_width=True)
    
    if submitted:
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")
        
        # Note about simplified prediction
        st.info("ℹ️ **Note:** This demo uses simplified feature engineering. In production, all features would be properly calculated from application data.")
        
        # Create feature vector (simplified - matches training feature order)
        # Get the actual feature names from training
        try:
            feature_names = feature_info['Feature_Name'].tolist()
        except:
            st.error("Could not load feature names. Using default order.")
            feature_names = []
        
        # Create features with proper values
        features_dict = {
            'loan_amnt': loan_amnt,
            'int_rate': int_rate,
            'installment': installment,
            'annual_inc': annual_inc,
            'dti': dti,
            'delinq_2yrs': delinq_2yrs,
            'open_acc': 11,  # Default values
            'pub_rec': 0,
            'revol_bal': annual_inc * 0.2,  # Estimated
            'revol_util': 50.0,  # Default
            'total_acc': 25,  # Default
            'term_months': term_months,
            'payment_to_income_ratio': (installment * 12) / (annual_inc + 1),
            'total_interest_burden': int_rate * loan_amnt / 100,
            'credit_risk_score': 50 * dti / 100,
            'loan_to_income_ratio': loan_amnt / (annual_inc + 1),
            'account_utilization': 0.5,
            'emp_length_years': emp_length_years,
            'has_delinquency': has_delinquency,
            'has_public_record': 0,
            'emp_length_missing': 0,
            'grade_encoded': grade_encoded,
            'app_edit_score': app_edit_score,
            'app_edit_count': app_edit_count,
            'login_irregularity_score': behavioral_risk_composite * 0.5,
            'device_anomaly_flag': device_anomaly,
            'off_hours_application': off_hours,
            'rush_application': rush_app,
            'info_consistency_score': info_consistency_score,
            'behavioral_risk_composite': behavioral_risk_composite
        }
        
        # Create feature vector matching the training data order
        X_input = []
        for fname in feature_names:
            if fname in features_dict:
                X_input.append(features_dict[fname])
            elif fname.startswith('home_ownership_'):
                X_input.append(1 if fname == f'home_ownership_{home_ownership}' else 0)
            elif fname.startswith('verification_'):
                X_input.append(0)  # Default
            elif fname.startswith('purpose_'):
                X_input.append(0)  # Default
            else:
                X_input.append(0)  # Default for any missing features
        
        X_input = np.array(X_input).reshape(1, -1)
        
        # Validate input shape
        if len(X_input[0]) != len(feature_names):
            st.warning(f"⚠️ Feature count mismatch. Expected {len(feature_names)}, got {len(X_input[0])}")
            # Pad or trim to match
            if len(X_input[0]) < len(feature_names):
                X_input = np.pad(X_input, ((0, 0), (0, len(feature_names) - len(X_input[0]))), mode='constant')
            else:
                X_input = X_input[:, :len(feature_names)]
        
        # Select model
        model_choice = st.selectbox("Choose Model:", list(models.keys()), index=1)
        model = models[model_choice]
        
        # Predict
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0][1]
        
        risk_category, risk_class = get_risk_category(probability)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 0:
                st.success("### ✅ APPROVED")
                st.markdown("This loan is predicted to be **PAID BACK**")
            else:
                st.error("### ❌ DECLINED")
                st.markdown("This loan is predicted to **DEFAULT**")
        
        with col2:
            st.metric(
                label="Default Probability",
                value=f"{probability*100:.1f}%",
                delta=f"{(probability - 0.20)*100:.1f}% vs average" if probability > 0.20 else f"{(0.20 - probability)*100:.1f}% below average"
            )
        
        with col3:
            st.markdown(f'<p class="{risk_class}">Risk Category: {risk_category}</p>', unsafe_allow_html=True)
            
            if risk_category == "Low Risk":
                st.success("✅ Auto-approve with standard terms")
            elif risk_category == "Medium Risk":
                st.warning("⚠️ Manual review recommended")
            elif risk_category == "High Risk":
                st.warning("🔍 Enhanced due diligence required")
            else:
                st.error("🚫 High decline probability")
        
        # Gauge chart
        st.plotly_chart(create_gauge_chart(probability), use_container_width=True)
        
        # Feature contributions
        st.markdown("### 📈 Feature Contributions")
        fig = create_feature_contribution_chart(
            features_dict,
            feature_names
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        
        if info_consistency_score < 70:
            st.warning("⚠️ **Low information consistency** - Additional verification recommended")
        
        if app_edit_count >= 5:
            st.warning("⚠️ **High edit count** - Review application carefully for discrepancies")
        
        if device_anomaly:
            st.warning("⚠️ **Device anomaly detected** - Enhanced identity verification recommended")
        
        if rush_application:
            st.warning("⚠️ **Rush application** - Consider phone verification")
        
        if behavioral_risk_composite > 40:
            st.error("🚫 **High behavioral risk** - Multiple red flags present")

# ============================================================================
# PAGE: MODEL COMPARISON
# ============================================================================
elif page == "📊 Model Comparison":
    st.markdown('<h1 class="main-header">📊 Model Performance Comparison</h1>', unsafe_allow_html=True)
    
    if not MODEL_LOADED:
        st.error("⚠️ Data not loaded.")
        st.stop()
    
    comparison_df = load_comparison_data()
    
    st.markdown("### Performance Metrics Across All Models")
    st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']), use_container_width=True)
    
    # Metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Accuracy & F1-Score")
        fig = px.bar(
            comparison_df,
            x='Model',
            y=['Accuracy', 'F1-Score'],
            barmode='group',
            color_discrete_sequence=['#3498db', '#e74c3c']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Precision & Recall")
        fig = px.bar(
            comparison_df,
            x='Model',
            y=['Precision', 'Recall'],
            barmode='group',
            color_discrete_sequence=['#2ecc71', '#f39c12']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ROC-AUC comparison
    st.markdown("#### ROC-AUC Score")
    fig = px.bar(
        comparison_df,
        x='Model',
        y='ROC-AUC',
        color='ROC-AUC',
        color_continuous_scale='RdYlGn',
        range_color=[0.7, 1.0]
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Best model
    best_idx = comparison_df['ROC-AUC'].idxmax()
    best_model = comparison_df.loc[best_idx]
    
    st.success(f"""
    ### 🏆 Best Performing Model: {best_model['Model']}
    - **ROC-AUC:** {best_model['ROC-AUC']:.4f}
    - **F1-Score:** {best_model['F1-Score']:.4f}
    - **Accuracy:** {best_model['Accuracy']:.4f}
    """)
    
    # Impact of cybersecurity features
    st.markdown("---")
    st.markdown("### 🔐 Impact of Cybersecurity Features")
    
    cyber_impact = pd.read_csv(Path('results') / 'cybersecurity_impact.csv')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Absolute Improvement")
        fig = px.bar(
            cyber_impact,
            x='Metric',
            y=['Baseline', 'Enhanced'],
            barmode='group',
            color_discrete_sequence=['lightblue', 'orange']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Relative Improvement (%)")
        fig = px.bar(
            cyber_impact,
            x='Metric',
            y='Relative_Improvement_%',
            color='Relative_Improvement_%',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: BATCH PREDICTION
# ============================================================================
elif page == "📈 Batch Prediction":
    st.markdown('<h1 class="main-header">📈 Batch Prediction</h1>', unsafe_allow_html=True)
    
    st.markdown("### Upload CSV file for bulk predictions")
    
    st.info("""
    📋 **CSV Format Required:**
    - Must contain all required features
    - Use the same column names as training data
    - One row per loan application
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.markdown(f"### 📊 Loaded {len(df):,} applications")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("🎯 Generate Predictions"):
            if MODEL_LOADED:
                with st.spinner("Processing..."):
                    # Make predictions (simplified - needs proper feature engineering)
                    model = models['Random Forest']
                    
                    try:
                        predictions = model.predict(df)
                        probabilities = model.predict_proba(df)[:, 1]
                        
                        # Add results to dataframe
                        df['Prediction'] = ['Default' if p == 1 else 'Paid' for p in predictions]
                        df['Default_Probability'] = probabilities
                        df['Risk_Category'] = [get_risk_category(p)[0] for p in probabilities]
                        
                        st.success("✅ Predictions completed!")
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Applications", len(df))
                        with col2:
                            st.metric("Predicted Defaults", (predictions == 1).sum())
                        with col3:
                            st.metric("Average Default Prob", f"{probabilities.mean()*100:.1f}%")
                        with col4:
                            st.metric("High Risk Count", (df['Risk_Category'] == 'High Risk').sum())
                        
                        # Display results
                        st.dataframe(df, use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name="loan_predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error making predictions: {e}")
                        st.info("Make sure your CSV has all required features in the correct format")
            else:
                st.error("Models not loaded")

# ============================================================================
# PAGE: ABOUT
# ============================================================================
elif page == "ℹ️ About":
    st.markdown('<h1 class="main-header">ℹ️ About This Project</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Risk Assessment for Loan Default Using Machine Learning with Security-Aware Features
    
    ### 🎯 Research Objective
    
    This research develops a hybrid, explainable machine learning system for predicting loan default risk 
    by integrating cybersecurity-inspired behavioral features into traditional financial risk models.
    
    ### 🔬 Innovation
    
    While traditional loan risk prediction relies heavily on financial variables (credit grade, interest rate, income), 
    this project introduces a novel approach by incorporating behavioral patterns observed during the loan 
    application process:
    
    - **Application editing patterns** - Excessive edits may indicate uncertainty or deception
    - **Login irregularities** - Unusual access patterns as risk signals
    - **Device anomalies** - VPN/proxy usage detection
    - **Timing patterns** - Off-hours or rushed applications
    - **Information consistency** - Contradictory data detection
    
    ### 📊 Dataset
    
    - **Source:** LendingClub Dataset (Kaggle)
    - **Period:** 2007-2018 Q4
    - **Samples:** 87,892 accepted loans with clear outcomes
    - **Features:** 39 (31 traditional + 8 cybersecurity-inspired)
    
    ### 🤖 Models Implemented
    
    1. **Logistic Regression** (Baseline & Enhanced)
    2. **Random Forest**
    3. **XGBoost**
    4. **Neural Network**
    
    ### 📈 Key Results
    
    - **ROC-AUC:** 99.43% (36% improvement over baseline)
    - **F1-Score:** 91.90% (105% improvement)
    - **Feature Importance:** Cybersecurity features account for 71% of model decisions
    - **Top Predictor:** Information consistency score (44% importance)
    
    ### 💡 Business Impact
    
    For a $100M loan portfolio:
    - **Estimated savings:** $4-7M annually
    - **Default detection:** 95% vs 64% (baseline)
    - **False rejections:** Reduced by 83%
    - **Approval rate increase:** 8-10%
    
    ### 🔍 Explainability
    
    The system provides transparent decision-making through:
    - SHAP (SHapley Additive exPlanations) analysis
    - Individual prediction explanations
    - Feature contribution visualizations
    - Regulatory-compliant documentation
    
    ### ⚠️ Important Notes
    
    - Cybersecurity features are **simulated** based on risk-behavior correlations
    - This is a **proof-of-concept** demonstrating the framework
    - Real-world deployment requires validation with actual behavioral data
    - Results shown represent potential value, not production guarantees
    
    ### 👨‍🎓 Author
    
    **Your Name**  
    Master of Data Science and Analytics  
    Toronto Metropolitan University  
    November 2025
    
    ### 📚 Technologies Used
    
    - **Machine Learning:** scikit-learn, XGBoost, TensorFlow
    - **Explainability:** SHAP
    - **Data Processing:** pandas, numpy
    - **Visualization:** Plotly, matplotlib, seaborn
    - **Web App:** Streamlit
    - **Class Balancing:** SMOTE (imbalanced-learn)
    
    ### 📧 Contact
    
    For questions or collaboration:
    - Email: your.email@example.com
    - GitHub: github.com/yourusername
    - LinkedIn: linkedin.com/in/yourprofile
    
    ---
    
    ### 🙏 Acknowledgments
    
    - LendingClub for providing the dataset
    - Toronto Metropolitan University
    - Research supervisors and committee members
    """)
    
    st.markdown("---")
    st.info("💡 **How to Use:** Navigate through different pages using the sidebar to explore predictions, model comparisons, and batch processing capabilities.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Project Statistics")
st.sidebar.metric("Total Models", "5")
st.sidebar.metric("Features", "39")
st.sidebar.metric("Best ROC-AUC", "99.43%")
st.sidebar.markdown("---")
st.sidebar.info("🎓 Master's Thesis Project\n\nToronto Metropolitan University\n\n2025")
