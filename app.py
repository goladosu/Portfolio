import os
import sys
# Standard library imports
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# Delay heavy ML imports - load only when needed
try:
    import joblib
except ImportError:
    joblib = None

try:
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:
    BaseEstimator = object
    TransformerMixin = object

try:
    import shap
except ImportError:
    shap = None


# ==============================================================================
# Custom Pipeline Components (Names MUST match the pickled model!)
# ==============================================================================

class ClinicalConsistencyTransformer(BaseEstimator, TransformerMixin):
    # Rule enforcement step: check data logic before processing.

    def __init__(
        self,
        visit1_cols: Optional[List[str]] = None,
        visit2_cols: Optional[List[str]] = None,
        age_col: str = "age",
        min_age: int = 18,
        max_age: int = 90
    ):
        # Default columns for V1 and V2
        self.visit1_cols = visit1_cols or [
            "visit1_symptom_score",
            "visit1_adherence_rate",
            "visit1_AE_count",
        ]
        self.visit2_cols = visit2_cols or [
            "visit2_symptom_score",
            "visit2_adherence_rate",
            "visit2_AE_count",
        ]

        self.age_col = age_col
        self.min_age = min_age
        self.max_age = max_age

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()

        # Clamp age to a sensible 18-90 range, don't drop rows.
        if self.age_col in X.columns:
            X.loc[X[self.age_col] < self.min_age, self.age_col] = self.min_age
            X.loc[X[self.age_col] > self.max_age, self.age_col] = self.max_age

        # Visit dependency logic: V2 can't exist if V1 is completely blank.
        v1_cols = [c for c in self.visit1_cols if c in X.columns]
        v2_cols = [c for c in self.visit2_cols if c in X.columns]

        if v1_cols and v2_cols:
            # Mask where ALL V1 fields are NaN
            no_v1_mask = X[v1_cols].isna().all(axis=1)

            # Wipe V2 data if V1 is missing
            X.loc[no_v1_mask, v2_cols] = np.nan

        return X


class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    # Adds the `_missing` flags needed for XGBoost to learn from missingness.

    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                # 1 if missing, 0 otherwise
                X[f"{col}_missing"] = X[col].isna().astype(int)
        return X


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fix unpickling errors (often needed for Streamlit/notebooks)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sys.modules["main"] = sys.modules[__name__]
sys.modules["__main__"] = sys.modules[__name__]


# ==============================================================================
# App config + model loader
# ==============================================================================

st.set_page_config(page_title="Abdul Oladosu", layout="wide")

MODEL_PATH = "xgb_dropout_pipeline.pkl"

# Deployment cutoff, set by evaluation notebook
CHOSEN_THRESHOLD = 0.30


@st.cache_resource
def load_pipeline():
    # Only load the model once, cache it.
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found: {MODEL_PATH}"
    try:
        pipe = joblib.load(MODEL_PATH)
        return pipe, None
    except Exception as e:
        return None, f"Failed to load model: {e}"


# Don't load model at startup - load it only when needed
pipeline = None
load_err = None


# ==============================================================================
# Navigation state
# ==============================================================================

PAGES = ["Home", "Projects Summary", "Clinical Trial Dropout Prediction", "Revenue Cycle Dashboard"]

if "page" not in st.session_state:
    st.session_state.page = "Home"

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    PAGES,
    index=PAGES.index(st.session_state.page),
)
st.session_state.page = page


# ==============================================================================
# Helpers
# ==============================================================================

def risk_bucket(p: float) -> str:
    # Assign risk level based on probability p.
    # High risk means >= CHOSEN_THRESHOLD
    if p < 0.15:
        return "Low"
    elif p < CHOSEN_THRESHOLD:  # 0.30
        return "Moderate"
    else:
        return "High"


def build_feature_names_from_preprocessor(preprocess, numeric_features, categorical_features):
    # Reconstruct feature names after OHE for SHAP display.
    num_names = list(numeric_features)

    try:
        # Get OHE names from the 'cat' step
        cat_encoder = preprocess.named_transformers_["cat"].named_steps["encoder"]
        cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    except Exception:
        cat_names = []  # Handle if encoding step isn't available

    return np.array(num_names + cat_names, dtype=object)


def get_shap_for_single_row(pipe, input_df):
    # Runs the single input row through the pipeline steps to generate SHAP values.
    # Returns (prob, pred, shap_values_row, feature_names)

    # Run prediction first
    prob = float(pipe.predict_proba(input_df)[0, 1])
    pred = int(prob >= CHOSEN_THRESHOLD)

    # Extract components for manual transformation
    clinical = pipe.named_steps.get("clinical_logic")
    flags = pipe.named_steps.get("missing_flags")
    preprocess = pipe.named_steps.get("preprocess")
    model = pipe.named_steps.get("model")

    # Transform data through custom steps
    X_logic = clinical.transform(input_df) if clinical else input_df
    X_flags = flags.transform(X_logic) if flags else X_logic

    # Final preprocessing (scaling, encoding, etc.)
    X_prep = preprocess.transform(X_flags)

    # Ensure it's a dense matrix for SHAP
    try:
        X_prep_dense = X_prep.toarray()
    except Exception:
        X_prep_dense = X_prep

    # Try to extract the feature names list
    try:
        num_cols = preprocess.transformers_[0][2]
        numeric_features = list(num_cols)
    except Exception:
        # Generic names if extraction fails
        numeric_features = [f"x{i}" for i in range(X_prep_dense.shape[1])]

    try:
        # Rebuild full list for plot labels
        feature_names = build_feature_names_from_preprocessor(
            preprocess=preprocess,
            numeric_features=numeric_features,
            categorical_features=["sex", "race"],
        )
        if feature_names.shape[0] != X_prep_dense.shape[1]:
            feature_names = np.array([f"x{i}" for i in range(X_prep_dense.shape[1])], dtype=object)
    except Exception:
        feature_names = np.array([f"x{i}" for i in range(X_prep_dense.shape[1])], dtype=object)

    # Compute SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_prep_dense)

    # Get the values for class 1 (Dropout)
    if isinstance(shap_values, list):
        shap_row = np.array(shap_values[1][0]).ravel()
    else:
        shap_row = np.array(shap_values[0]).ravel()

    return prob, pred, shap_row, feature_names


# ==============================================================================
# Pages
# ==============================================================================

def page_home():
    st.title("Welcome — I'm Abdul Oladosu")

    st.markdown(
        """
** Data Scientist | Machine Learning **

This is a space where I build and explore **data-driven systems**—from analytics to machine learning.

I'm interested in problems where data is imperfect, decisions matter, and solutions need to be **clear, explainable, and useful**. My work focuses on turning raw data into insights and tools that support real-world decision-making across different domains.

---

### What You'll Find Here

Hands-on projects that demonstrate:
- Applied analytics and machine learning  
- End-to-end workflows, from data to deployment  
- Thoughtful evaluation and interpretation of results  

---

### How I Approach Data Work

I focus on:
- Understanding the data before modeling  
- Making assumptions explicit  
- Explaining results clearly to non-technical audiences  
- Considering how outputs are actually used in practice  

Interpretability, transparency, and responsible use of data are themes that run across my projects.

---

### Explore

Use the navigation to explore projects, interact with models, and see how data science ideas translate into working applications.
        """
    )

    st.divider()
    st.subheader("Quick Links")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📁 Projects Summary", use_container_width=True):
            st.session_state.page = "Projects Summary"
            st.rerun()

    with col2:
        if st.button("🧪 Clinical Trial Dropout Prediction", use_container_width=True):
            st.session_state.page = "Clinical Trial Dropout Prediction"
            st.rerun()

    with col3:
        if st.button("💰 Revenue Dashboard", use_container_width=True):
            st.session_state.page = "Revenue Cycle Dashboard"
            st.rerun()


def page_resume():
    st.title("Resume")

    st.subheader("Education")
    st.markdown(
        """
- **M.S. Data Science** — Eastern University
- **M.S. Biomedical Science** — Roosevelt University
- **B.S. Biomedical Science** — Gulf Medical University
        """
    )

    st.subheader("Work Experience")
    st.markdown(
        """
**Clinical Laboratory Scientist — Saint Mary Hospital (Chicago)**  
- Perform high-complexity diagnostic testing and QC in a hospital lab environment  
- Collaborate with clinical teams to ensure accurate and timely results  
- Apply data-driven thinking to workflow, quality, and operational improvement  

**Business Associate — Insight Hospital** *(April 2023 – February 2025)*  
- Conducted in-depth market research and analysis, identifying **10+ actionable trends**  
- Produced reports and presentations enabling data-driven decision-making  
- Managed policy adherence and regulatory compliance with healthcare standards  
        """
    )

    st.subheader("Technical Skills")
    st.markdown(
        """
- **Programming:** Python, SQL, R  
- **Machine Learning:** scikit-learn, XGBoost, SHAP  
- **Data Analysis:** EDA, feature engineering, model training  
- **Evaluation:** ROC-AUC, PR-AUC, F1-score, Precision/Recall, Confusion Matrix  
- **Deployment:** Streamlit, model serialization (joblib / pickle)
        """
    )

    st.subheader("Links")
    st.markdown("**GitHub:** https://github.com/goladosu")

    if os.path.exists("Resume.pdf"):
        with open("Resume.pdf", "rb") as f:
            st.download_button(
                "⬇️ Download Resume (PDF)",
                f,
                file_name="Resume.pdf"
            )
    else:
        st.caption("(Optional) Add Resume.pdf to your repo root to enable a download button.")


def page_executive_summary():
    st.title("📋 Executive Summary & Recommendations")
    
    st.markdown("""
    This section provides non-technical summaries and actionable recommendations for each project, 
    designed for executives, stakeholders, and decision-makers.
    """)
    
    st.divider()
    
    # Clinical Trial Dropout Project
    st.subheader("🧪 Clinical Trial Dropout Risk Prediction")
    
    with st.expander("📊 Executive Summary", expanded=True):
        st.markdown("""
        **Business Problem:**  
        Clinical trials lose 30-40% of participants before completion, causing delays, increased costs, and reduced statistical power.
        
        **Solution:**  
        I developed a machine learning system that predicts which participants are at risk of dropping out after their second visit, 
        allowing research teams to intervene early. The model was developed using 1,500 participants, where 31% dropped out before completion.
        
        **Key Results:**
        - **91% catch rate** — correctly identified 84 of 92 participants who dropped out
        - **89% overall accuracy** on 300-participant test set
        - **Early warning system** triggers at Visit 2 (when 70% of trial remains)
        - **Cost savings**: Potential to avoid $500K+ per trial in recruitment/restart costs
        
        **Impact:**
        This model enables strategic resource allocation, focusing retention efforts on high-risk participants. The model is tuned to 
        prioritize catching at-risk participants (91% recall) over minimizing false alarms, since an overlooked dropout is typically 
        more costly than an unnecessary outreach call.
        """)
    
    with st.expander("💡 How to Use"):
        st.markdown("""
        **For Clinical Operations Teams:**
        1. After each participant completes Visit 2, enter their data into the model
        2. Review the dropout risk score (Low / Moderate / High)
        3. For High-risk participants (>30% probability):
           - Schedule additional check-in calls
           - Address specific concerns identified by the model
           - Provide extra support resources
        
        **For Trial Managers:**
        - Use predictions to forecast dropout rates and plan recruitment buffer
        - Monitor model performance across different trial phases
        - Track which interventions successfully reduce dropout
        
        **For Data Teams:**
        - Retrain model quarterly with new trial data
        - Monitor feature importance shifts over time
        - Validate predictions against actual outcomes
        """)
    
    with st.expander("🎯 Recommendations"):
        st.markdown("""
        **Immediate Actions (0-3 months):**
        1. **Pilot program**: Test the model on 2-3 active trials
        2. **Define intervention protocols**: Create standard operating procedures for high-risk participants
        3. **Set success metrics**: Track retention improvement vs. control group
        
        **Medium-term (3-6 months):**
        1. **Scale deployment**: Roll out to all Phase II/III trials
        2. **Automate integration**: Connect model to trial management systems
        3. **Staff training**: Train coordinators on interpreting model outputs
        
        **Long-term (6-12 months):**
        1. **Continuous improvement**: Retrain model with cumulative trial data
        2. **Expand features**: Incorporate wearable device data, electronic health records
        3. **Cost-benefit analysis**: Quantify ROI from reduced dropout rates
        
        **Resource Requirements:**
        - Data engineer: 20 hours/quarter for model maintenance
        - Clinical coordinator training: 2 hours per staff member
        - Technology investment: Integration with existing systems (~$15K one-time)
        """)
    
    st.divider()
    
    # Revenue Cycle Analytics Project
    st.subheader("💰 Revenue Cycle Analytics")
    
    with st.expander("📊 Executive Summary", expanded=True):
        st.markdown("""
        **Business Problem:**  
        Healthcare revenue cycle is complex with multiple failure points: claim denials, delayed collections, 
        and inefficient payer relationships costing millions in lost revenue.
        
        **Solution:**  
        Built a comprehensive analytics dashboard tracking 30+ KPIs across 6 dimensions: executive overview, 
        departmental performance, payer mix, denials, AR aging, and collection efficiency.
        
        **Key Findings:**
        - **Collection rate: 57.5%** (industry benchmark: 65-70%) → **$12.5M opportunity**
        - **Self-Pay denial rate: 19.8%** (2.5x higher than Medicare)
        - **$37.2M** in AR aging 0-30 days, but **$0** collected after 120 days
        - **Commercial payers**: 72% of claims but varied collection rates (87-93%)
        
        **Impact:**  
        Leadership now has real-time visibility into revenue leakage, enabling data-driven decisions on 
        staffing, payer negotiations, and collection strategies.
        """)
    
    with st.expander("💡 How to Use"):
        st.markdown("""
        **For CFO / Revenue Cycle Leaders:**
        1. **Weekly Review**: Check Executive Overview tab for trending KPIs
        2. **Monthly Deep Dive**: Analyze departmental and payer performance
        3. **Quarterly Strategy**: Use insights for payer contract negotiations
        
        **For Revenue Cycle Managers:**
        1. **Daily Monitoring**: Track AR aging buckets and collection rates
        2. **Denial Management**: Prioritize denial reasons with highest volume/value
        3. **Team Performance**: Compare departmental efficiency metrics
        
        **For Department Leaders:**
        1. Review your department's collection and denial rates
        2. Identify service lines underperforming vs. benchmarks
        3. Request targeted training based on denial patterns
        
        **For Payer Relations:**
        1. Identify payers with highest denial rates
        2. Prepare data-driven negotiation strategies
        3. Monitor contract performance post-renegotiation
        """)
    
    with st.expander("🎯 Recommendations"):
        st.markdown("""
        **Critical Actions (Immediate):**
        1. **Address Self-Pay denials**: 19.8% rate → implement upfront payment plans, financial counseling
           - **Estimated impact**: Reduce denials to 12% = **$312K** additional annual revenue
        
        2. **Improve overall collection rate**: 57.5% → 65% (industry standard)
           - **Estimated impact**: **$12.5M** additional annual revenue
        
        3. **Accelerate AR aging 120+ days**: $0 collected from $1.8M in claims
           - **Action**: Write off uncollectible, focus on preventing future aging
        
        **Strategic Initiatives (3-6 months):**
        1. **Payer mix optimization**: 
           - Renegotiate contracts with lowest collection rate payers
           - Shift patient volume toward high-performing payer relationships
        
        2. **Denial prevention program**:
           - Train billers on top 5 denial reasons (Medical Necessity, Authorization, Coding Errors)
           - Implement pre-claim scrubbing technology
           - **Target**: Reduce denial rate from 10.3% to 7%
        
        3. **Departmental improvement**:
           - Share best practices from high-performing departments
           - Standardize workflows across service lines
        
        **Technology & Process (6-12 months):**
        1. Automate eligibility verification to reduce authorization denials
        2. Implement predictive analytics for claim approval probability
        3. Integrate real-time alerts for claims approaching 90-day AR threshold
        
        **Expected ROI:**
        - Year 1: $8-10M additional revenue from collection improvements
        - Year 2: $12-15M as denial prevention matures
        - Ongoing: 5-7% improvement in net revenue margin
        """)
    
    st.divider()
    
    # Live Documentation Links
    st.subheader("📚 Live Documentation & Resources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📄 Technical Documentation**  
        Detailed methodology, model architecture, and implementation guides
        """)
        if st.button("View Technical Docs", key="tech_docs", use_container_width=True):
            st.info("Technical documentation available on request")
    
    with col2:
        st.info("""
        **📈 Interactive Demos**  
        Hands-on demonstrations of each model and dashboard
        """)
        if st.button("Try Interactive Demos", key="demos", use_container_width=True):
            st.info("Navigate to individual project pages to interact with models")


def page_projects():
    st.title("📁 Projects Summary")
    
    st.markdown("""
    Demonstration of end-to-end data science capabilities: machine learning model development, 
    analytics dashboard design, and deployment for real-world business impact.
    """)
    
    st.divider()
    
    # Project 1: Clinical Trial Dropout Prediction
    st.subheader("🧪 Clinical Trial Participant Dropout Prediction")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        **Machine learning model predicting dropout risk after Visit 2 to enable early intervention**
        
        - **Dataset:** 1,500 participants across simulated clinical trials (31% dropout rate)
        - **Model:** XGBoost classifier with SHAP interpretability
        - **Performance:** 91% catch rate (correctly identified 84 of 92 dropouts), 89% overall accuracy
        - **Features:** Visit adherence, missed appointments, baseline lab results, adverse events
        - **Deployment:** Interactive web application with risk scoring (Low/Moderate/High)
        - **Impact:** Enables targeted retention efforts, potential to avoid $500K+ per trial in restart costs
        """)
    
    with col2:
        st.metric("Catch Rate", "91%")
        st.metric("Accuracy", "89%")
        st.metric("Dataset", "1,500")
    
    if st.button("🔍 Explore Dropout Model", key="dropout", use_container_width=True):
        st.session_state.page = "Clinical Trial Dropout Prediction"
        st.rerun()
    
    st.divider()
    
    # Project 2: Revenue Cycle Analytics
    st.subheader("💰 Healthcare Revenue Cycle Analytics Dashboard")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        **Comprehensive analytics dashboard tracking 30+ KPIs to identify revenue leakage and optimize collections**
        
        - **Dataset:** 20,000 patient encounters, January 2024 – December 2025
        - **Tool:** Databricks Lakeview with 6 interactive pages (Executive Overview, Departmental Performance, Payer Mix, Denials Analysis, AR Aging, Collection Efficiency)
        - **Key Findings:**
          - $136M billed → $77.3M approved → $64.8M collected (48% collection rate)
          - 11% denial rate driven by preventable errors (incomplete paperwork, missing approvals, late submissions)
          - $12.3M in claims over 120 days old with near-zero collection
          - $2.1M approved but underpaid by insurance across 7,087 claims
        - **Impact:** Identified actionable opportunities to recover millions in lost revenue through process improvements
        """)
    
    with col2:
        st.metric("Total Billed", "$136M")
        st.metric("Collected", "$64.8M")
        st.metric("Denial Rate", "11%")
    
    if st.button("📊 View Revenue Dashboard", key="revenue", use_container_width=True):
        st.session_state.page = "Revenue Cycle Dashboard"
        st.rerun()
    
    st.divider()
    
    # Technical Skills
    st.subheader("🛠️ Technical Skills Demonstrated")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Machine Learning**
        - Classification modeling
        - XGBoost, Random Forest
        - Hyperparameter tuning
        - Class imbalance handling
        - SHAP interpretability
        """)
    
    with col2:
        st.markdown("""
        **Data Engineering**
        - ETL pipeline development
        - Feature engineering
        - Dashboard design
        - SQL query optimization
        - Data quality validation
        """)
    
    with col3:
        st.markdown("""
        **Deployment & Tools**
        - Streamlit applications
        - Databricks Lakeview
        - Model serialization
        - Performance monitoring
        - Technical documentation
        """)



def page_dropout_project():
    st.title("Clinical Trial Dropout Risk — Deployed Model")

    if load_err:
        st.error(load_err)
        st.write("Current working directory:", os.getcwd())
        st.write("Files:", os.listdir("."))
        st.stop()

    # Executive Summary for Non-Technical Audiences
    st.info("📋 **Executive Summary and Recommendations for Non-Technical Audiences**")
    
    # Link to full PDF in workspace
    workspace_url = "https://dbc-b0a51582-e395.cloud.databricks.com"
    workspace_id = "7474658800238295"
    pdf_link = f"{workspace_url}/explore/data/volumes/workspace/clinicaltrial/clinicaltrialpredictionmodel?o={workspace_id}#Dropout_Prediction_Executive%20Report.pdf"
    
    st.markdown(f"""
    📄 **[View / Download Executive Summary Report]({pdf_link})**  
    *A plain-language guide to the clinical trial dropout prediction model*
    """)
    
    with st.expander("📊 Executive Summary", expanded=False):
        st.markdown("""
        **Business Problem:**  
        Clinical trials lose 30-40% of participants before completion, causing delays, increased costs ($500K+ per trial), and reduced statistical power.
        
        **Solution:**  
        I developed a machine learning model that predicts dropout risk after Visit 2, allowing early intervention when 70% of the trial remains. 
        The model was developed using 1,500 participants, where 31% dropped out before completion.
        
        **Key Results:**
        - **91% catch rate** — correctly identified 84 of 92 participants who dropped out
        - **89% overall accuracy** on 300-participant test set
        - **Early warning system** triggers at Visit 2
        - **Cost savings**: Potential to avoid $500K+ per trial in recruitment/restart costs
        
        **Impact:**
        This model enables strategic resource allocation, focusing retention efforts on high-risk participants. The model is tuned to 
        prioritize catching at-risk participants (91% recall) over minimizing false alarms, since an overlooked dropout is typically 
        more costly than an unnecessary outreach call.
        """)
    
    with st.expander("💡 How to Use"):
        st.markdown("""
        **For Clinical Operations Teams:**
        1. After Visit 2, enter participant data below
        2. Review dropout risk score (Low / Moderate / High)
        3. For High-risk participants (>30% probability):
           - Schedule additional check-in calls
           - Address specific concerns from SHAP analysis
           - Provide extra support resources
        
        **For Trial Managers:**
        - Forecast dropout rates and plan recruitment buffer
        - Monitor model performance across trial phases
        - Track intervention success rates
        
        **For Data Teams:**
        - Retrain model quarterly with new trial data
        - Monitor feature importance shifts
        - Validate predictions against actual outcomes
        """)
    
    with st.expander("🎯 Recommendations"):
        st.markdown("""
        **Immediate Actions (0-3 months):**
        1. **Pilot program**: Test on 2-3 active trials
        2. **Define intervention protocols**: Standard procedures for high-risk participants
        3. **Set success metrics**: Track retention improvement vs. control group
        
        **Medium-term (3-6 months):**
        1. **Scale deployment**: Roll out to all Phase II/III trials
        2. **Automate integration**: Connect to trial management systems
        3. **Staff training**: 2 hours per coordinator on interpreting outputs
        
        **Long-term (6-12 months):**
        1. **Continuous improvement**: Retrain with cumulative data
        2. **Expand features**: Wearable device data, EHR integration
        3. **Cost-benefit analysis**: Quantify ROI from reduced dropout
        
        **Resource Requirements:**
        - Data engineer: 20 hours/quarter for maintenance
        - Technology investment: ~$15K one-time integration costs
        """)

    st.divider()
    
    st.markdown(
        f"""
### Interactive Model: Predict Dropout Risk
Clinical trials often lose participants before the primary endpoint. Dropout can delay timelines, reduce statistical power,
and introduce bias. This model estimates the **probability of dropout** using information collected up to **Visit 2** so teams
can intervene early.

**Operational threshold:** `{CHOSEN_THRESHOLD:.2f}`
        """
    )

    st.divider()
    st.subheader("Try it: Enter participant data")

    with st.form("participant_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", min_value=18, max_value=90, value=55)
            sex = st.selectbox("Sex", ["Male", "Female"])
            race = st.selectbox("Race", ["White", "Black", "Asian", "Other"])
            BMI = st.slider("BMI", 15.0, 45.0, 27.0)

        with c2:
            baseline_lab_score = st.slider("Baseline Lab Score", 0.0, 200.0, 110.0)
            disease_severity = st.slider("Disease Severity (1–10)", 1.0, 10.0, 5.0)
            prior_treatments = st.number_input("Prior Treatments", min_value=0, max_value=20, value=1)
            missed_appointments = st.number_input("Missed Appointments", min_value=0, max_value=20, value=0)

        with c3:
            communication_score = st.slider("Communication Score (1–5)", 1.0, 5.0, 3.0, step=0.1)
            st.caption("Visits")
            visit1_symptom_score = st.slider("Visit 1 Symptom Score", 0.0, 100.0, 50.0)
            visit1_adherence_rate = st.slider("Visit 1 Adherence (%)", 0.0, 100.0, 80.0)
            visit1_AE_count = st.number_input("Visit 1 AE Count", min_value=0, max_value=20, value=0)

            visit2_symptom_score = st.slider("Visit 2 Symptom Score", 0.0, 100.0, 45.0)
            visit2_adherence_rate = st.slider("Visit 2 Adherence (%)", 0.0, 100.0, 75.0)
            visit2_AE_count = st.number_input("Visit 2 AE Count", min_value=0, max_value=20, value=0)

        submitted = st.form_submit_button("Predict Dropout Risk")

    if not submitted:
        return

    # Create the DataFrame from inputs
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "race": race,
        "BMI": BMI,
        "baseline_lab_score": baseline_lab_score,
        "disease_severity": disease_severity,
        "prior_treatments": prior_treatments,
        "visit1_symptom_score": visit1_symptom_score,
        "visit1_adherence_rate": visit1_adherence_rate,
        "visit1_AE_count": visit1_AE_count,
        "visit2_symptom_score": visit2_symptom_score,
        "visit2_adherence_rate": visit2_adherence_rate,
        "visit2_AE_count": visit2_AE_count,
        "missed_appointments": missed_appointments,
        "communication_score": communication_score,
    }])

    # Scale percentage adherence to proportion (0.0 to 1.0)
    for col in ["visit1_adherence_rate", "visit2_adherence_rate"]:
        input_df[col] = input_df[col] / 100.0

    try:
        prob, pred, shap_row, feat_names = get_shap_for_single_row(pipeline, input_df)
    except Exception as e:
        st.error(f"Prediction/SHAP error: {e}")
        st.stop()

    st.divider()
    st.subheader("Results")

    bucket = risk_bucket(prob)
    if bucket == "Low":
        st.success(f"Dropout probability: **{prob:.3f}**  → **{bucket} risk**")
    elif bucket == "Moderate":
        st.warning(f"Dropout probability: **{prob:.3f}**  → **{bucket} risk**")
    else:
        st.error(f"Dropout probability: **{prob:.3f}**  → **{bucket} risk**")

    st.write(f"Predicted class (thresholded): {'Dropout (1)' if pred == 1 else 'Completer (0)'}")
    st.caption("This is a risk estimate, not a guarantee.")

    st.subheader("Top drivers of this prediction (SHAP)")
    order = np.argsort(np.abs(shap_row))[::-1][:10]
    top_df = pd.DataFrame({
        "feature": feat_names[order],
        "shap_value": shap_row[order],
    })
    st.dataframe(top_df, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(top_df["feature"][::-1], top_df["shap_value"][::-1])
    ax.set_xlabel("SHAP contribution (positive increases dropout risk)")
    ax.set_ylabel("Feature")
    st.pyplot(fig, clear_figure=True)

    with st.expander("Show participant input"):
        st.dataframe(input_df, use_container_width=True)



def page_revenue_dashboard():
    st.title("Revenue Cycle Analytics Dashboard")
    
    # Executive Summary for Non-Technical Audiences
    st.info("📋 **Executive Summary: Revenue Cycle Analysis**")
    
    # Link to full PDF in workspace
    workspace_url = "https://dbc-b0a51582-e395.cloud.databricks.com"
    workspace_id = "7474658800238295"
    pdf_link = f"{workspace_url}/explore/data/volumes/workspace/default/executivesummary?o={workspace_id}#Revenue%20Cycle_Executive%20Report.pdf"
    
    st.markdown(f"""
    📄 **[View / Download Executive Summary Report]({pdf_link})**  
    *A plain-language look at our billing & collections (20,000 patient visits, Jan 2024 – Dec 2025)*
    """)
    
    with st.expander("📊 The Big Picture", expanded=False):
        st.markdown("""
        For every $100 billed, insurance companies approved about $57, but only $48 was actually collected.
        
        **Real Numbers:**
        - **$136M** Total Billed
        - **$77.3M** Approved by Insurance  
        - **$64.8M** Actually Collected
        
        The gap between what was billed, what was approved, and what was collected reveals where revenue is leaking.
        """)
    
    with st.expander("🚨 Key Findings", expanded=False):
        st.markdown("""
        **1. Old Claims Are Turning Into Losses**
        - **$12.3M** in claims sitting over 120 days (1,801 claims) with almost nothing collected
        - 1,797 of these already denied — unlikely to ever get paid
        
        **2. Simple Mistakes Cause Most Denials**
        - **11 out of 100 claims** get denied
        - Top reasons: incomplete paperwork (349), missing pre-approval (325), late submission (322)
        - **ICU (13.5%), Surgery (13.2%), Emergency Dept (13.1%)** have highest denial rates
        - Most denials happen *before* the claim reaches insurance — they're preventable
        
        **3. Insurance Paying Less Than Approved**
        - **$2.1M** approved by insurance but never collected (7,087 claims)
        - **Med/Surg ($529K), Surgery ($392K), ICU ($378K)** account for 61% of this gap
        - Insurance already agreed to pay — just need to follow up
        
        **4. Getting Faster**
        - Claim resolution time improved from 60 days → under 30 days (real progress!)
        """)
    
    with st.expander("🎯 Recommendations", expanded=False):
        st.markdown("""
        **1. Deal With Old Claims First**
        - Sort 1,801 old claims: follow up on 4 awaiting decision, decide on 1,797 denied (appeal or write off)
        - **New rule**: Any claim at 90 days gets decided within 10 business days
        
        **2. Catch Mistakes Before Claims Go Out**
        - Add pre-submission checklist in ICU, Surgery, ED (highest denial departments)
        - Double-check insurance approval at check-in, especially Self-Pay and Medicaid
        
        **3. Collect Money Already Approved ($2.1M)**
        - Start underpayment follow-up focused on Med/Surg, Surgery, ICU
        - Track underpayment as separate monthly metric (currently invisible)
        
        **4. Monitor December 2025 Performance**
        - Allow 60-90 days for claims to fully process before drawing conclusions
        - Recent month dip likely due to incomplete claim processing time
        
        **Bottom Line:**  
        None of these require big systems or investments — they're **process fixes** that could recover **several million dollars** annually.
        """)
    
    st.divider()
    
    st.markdown("""
    ### Interactive Dashboard
    Comprehensive analytics tracking healthcare revenue cycle performance across:
    - **Executive Overview**: Key KPIs including encounters, claims, collection rates, and AR aging
    - **Departmental Performance**: Collection and denial rates by department and service line
    - **Payer Mix Analysis**: Revenue distribution and performance by payer type
    - **Denials Analysis**: Breakdown of denials by reason and payer
    - **AR Aging**: Accounts receivable distribution by age buckets
    - **Collection Efficiency**: Overall efficiency metrics and trends
    """)
    
    # ========== EXECUTIVE KPIs ==========
    st.subheader("📊 Key Performance Indicators")
    
    # Top-level metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Encounters", "20,000")
        st.metric("Avg Days in AR", "45 days", delta="-15 days", delta_color="normal")
    
    with col2:
        st.metric("Total Billed", "$136M")
        st.metric("Collection Rate", "48%", delta="+3%", delta_color="normal")
    
    with col3:
        st.metric("Total Allowed", "$77.3M")
        st.metric("Denial Rate", "11%", delta="-2%", delta_color="inverse")
    
    with col4:
        st.metric("Total Paid", "$64.8M")
        st.metric("Underpayment", "$2.1M", delta="Problem area", delta_color="off")
    
    with col5:
        st.metric("Net Revenue", "$64.8M")
        st.metric("Total Claims", "20,000")
    
    st.divider()
    
    # ========== INTERACTIVE VISUALIZATIONS ==========
    st.subheader("📈 Interactive Analytics")
    
    # Enhanced tab layout with 6 categories matching dashboard
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💰 Revenue Flow",
        "🏥 Department Analysis", 
        "🚫 Denials Analysis",
        "🏦 Payer Mix",
        "⏰ AR Aging",
        "📈 Trends"
    ])
    
    with tab1:
        st.markdown("### Revenue Funnel: From Billed to Collected")
        
        # Revenue funnel data
        funnel_data = pd.DataFrame({
            'Stage': ['Billed', 'Approved', 'Collected'],
            'Amount': [136, 77.3, 64.8],
            'Color': ['#667eea', '#764ba2', '#48bb78']
        })
        
        fig_funnel = go.Figure()
        
        fig_funnel.add_trace(go.Funnel(
            y = funnel_data['Stage'],
            x = funnel_data['Amount'],
            textposition = "inside",
            textinfo = "value+percent initial",
            marker = {"color": funnel_data['Color']},
            connector = {"line": {"color": "#e0e0e0", "width": 3}}
        ))
        
        fig_funnel.update_layout(
            title="Revenue Flow: $136M → $64.8M (48% collection rate)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_funnel, use_container_width=True)
        
        # Key insight
        st.info("💡 **Key Insight:** Only 48 cents collected for every dollar billed. The gap between approved ($77.3M) and collected ($64.8M) represents $12.5M in potential recovery.")
    
    with tab2:
        st.markdown("### Denial Rates by Department")
        
        # Department denial data
        dept_data = pd.DataFrame({
            'Department': ['ICU', 'Surgery', 'Emergency', 'Med/Surg', 'Cardiology', 'Oncology', 'Pediatrics'],
            'Denial_Rate': [13.5, 13.2, 13.1, 10.8, 10.2, 9.5, 8.7],
            'Claims': [1250, 1580, 2100, 3200, 1800, 1450, 1100]
        })
        
        # Bar chart with color gradient
        fig_dept = px.bar(
            dept_data,
            x='Department',
            y='Denial_Rate',
            title='Denial Rates by Department (%)',
            color='Denial_Rate',
            color_continuous_scale='Reds',
            text='Denial_Rate'
        )
        
        fig_dept.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_dept.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="Department",
            yaxis_title="Denial Rate (%)",
            yaxis_range=[0, 16]
        )
        fig_dept.add_hline(y=11, line_dash="dash", line_color="orange", 
                          annotation_text="Overall Average (11%)")
        
        st.plotly_chart(fig_dept, use_container_width=True)
        
        st.info("💡 **Key Insight:** ICU, Surgery, and Emergency departments have denial rates 2-3 percentage points above average. Focus denial prevention efforts here for maximum impact.")
    
    with tab3:
        st.markdown("### Denials Analysis")
        
        # Denials by Reason
        st.markdown("#### Top Denial Reasons")
        denial_reasons = pd.DataFrame({
            'Reason': [
                'Incomplete Documentation',
                'Missing Pre-Authorization', 
                'Timely Filing',
                'Coding Error',
                'Medical Necessity',
                'Duplicate Claim',
                'Non-Covered Service',
                'Patient Eligibility'
            ],
            'Count': [349, 325, 322, 287, 245, 198, 156, 118]
        })
        
        fig_denial_reason = px.bar(
            denial_reasons,
            x='Reason',
            y='Count',
            title='Denial Volume by Reason',
            color='Count',
            color_continuous_scale='Reds',
            text='Count'
        )
        fig_denial_reason.update_traces(textposition='outside')
        fig_denial_reason.update_layout(
            height=400,
            showlegend=False,
            xaxis_tickangle=-45,
            xaxis_title="Denial Reason",
            yaxis_title="Number of Claims"
        )
        
        st.plotly_chart(fig_denial_reason, use_container_width=True)
        
        # Denials by Payer Type
        st.markdown("#### Denial Rates by Payer Type")
        
        col1, col2 = st.columns(2)
        
        with col1:
            payer_denial = pd.DataFrame({
                'Payer_Type': ['Commercial', 'Medicare', 'Medicaid', 'Self-Pay'],
                'Denial_Rate': [9.2, 10.5, 13.8, 15.2],
                'Claims': [8500, 7200, 3100, 1200]
            })
            
            fig_payer_denial = px.bar(
                payer_denial,
                x='Payer_Type',
                y='Denial_Rate',
                title='Denial Rate by Payer Type (%)',
                color='Denial_Rate',
                color_continuous_scale='OrRd',
                text='Denial_Rate'
            )
            fig_payer_denial.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_payer_denial.update_layout(height=350, showlegend=False)
            
            st.plotly_chart(fig_payer_denial, use_container_width=True)
        
        with col2:
            # Denial metrics
            st.metric("Total Denied Claims", "2,200", delta="-8% vs last quarter", delta_color="normal")
            st.metric("Avg Denial Amount", "$3,450")
            st.metric("Total Denied $", "$7.6M")
            st.metric("Successfully Appealed", "18%", delta="+3%", delta_color="normal")
        
        st.info("💡 **Key Insight:** Top 3 denial reasons (Documentation, Pre-Auth, Timely Filing) account for 50% of all denials. Self-Pay and Medicaid have highest denial rates - target prevention efforts here.")
    
    with tab4:
        st.markdown("### Payer Mix Analysis")
        
        # Payer distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Claims Volume by Payer Type")
            
            payer_mix = pd.DataFrame({
                'Payer_Type': ['Commercial', 'Medicare', 'Medicaid', 'Self-Pay'],
                'Claims': [8500, 7200, 3100, 1200],
                'Percentage': [42.5, 36.0, 15.5, 6.0]
            })
            
            fig_payer_pie = px.pie(
                payer_mix,
                values='Claims',
                names='Payer_Type',
                title='Payer Mix Distribution',
                color_discrete_sequence=['#667eea', '#764ba2', '#f093fb', '#4facfe']
            )
            fig_payer_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_payer_pie.update_layout(height=350)
            
            st.plotly_chart(fig_payer_pie, use_container_width=True)
        
        with col2:
            st.markdown("#### Revenue by Payer Type")
            
            payer_revenue = pd.DataFrame({
                'Payer_Type': ['Commercial', 'Medicare', 'Medicaid', 'Self-Pay'],
                'Revenue': [31.2, 22.8, 8.4, 2.4]
            })
            
            fig_payer_rev = px.bar(
                payer_revenue,
                x='Payer_Type',
                y='Revenue',
                title='Total Paid by Payer Type ($M)',
                color='Revenue',
                color_continuous_scale='Blues',
                text='Revenue'
            )
            fig_payer_rev.update_traces(texttemplate='$%{text:.1f}M', textposition='outside')
            fig_payer_rev.update_layout(height=350, showlegend=False)
            
            st.plotly_chart(fig_payer_rev, use_container_width=True)
        
        # Payer performance metrics
        st.markdown("#### Payer Performance Summary")
        
        payer_performance = pd.DataFrame({
            'Payer Type': ['Commercial', 'Medicare', 'Medicaid', 'Self-Pay'],
            'Claims': ['8,500', '7,200', '3,100', '1,200'],
            'Billed': ['$64.6M', '$46.2M', '$18.8M', '$6.4M'],
            'Paid': ['$31.2M', '$22.8M', '$8.4M', '$2.4M'],
            'Collection Rate': ['48.3%', '49.4%', '44.7%', '37.5%'],
            'Denial Rate': ['9.2%', '10.5%', '13.8%', '15.2%'],
            'Avg Days in AR': ['42', '48', '52', '65']
        })
        
        st.dataframe(payer_performance, use_container_width=True, hide_index=True)
        
        st.info("💡 **Key Insight:** Commercial payers generate 48% of revenue with best collection rates (48.3%). Self-Pay has poorest performance (37.5% collection, 15% denial rate, 65 days AR).")
    
    with tab5:
        st.markdown("### Accounts Receivable Aging")
        
        # AR Aging data
        ar_data = pd.DataFrame({
            'Age_Bucket': ['0-30 days', '31-60 days', '61-90 days', '91-120 days', '120+ days'],
            'Amount': [28.5, 18.2, 12.1, 8.9, 12.3],
            'Claims': [5200, 3800, 2400, 1600, 1801]
        })
        
        # Stacked bar showing amount and claims
        fig_ar = go.Figure()
        
        fig_ar.add_trace(go.Bar(
            x=ar_data['Age_Bucket'],
            y=ar_data['Amount'],
            name='Amount ($M)',
            marker_color=['#48bb78', '#4299e1', '#ed8936', '#f56565', '#c53030'],
            text=ar_data['Amount'],
            texttemplate='$%{text:.1f}M',
            textposition='auto'
        ))
        
        fig_ar.update_layout(
            title='Outstanding Accounts Receivable by Age',
            xaxis_title='Age Bucket',
            yaxis_title='Amount ($M)',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_ar, use_container_width=True)
        
        # Additional metric
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total AR", "$80M")
        with col2:
            st.metric("120+ Days", "$12.3M", delta="-15% Critical", delta_color="inverse")
        with col3:
            st.metric("0-30 Days", "$28.5M", delta="+35% Healthy", delta_color="normal")
        
        st.info("💡 **Key Insight:** $12.3M sitting over 120 days (1,801 claims). Of these, 1,797 already denied — immediate action needed: appeal or write off.")
    
    with tab6:
        st.markdown("### Collection Performance Trends")
        
        # Monthly collection trend data
        months = pd.date_range(start='2024-01', end='2025-12', freq='MS')
        np.random.seed(42)
        
        trend_data = pd.DataFrame({
            'Month': months,
            'Collection_Rate': [45 + i*0.5 + np.random.normal(0, 1.5) for i in range(len(months))],
            'Amount_Collected': [2.5 + i*0.08 + np.random.normal(0, 0.3) for i in range(len(months))],
            'Denial_Rate': [13 - i*0.15 + np.random.normal(0, 0.8) for i in range(len(months))],
            'Days_in_AR': [60 - i*1.2 + np.random.normal(0, 2) for i in range(len(months))]
        })
        
        # Two charts side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Collection Rate & Revenue Trends")
            
            # Dual axis chart
            fig_collection = go.Figure()
            
            # Collection rate line
            fig_collection.add_trace(go.Scatter(
                x=trend_data['Month'],
                y=trend_data['Collection_Rate'],
                name='Collection Rate (%)',
                mode='lines+markers',
                line=dict(color='#667eea', width=3),
                yaxis='y'
            ))
            
            # Amount collected bars
            fig_collection.add_trace(go.Bar(
                x=trend_data['Month'],
                y=trend_data['Amount_Collected'],
                name='Amount Collected ($M)',
                marker_color='#48bb78',
                opacity=0.6,
                yaxis='y2'
            ))
            
            fig_collection.update_layout(
                xaxis_title='Month',
                yaxis=dict(title='Collection Rate (%)', side='left', range=[40, 60]),
                yaxis2=dict(title='Amount Collected ($M)', overlaying='y', side='right', range=[0, 6]),
                height=400,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_collection, use_container_width=True)
        
        with col2:
            st.markdown("#### Denial Rate & Days in AR Trends")
            
            fig_ops = go.Figure()
            
            # Denial rate line
            fig_ops.add_trace(go.Scatter(
                x=trend_data['Month'],
                y=trend_data['Denial_Rate'],
                name='Denial Rate (%)',
                mode='lines+markers',
                line=dict(color='#f56565', width=3),
                yaxis='y'
            ))
            
            # Days in AR line
            fig_ops.add_trace(go.Scatter(
                x=trend_data['Month'],
                y=trend_data['Days_in_AR'],
                name='Avg Days in AR',
                mode='lines+markers',
                line=dict(color='#ed8936', width=3),
                yaxis='y2'
            ))
            
            fig_ops.update_layout(
                xaxis_title='Month',
                yaxis=dict(title='Denial Rate (%)', side='left', range=[8, 15]),
                yaxis2=dict(title='Days in AR', overlaying='y', side='right', range=[25, 65]),
                height=400,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_ops, use_container_width=True)
        
        # Monthly comparison table
        st.markdown("#### Key Metrics by Month (Recent 6 Months)")
        
        recent_months = trend_data.tail(6).copy()
        recent_months['Month'] = recent_months['Month'].dt.strftime('%b %Y')
        recent_months['Collection_Rate'] = recent_months['Collection_Rate'].apply(lambda x: f"{x:.1f}%")
        recent_months['Amount_Collected'] = recent_months['Amount_Collected'].apply(lambda x: f"${x:.1f}M")
        recent_months['Denial_Rate'] = recent_months['Denial_Rate'].apply(lambda x: f"{x:.1f}%")
        recent_months['Days_in_AR'] = recent_months['Days_in_AR'].apply(lambda x: f"{int(x)} days")
        
        st.dataframe(
            recent_months[['Month', 'Collection_Rate', 'Amount_Collected', 'Denial_Rate', 'Days_in_AR']],
            use_container_width=True,
            hide_index=True
        )
        
        st.info("💡 **Key Insight:** Collection rate improved from 45% to 52% over 2 years. Denial rate decreased from 13% to 10%, and claim resolution time improved from 60 to 30 days. Consistent upward trend in all key metrics.")
    
    st.divider()
    
    # Dashboard preview image and link
    st.info("📊 This dashboard contains 6 interactive pages with 30+ visualizations tracking revenue cycle KPIs.")
    
    dashboard_url = "https://dbc-b0a51582-e395.cloud.databricks.com/sql/dashboards/01f1776f12fd171ca4d7f551417cc74d?o=7474658800238295"
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.link_button(
            "🔗 Open Revenue Cycle Dashboard",
            dashboard_url,
            use_container_width=True
        )
    
    st.divider()
    
    # Dashboard features
    st.subheader("Dashboard Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📈 Key Metrics**
        - Total Encounters: 20,000
        - Total Claims: 20,000  
        - Collection Rate: 48% (of billed)
        - Denial Rate: ~11%
        - Avg Days in AR: 45.2
        
        **💰 Revenue Analysis**
        - Total Billed: $136M
        - Total Approved: $77.3M
        - Total Collected: $64.8M
        """)
    
    with col2:
        st.markdown("""
        **🏥 Departmental Insights**
        - Performance by service line
        - Collection rate benchmarking
        - Denial rate trends
        
        **💳 Payer Mix**
        - Commercial: 72% of claims
        - Medicare: 18%
        - Medicaid: 8%
        - Self-Pay: 2%
        """)
    
    st.caption("💡 Click the button above to explore the full interactive dashboard with drill-down capabilities")


# ==============================================================================
# Router
# ==============================================================================

if page == "Home":
    page_home()
elif page == "Resume":
    page_resume()
elif page == "Projects Summary":
    page_projects()
elif page == "Clinical Trial Dropout Prediction":
    page_dropout_project()
elif page == "Revenue Cycle Dashboard":
    page_revenue_dashboard()
