import streamlit as st

st.title("Portfolio Test - Minimal Version")

st.write("✅ Streamlit is working!")

# Test imports one by one
try:
    import numpy as np
    st.write("✅ numpy imported")
except Exception as e:
    st.error(f"❌ numpy failed: {e}")

try:
    import pandas as pd
    st.write("✅ pandas imported")
except Exception as e:
    st.error(f"❌ pandas failed: {e}")

try:
    import plotly.graph_objects as go
    st.write("✅ plotly imported")
except Exception as e:
    st.error(f"❌ plotly failed: {e}")

try:
    import joblib
    st.write("✅ joblib imported")
except Exception as e:
    st.error(f"❌ joblib failed: {e}")

try:
    from sklearn.base import BaseEstimator
    st.write("✅ sklearn imported")
except Exception as e:
    st.error(f"❌ sklearn failed: {e}")

try:
    import xgboost
    st.write("✅ xgboost imported")
except Exception as e:
    st.error(f"❌ xgboost failed: {e}")

try:
    import shap
    st.write("✅ shap imported")
except Exception as e:
    st.error(f"❌ shap failed: {e}")

try:
    from imblearn.over_sampling import SMOTE
    st.write("✅ imbalanced-learn imported")
except Exception as e:
    st.error(f"❌ imbalanced-learn failed: {e}")

st.divider()
st.subheader("File Check")
import os
st.write("Files in directory:", os.listdir("."))
