import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from databricks import sql
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="RCM Denial Rate Forecast",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    </style>
""", unsafe_allow_html=True)

# Databricks connection
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_forecast_data():
    """
    Fetch latest forecast data from Databricks.
    Credentials are stored in Streamlit secrets.
    """
    try:
        # Connect to Databricks SQL warehouse
        connection = sql.connect(
            server_hostname=st.secrets["databricks"]["host"],
            http_path=st.secrets["databricks"]["http_path"],
            access_token=st.secrets["databricks"]["token"]
        )
        
        # Query the forecast table
        query = """
        SELECT 
            forecast_date,
            predicted_denial_rate,
            lower_bound,
            upper_bound,
            model_version,
            created_at
        FROM workspace.default.denial_rate_forecast
        ORDER BY forecast_date
        """
        
        cursor = connection.cursor()
        cursor.execute(query)
        
        # Fetch all results
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(rows, columns=columns)
        
        cursor.close()
        connection.close()
        
        return df
        
    except Exception as e:
        st.error(f"Error connecting to Databricks: {e}")
        return None

# Main app
def main():
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 40px 0;'>
            <h1 style='color: #667eea; font-size: 3rem;'>📊 RCM Denial Rate Forecast</h1>
            <p style='font-size: 1.2rem; color: #666;'>Machine Learning-Powered Revenue Cycle Predictions</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Fetch data
    with st.spinner("Loading forecast data from Databricks..."):
        forecast_df = fetch_forecast_data()
    
    if forecast_df is None or len(forecast_df) == 0:
        st.error("Unable to load forecast data. Please check your Databricks connection.")
        st.info("Make sure your Databricks credentials are set in Streamlit secrets.")
        return
    
    # Extract metadata
    model_version = forecast_df['model_version'].iloc[0]
    last_updated = forecast_df['created_at'].iloc[0]
    forecast_months = len(forecast_df)
    avg_rate = forecast_df['predicted_denial_rate'].mean()
    
    # Display last updated
    st.markdown(f"<p style='text-align: center; color: #666;'>Last Updated: {last_updated}</p>", 
                unsafe_allow_html=True)
    
    # KPI Cards
    st.markdown("### 📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Forecast Horizon",
            value=f"{forecast_months} Months",
            delta="Ahead"
        )
    
    with col2:
        st.metric(
            label="Average Predicted Rate",
            value=f"{avg_rate:.2f}%",
            delta="Next 6 Months"
        )
    
    with col3:
        st.metric(
            label="Model Used",
            value=model_version,
            delta="Walk-Forward Validated"
        )
    
    with col4:
        st.metric(
            label="Confidence Interval",
            value="95%",
            delta="Prediction Bounds"
        )
    
    st.markdown("---")
    
    # Interactive Forecast Chart
    st.markdown("### 📊 6-Month Denial Rate Forecast")
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add predicted line
    fig.add_trace(go.Scatter(
        x=forecast_df['forecast_date'],
        y=forecast_df['predicted_denial_rate'],
        mode='lines+markers',
        name='Predicted Denial Rate',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8)
    ))
    
    # Add confidence interval (shaded area)
    fig.add_trace(go.Scatter(
        x=forecast_df['forecast_date'],
        y=forecast_df['upper_bound'],
        mode='lines',
        name='95% Upper Bound',
        line=dict(color='rgba(245, 101, 101, 0.3)', dash='dash'),
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['forecast_date'],
        y=forecast_df['lower_bound'],
        mode='lines',
        name='95% Lower Bound',
        line=dict(color='rgba(72, 187, 120, 0.3)', dash='dash'),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.1)',
        showlegend=True
    ))
    
    # Update layout
    fig.update_layout(
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            title="Month",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)'
        ),
        yaxis=dict(
            title="Denial Rate (%)",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed Breakdown Table
    st.markdown("### 📋 Detailed Forecast Breakdown")
    
    # Format the dataframe for display
    display_df = forecast_df[['forecast_date', 'predicted_denial_rate', 'lower_bound', 'upper_bound']].copy()
    display_df['forecast_date'] = pd.to_datetime(display_df['forecast_date']).dt.strftime('%B %Y')
    display_df['predicted_denial_rate'] = display_df['predicted_denial_rate'].apply(lambda x: f"{x:.2f}%")
    display_df['confidence_range'] = display_df.apply(
        lambda row: f"{row['lower_bound']:.2f}% - {row['upper_bound']:.2f}%", 
        axis=1
    )
    
    # Display table
    st.dataframe(
        display_df[['forecast_date', 'predicted_denial_rate', 'confidence_range']].rename(columns={
            'forecast_date': 'Month',
            'predicted_denial_rate': 'Predicted Denial Rate',
            'confidence_range': '95% Confidence Range'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # Methodology Section
    st.markdown("### 🔬 Methodology")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📊 Data Source**
        - 20,000 historical claims
        - 24 months of denial data
        - Monthly aggregation
        
        **🤖 Models Tested**
        - Baseline (3-Month Moving Average)
        - Prophet (Facebook's forecaster)
        - ARIMA (Auto-tuned)
        - SARIMA (Seasonal ARIMA)
        - XGBoost (Machine Learning)
        """)
    
    with col2:
        st.markdown("""
        **✅ Validation**
        - Walk-forward cross-validation
        - Expanding training windows
        - 12 test folds
        
        **🎯 Selection Criteria**
        - Lowest Mean Absolute Percentage Error (MAPE)
        - 95% confidence intervals
        - Production-ready pipeline
        """)
    
    st.markdown("---")
    
    # Technical Details (Expandable)
    with st.expander("🛠️ Technical Details"):
        st.markdown("""
        **Platform**: Databricks with PySpark, Prophet, and XGBoost
        
        **Automation**: 
        - Databricks Job runs every Monday at midnight
        - Retrains all 5 models on latest data
        - Selects best performer automatically
        - Updates forecast table for real-time access
        
        **Data Pipeline**:
        1. Extract claims data from warehouse
        2. Aggregate to monthly denial rates
        3. Train and validate 5 forecasting models
        4. Select model with lowest MAPE
        5. Generate 6-month forecast with confidence intervals
        6. Save to Unity Catalog Delta table
        
        **Query**: `workspace.default.denial_rate_forecast`
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <p><strong>Built by Surahman Golado</strong></p>
            <p style='margin-top: 10px;'>
                <a href='https://github.com/goladosu' target='_blank'>GitHub</a> | 
                <a href='https://linkedin.com/in/yourprofile' target='_blank'>LinkedIn</a>
            </p>
            <p style='margin-top: 20px; font-size: 0.9rem; color: #666;'>
                © 2026 | Data Science Portfolio Project
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()