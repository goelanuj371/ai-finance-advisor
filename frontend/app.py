import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Personal Finance Advisor",
    page_icon="💡",
    layout="wide"
)

# --- API URL ---
API_URL = "https://finance-advisor-api-871473564034.us-central1.run.app/upload-and-process/"

# --- Helper Functions for UI ---

def display_metrics(metrics):
    """Displays the key financial metrics."""
    st.subheader("Key Financial Metrics")
    
    # Separate metrics for display
    top_level_metrics = {k: v for k, v in metrics.items() if 'categories' not in k}
    category_metrics = metrics.get('top_5_categories_total')

    cols = st.columns(len(top_level_metrics))
    for i, (key, value) in enumerate(top_level_metrics.items()):
        metric_title = key.replace('_', ' ').title()
        cols[i].metric(metric_title, f"₹{value}")

def create_forecast_chart(original_forecast, simulated_forecast=None):
    """Creates an interactive Plotly chart showing original and simulated forecasts."""
    df_orig = pd.DataFrame(original_forecast)
    df_orig['ds'] = pd.to_datetime(df_orig['ds'])
    
    fig = go.Figure()

    # Original Forecast Line
    fig.add_trace(go.Scatter(
        x=df_orig['ds'], y=df_orig['yhat'], mode='lines',
        line=dict(color='#6a1b9a', width=3, dash='dash'), name='Original Forecast'
    ))

    # Simulated Forecast Line (if available)
    if simulated_forecast:
        df_sim = pd.DataFrame(simulated_forecast)
        df_sim['ds'] = pd.to_datetime(df_sim['ds'])
        fig.add_trace(go.Scatter(
            x=df_sim['ds'], y=df_sim['yhat'], mode='lines',
            line=dict(color='#ff6f00', width=4), name='Simulated Forecast'
        ))

    fig.update_layout(
        title_text='90-Day Cash Flow Forecast',
        xaxis_title='Date', yaxis_title='Projected Balance',
        hovermode='x unified', legend=dict(x=0.01, y=0.99)
    )
    return fig

def display_risks(risks_data):
    """Displays the top spending risks in a clean table."""
    if not risks_data:
        st.info("No significant spending anomalies were detected.")
        return
    st.subheader("Top Spending Anomalies Detected")
    df = pd.DataFrame(risks_data)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df['amount'] = df['amount'].apply(lambda x: f"₹{x:,.2f}")
    st.dataframe(df, use_container_width=True)

def create_spending_chart(metrics):
    """Creates a pie chart for spending categories."""
    category_metrics = metrics.get('top_5_categories_total')
    if not category_metrics:
        st.info("Not enough data to generate a spending breakdown.")
        return

    st.subheader("Top 5 Spending Categories")
    category_df = pd.DataFrame(list(category_metrics.items()), columns=['Category', 'Amount'])
    category_df['Amount'] = pd.to_numeric(category_df['Amount'])
    
    # Pie Chart
    pie_fig = go.Figure(data=[go.Pie(labels=category_df['Category'], values=category_df['Amount'], hole=.4)])
    pie_fig.update_traces(textposition='inside', textinfo='percent+label')
    pie_fig.update_layout(showlegend=False, title_text='Overall Spending Distribution')
    st.plotly_chart(pie_fig, use_container_width=True)

# --- Main App UI ---
st.title("Personalized AI Finance Advisor �")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

uploaded_file = st.file_uploader("Upload your transaction history (CSV or XLSX)", type=['csv', 'xlsx'])

with st.expander("Click here for file format instructions"):
    st.write("Your file must contain columns named exactly: `DATE`, `TRANSACTION DETAILS`, `WITHDRAWAL AMT`, and `DEPOSIT AMT`.")
    sample_data = {'DATE': ['2025-07-01'], 'TRANSACTION DETAILS': ['Salary'], 'WITHDRAWAL AMT': [0], 'DEPOSIT AMT': [5000]}
    sample_df = pd.DataFrame(sample_data)
    st.download_button(label="Download Sample CSV", data=sample_df.to_csv(index=False).encode('utf-8'), file_name='sample.csv', mime='text/csv')

if uploaded_file is not None:
    if st.button("Analyze My Finances", type="primary"):
        with st.spinner("Analyzing... This may take a minute."):
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post(API_URL, files=files, timeout=600)
                if response.status_code == 200:
                    st.session_state.results = response.json()
                    st.success("Analysis Complete!")
                else:
                    st.session_state.results = None
                    error_detail = response.json().get('detail', response.text)
                    st.error(f"Error from API: {error_detail}")
            except requests.exceptions.RequestException as e:
                st.session_state.results = None
                st.error(f"Could not connect to the analysis API. Error: {e}")

# Display results if available in session state
if st.session_state.results:
    results = st.session_state.results
    
    st.header("Your Financial Dashboard")
    display_metrics(results.get("metrics", {}))
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2]) # Give more space to the forecast
    
    with col1:
        create_spending_chart(results.get("metrics", {}))
        
        # --- Interactive "What-If" Sliders ---
        st.subheader("Simulate Your Future")
        st.write("Adjust your monthly spending in top categories to see the impact on your forecast.")
        
        monthly_avg = results.get('metrics', {}).get('top_5_categories_monthly_avg', {})
        sim_changes = {}
        for category, avg_spend in monthly_avg.items():
            sim_changes[category] = st.slider(
                f"Change in {category} spending (%)", 
                -100, 100, 0, 10,
                key=f"slider_{category}"
            )

    # --- Simulation Logic ---
    original_forecast = results.get('forecast', [])
    simulated_forecast = None
    
    if any(v != 0 for v in sim_changes.values()):
        simulated_forecast = pd.DataFrame(original_forecast)
        
        total_daily_reduction = 0
        for category, change_percent in sim_changes.items():
            monthly_change = monthly_avg.get(category, 0) * (change_percent / 100.0)
            daily_change = monthly_change / 30.44 # Average days in a month
            total_daily_reduction -= daily_change # Subtract because it's an expense
        
        future_dates = pd.to_datetime(simulated_forecast['ds'])
        start_date = future_dates.min()
        
        # --- THIS IS THE FIX ---
        # Use .dt.days to correctly calculate the number of days for each row
        cumulative_change = (future_dates - start_date).dt.days * total_daily_reduction
        
        simulated_forecast['yhat'] += cumulative_change
        
        simulated_forecast = simulated_forecast.to_dict(orient='records')


    with col2:
        st.plotly_chart(create_forecast_chart(original_forecast, simulated_forecast), use_container_width=True)

    st.markdown("---")
    tab1, tab2 = st.tabs(["📄 AI-Powered Report", "⚠️ Spending Risks"])
    with tab1:
        st.markdown(results.get("advice", "No advice generated."))
    with tab2:
        display_risks(results.get("risks", []))
