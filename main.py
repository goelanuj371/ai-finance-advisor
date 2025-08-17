from fastapi import FastAPI, UploadFile, File, HTTPException
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import io
import logging
import os
import google.generativeai as genai
import json
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
genai.configure(api_key=GOOGLE_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Personal Finance Advisor API")

# --- Categorization Logic ---
def categorize_transaction(description):
    """Categorizes a transaction based on keywords in its description."""
    description = str(description).lower()
    if any(keyword in description for keyword in ['food', 'restaurant', 'groceries', 'swiggy', 'zomato']):
        return 'Food & Dining'
    if any(keyword in description for keyword in ['transport', 'ola', 'uber', 'gasoline', 'metro']):
        return 'Transport'
    if any(keyword in description for keyword in ['bill', 'utility', 'internet', 'phone']):
        return 'Bills & Utilities'
    if any(keyword in description for keyword in ['amazon', 'flipkart', 'shopping', 'store']):
        return 'Shopping'
    if 'rent' in description:
        return 'Rent'
    if 'salary' in description or 'deposit' in description:
        return 'Income'
    return 'Miscellaneous'

# --- Data Cleaning and ML Functions ---
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and preprocesses the transaction data from the user's file."""
    logger.info("Starting data cleaning process...")
    df = df.rename(columns={
        'DATE': 'date', 'TRANSACTION DETAILS': 'description',
        'WITHDRAWAL AMT': 'withdrawal', 'DEPOSIT AMT': 'deposit'
    })
    required_cols = ['date', 'description', 'withdrawal', 'deposit']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df[required_cols].copy()
    df['date'] = pd.to_datetime(df['date'])
    df['withdrawal'].fillna(0, inplace=True)
    df['deposit'].fillna(0, inplace=True)
    df['withdrawal'] = pd.to_numeric(df['withdrawal'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df['deposit'] = pd.to_numeric(df['deposit'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df['amount'] = df['deposit'] - df['withdrawal']
    df['category'] = df['description'].apply(categorize_transaction)
    logger.info("Data cleaning successful.")
    return df[['date', 'description', 'amount', 'category']]

def calculate_key_metrics(df: pd.DataFrame) -> dict:
    """Calculates key financial metrics from the transaction history."""
    logger.info("Calculating key financial metrics...")
    metrics = {}
    
    income_df = df[df['amount'] > 0]
    if not income_df.empty:
        total_income = income_df['amount'].sum()
        avg_monthly_income = income_df.set_index('date')['amount'].resample('M').sum().mean()
        metrics['total_income'] = f"{total_income:,.2f}"
        metrics['avg_monthly_income'] = f"{avg_monthly_income:,.2f}"

    expenses_df = df[df['amount'] < 0]
    if not expenses_df.empty:
        total_expenses = expenses_df['amount'].sum()
        avg_monthly_expenses = expenses_df.set_index('date')['amount'].resample('M').sum().mean()
        metrics['total_expenses'] = f"{total_expenses:,.2f}"
        metrics['avg_monthly_expenses'] = f"{avg_monthly_expenses:,.2f}"
        
        category_spending = expenses_df.groupby('category')['amount'].sum().abs().sort_values(ascending=False)
        metrics['top_5_categories_total'] = category_spending.head(5).to_dict()
        
        monthly_category_spend = expenses_df.set_index('date').groupby('category')['amount'].resample('M').sum()
        avg_monthly_category_spend = monthly_category_spend.groupby('category').mean()
        metrics['top_5_categories_monthly_avg'] = avg_monthly_category_spend.loc[category_spending.head(5).index].abs().to_dict()


    if 'total_income' in metrics and 'total_expenses' in metrics:
        net_savings = total_income + total_expenses
        metrics['net_savings'] = f"{net_savings:,.2f}"

    logger.info(f"Calculated metrics: {metrics}")
    return metrics

def get_prophet_forecast(df: pd.DataFrame) -> dict:
    """Generates a 90-day cash flow forecast using Prophet."""
    logger.info("Generating Prophet forecast...")
    prophet_df = df.set_index('date')['amount'].resample('D').sum().reset_index()
    prophet_df.columns = ['ds', 'y']
    prophet_df = prophet_df.sort_values('ds')
    prophet_df['y'] = prophet_df['y'].cumsum()
    logger.info(f"Resampled data to {len(prophet_df)} daily records for forecasting.")
    model = Prophet(interval_width=0.95, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    forecast['ds'] = forecast['ds'].dt.strftime('%Y-%m-%d')
    forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')
    logger.info("Prophet forecast generated successfully.")
    return forecast_data

def get_risk_flags(df: pd.DataFrame) -> dict:
    """Identifies the top 10 most significant anomalous transactions."""
    logger.info("Identifying risk flags with IsolationForest...")
    expenses_df = df[df['amount'] < 0].copy()
    if len(expenses_df) < 2: return []
    X = expenses_df[['amount']]
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    scores = model.decision_function(X)
    expenses_df['anomaly_score'] = scores
    anomalies = expenses_df[model.predict(X) == -1].sort_values('anomaly_score').head(10)
    anomalies['date'] = anomalies['date'].dt.strftime('%Y-%m-%d')
    risk_flags = anomalies[['date', 'description', 'amount']].to_dict(orient='records')
    logger.info(f"Found {len(risk_flags)} most significant risk flags.")
    return risk_flags

# --- Gemini Function ---
def get_gemini_advice(forecast_data: list, risks_data: list, metrics: dict) -> str:
    """
    Generates a detailed, multi-part financial report by chaining multiple calls to the Gemini API.
    """
    logger.info("Generating multi-part financial report with Gemini...")
    
    final_balance = forecast_data[-1]['yhat'] if forecast_data else 0
    risk_summary = [f"- A transaction of {r['amount']:.2f} on {r['date']} for '{r['description']}'" for r in risks_data]
    risk_text = "\n".join(risk_summary)
    metrics_text = "\n".join([f"- **{key.replace('_', ' ').title()}:** {value}" for key, value in metrics.items() if 'categories' not in key])
    category_text = "\n".join([f"- **{cat}:** {amount:,.2f}" for cat, amount in metrics.get('top_5_categories_total', {}).items()])
    
    base_context = f"""
    Here is the summary of the client's financial data analysis:
    1. Key Financial Metrics:
    {metrics_text}
    2. Top 5 Spending Categories (by total amount):
    {category_text if category_text else "Not enough data to determine spending categories."}
    3. 90-Day Cash Flow Forecast: The projected balance is {final_balance:,.2f} in 90 days.
    4. Spending Anomaly Detection: The following large or unusual expenses were flagged:
    {risk_text if risk_text else "No significant spending anomalies were detected."}
    """

    prompt1 = f"You are a financial analyst... {base_context} ... write a detailed analysis for 'Section 1: Financial Health Analysis'..."
    prompt2 = f"You are a forensic accountant... {base_context} ... write a detailed analysis for 'Section 2: Analysis of Spending Anomalies'..."
    prompt3 = f"You are a financial strategist... {base_context} ... write a detailed 'Section 3: Actionable Recommendations'..."

    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        
        # Increase the maximum output tokens to allow for a longer response
        generation_config = genai.types.GenerationConfig(max_output_tokens=8192)

        logger.info("Generating Section 1: Health Analysis...")
        response1 = model.generate_content("### Section 1: Detailed Financial Health Analysis\n" + prompt1, generation_config=generation_config)
        
        logger.info("Generating Section 2: Anomaly Analysis...")
        response2 = model.generate_content("### Section 2: Analysis of Spending Anomalies\n" + prompt2, generation_config=generation_config)
        
        logger.info("Generating Section 3: Recommendations...")
        response3 = model.generate_content("### Section 3: Detailed Actionable Recommendations\n" + prompt3, generation_config=generation_config)
        
        # Add a safety check before accessing .text
        part1 = response1.text if response1.parts else ""
        part2 = response2.text if response2.parts else ""
        part3 = response3.text if response3.parts else ""

        full_report = f"{part1}\n\n{part2}\n\n{part3}"
        
        logger.info("Successfully generated multi-part report from Gemini.")
        return full_report
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return "Could not generate financial advice at this time due to an API error."

@app.get("/")
def read_root(): return {"status": "ok"}

@app.post("/upload-and-process/")
async def upload_and_process(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")
        contents = await file.read()
        df = None
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension == 'csv':
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    df = pd.read_csv(io.BytesIO(contents), encoding=encoding)
                    break
                except UnicodeDecodeError:
                    contents.seek(0)
            if df is None: raise HTTPException(status_code=400, detail="Could not parse CSV.")
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        cleaned_df = clean_data(df)
        metrics = calculate_key_metrics(cleaned_df)
        forecast_data = get_prophet_forecast(cleaned_df)
        risk_flags = get_risk_flags(cleaned_df)
        advice = get_gemini_advice(forecast_data, risk_flags, metrics)
        
        logger.info("Full pipeline executed successfully.")
        return {
            "filename": file.filename,
            "metrics": metrics,
            "forecast": forecast_data,
            "risks": risk_flags,
            "advice": advice
        }
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
