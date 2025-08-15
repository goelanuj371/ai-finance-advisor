**Personalized AI Finance Advisor**
A web-based financial advisor that leverages machine learning and generative AI to provide users with deep insights into their financial health, including predictive forecasts, risk analysis, and interactive "what-if" scenario planning.

**Video Demonstration & Live Application**
Live Demo URL: https://personal-finance-advisor-ai.streamlit.app/

**Core Features**
Robust Data Ingestion: Securely accepts user transaction history in both CSV and XLSX formats, with intelligent handling of various text encodings.

Predictive Cash Flow Forecast: Utilizes the Prophet time-series model to generate a 90-day forecast of the user's account balance, complete with uncertainty intervals for a realistic outlook.

Intelligent Anomaly Detection: Employs an IsolationForest model to identify and rank the top 10 most significant spending anomalies, flagging potential risks or unusual transactions that require review.

Comprehensive Financial Metrics: Automatically calculates key metrics, including average monthly income vs. expenses, net savings, and a breakdown of spending by category, providing a quick snapshot of financial health.

Interactive "What-If" Scenarios: Allows users to actively engage with their forecast. Interactive sliders for top spending categories enable users to see in real-time how changes in their spending habits (e.g., reducing dining out by 20%) will impact their future account balance.

Advanced AI-Powered Reporting: Integrates with Google's Gemini 1.5 Pro API using a multi-prompt chain to generate a detailed, multi-section financial report. The report offers deep analysis and actionable advice, complete with reasoning, pros, and cons for each recommendation.

**System Architecture**
This project is built on a modern, scalable hybrid architecture that separates the user interface from the heavy computational logic. This design ensures the frontend remains fast and responsive, even when processing large, multi-year transaction histories.

Frontend: A lightweight Streamlit application serves as the complete user interface. It is responsible for all user interactions, including file uploads, handling the interactive "what-if" sliders, and rendering all visualizations and reports.

Backend: A robust FastAPI service exposes a single, powerful API endpoint. This service handles all the heavy lifting:

Data cleaning and rule-based transaction categorization with Pandas.

Time-series forecasting with Prophet.

Anomaly detection with Scikit-learn.

Report generation via a multi-prompt chain to the Gemini API.

Deployment & Hosting:

The FastAPI backend is containerized using Docker and deployed on Google Cloud Run, configured with 4GB of memory and a 10-minute timeout to handle large computations efficiently.

The Streamlit frontend is deployed on Streamlit Community Cloud for fast, global access.

Security: API keys and other secrets are managed securely using a local .env file for development and as environment variables on the server for deployment, ensuring they are never exposed in the codebase.

**Technology Stack**
Backend: Python, FastAPI, Pandas, Prophet, Scikit-learn, Google Generative AI

Frontend: Streamlit, Plotly

Deployment: Google Cloud Run, Docker, Streamlit Community Cloud

**How to Use the Live Demo**
Navigate to the live application URL provided above.

Use the file uploader to select your transaction history file.

Note: The file must contain the columns named exactly: DATE, TRANSACTION DETAILS, WITHDRAWAL AMT, and DEPOSIT AMT. A sample CSV file is available for download directly in the app for your convenience.

Click the "Analyze My Finances" button and allow a few moments for the complete analysis.

Explore your financial dashboard, interact with the "what-if" sliders, and read your detailed AI-powered report.

**Future Enhancements**
This prototype serves as a strong foundation. Given more time, the following features would be implemented to create a production-ready application:

Advanced Transaction Categorization: Implement a machine learning model (e.g., using a pre-trained NLP model) to automatically categorize transactions with higher accuracy and less reliance on simple keywords.

Secure User Authentication: Integrate an OAuth 2.0 system to allow users to create accounts, save their analysis history, and track their progress over time.

Database Integration: Connect the application to a persistent database (like Firebase Firestore or Supabase) to store user data securely.

Direct Bank Connections: Utilize APIs like Plaid to allow users to connect their bank accounts directly for a seamless, real-time experience.
