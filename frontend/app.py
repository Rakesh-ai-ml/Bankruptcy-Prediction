import streamlit as st
import requests


st.set_page_config(
    page_title="ML Predictor",
    page_icon="🤖",
    layout="wide",              
    initial_sidebar_state="collapsed",
)

st.header("Bankruptcy Prediction")

Currentassets = st.number_input("Current assets",value=511.26)
Costofgoodssold = st.number_input("Cost of goods sold",value=833.10)
Depreciationandamortization = st.number_input("Depreciation and amortization",value=25.40)
EBITDA = st.number_input("EBITDA",value=112.50)
Inventory = st.number_input("Inventory",value=85.20)
NetIncome = st.number_input("NetIncome",value=35.16)
TotalReceivables = st.number_input("Total Receivables",value=120.40)
Marketvalue = st.number_input("Market value",value=450.00)
Netsales = st.number_input("Net sales",value=950.00)
Totalassets = st.number_input("Total assets",value=740.99)
TotalLongtermdebt = st.number_input("Total Longterm debt",value=150.00)
EBIT = st.number_input("EBIT",value=85.00)
GrossProfit = st.number_input("Gross Profit",value=116.90)
TotalCurrentLiabilities = st.number_input("Total Current Liabilities",value=220.00)
RetainedEarnings = st.number_input("Retained Earnings",value=310.00)
TotalRevenue = st.number_input("Total Revenue",value=950.00)
TotalLiabilities = st.number_input("Total Liabilities",value=370.00)
TotalOperatingExpenses = st.number_input("Total Operating Expenses",value=865.00)

st.markdown("""
<style>
/* Style the container that has key="result_card" */
.st-key-result_card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    margin-top: 24px;
}
/* Title */
.st-key-result_card h3 {
    color: white;
    font-weight: 400;
    margin: 0 0 16px 0;
    text-align: center;
}
/* Make the metric labels + values readable on the gradient */
.st-key-result_card [data-testid="stMetricLabel"] {
    color: rgba(255, 255, 255, 0.85) !important;
}
.st-key-result_card [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 2rem !important;
}
.st-key-result_card [data-testid="stMetricDelta"] {
    color: rgba(255, 255, 255, 0.9) !important;
}
</style>
""", unsafe_allow_html=True)



if st.button("Predict Bankruptcy"):

    input_financials ={
        "Currentassets":Currentassets,
        "Costofgoodssold":Costofgoodssold,
        "Depreciationandamortization":Depreciationandamortization,
        "EBITDA":EBITDA,
        "Inventory":Inventory,
        "NetIncome":NetIncome,
        "TotalReceivables":TotalReceivables,
        "Marketvalue":Marketvalue,
        "Netsales":Netsales,
        "Totalassets":Totalassets,
        "TotalLongtermdebt":TotalLongtermdebt,
        "EBIT":EBIT,
        "GrossProfit":GrossProfit,
        "TotalCurrentLiabilities":TotalCurrentLiabilities,
        "RetainedEarnings":RetainedEarnings,
        "TotalRevenue":TotalRevenue,
        "TotalLiabilities":TotalLiabilities,
        "TotalOperatingExpenses":TotalOperatingExpenses,
    }
    try:
        
        with st.spinner("Predicting ...."):
            response= requests.post("http://backend:8000/predict", json=input_financials)
            if response.status_code == 200:
                result=response.json()
            
        with st.container(key="result_card"):
            st.markdown("<h3>Predicted Value</h3>", unsafe_allow_html=True)

            cols = st.columns(len(result))
            for col, (label, value) in zip(cols, result.items()):
                with col:
                    st.metric(label=label, value=value)

    except Exception as e:
        print(" error while connecting to api : ", e)
    