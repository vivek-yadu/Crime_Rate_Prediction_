# Run: streamlit run crime_prediction_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="Crime Rate Predictor",
    page_icon="🕵️‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    /* Main Background */
    .main {
        background-color: #f5f7f9;
    }
    /* Titles and Headers */
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #34495e;
    }
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #2c3e50;
        color: white;
    }
    /* Button Styling */
    div.stButton > button {
        background-color: #e74c3c;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
    }
    div.stButton > button:hover {
        background-color: #c0392b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Apply a nicer plot style
plt.style.use('ggplot')

# --- HEADER ---
st.title("🕵️‍♀️ Crime Rate Predictor – India")
st.markdown("---")

# --- SESSION STATE ---
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
    st.session_state.last_city = None
    st.session_state.last_year = None
    st.session_state.last_month = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go To",
        ["Upload & Clean", "Dashboard", "Model & Prediction",
         "ARIMA Forecast", "City Comparison", "Download Report"]
    )
    st.markdown("---")
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload crime_dataset_india.csv", type=["csv"])

if uploaded_file:
    # Data Processing (Unchanged)
    df = pd.read_csv(uploaded_file)
    df = df[['Time of Occurrence', 'City']]
    df['Time of Occurrence'] = pd.to_datetime(df['Time of Occurrence'], errors='coerce')
    df = df.dropna()
    df['year'] = df['Time of Occurrence'].dt.year
    df['month'] = df['Time of Occurrence'].dt.month

    crime_df = df.groupby(['City','year','month']).size().reset_index(name='crime_count')

    # ---------------- PAGE 1: UPLOAD & CLEAN ----------------
    if page == "Upload & Clean":
        st.subheader("📂 Data Overview")
        st.info("Data successfully loaded and cleaned. Expand sections below to view details.")
        
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("View Raw Data (First 50 rows)", expanded=True):
                st.dataframe(df.head(50), use_container_width=True)
        with col2:
            with st.expander("View Aggregated Data (First 50 rows)", expanded=True):
                st.dataframe(crime_df.head(50), use_container_width=True)

    # ---------------- PAGE 2: DASHBOARD ----------------
    if page == "Dashboard":
        st.subheader("📊 Crime Analytics Dashboard")

        # Top Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        total_crimes = crime_df['crime_count'].sum()
        total_cities = crime_df['City'].nunique()
        peak_year = crime_df.groupby('year')['crime_count'].sum().idxmax()
        top_crime_city = crime_df.groupby('City')['crime_count'].sum().idxmax()

        m1.metric("Total Reported Crimes", f"{total_crimes:,}")
        m2.metric("Cities Covered", total_cities)
        m3.metric("Peak Crime Year", peak_year)
        m4.metric("Highest Crime City", top_crime_city)

        st.markdown("### 📈 Trend Analysis")
        
        # Row 1: Line and Bar Charts
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            yearly = crime_df.groupby('year')['crime_count'].sum().reset_index()
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(yearly['year'], yearly['crime_count'], marker='o', color='#e74c3c')
            ax1.set_title("Yearly Crime Trend", fontsize=10)
            st.pyplot(fig1)

        with row1_col2:
            monthly = crime_df.groupby('month')['crime_count'].sum().reset_index()
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.bar(monthly['month'], monthly['crime_count'], color='#3498db')
            ax2.set_title("Monthly Crime Distribution", fontsize=10)
            st.pyplot(fig2)

        st.markdown("### 🏙️ City Analysis")

        # Row 2: Top 10 and Pie Chart
        row2_col1, row2_col2 = st.columns(2)
        
        with row2_col1:
            top10 = crime_df.groupby('City')['crime_count'].sum().sort_values(ascending=False).head(10)
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            top10.plot(kind='bar', ax=ax3, color='#f1c40f')
            ax3.set_title("Top 10 Crime Cities", fontsize=10)
            plt.xticks(rotation=45)
            st.pyplot(fig3)
        
        with row2_col2:
            top5 = crime_df.groupby('City')['crime_count'].sum().sort_values(ascending=False).head(5)
            fig4, ax4 = plt.subplots(figsize=(8, 4))
            ax4.pie(top5.values, labels=top5.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
            ax4.set_title("Crime Share of Top 5 Cities", fontsize=10)
            st.pyplot(fig4)

        st.markdown("### 🔥 Seasonal Heatmap")
        heat = crime_df.pivot_table(values='crime_count', index='year', columns='month', aggfunc='sum')
        st.dataframe(heat.fillna(0), use_container_width=True)

    # ---------------- PAGE 3: MODEL & PREDICTION ----------------
    if page == "Model & Prediction":
        st.subheader("🤖 AI Crime Predictor (Random Forest)")
        st.write("Train a Random Forest model to predict future crime counts based on City and Date.")

        # Logic
        tfidf = TfidfVectorizer()
        city_vec = tfidf.fit_transform(crime_df['City'])
        X_num = crime_df[['year','month']].values
        X = np.hstack([city_vec.toarray(), X_num])
        y = crime_df['crime_count'].values

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        model = RandomForestRegressor(n_estimators=120, max_depth=15)
        model.fit(X_train,y_train)

        # UI for Inputs
        st.markdown("#### Input Parameters")
        p_col1, p_col2, p_col3 = st.columns(3)
        
        with p_col1:
            city = st.selectbox("Select City", sorted(crime_df['City'].unique()))
        with p_col2:
            year = st.number_input("Year", 2024, 2035, 2027)
        with p_col3:
            month = st.number_input("Month", 1, 12, 6)

        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔮 Predict Crime Count"):
            v = tfidf.transform([city]).toarray()
            future = np.hstack([v, [[year,month]]])
            pred = model.predict(future)[0]
            st.session_state.last_prediction = int(pred)
            st.session_state.last_city = city
            st.session_state.last_year = year
            st.session_state.last_month = month
            
            st.success(f"### Result: Predicted Crimes: {int(pred)}")
            st.info(f"Prediction for **{city}** in **{month}/{year}**")

    # ---------------- PAGE 4: ARIMA FORECAST ----------------
    if page == "ARIMA Forecast":
        st.subheader("📉 Time Series Forecasting (ARIMA)")
        
        col_arima_1, col_arima_2 = st.columns([1, 3])
        
        with col_arima_1:
            city = st.selectbox("Select City for Forecast", sorted(crime_df['City'].unique()))
            
        city_data = crime_df[crime_df['City']==city]
        ts = city_data.groupby(['year','month'])['crime_count'].sum().reset_index()
        ts['date'] = pd.to_datetime(ts[['year','month']].assign(day=1))
        ts = ts.sort_values('date')
        ts_series = ts.set_index('date')['crime_count']

        with col_arima_2:
            st.markdown(f"**Historical Data for {city}**")
            st.line_chart(ts_series)

        if len(ts_series)>=12:
            if st.button("🚀 Run ARIMA Forecast (24 Months)"):
                with st.spinner("Calculating Forecast..."):
                    model_arima = ARIMA(ts_series, order=(1,1,1))
                    model_fit = model_arima.fit()
                    forecast = model_fit.forecast(steps=24)
                    future_dates = pd.date_range(ts_series.index[-1], periods=25, freq='MS')[1:]
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(ts_series.index, ts_series.values, label="Historical")
                    ax.plot(future_dates, forecast.values, linestyle="--", label="Forecast", color='green')
                    ax.legend()
                    ax.set_title(f"24-Month Forecast for {city}")
                    st.pyplot(fig)
        else:
            st.warning("Not enough data to run ARIMA for this city.")

    # ---------------- PAGE 5: CITY COMPARISON ----------------
    if page == "City Comparison":
        st.subheader("⚔️ City vs City Comparison")
        
        col1, col2 = st.columns(2)
        with col1:
            c1 = st.selectbox("City 1", sorted(crime_df['City'].unique()), key='c1')
        with col2:
            c2 = st.selectbox("City 2", sorted(crime_df['City'].unique()), index=1, key='c2')

        d1 = crime_df[crime_df['City']==c1].groupby('year')['crime_count'].sum()
        d2 = crime_df[crime_df['City']==c2].groupby('year')['crime_count'].sum()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(d1.index, d1.values, label=c1, marker='o')
        ax.plot(d2.index, d2.values, label=c2, marker='s')
        ax.legend()
        ax.set_title("Crime Trends Comparison")
        st.pyplot(fig)

    # ---------------- PAGE 6: DOWNLOAD REPORT ----------------
    if page == "Download Report":
        st.subheader("📄 Generate Intelligence Report")
        st.write("Download a PDF report containing the latest charts and your last prediction.")

        # Logic for report generation
        yearly = crime_df.groupby('year')['crime_count'].sum().reset_index()
        plt.figure(); plt.plot(yearly['year'], yearly['crime_count']); plt.savefig("yearly.png"); plt.close()

        monthly = crime_df.groupby('month')['crime_count'].sum().reset_index()
        plt.figure(); plt.bar(monthly['month'], monthly['crime_count']); plt.savefig("monthly.png"); plt.close()

        top5 = crime_df.groupby('City')['crime_count'].sum().sort_values(ascending=False).head(5)
        plt.figure(); top5.plot(kind='bar'); plt.savefig("top5.png"); plt.close()

        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.session_state.last_prediction:
                st.success(f"**Last Prediction Included:** {st.session_state.last_prediction} crimes in {st.session_state.last_city}")
            else:
                st.warning("No prediction made yet. The report will only contain general stats.")

        def create_pdf():
            doc = SimpleDocTemplate("crime_report.pdf")
            styles = getSampleStyleSheet()
            e = []
            e.append(Paragraph("Crime Intelligence System Report", styles['Title']))
            if st.session_state.last_prediction:
                e.append(Paragraph(
                    f"Last Prediction: {st.session_state.last_prediction} crimes in "
                    f"{st.session_state.last_city} - {st.session_state.last_month}/{st.session_state.last_year}",
                    styles['Normal']
                ))
            e.append(Spacer(1,10))
            e.append(Image("yearly.png",400,250))
            e.append(Image("monthly.png",400,250))
            e.append(Image("top5.png",400,250))
            e.append(Paragraph("Conclusion: AI and ARIMA based crime forecasting system.", styles['Normal']))
            doc.build(e)

        with col2:
            st.markdown("<br>", unsafe_allow_html=True) # Spacer
            if st.button("📥 Generate & Download PDF"):
                create_pdf()
                with open("crime_report.pdf","rb") as f:
                    st.download_button("Download PDF", f, "crime_report.pdf", "application/pdf")

else:
    # Empty State with illustration
    st.info("👋 Welcome! Please upload the `crime_dataset_india.csv` file from the sidebar to begin analysis.")
    st.markdown("""
        ### Features:
        - 📊 **Interactive Dashboard**
        - 🤖 **AI Prediction (Random Forest)**
        - 📉 **Future Forecasting (ARIMA)**
        - ⚔️ **City Comparisons**
        - 📄 **PDF Reporting**
    """)