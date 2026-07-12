import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# ==========================================
# 1. GLOBAL SETTINGS & STYLING
# ==========================================
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom enterprise styling
st.markdown("""
    <style>
    .main-header { font-size:32px !important; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .card { background-color: #F8FAFC; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

# Best Model Metrics (Facebook Prophet from Task 4)
BEST_MODEL_NAME = "Facebook Prophet"
MAE_VAL = 13434.28
RMSE_VAL = 14049.75

# Hardcoded 3-Month Forecast Arrays (Base dynamic offsets off these vectors)
FORECAST_DATA = {
    "Month 1 (Next Month)": 61590.52,
    "Month 2 (Two Months Ahead)": 101136.34,
    "Month 3 (Three Months Ahead)": 90673.80
}

# ==========================================
# 2. DATA CACHING & INITIALIZATION
# ==========================================
@st.cache_data
def load_and_preprocess(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Standardize typical column naming schemas automatically
        for col in df.columns:
            if 'date' in col.lower():
                # FIX: format='mixed' handles any DD/MM/YYYY or MM/DD/YYYY variations seamlessly
                df[col] = pd.to_datetime(df[col], format='mixed')
                df = df.sort_values(by=col)
                break
        return df
    return None

# Sidebar Dataset Upload panel
st.sidebar.title("Configuration Panel")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload your 'train.csv' dataset:", type=["csv"])
df = load_and_preprocess(uploaded_file)

# Sidebar Page Router Selection Map
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Select App Section:",
    [
        "Page 1 — Sales Overview Dashboard",
        "Page 2 — Forecast Explorer",
        "Page 3 — Anomaly Report",
        "Page 4 — Product Demand Segments"
    ]
)

# Enforce file upload barrier globally across sub-panels
if df is None:
    st.markdown("<div class='main-header'>Sales Forecasting & Operations Hub</div>", unsafe_allow_html=True)
    st.warning("⚠️ Welcome! Please upload your project dataset file (`train.csv`) via the sidebar to initialize the analytical engine dashboards.")
    st.stop()

# Auto-identify tracking targets dynamically 
date_c = [c for c in df.columns if 'date' in c.lower()][0]
sales_c = [c for c in df.columns if 'sales' in c.lower() or 'revenue' in c.lower()][0]
cat_c = [c for c in df.columns if 'category' in c.lower() and 'sub' not in c.lower()][0]
subcat_c = [c for c in df.columns if 'sub' in c.lower() or 'item' in c.lower() or 'product' in c.lower()][0]
region_c = [c for c in df.columns if 'region' in c.lower() or 'zone' in c.lower() or 'state' in c.lower()][0]

# ==========================================
# PAGE 1: SALES OVERVIEW DASHBOARD
# ==========================================
if page == "Page 1 — Sales Overview Dashboard":
    st.markdown("<div class='main-header'>Sales Overview Dashboard</div>", unsafe_allow_html=True)
    st.write("Analyze structural historical sales baselines and trend movements.")
    
    # Global Interactive Cross-Filters for deep drills
    st.markdown("### 🔍 Interactive Visual Cross-Filters")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_regions = st.multiselect(f"Filter by Region ({region_c}):", options=df[region_c].unique().tolist(), default=df[region_c].unique().tolist())
    with col_f2:
        selected_cats = st.multiselect(f"Filter by Main Category ({cat_c}):", options=df[cat_c].unique().tolist(), default=df[cat_c].unique().tolist())
        
    # Apply selected interactive cross filters to current session frame
    f_df = df[df[region_c].isin(selected_regions) & df[cat_c].isin(selected_cats)].copy()
    
    if f_df.empty:
        st.error("❌ No matching metrics remain under current active cross filters. Broaden your bounds.")
        st.stop()
        
    # Create Year and Month periods 
    f_df['Year'] = f_df[date_c].dt.year
    f_df['Year-Month'] = f_df[date_c].dt.to_period('M').astype(str)
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("#### Total Sales by Year")
        yearly_sales = f_df.groupby('Year')[sales_c].sum().reset_index()
        fig_year = px.bar(yearly_sales, x='Year', y=sales_c, text_auto='.3s', color_discrete_sequence=['#1E3A8A'])
        fig_year.update_layout(xaxis_type='category', yaxis_title="Gross Value ($)")
        st.plotly_chart(fig_year, use_container_width=True)
        
    with col_g2:
        st.markdown("#### Monthly Sales Evolution Trend")
        monthly_sales = f_df.groupby('Year-Month')[sales_c].sum().reset_index()
        fig_month = px.line(monthly_sales, x='Year-Month', y=sales_c, markers=True, color_discrete_sequence=['#EF4444'])
        fig_month.update_layout(xaxis_title="Timeline", yaxis_title="Monthly Gross Total ($)")
        st.plotly_chart(fig_month, use_container_width=True)

# ==========================================
# PAGE 2: FORECAST EXPLORER
# ==========================================
elif page == "Page 2 — Forecast Explorer":
    st.markdown("<div class='main-header'>Operational Forecast Explorer</div>", unsafe_allow_html=True)
    
    # Interactive selection criteria drop down controls
    col_fc1, col_fc2 = st.columns(2)
    with col_fc1:
        dimension_choice = st.selectbox("Select Forecast View Aggregation Target Group:", ["Category", "Region"])
    
    target_col = cat_c if dimension_choice == "Category" else region_c
    
    with col_fc2:
        specific_element = st.selectbox(f"Select Target Specific Dimension Segment Element ({target_col}):", options=df[target_col].unique().tolist())
        
    # Horizon slider parameterizer selection range
    horizon_months = st.slider("Select Forecast Horizon window step (Months Ahead):", min_value=1, max_value=3, value=3)
    
    # Establish historic and projected timeline pathways
    slice_df = df[df[target_col] == specific_element].copy()
    hist_monthly = slice_df.set_index(date_c).resample('ME')[sales_c].sum().reset_index()
    
    last_date = hist_monthly[date_c].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_months, freq='ME')
    
    # Generate scaled dynamic projections based on slice ratios
    total_baseline = df.set_index(date_c).resample('ME')[sales_c].sum().values[-1]
    slice_ratio = hist_monthly[sales_c].values[-1] / total_baseline if total_baseline > 0 else 0.1
    
    forecast_keys = list(FORECAST_DATA.keys())[:horizon_months]
    projected_vals = [FORECAST_DATA[k] * slice_ratio for k in forecast_keys]
    
    # Visualization setup
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=hist_monthly[date_c], y=hist_monthly[sales_c], name="Historical Base", line=dict(color="#1E3A8A")))
    fig_fc.add_trace(go.Scatter(x=future_dates, y=projected_vals, name=f"Model Prediction Target Pipeline", line=dict(color="#EF4444", dash='dash'), marker=dict(size=8)))
    
    fig_fc.update_layout(title=f"{specific_element} Out-of-Sample Market Trajectory Pathway", yaxis_title="Revenue ($)")
    st.plotly_chart(fig_fc, use_container_width=True)
    
    # Display the metrics of the best model underneath as required
    st.markdown(f"### 🏆 Model Performance Analytics Pipeline: `{BEST_MODEL_NAME}`")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Mean Absolute Error (MAE)", f"{MAE_VAL:,.2f}")
    with col_m2:
        st.metric("Root Mean Squared Error (RMSE)", f"{RMSE_VAL:,.2f}")

# ==========================================
# PAGE 3: ANOMALY REPORT
# ==========================================
elif page == "Page 3 — Anomaly Report":
    st.markdown("<div class='main-header'>Operational Anomaly Report (Task 5 Isolation Forest)</div>", unsafe_allow_html=True)
    
    # Isolation Forest implementation
    ts_df = df.set_index(date_c).resample('ME')[sales_c].sum().reset_index()
    
    iso = IsolationForest(contamination=0.1, random_state=42)
    ts_df['Cluster_ID'] = iso.fit_predict(ts_df[[sales_c]])
    ts_df['Status'] = ts_df['Cluster_ID'].apply(lambda x: 'Anomaly Outlier' if x == -1 else 'Normal Day Target')
    
    # Anomaly Chart
    fig_anom = px.scatter(ts_df, x=date_c, y=sales_c, color='Status', color_discrete_map={'Normal Day Target': '#1E3A8A', 'Anomaly Outlier': '#EF4444'}, size=ts_df['Status'].apply(lambda x: 12 if x=='Anomaly Outlier' else 6), title="Identified Systemic Volume Variations")
    fig_anom.add_trace(go.Scatter(x=ts_df[date_c], y=ts_df[sales_c], mode='lines', line=dict(color='rgba(30,58,138,0.2)'), showlegend=False))
    st.plotly_chart(fig_anom, use_container_width=True)
    
    # Table listing anomaly dates and values
    st.markdown("### 📋 Detected Anomalies Timeline Ledger")
    anomaly_table = ts_df[ts_df['Status'] == 'Anomaly Outlier'][[date_c, sales_c]].copy()
    anomaly_table[date_c] = anomaly_table[date_c].dt.strftime('%B %Y')
    anomaly_table.columns = ['Historical Month Block Period', 'Recorded Deviation Total Revenue Volume ($)']
    
    st.dataframe(anomaly_table.reset_index(drop=True), use_container_width=True)

# ==========================================
# PAGE 4: PRODUCT DEMAND SEGMENTS
# ==========================================
elif page == "Page 4 — Product Demand Segments":
    st.markdown("<div class='main-header'>Product Demand Segments (Task 6 K-Means Clustering)</div>", unsafe_allow_html=True)
    
    # Perform grouping on sub-categories to keep the dashboard dynamic and functional
    seg_base = df.groupby(subcat_c).agg({
        sales_c: 'sum',
        cat_c: 'count' # proxy for volume/frequency
    }).rename(columns={cat_c: 'Frequency Volume'}).dropna()
    
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(seg_base)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    seg_base['Cluster_ID'] = kmeans.fit_predict(scaled_feats)
    
    cluster_names = {0: "Core Fast Movers", 1: "High Value Premium Assets", 2: "Long-Tail Volatile Items"}
    seg_base['Strategic Demand Cluster Classification'] = seg_base['Cluster_ID'].map(cluster_names)
    
    # Cluster Scatter Chart
    fig_clus = px.scatter(seg_base.reset_index(), x='Frequency Volume', y=sales_c, color='Strategic Demand Cluster Classification', hover_data=[subcat_c], color_discrete_sequence=px.colors.qualitative.Dark2, title="K-Means Cluster Optimization Mapping Profiling")
    st.plotly_chart(fig_clus, use_container_width=True)
    
    # Subcategory Mapping Table
    st.markdown("### 📦 Sub-Category Allocation Cluster Mapping Table")
    mapping_table = seg_base.reset_index()[[subcat_c, 'Strategic Demand Cluster Classification', sales_c]].sort_values(by='Strategic Demand Cluster Classification')
    mapping_table.columns = ['Sub-Category Item Segment Name', 'Assigned Strategic Demand Cluster Grouping Target', 'Total Historical Financial Turnover Contribution ($)']
    
    st.dataframe(mapping_table.reset_index(drop=True), use_container_width=True)