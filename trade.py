#----------------------------------------
# DASHBOARD SCRIPT APP FOR RWANDA EXPORT ANALYSIS
#----------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import joblib
import io
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


#1. Now the first thing is to make a page and its headers
#--------------------------------------------------------
st.set_page_config(page_title="Export opportunity in Rwanda", layout ='wide')
st.title('Analysis on Rwandan Exports')

#-------------------------------------------------------------
#3. SETTING SIDE BAR
#-------------------------------------------------------------
st.sidebar.header("NISR Trade opportunity project")
try: 
    image = "https://github.com/Julesmugabo/Future-trade-opportunity/blob/main/mufaxa.jpg"
    st.sidebar.image(image, caption='Mufaxa Traders', width = 200)
except:
    st.sidebar.markdown("Mufaxa traders")



#--------------------------------------------------------------
#2. LOADING DATA
#--------------------------------------------------------------
file_path = "https://raw.githubusercontent.com/Julesmugabo/Future-trade-opportunity/main/2025Q2_Trade_report_annexTables.xlsx"
df_exports = pd.read_excel(file_path, sheet_name= 'ExportCountry')
df_commodity = pd.read_excel(file_path,sheet_name = 'ExportsCommodity')

all_sheets = {
    'ExportCountry': df_exports,
    'ExportsCommodity': df_commodity,
}
selected_sheet = ['ExportCountry', 'ExportsCommodity']
df_raw = pd.concat(all_sheets.values(), ignore_index=True)


#----------------------------------------
# SIDEBAR ARRANGEMENT
#----------------------------------------
available_pages = ["overview","ExportsCommodity", "ExportCountry", "Machine Learning Model"]
selected_page = st.sidebar.selectbox("Select the page to explore", available_pages)
#-------------------------------------------------------------
# PAGE ARRANGEMENT
#-------------------------------------------------------------
if selected_page == "overview":
    st.title(" overview of the data that we have")
    st.markdown("""This dashborad reveals Rwanda's export trends,forecast on how opportunity can be got using Machine Learning models.""")
    st.markdown("Developed by Jules Mugabushaka")
    st.dataframe(df_raw)
    st.download_button("Download Combined Data", df_raw.to_csv(index=False).encode(), "exports_combined.csv")

if selected_page == "ExportsCommodity":
    st.title(" Exports Commodity Analysis")
    df = pd.read_excel(file_path, sheet_name="ExportsCommodity")
    st.dataframe(df)

if selected_page == "ExportCountry":
    st.title(" Export Country Page")
    dff = pd.read_excel(file_path, sheet_name="ExportCountry")
    st.dataframe(dff)

if selected_page == "Machine Learning Model":
    st.title("Machine Learning Forecast – Export Growth Prediction")
    df_predictions = pd.read_csv("predictions.csv")
    st.write("""After computing machine learning models, here are the summary of the findings we can make good decision from""")

#----------------------------------------
# 4. LETS DO DATA CLEANING
#----------------------------------------
# 4.a renaming the first columns
# Loop over the sheets in the list and rename their first column
for sheet_name in selected_sheet:
    all_sheets[sheet_name].rename(columns={all_sheets[sheet_name].columns[0]: 'Label'}, inplace=True)

#4.c arranging well time columns
# Identify time columns (e.g. 2023Q1, 2024Q2, etc.)
time_columns_country = [c for c in all_sheets['ExportCountry'].columns if str(c).strip().startswith('202') and 'Q' in str(c)]
if not time_columns_country:
    time_columns_country = list(all_sheets['ExportCountry'].columns[1:11])

# Identify time columns for ExportsCommodity
time_columns_commodity = [c for c in all_sheets['ExportsCommodity'].columns if str(c).strip().startswith('202') and 'Q' in str(c)]
if not time_columns_commodity:
    time_columns_commodity = list(all_sheets['ExportsCommodity'].columns[1:11])


# 4.5 creating working dataframe
#--------------------------------------------------------------
# For ExportCountry 
#--------------------------------------------------------------
sheet_name = 'ExportCountry'
df0_country = all_sheets[sheet_name]
df0_country.rename(columns={df0_country.columns[0]: 'Label'}, inplace=True)
time_columns_country = [c for c in df0_country.columns if str(c).strip().startswith('202') and 'Q' in str(c)]
if not time_columns_country:
    time_columns_country = list(df0_country.columns[1:11])
working_df_country = df0_country[['Label'] + time_columns_country].copy()
#--------------------------------------------------------------
# For ExportsCommodity
#--------------------------------------------------------------
sheet_name = 'ExportsCommodity'
df0_commodity = all_sheets[sheet_name]
df0_commodity.rename(columns={df0_commodity.columns[0]: 'Label'}, inplace=True)
time_columns_commodity = [c for c in df0_commodity.columns if str(c).strip().startswith('202') and 'Q' in str(c)]
if not time_columns_commodity:
    time_columns_commodity = list(df0_commodity.columns[1:11])
working_df_commodity = df0_commodity[['Label'] + time_columns_commodity].copy()

# For ExportCountry 
working_country = working_df_country.copy()
working_numeric_country = working_country.set_index('Label').apply(pd.to_numeric, errors='coerce').fillna(0)

latest_col_country = working_numeric_country.columns[-1]
top_labels_by_latest_country = working_numeric_country[latest_col_country].nlargest(8).index.tolist()

data_for_area_country = working_numeric_country.loc[top_labels_by_latest_country].copy()
data_T_country = data_for_area_country.T


# For ExportsCommodity 
working_commodity = working_df_commodity.copy()
working_numeric_commodity = working_commodity.set_index('Label').apply(pd.to_numeric, errors='coerce').fillna(0)

latest_col_commodity = working_numeric_commodity.columns[-1]
top_labels_by_latest_commodity = working_numeric_commodity[latest_col_commodity].nlargest(8).index.tolist()

data_for_area_commodity = working_numeric_commodity.loc[top_labels_by_latest_commodity].copy()
data_T_commodity = data_for_area_commodity.T


#------------------------------
# GRAPHS FOR COMMODITYEXPORTS
#------------------------------
if selected_page == "ExportsCommodity":
    import plotly.express as px
    import plotly.graph_objects as go

    st.title("Exports Commodity Page")
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Stacked Area", "Top 10 Bar", "Pie Share"])
    with tab1:
        st.caption("Stacked area for big 8 lables")
        try:
            x = data_T_commodity.index
            y = data_T_commodity.values.T  

            fig = go.Figure()
            for col in data_T_commodity.columns:
                fig.add_trace(go.Scatter(
                    x=x, 
                    y=data_T_commodity[col], 
                    mode='lines',
                    stackgroup='one', 
                    name=col
                ))

            fig.update_layout(
                title="Top 8 Labels",
                xaxis_title="Period",
                yaxis_title="Value",
                legend_title="Labels",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Couldn't draw stacked area: {e}")

    with tab2:
        st.caption("Big 10 latest values")
        try:
            latest_col = working_numeric_commodity.columns[-1]
            top10 = working_numeric_commodity[latest_col].nlargest(10).sort_values()

            fig2 = px.bar(
                x=top10.values,
                y=top10.index,
                orientation='h',
                title=f"Top 10 — {latest_col}",
                color=top10.values,
                color_continuous_scale='Blues'
            )
            fig2.update_layout(height=300, xaxis_title="Value", yaxis_title="")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Couldn't draw top 10 bar: {e}")

    with tab3:
        st.caption("Distribution sharing")
        try:
            top6 = working_numeric_commodity.iloc[:, -1].astype(float).nlargest(6)

            fig3 = px.pie(
                names=top6.index,
                values=top6.values,
                title="Top 6 share (latest)",
                hole=0.3
            )
            fig3.update_traces(textinfo='percent+label')
            fig3.update_layout(height=300)
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:  
            st.warning(f"Couldn't draw pie: {e}")

    st.download_button("Download Commodity Data",
                       working_numeric_commodity.to_csv().encode(),
                       "commodity_exports.csv")

#------------------------------
# GRAPHS FOR EXPORTCOUNTRIES
#------------------------------
if selected_page == "ExportCountry":
    import plotly.express as px
    import plotly.graph_objects as go

    st.title("Export Country Page")
    tab1, tab2, tab3 = st.tabs(["Stacked Area", "Top 10 Bar", "Pie Share"])
    with tab1:
        st.caption("Stacked area chart")
        try:
            x = data_T_country.index
            y = data_T_country.values.T  # shape: (n_labels, n_periods)

            fig = go.Figure()
            for col in data_T_country.columns:
                fig.add_trace(go.Scatter(
                    x=x, 
                    y=data_T_country[col], 
                    mode='lines',
                    stackgroup='one', 
                    name=col
                ))

            fig.update_layout(
                title="Top 8 Labels",
                xaxis_title="Period",
                yaxis_title="Value",
                legend_title="Labels",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Couldn't draw stacked area: {e}")

    with tab2:
        st.caption("Top 10 latest values")
        try:
            latest_col = working_numeric_country.columns[-1]
            top10 = working_numeric_country[latest_col].nlargest(10).sort_values()

            fig2 = px.bar(
                x=top10.values,
                y=top10.index,
                orientation='h',
                title=f"Top 10 — {latest_col}",
                color=top10.values,
                color_continuous_scale='Blues'
            )
            fig2.update_layout(height=300, xaxis_title="Value", yaxis_title="")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Couldn't draw top 10 bar: {e}")

    with tab3:
        st.caption("Distribution sharing")
        try:
            top6 = working_numeric_country.iloc[:, -1].astype(float).nlargest(6)

            fig3 = px.pie(
                names=top6.index,
                values=top6.values,
                title="Top 6 share (latest)",
                hole=0.3
            )
            fig3.update_traces(textinfo='percent+label')
            fig3.update_layout(height=300)
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:  
            st.warning(f"Couldn't draw pie: {e}")

    st.download_button(" Download Country Data",
                       working_numeric_country.to_csv().encode(),
                       "export_country.csv")

#---------------------------------
# MACHINE LEARNING SECTION
#---------------------------------
if selected_page == "Machine Learning Model":
    df1 = pd.read_excel(file_path, sheet_name="ExportsCommodity")
    df2 = pd.read_excel(file_path, sheet_name="ExportCountry")
    df2 = df2[df1.columns]
    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df = merged_df.rename(columns={'Year and Period': 'Label'})

    df = merged_df

    # Create growth change features
    df['2023Q2_growth'] = (df['2023Q2'] - df['2023Q1']) / df['2023Q1']
    df['2023Q3_growth'] = (df['2023Q3'] - df['2023Q2']) / df['2023Q2']
    df['2023Q4_growth'] = (df['2023Q4'] - df['2023Q3']) / df['2023Q3']
    df['2024Q1_growth'] = (df['2024Q1'] - df['2023Q4']) / df['2023Q4']
    df['2024Q2_growth'] = (df['2024Q2'] - df['2024Q1']) / df['2024Q1']
    df['2024Q3_growth'] = (df['2024Q3'] - df['2024Q2']) / df['2024Q2']
    df['2024Q4_growth'] = (df['2024Q4'] - df['2024Q3']) / df['2024Q3']
    df['2025Q1_growth'] = (df['2025Q1'] - df['2024Q4']) / df['2024Q4']
    df['2025Q2_growth'] = (df['2025Q2'] - df['2025Q1']) / df['2025Q1']

    # Total growth
    df['total_growth'] = (df['2025Q2'] - df['2023Q1']) / df['2023Q1']

    # Features and target
    X = df[['2023Q1', '2023Q2', '2023Q3', '2023Q4',
            '2024Q1', '2024Q2', '2024Q3', '2024Q4',
            '2025Q1', '2025Q2',
            '2023Q2_growth', '2023Q3_growth', '2023Q4_growth',
            '2024Q1_growth', '2024Q2_growth', '2024Q3_growth',
            '2024Q4_growth', '2025Q1_growth', '2025Q2_growth']]

    y = df['total_growth']

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model evaluation
    st.write("R² Score:", r2_score(y_test, y_pred))
    st.write("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

    # Predict total growth
    df['Predicted_Growth'] = model.predict(X)

    df_sorted = df.sort_values(by='Predicted_Growth', ascending=False)
    st.dataframe(df_sorted[['Label','Predicted_Growth', 'total_growth']])
    df_predictions = "https://github.com/Julesmugabo/Future-trade-opportunity/blob/main/predictions.csv"


    #------------------------
    # GRAPHS
    #-------------------------

    # 1️⃣ Comparison: Actual vs Predicted Growth
    st.subheader("Comparison: Actual vs Predicted Growth")

    df_comparison = df.sort_values(by='Predicted_Growth', ascending=False).head(10)

    fig1 = go.Figure(data=[
        go.Bar(name='Actual Growth', x=df_comparison['Label'], y=df_comparison['total_growth'],
               marker_color='lightgreen'),
        go.Bar(name='Predicted Growth', x=df_comparison['Label'], y=df_comparison['Predicted_Growth'],
               marker_color='steelblue')
    ])
    fig1.update_layout(
        barmode='group',
        title="Actual vs Predicted Growth (Top 10)",
        xaxis_title="Country / Commodity",
        yaxis_title="Growth Rate",
        xaxis_tickangle=-45,
        legend_title_text='Growth Type',
        height=400
    )
    st.plotly_chart(fig1, use_container_width=True)


    # 2️⃣ Top 10 Predicted Growth – Investment Opportunities
    st.write("Feature opportunities where we can invest either as a country or a certain commodity")
    top10 = df_sorted[['Label', 'Predicted_Growth', 'total_growth']].head(10)

    fig2 = px.bar(
        top10,
        y='Label',
        x='Predicted_Growth',
        orientation='h',
        color='Predicted_Growth',
        color_continuous_scale='Blues',
        title="Top 10 Predicted Growth – Investment Opportunities"
    )
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)


    # 3️⃣ Model performance: Actual vs Predicted scatter
    st.write('This is a graph that shows how well our model is fitting and can be trusted')

    fig3 = px.scatter(
        x=y_test,
        y=y_pred,
        labels={'x': 'Actual Growth', 'y': 'Predicted Growth'},
        title="Actual vs Predicted Growth"
    )
    fig3.add_trace(go.Scatter(
        x=[min(y_test), max(y_test)],
        y=[min(y_test), max(y_test)],
        mode='lines',
        name='Perfect Fit',
        line=dict(color='red', dash='dash')
    ))
    fig3.update_traces(marker=dict(size=8, opacity=0.7))
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)
