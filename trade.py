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
st.title('Rwandan Exports Analysis')

#-------------------------------------------------------------
#3. SETTING SIDE BAR
#-------------------------------------------------------------
st.sidebar.header("NISR Trade opportunity project")
try: 
    image = "https://raw.githubusercontent.com/Julesmugabo/Future-trade-opportunity/main/Logo.png"
    st.sidebar.image(image, caption='Mufaxa Traders', width=200)
except:
    st.sidebar.markdown("Mufaxa Traders")



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
available_pages = ["overview","Commodity exports.", "Country exports", "Commodity prediction model", "Country prediction model", "Youth & SME Engagement"]
selected_page = st.sidebar.selectbox("Select the page to explore", available_pages)
#-------------------------------------------------------------
# PAGE ARRANGEMENT
#-------------------------------------------------------------
if selected_page == "overview":
    st.title(" Data overview")
    
    st.markdown("""This dashboard walks you through the analysis on Rwanda's export performance either in commodities exported or destination of those commodities.
                 **Its aim is to predict countries and commodites that if we invest in now we shall make a good profit out of it**.
                This analysis is to help us make a good choice of a country or a commodity.""")
    
    st.markdown("""This is the dataset that contains the data for country destination of our exports.""")
    
    st.dataframe(df_exports)
    
    st.download_button("Download export destination Data", df_exports.to_csv(index=False).encode(), "exports_destination.csv")
    st.markdown("""This is the dataset that contains the data for commodities exported from Rwanda""")
    st.dataframe(df_commodity)
    st.download_button("Download commodities exported Data", df_exports.to_csv(index=False).encode(), "Commodity_exported.csv")


if selected_page == "Commodity exports":
    st.title(" Exports Commodity Analysis")
    df = pd.read_excel(file_path, sheet_name="ExportsCommodity")
    st.dataframe(df)

if selected_page == "Country exports":
    st.title(" Export Country Page")
    dff = pd.read_excel(file_path, sheet_name="ExportCountry")
    st.dataframe(dff)

if selected_page == "Commodity prediction model":
    st.title("Machine Learning Forecast – ExportsCommodity Growth Prediction")
    df_predictions = pd.read_csv("predictions.csv")

if selected_page == "Youth & SME Engagement":
    st.title("Youth and SME engagement")
    
    


   
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
if selected_page == "Commodity exports":

    st.title("Exports Commodity Page")
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Line chart", "Bar chart", "Pie Shart"])
    with tab1:
        st.caption("Line chart for top 8 commodities")
        try:
            x = data_T_commodity.index
            y = data_T_commodity.values.T  

            fig = go.Figure()
            for col in data_T_commodity.columns:
                fig.add_trace(go.Scatter(
                    x=x, 
                    y=data_T_commodity[col], 
                    mode='lines',
                    name=col
                ))

            fig.update_layout(
                title="Top 8 Labels",
                xaxis_title="Period",
                yaxis_title="Value",
                legend_title="Labels",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Couldn't draw line chart: {e}")
        st.markdown(""" This graph tells about quartely trends of top export labels from 2023q1 to 2025q2.
                    
                    From the graph, we see that Food and live animals stayed leading in consistency.
                    
                    Other commodities & transactionss and crude materials... remained stable with little flactuations.""")

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

        st.markdown("""This graph clearly shows ascending order of top Rwanda export commodities from 2023 to 2025.
                    
                    From this graph we easly discover leading sectors that contributes to the GDP of Rwanda.
                    
                    we see that Rwanda's top Export commodity is manufactured materials and goods.  """)
    with tab3:
        st.caption("Distribution sharing")
        try:
            top6 = working_numeric_commodity.iloc[:, -1].astype(float).nlargest(11)

            fig3 = px.pie(
                names=top6.index,
                values=top6.values,
                title="Top 6 share (latest)",
                hole=0.3
            )
            fig3.update_traces(textinfo='percent+label')
            fig3.update_layout(height=500)
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:  
            st.warning(f"Couldn't draw pie: {e}")
            st.markdown("""This graph shows its percentage contribution to others in same set.""")
    st.download_button("Download Commodity Data",
                       working_numeric_commodity.to_csv().encode(),
                       "commodity_exports.csv")
                
#------------------------------     
# GRAPHS FOR EXPORTCOUNTRIES
#------------------------------
if selected_page == "Country exports":
    import plotly.express as px
    import plotly.graph_objects as go

    st.title("Export Country Page")
    tab1, tab2, tab3 = st.tabs(["Stacked Area", "Bar chart", "Pie Shart"])
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
# EXPORT COUNTRIES MACHINE LEARNING SECTION
#---------------------------------
if selected_page == "Country prediction model":
    df2 = pd.read_excel(file_path, sheet_name="ExportCountry")
    df2 = df2.rename(columns={'Year and Period': 'Label'})

    df = df2

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


    #------------------------
    # GRAPHS
    #-------------------------
    tab1, tab2, tab3 = st.tabs(["Prediction comparison", "Countries with opportunity", "Model perfomance"])
    with tab1:
        # 1️ Comparison: Actual vs Predicted Growth
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
        st.markdown("""From this graph we can see there is a hidden opportunity in investing our exports to Thailand. so our next move would be to define which goods are predicted to be imported in Thailaand so that we fcan start preparing them now.""")
    with tab2:
        # 2️ Top 10 Predicted Growth – Investment Opportunities
        st.write("Top 10 countries that will be good for our next investment")
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

    with tab3:
        # 3️ Model performance: Actual vs Predicted scatter
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
#---------------------------------
# COMMODITY MACHINE LEARNING SECTION 
#---------------------------------
if selected_page == "Commodity prediction model":
    df1 = pd.read_excel(file_path, sheet_name="ExportsCommodity")
    df1 = df1.rename(columns={'Year and Period': 'Label'})
    # Create growth change features
    df1['2023Q2_growth'] = (df1['2023Q2'] - df1['2023Q1']) / df1['2023Q1']
    df1['2023Q3_growth'] = (df1['2023Q3'] - df1['2023Q2']) / df1['2023Q2']
    df1['2023Q4_growth'] = (df1['2023Q4'] - df1['2023Q3']) / df1['2023Q3']
    df1['2024Q1_growth'] = (df1['2024Q1'] - df1['2023Q4']) / df1['2023Q4']
    df1['2024Q2_growth'] = (df1['2024Q2'] - df1['2024Q1']) / df1['2024Q1']
    df1['2024Q3_growth'] = (df1['2024Q3'] - df1['2024Q2']) / df1['2024Q2']
    df1['2024Q4_growth'] = (df1['2024Q4'] - df1['2024Q3']) / df1['2024Q3']
    df1['2025Q1_growth'] = (df1['2025Q1'] - df1['2024Q4']) / df1['2024Q4']
    df1['2025Q2_growth'] = (df1['2025Q2'] - df1['2025Q1']) / df1['2025Q1']

    # Total growth
    df1['total_growth'] = (df1['2025Q2'] - df1['2023Q1']) / df1['2023Q1']

    # Features and target
    X_df1= df1[['2023Q1', '2023Q2', '2023Q3', '2023Q4',
            '2024Q1', '2024Q2', '2024Q3', '2024Q4',
            '2025Q1', '2025Q2',
            '2023Q2_growth', '2023Q3_growth', '2023Q4_growth',
            '2024Q1_growth', '2024Q2_growth', '2024Q3_growth',
            '2024Q4_growth', '2025Q1_growth', '2025Q2_growth']]

    y_df1 = df1['total_growth']

    # Train model
    X_train_df1, X_test_df1, y_train_df1, y_test_df1 = train_test_split(
        X_df1, y_df1, test_size=0.2, random_state=42
    )

    model_df1 = RandomForestRegressor(random_state=42)
    model_df1.fit(X_train_df1, y_train_df1)
    y_pred_df1 = model_df1.predict(X_test_df1)

    # Model evaluation
    st.write("R² Score:", r2_score(y_test_df1, y_pred_df1))
    st.write("Mean Absolute Error:", mean_absolute_error(y_test_df1, y_pred_df1))
    st.markdown("""The R² Score = 0.987 means that it explains about 98.7% of the target variable. so the model fits our data accurately""")
    st.markdown("""Mean Absolute Error = 0.65 means that our model has high accuracy because it deviate from true values by only 0.66 units""")
    # Predict total growth
    df1['Predicted_Growth'] = model_df1.predict(X_df1)

    df1_sorted = df1.sort_values(by='Predicted_Growth', ascending=False)
    st.dataframe(df1_sorted[['Label','Predicted_Growth', 'total_growth']])


    #------------------------
    # GRAPHS
    #-------------------------
    tab1, tab2, tab3 = st.tabs(["Prediction comparison", "Commodities with opportunity", "Model perfomance"])
    with tab1:
        # 1 Comparison: Actual vs Predicted Growth
        st.subheader("Comparison: Actual vs Predicted Growth")

        df1_comparison = df1.sort_values(by='Predicted_Growth', ascending=False).head(10)

        fig1 = go.Figure(data=[
            go.Bar(name='Actual Growth', x=df1_comparison['Label'], y=df1_comparison['total_growth'],
                marker_color='lightgreen'),
            go.Bar(name='Predicted Growth', x=df1_comparison['Label'], y=df1_comparison['Predicted_Growth'],
                marker_color='steelblue')
        ])
        fig1.update_layout(
            barmode='group',
            title="Actual vs Predicted Growth (Top 10)",
            xaxis_title="Country / Commodity",
            yaxis_title="Growth Rate",
            xaxis_tickangle=-45,
            legend_title_text='Growth Type',
            height=600
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("""Based on this graph, 
                    
                    animals and vegetable oils,fats & waxes will grow by 21.9%
                    
                    Beverages and tobacco wll grow by 11% 
                    
                    since we expect to export in high quantity in the future. so its better we can invest in their preparation so that when time comes, we can be able to make good sales out of them.
                    """)
    with tab2:
        # 2 Top 10 Predicted Growth – Investment Opportunities
        st.write("Feature opportunities where we can invest either as a country or a certain commodity")
        Top_10_commodities = df1_sorted[['Label', 'Predicted_Growth', 'total_growth']].head(10)

        fig2 = px.bar(
            Top_10_commodities,
            y='Label',
            x='Predicted_Growth',
            orientation='h',
            color='Predicted_Growth',
            color_continuous_scale='Blues',
            title="Top 10 Predicted Growth – Investment Opportunities"
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("""Based on this graph, 
                    
                    animals and vegetable oils,fats & waxes will grow by 21.9%
                    
                    Beverages and tobacco wll grow by 11% 
                    
                    since we expect to export in high quantity in the future. so its better we can invest in their preparation so that when time comes, we can be able to make good sales out of them.
                    """)
    with tab3:
        # Model performance: Actual vs Predicted scatter
        st.write('This is a graph that shows how well our model is fitting and can be trusted')

        fig3 = px.scatter(
            x=y_test_df1,
            y=y_pred_df1,
            labels={'x': 'Actual Growth', 'y': 'Predicted Growth'},
            title="Actual vs Predicted Growth"
        )
        fig3.add_trace(go.Scatter(
            x=[min(y_test_df1), max(y_test_df1)],
            y=[min(y_test_df1), max(y_test_df1)],
            mode='lines',
            name='Perfect Fit',
            line=dict(color='red', dash='dash')
        ))
        fig3.update_traces(marker=dict(size=8, opacity=0.7))
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

#-----------------------------------------------------------
# Youth and SME Engagement
#-----------------------------------------------------------
if selected_page == "Youth & SME Engagement":
    st.markdown("""
    ### Engaging in Rwanda’s Growing Trade Opportunities by  


    This is a **call to action** for all **Rwandan youth** and **small & medium-sized enterprises (SMEs)** to seize the opportunities available in **export trade**.  
    Through informed decision-making and data-driven insights, every participant can **grow an existing business** or **launch a new one** with confidence about **where and how to export**.  

    ---

    ###  Key Insights from Our Analysis
    Our analysis reveals **promising export opportunities** for Rwandan investors:  

    - **Thailand and Sweden** have emerged as highly potential markets:  
    - Exports to **Thailand** are projected to **grow by 93.3%**.  
    - Exports to **Sweden** have already **grown by 47%**, with predictions showing continued upward trends.  

    - **Agricultural and farming products** remain a major opportunity:  
    - Exports of **animal and vegetable oils, fats, and waxes** are expected to **increase by 21%**.  

    - **Tobacco and beverage-related SMEs** also show strong potential,  
    with expected export growth of around **11%**.

    These insights confirm that **Rwanda’s export sector is expanding** — and those who act now will benefit the most.

    ---

    ###  Opportunity for Youth and Entrepreneurs
    This initiative offers a pathway for:
    - **Unemployed youth** to start export-oriented agricultural ventures.  
    - **Small businesses** to scale and reach international markets.  
    - **Innovators** to make data-informed trade decisions that transform local industries.

    ---

    ###  About Mufaxa Traders
    At **Mufaxa Traders**, we empower **unemployed youth** engaged in agriculture through:  
    - **Financial support** — providing startup and expansion loans.  
    - **Capacity building** — training in innovation and **data-informed decision-making**.  
 

    ---

    ###  Get Involved
    If you wish to take part in this opportunity and grow your business through export trade,  
    **fill out the form below to get started**.  

    """)

    name = st.text_input("Your name")
    status = st.text_input("youth or SME")
    business = st.text_input("Your Business or Startup Name")
    email = st.text_input("Email:")
    phone_number = st.text_input("Phone number")
    interest = st.text_area(" Are you intrested in getting a loan from Mufaxa traders or capacity building")
    if st.button("Submit"):
        st.success("Thank you for your submission.")
