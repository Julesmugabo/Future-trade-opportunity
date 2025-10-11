import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import joblib
import io
from PIL import Image
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#1. Now the first thing is to make a page and its headers
st.set_page_config(page_title="New opportunity in Rwandan Exports", layout ='wide')
st.title('Analysis on Rwandan Exports')
st.markdown("""This dashborad reveals Rwanda's export trends,forecast on how opportunity can be got using Machine Learning models.
             The data used is from the National Bank of Rwanda and the World Bank.""")
st.markdown("Developed by Jules Mugabushaka")

#2. loading the data
file_path= (r"C:\Users\PC\OneDrive\Desktop\Nisr project\2025Q2_Trade_report_annexTables.xlsx")

#3. setting sidebar header
st.sidebar.header("Mufaxa Traders")
try: 
    image = Image.open(r"C:\Users\PC\OneDrive\Desktop\Nisr project\mufaxa.jpg")
    st.sidebar.image(image, caption='Mufaxa Traders', width = 200)
except:
    st.sidebar.markdown("### Your Company Logo")
    st.sidebar.caption("Add logo.png to project folder")

st.sidebar.header("NISR Trade opportunity project")



# dictionary to store specific sheets that shall be used
df_exports = pd.read_excel(file_path, sheet_name= 'ExportCountry')
df_commodity = pd.read_excel(file_path,sheet_name = 'ExportsCommodity')


all_sheets = {
    'ExportCountry': df_exports,
    'ExportsCommodity': df_commodity,
}
selected_sheet = ['ExportCountry', 'ExportsCommodity']
df_raw = pd.concat(all_sheets.values(), ignore_index=True)


#----------------------------------------
# SELECT BOX ARRANGEMENT
#----------------------------------------
# Existing pages (like sheets)
available_pages = ["overview","ExportsCommodity", "ExportCountry", "Machine Learning Model"]

# Sidebar box with new page added
selected_page = st.sidebar.selectbox("Select the page to explore", available_pages)

# --- PAGE LOGIC ---
if selected_page == "overview":
    st.title(" overview of the data that we have")
    st.dataframe(df_raw)

if selected_page == "ExportsCommodity":
    st.title(" Exports Commodity Page")
    df = pd.read_excel(file_path, sheet_name="ExportsCommodity")
    st.dataframe(df)

if selected_page == "ExportCountry":
    st.title(" Export Country Page")
    df = pd.read_excel(file_path, sheet_name="ExportCountry")
    st.dataframe(df_raw)

if selected_page == "Machine Learning Model":
    st.title(" Machine Learning Model Results")
    df_predictions = pd.read_csv("predictions.csv")
    st.dataframe(df_predictions)
    st.write("Here you can add charts or ranking results from your model.")

#----------------------------------------
# 4. LETS DO DATA CLEANING
#----------------------------------------
# 4.a renaming the first columns
# Loop over the sheets in the list and rename their first column
for sheet_name in selected_sheet:
    all_sheets[sheet_name].rename(columns={all_sheets[sheet_name].columns[0]: 'Label'}, inplace=True)

#4.c arranging well time columns
# Identify time columns (e.g. 2023Q1, 2024Q2, etc.)
# Identify time columns for ExportCountry
time_columns_country = [c for c in all_sheets['ExportCountry'].columns if str(c).strip().startswith('202') and 'Q' in str(c)]
if not time_columns_country:
    time_columns_country = list(all_sheets['ExportCountry'].columns[1:11])

# Identify time columns for ExportsCommodity
time_columns_commodity = [c for c in all_sheets['ExportsCommodity'].columns if str(c).strip().startswith('202') and 'Q' in str(c)]
if not time_columns_commodity:
    time_columns_commodity = list(all_sheets['ExportsCommodity'].columns[1:11])


# 4.5 creating working dataframe

# --- For ExportCountry ---
sheet_name = 'ExportCountry'
df0_country = all_sheets[sheet_name]
df0_country.rename(columns={df0_country.columns[0]: 'Label'}, inplace=True)
time_columns_country = [c for c in df0_country.columns if str(c).strip().startswith('202') and 'Q' in str(c)]
if not time_columns_country:
    time_columns_country = list(df0_country.columns[1:11])
working_df_country = df0_country[['Label'] + time_columns_country].copy()

# --- For ExportsCommodity ---
sheet_name = 'ExportsCommodity'
df0_commodity = all_sheets[sheet_name]
df0_commodity.rename(columns={df0_commodity.columns[0]: 'Label'}, inplace=True)
time_columns_commodity = [c for c in df0_commodity.columns if str(c).strip().startswith('202') and 'Q' in str(c)]
if not time_columns_commodity:
    time_columns_commodity = list(df0_commodity.columns[1:11])
working_df_commodity = df0_commodity[['Label'] + time_columns_commodity].copy()

# --- For ExportCountry ---
working_country = working_df_country.copy()
working_numeric_country = working_country.set_index('Label').apply(pd.to_numeric, errors='coerce').fillna(0)

latest_col_country = working_numeric_country.columns[-1]
top_labels_by_latest_country = working_numeric_country[latest_col_country].nlargest(8).index.tolist()

data_for_area_country = working_numeric_country.loc[top_labels_by_latest_country].copy()
data_T_country = data_for_area_country.T


# --- For ExportsCommodity ---
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
    st.title("Exports Commodity Page")
    r1c1, r1c2 = st.columns([1,1])
    with r1c1:
        st.caption("Stacked area: Top 8 labels")
        try:
            fig, ax = plt.subplots(figsize=(5,3))
            # Make sure shapes match: x length = rows of data_T, y arrays per label must align
            x = data_T_commodity.index
            y = data_T_commodity.values.T  # shape: (n_labels, n_periods)
            ax.stackplot(x, y, labels=data_T_commodity.columns)
            ax.set_xticks(x[::max(1, int(len(x)/4))])
            ax.tick_params(axis='x', labelrotation=45)
            ax.set_title("Top 8 Labels")
            ax.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1,1))
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Couldn't draw stacked area: {e}")

    with r1c2:
        st.caption("Top 10 latest values")
        try:
            fig2, ax2 = plt.subplots(figsize=(5,3))
            latest_col =working_numeric_commodity.columns[-1]
            working_numeric_commodity[latest_col].nlargest(10).sort_values().plot(kind='barh', ax=ax2)
            ax2.set_title(f"Top 10 — {-1}")
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"Couldn't draw top 10 bar: {e}")

    # Row 2
    r2c1, r2c2 = st.columns([1,1])
    with r2c1:
        st.caption("Share distribution among top 6 (compact pie)")
        try:
            top6 =top6 = working_numeric_commodity.iloc[:, -1].astype(float).nlargest(6)
            fig3, ax3 = plt.subplots(figsize=(5,3))
            ax3.pie(top6, labels=top6.index, autopct='%1.0f%%', startangle=140)
            ax3.set_title("Top 6 share (latest)")
            st.pyplot(fig3)
        except Exception as e:  
            st.warning(f"Couldn't draw pie: {e}")


# GRAPHS FOR EXPORTCOUNTRIES
#------------------------------
if selected_page == "ExportCountry":
    st.title("Export Country Page")
    r1c1, r1c2 = st.columns([1,1])
    with r1c1:
        st.caption("Stacked area: Top 8 labels")
        try:
            fig, ax = plt.subplots(figsize=(5,3))
            # Make sure shapes match: x length = rows of data_T, y arrays per label must align
            x =data_T_country.index
            y = data_T_country.values.T  # shape: (n_labels, n_periods)
            ax.stackplot(x, y, labels=data_T_country.columns)
            ax.set_xticks(x[::max(1, int(len(x)/4))])
            ax.tick_params(axis='x', labelrotation=45)
            ax.set_title("Top 8 Labels")
            ax.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1,1))
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Couldn't draw stacked area: {e}")

    with r1c2:
        st.caption("Top 10 latest values")
        try:
            fig2, ax2 = plt.subplots(figsize=(5,3))
            latest_col = working_numeric_country.columns[-1]
            working_numeric_country[latest_col].nlargest(10).sort_values().plot(kind='barh', ax=ax2)
            ax2.set_title(f"Top 10 — {-1}")
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"Couldn't draw top 10 bar: {e}")

    # Row 2
    r2c1, r2c2 = st.columns([1,1])
    with r2c1:
        st.caption("Share distribution among top 6 (compact pie)")
        try:
            top6 =top6 = working_numeric_country.iloc[:, -1].astype(float).nlargest(6)
            fig3, ax3 = plt.subplots(figsize=(5,3))
            ax3.pie(top6, labels=top6.index, autopct='%1.0f%%', startangle=140)
            ax3.set_title("Top 6 share (latest)")
            st.pyplot(fig3)
        except Exception as e:  
            st.warning(f"Couldn't draw pie: {e}")

# -------------------------------
# MACHINE LEARNING SECTION
#---------------------------------

# Load the two sheets
# Read and merge
if selected_page == "Machine Learning Model":
    df1 = pd.read_excel(file_path, sheet_name="ExportsCommodity")
    df2 = pd.read_excel(file_path, sheet_name="ExportCountry")
    df2 = df2[df1.columns]
    merged_df = pd.concat([df1, df2], ignore_index=True)
    merged_df = merged_df.rename(columns={'Year and Period': 'Label'})

    df = merged_df
    # lets create growth change featuress of all quarters

    df['2023Q2_growth'] = (df['2023Q2'] - df['2023Q1']) / df['2023Q1']
    df['2023Q3_growth'] = (df['2023Q3'] - df['2023Q2']) / df['2023Q2']
    df['2023Q4_growth'] = (df['2023Q4'] - df['2023Q3']) / df['2023Q3']
    df['2024Q1_growth'] = (df['2024Q1'] - df['2023Q4']) / df['2023Q4']
    df['2024Q2_growth'] = (df['2024Q2'] - df['2024Q1']) / df['2024Q1']
    df['2024Q3_growth'] = (df['2024Q3'] - df['2024Q2']) / df['2024Q2']
    df['2024Q4_growth'] = (df['2024Q4'] - df['2024Q3']) / df['2024Q3']
    df['2025Q1_growth'] = (df['2025Q1'] - df['2024Q4']) / df['2024Q4']
    df['2025Q2_growth'] = (df['2025Q2'] - df['2025Q1']) / df['2025Q1']

    # Create total overall growth from first to last quarter
    df['total_growth'] = (df['2025Q2'] - df['2023Q1']) / df['2023Q1']

    # Select features (all quarterly values and growth columns except total_growth)
    X = df[['2023Q1', '2023Q2', '2023Q3', '2023Q4',
            '2024Q1', '2024Q2', '2024Q3', '2024Q4',
            '2025Q1', '2025Q2',
            '2023Q2_growth', '2023Q3_growth', '2023Q4_growth',
            '2024Q1_growth', '2024Q2_growth', '2024Q3_growth',
            '2024Q4_growth', '2025Q1_growth', '2025Q2_growth']]

    # Select target (the growth we want to predict or rank by)
    y = df['total_growth']


    # now we are going to train a regression model that will predict growth

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate how well the model performs
    st.write("R² Score:", r2_score(y_test, y_pred))
    st.write("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

    # Predict total growth for all entities (for ranking later)
    df['Predicted_Growth'] = model.predict(X)

    # then lets rank and see top investment opportunities
    # Sort all entities by predicted growth (highest first)
    df_sorted = df.sort_values(by='Predicted_Growth', ascending=False)

    # Display the top 10 opportunities
    print("Top 10 Investment Opportunities:")
    print(df_sorted[['Predicted_Growth', 'total_growth']].head(10))

    # Optionally, display the bottom 10 (lowest predicted growth)
    st.dataframe(df_sorted[['Label','Predicted_Growth', 'total_growth']])
    df_predictions = pd.read_csv(r"C:\Users\PC\OneDrive\Desktop\Nisr project\predictions.csv")


    #------------------------
    # ml graphs
    #-------------------------
    # Lets make graph that shows comparison of growth
    st.subheader("Comparison: Actual vs Predicted Growth")

    df_comparison = df.sort_values(by='Predicted_Growth', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(4, 3),dpi=100)
    x = np.arange(len(df_comparison['Label']))
    width = 0.35

    ax.bar(x - width/2, df_comparison['total_growth'], width, label='Actual Growth', color='lightgreen')
    ax.bar(x + width/2, df_comparison['Predicted_Growth'], width, label='Predicted Growth', color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(df_comparison['Label'], rotation=45, ha='right')
    ax.set_xlabel("Country / Commodity")
    ax.set_ylabel("Growth Rate")
    ax.set_title("Actual vs Predicted Growth (Top 10)")
    ax.legend()
    st.pyplot(fig, use_container_width=False)

    # we are going to make another graph
    st.write(" Feature opportunities where we can invest either as a country or a certain commodity") 
    # Select top 10
    top10 = df_sorted[['Label', 'Predicted_Growth', 'total_growth']].head(10)
    # --- Plot bar chart for top 10 predicted growth ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top10['Label'], top10['Predicted_Growth'], color='skyblue')
    ax.invert_yaxis()  # So highest value appears at top
    ax.set_xlabel("Predicted Growth Rate")
    ax.set_ylabel("Country / Commodity")
    ax.set_title("Top 10 Predicted Growth – Investment Opportunities")
    st.pyplot(fig)

    # Model performance graph and how its trust worthy
    st.write('This is a graph that shows how well our model is fitting and be trusted')
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual Growth")
    ax.set_ylabel("Predicted Growth")
    ax.set_title("Actual vs Predicted Growth")
    st.pyplot(fig)
