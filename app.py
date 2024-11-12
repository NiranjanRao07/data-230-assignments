import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.title("# EDA Basketball")

# Sidebar: Input Year
st.sidebar.header("Input Parameters")
selected_year = st.sidebar.selectbox("Year", list(reversed(range(1950, 2020))))

"""
Parsing basketball players info from https://www.basketball-reference.com
"""


@st.cache_data(show_spinner=True)
def parse_data(year: str):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    parsed_df = pd.read_html(url, header=0)[0]
    parsed_df = parsed_df[parsed_df["Age"] != "Age"]  # Remove headers in body
    parsed_df = parsed_df.fillna(0)
    parsed_df = parsed_df.drop(["Rk"], axis=1)

    # Ensure percentage columns are strings
    percentage_cols = ["FG%", "3P%", "2P%", "eFG%", "FT%"]
    parsed_df[percentage_cols] = parsed_df[percentage_cols].astype(str)

    # Handle 'Age' conversion safely
    parsed_df["Age"] = (
        pd.to_numeric(parsed_df["Age"], errors="coerce").fillna(0).astype(int)
    )

    for col in parsed_df.columns:
        if parsed_df[col].dtype == "object":
            parsed_df[col] = parsed_df[col].astype(str)

    return parsed_df


df_player_stat_dataset = parse_data(str(selected_year))

# Debugging: Check columns and head
print(df_player_stat_dataset.columns)
print(df_player_stat_dataset.head())

# Team Filter
sorted_dataset_by_team = []
if "Team" in df_player_stat_dataset.columns:
    df_player_stat_dataset["Team"] = df_player_stat_dataset["Team"].astype(str)
    sorted_dataset_by_team = sorted(df_player_stat_dataset["Team"].unique())
else:
    st.warning("Column 'Team' not found in the dataset.")

selected_team = st.sidebar.multiselect(
    "Team", sorted_dataset_by_team, default=sorted_dataset_by_team
)

# Position Filter
player_positions = ["C", "PF", "SF", "PG", "SG"]
selected_position = st.sidebar.multiselect(
    "Position", player_positions, default=player_positions
)

# Age Filter Slider
unique_age_values = df_player_stat_dataset["Age"].unique()
minValue, maxValue = min(unique_age_values), max(unique_age_values)
selected_age = st.sidebar.slider(
    "Age", int(minValue), int(maxValue), (int(minValue), int(maxValue)), 1
)
min_age, max_age = selected_age

# Filtered Dataset
df_selected_dataset = df_player_stat_dataset[
    (df_player_stat_dataset["Team"].isin(selected_team))
    & (df_player_stat_dataset["Pos"].isin(selected_position))
    & (df_player_stat_dataset["Age"].between(min_age, max_age))
]

# Display Filtered Dataset
st.header("Display Player Stats of Selected Team(s)")
st.write(
    f"Data Dimension: {df_selected_dataset.shape[0]} rows, {df_selected_dataset.shape[1]} columns"
)
st.dataframe(df_selected_dataset)


# Download Dataset Function
def download_dataset(dataset):
    csv = dataset.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encode as Base64
    return f'<a href="data:file/csv;base64,{b64}" download="player_stats.csv">Download CSV File</a>'


st.markdown(download_dataset(df_selected_dataset), unsafe_allow_html=True)

# Inter-correlation Heatmap Button
if st.button("Inter-correlation Heatmap"):
    st.header("Inter-correlation Heatmap")

    numeric_df = df_selected_dataset.select_dtypes(include=["float64", "int64"])

    if not df_selected_dataset.empty:
        corr = numeric_df.corr()  # Calculate correlations

        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True  # Mask upper triangle

        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(
                corr,
                mask=mask,
                vmax=1,
                square=True,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                annot_kws={"size": 8},
                cbar_kws={"shrink": 0.75},
            )
        st.pyplot(f)
    else:
        st.warning("No data available to display the heatmap.")
