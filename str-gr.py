import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import gradio as gr
from io import BytesIO
from PIL import Image


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


def update_team_choices(year):
    df = parse_data(year)
    teams = sorted(df["Team"].unique()) if "Team" in df.columns else []
    return gr.update(choices=teams, value=teams)


def filter_data(selected_year, selected_team, selected_position, min_age, max_age):
    df_player_stat_dataset = parse_data(selected_year)

    # Filter based on selections
    df_selected_dataset = df_player_stat_dataset[
        (df_player_stat_dataset["Team"].isin(selected_team))
        & (df_player_stat_dataset["Pos"].isin(selected_position))
        & (df_player_stat_dataset["Age"].between(min_age, max_age))
    ]

    data_dim = f"Data Dimension: {df_selected_dataset.shape[0]} rows, {df_selected_dataset.shape[1]} columns"
    return df_selected_dataset, data_dim


def download_dataset(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="player_stats.csv">Download CSV File</a>'


def generate_heatmap(df_selected_dataset):
    if not df_selected_dataset.empty:
        numeric_df = df_selected_dataset.select_dtypes(include=["float64", "int64"])
        corr = numeric_df.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True

        plt.figure(figsize=(12, 8))
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

        # Save plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format="PNG")
        buf.seek(0)
        plt.close()  # Close the plot to free up memory

        # Open the image with PIL and return
        return Image.open(buf)
    else:
        return "No data available to display the heatmap."


# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("# EDA Basketball")

    # Year Dropdown
    selected_year = gr.Dropdown(
        label="Year", choices=list(map(str, reversed(range(1950, 2020)))), value="2019"
    )

    # Team and Position Filters
    selected_team = gr.CheckboxGroup(label="Team", choices=[])
    selected_position = gr.CheckboxGroup(
        label="Position",
        choices=["C", "PF", "SF", "PG", "SG"],
        value=["C", "PF", "SF", "PG", "SG"],
    )

    # Age Range Sliders (Separate Min and Max Age)
    min_age = gr.Slider(label="Min Age", minimum=18, maximum=40, step=1, value=18)
    max_age = gr.Slider(label="Max Age", minimum=18, maximum=40, step=1, value=40)

    # Display Data and Filter Button
    df_display = gr.Dataframe(label="Filtered Player Stats")
    data_dim = gr.Markdown()

    # Update team choices based on selected year
    selected_year.change(
        update_team_choices, inputs=selected_year, outputs=selected_team
    )

    # Filter data
    filter_button = gr.Button("Filter Data")
    filter_button.click(
        filter_data,
        inputs=[selected_year, selected_team, selected_position, min_age, max_age],
        outputs=[df_display, data_dim],
    )

    # Generate Heatmap Button
    heatmap_output = gr.Image(label="Inter-correlation Heatmap")
    heatmap_button = gr.Button("Generate Inter-correlation Heatmap")
    heatmap_button.click(generate_heatmap, inputs=[df_display], outputs=heatmap_output)

    # Download link
    download_link = gr.Markdown()
    download_button = gr.Button("Download Data")
    download_button.click(download_dataset, inputs=[df_display], outputs=download_link)

# Launch the Gradio app
app.launch()
