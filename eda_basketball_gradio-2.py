import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import gradio as gr


# Parsing basketball player data from the web
def parse_data(year: int):

    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    parsed_df = pd.read_html(url, header=0)[0]
    parsed_df = parsed_df.drop(parsed_df[parsed_df['Age'] == 'Age'].index)
    parsed_df = parsed_df.fillna(0)
    parsed_df = parsed_df.drop(['Rk'], axis=1)

    parsed_df['FG%'] = parsed_df['FG%'].astype(str)
    #parsed_df['3P%'] = parsed_df['3P%'].astype(str)
    #parsed_df['2P%'] = parsed_df['2P%'].astype(str)
    #parsed_df['eFG%'] = parsed_df['eFG%'].astype(str)
    parsed_df['FT%'] = parsed_df['FT%'].astype(str)
    parsed_df['Age'] = parsed_df['Age'].astype(float).astype(int)
    return parsed_df


# Displaying selected player stats and heatmap based on filters
def display_data(selected_year, selected_team, selected_position, min_age, max_age, show_heatmap):
    
    df = parse_data(selected_year)

    df['Team'] = df['Team'].astype(str)
    df_filtered = df[
        (df['Team'].isin(selected_team)) &
        (df['Pos'].isin(selected_position)) &
        (df['Age'].between(min_age, max_age))
    ]

    data_shape = f"Data Dimension: Rows - {df_filtered.shape[0]}, Columns - {df_filtered.shape[1]}"
    download_link = download_dataset(df_filtered)

    fig = None
    if show_heatmap:
        relevant_columns = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%']
        df_filtered_corr = df_filtered[relevant_columns].astype(float)

        corr = df_filtered_corr.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, mask=mask, vmax=1, square=True, ax=ax)

    

    return data_shape, df_filtered, download_link, fig


# Utility function to generate a CSV download link
def download_dataset(dataset):
    csv = dataset.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="player_stats.csv">Download CSV File</a>'


# Gradio UI components
def gradio_app():

    teams = parse_data(2020)['Team'].unique().tolist()

    selected_year = gr.components.Slider(minimum=1950, maximum=2019, step=1, label="Year")
    selected_team = gr.components.CheckboxGroup(teams, label="Team")
    selected_position = gr.components.CheckboxGroup(['C', 'PF', 'SF', 'PG', 'SG'], label="Position")
    min_age = gr.components.Slider(minimum=18, maximum=40, label="Minimum Age")
    max_age = gr.components.Slider(minimum=18, maximum=40, label="Maximum Age")
    show_heatmap = gr.components.Checkbox(label="Show Inter-correlation Heatmap")

    output_shape = gr.components.Textbox(label="Dataset Dimensions")
    output_df = gr.components.Dataframe(label="Filtered Player Stats")
    output_link = gr.components.HTML(label="Download Link")
    output_plot = gr.components.Plot(label="Heatmap")

    return gr.Interface(
        fn=display_data,
        inputs=[selected_year, selected_team, selected_position, min_age, max_age, show_heatmap],
        outputs=[output_shape, output_df, output_link, output_plot],
        live=True,
        title="EDA Basketball"
    )


gradio_app().launch()
