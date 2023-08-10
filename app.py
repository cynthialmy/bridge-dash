import dash
from dash import dcc
from dash import html
import holoviews as hv
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from flask_caching import Cache

hv.extension('bokeh', 'matplotlib')

app = dash.Dash(__name__, external_stylesheets=[
                dbc.themes.BOOTSTRAP])  # Use bootstrap stylesheet
server = app.server

# Cache configuration
cache = Cache(server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})
app.config.suppress_callback_exceptions = True


def load_and_process_data():
    df = pd.read_csv(
        'https://raw.githubusercontent.com/cynthialmy/NBI_Data/main/BridgesExport_AllYear.csv',
        dtype={3: str, 4: str, 41: str},
        low_memory=True,
        usecols=['8 - Structure Number', '16 - Latitude (decimal)', '17 - Longitude (decimal)',
                 '27 - Year Built', '29 - Average Daily Traffic', '3 - County Name', 'CAT10 - Bridge Condition',
                 '43B - Main Span Design', '43A - Main Span Material', '49 - Structure Length (ft.)',
                 '91 - Designated Inspection Frequency', '96 - Total Project Cost', '1 - State Name',
                 '58 - Deck Condition Rating', '59 - Superstructure Condition Rating',
                 '60 - Substructure Condition Rating', '64 - Operating Rating (US tons)',
                 '66 - Inventory Rating (US tons)', '114 - Future Average Daily Traffic',
                 '34 - Skew Angle (degrees)', '48 - Length of Maximum Span (ft.)',
                 '51 - Bridge Roadway Width Curb to Curb (ft.)', 'Computed - Average Daily Truck Traffic (Volume)',
                 'Average Relative Humidity', 'Average Temperature', 'Maximum Temperature', 'Minimum Temperature',
                 '106 - Year Reconstructed', 'CAT29 - Deck Area (sq. ft.)'])

    df.drop_duplicates(subset='8 - Structure Number', inplace=True)

    def condition_mapping(val):
        if val == 'Good':
            return 3
        elif val == 'Fair':
            return 2
        elif val == 'Poor':
            return 1
        else:
            return np.nan

    # df['CAT10 - Bridge Condition Numeric'] = df['CAT10 - Bridge Condition'].apply(condition_mapping)
    # df['Bridge Age (yr)'] = 2023 - df['27 - Year Built']  # Assume current year is 2023

    df.loc[:, 'CAT10 - Bridge Condition Numeric'] = df.loc[:,
                                                           'CAT10 - Bridge Condition'].apply(condition_mapping)
    df.loc[:, 'Bridge Age (yr)'] = 2023 - df.loc[:, '27 - Year Built']

    return df


# Load data
df = load_and_process_data()

# Define the options for each dropdown
x_options = [{'label': col, 'value': col} for col in
             ['17 - Longitude (decimal)', '29 - Average Daily Traffic', '27 - Year Built', 'Average Temperature',
              '114 - Future Average Daily Traffic', 'Bridge Age (yr)', 'CAT29 - Deck Area (sq. ft.)',
              '106 - Year Reconstructed', '34 - Skew Angle (degrees)', '48 - Length of Maximum Span (ft.)',
              '51 - Bridge Roadway Width Curb to Curb (ft.)', 'Computed - Average Daily Truck Traffic (Volume)',
              'Average Relative Humidity', 'Average Temperature', 'Maximum Temperature', 'Minimum Temperature']]
y_options = [{'label': col, 'value': col} for col in
             ['16 - Latitude (decimal)', '96 - Total Project Cost', '58 - Deck Condition Rating',
              '91 - Designated Inspection Frequency', '58 - Deck Condition Rating',
              '59 - Superstructure Condition Rating', '60 - Substructure Condition Rating',
              '64 - Operating Rating (US tons)', '66 - Inventory Rating (US tons)']]
color_options = [{'label': col, 'value': col} for col in
                 ['CAT10 - Bridge Condition', '43A - Main Span Material', '43B - Main Span Design', '3 - County Name',
                  '59 - Superstructure Condition Rating', '49 - Structure Length (ft.)', 'CAT10 - Bridge Condition',
                  '91 - Designated Inspection Frequency', '96 - Total Project Cost']]
size_options = [{'label': col, 'value': col} for col in ['49 - Structure Length (ft.)', '29 - Average Daily Traffic',
                                                         'Average Daily Truck Traffic (Percent ADT)',
                                                         '51 - Bridge Roadway Width Curb to Curb (ft.)',
                                                         '96 - Total Project Cost',
                                                         '91 - Designated Inspection Frequency',
                                                         '96 - Total Project Cost']]

# 'size' dropdown
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
size_dropdown_options = [{'label': col, 'value': col}
                         for col in numeric_columns]

# Define Georgia coordinates
georgia_coordinates = {
    'lat_min': 30.3556,
    'lat_max': 35.0000,
    'lon_min': -85.6052,
    'lon_max': -80.7514
}


def filter_georgia_coordinates_inplace(dataframe):
    mask = ((dataframe['16 - Latitude (decimal)'] >= georgia_coordinates['lat_min']) &
            (dataframe['16 - Latitude (decimal)'] <= georgia_coordinates['lat_max']) &
            (dataframe['17 - Longitude (decimal)'] >= georgia_coordinates['lon_min']) &
            (dataframe['17 - Longitude (decimal)'] <= georgia_coordinates['lon_max']))

    dataframe.drop(dataframe[~mask].index, inplace=True)


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Bridge Dash", style={'text-align': 'center'}),
            html.P(
                "This dashboard is a visual representation of bridge data. You can view and interact with various charts and graphs, and filter the data as per your needs.",
                style={'text-align': 'center', 'font-size': '24px'}),
            html.Br(),
            html.P(
                "You can select data for different years, enter county names, choose main span categories and much more. Each selection will dynamically update the charts below.",
                style={'text-align': 'left'}),
            html.Label('Select Year Built:'),
            dcc.RangeSlider(id="slct_year",
                            min=df["27 - Year Built"].min(),
                            max=df["27 - Year Built"].max(),
                            step=1,
                            value=[df["27 - Year Built"].min(),
                                   df["27 - Year Built"].max()],
                            marks={str(year): str(year) for year in
                                   range(int(df["27 - Year Built"].min()), int(df["27 - Year Built"].max()) + 1, 10)}
                            ),
            html.Div(id='output_container', children=[]),
            html.Br(),
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            html.P(
                "Please note that you can also export the charts as PNG images. Just hover over the top right corner of any chart, click on the camera icon, and the chart will be downloaded as a PNG image.",
                style={'text-align': 'left'}),
            html.Br(),
            html.Label('Input County Name:'),
            dcc.Input(id="input-county", type="text",
                      placeholder="Enter county name"),
            dcc.Graph(id='my-treemap'),
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Select Counties:"),  # Add label for county dropdown
            dcc.Dropdown(
                id='county_dropdown',
                # options=[{'label': i, 'value': i} for i in df['3 - County Name'].unique()],
                options=[{'label': i if pd.notna(i) else 'Unknown', 'value': i if pd.notna(i) else 'Unknown'} for i in
                         df['3 - County Name'].unique()],
                value=[],
                multi=True,
                placeholder='Select counties...'
            ),
            dcc.Graph(id='line_chart'),
            # dcc.Graph(id='second_line_chart'),
            # dcc.Graph(id='third_line_chart'),
            html.Div(className="two-columns", children=[
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='second_line_chart')
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='third_line_chart')
                    ]),
                ]),
            ]),
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            html.Label('Select Main Span Category:'),
            dcc.Dropdown(id="slct_span",
                         options=[
                             {'label': 'Main Span Material',
                                 'value': '43A - Main Span Material'},
                             {'label': 'Main Span Design',
                                 'value': '43B - Main Span Design'}
                         ],
                         value='43A - Main Span Material'
                         ),
            html.Div(id='stacked_bar'),
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Row([html.Div('x', style={'textAlign': 'center'})]),
            dbc.Row(
                [dcc.Dropdown(id='xaxis', options=x_options, value=x_options[0]['value'])]),
        ], width={"size": 6, "offset": 3}, md={"size": 3, "offset": 0}),
        dbc.Col([
            dbc.Row([html.Div('y', style={'textAlign': 'center'})]),
            dbc.Row(
                [dcc.Dropdown(id='yaxis', options=y_options, value=y_options[0]['value'])]),
        ], width={"size": 6, "offset": 3}, md={"size": 3, "offset": 0}),
        dbc.Col([
            dbc.Row([html.Div('Color', style={'textAlign': 'center'})]),
            dbc.Row([dcc.Dropdown(id='color', options=color_options,
                    value=color_options[0]['value'])]),
        ], width={"size": 6, "offset": 3}, md={"size": 3, "offset": 0}),
        dbc.Col([
            dbc.Row([html.Div('Size', style={'textAlign': 'center'})]),
            dbc.Row([dcc.Dropdown(id='size', options=size_options,
                    value=size_options[0]['value'])]),
        ], width={"size": 6, "offset": 3}, md={"size": 3, "offset": 0}),
    ]),
    dbc.Row([html.Div(id='scatter')]),
    html.Div(className="two-columns", children=[
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='dropdown',
                    options=[
                        {'label': 'Bridge Condition',
                            'value': 'CAT10 - Bridge Condition Numeric'},
                        {'label': 'Operating Rating (US tons)',
                         'value': '64 - Operating Rating (US tons)'},
                        {'label': 'Number of Spans in Main Unit',
                            'value': '45 - Number of Spans in Main Unit'},
                        {'label': 'Bridge Age (yr)',
                         'value': 'Bridge Age (yr)'}
                    ],
                    value='CAT10 - Bridge Condition Numeric'
                ),
                html.Div(id='density_map')
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='heatmap_dropdown',
                    options=[
                        {'label': 'Bridge Condition',
                            'value': 'CAT10 - Bridge Condition Numeric'},
                        {'label': 'Operating Rating (US tons)',
                         'value': '64 - Operating Rating (US tons)'},
                        {'label': 'Number of Spans in Main Unit',
                            'value': '45 - Number of Spans in Main Unit'},
                        {'label': 'Bridge Age (yr)',
                         'value': 'Bridge Age (yr)'}
                    ],
                    multi=False,
                    value='CAT10 - Bridge Condition Numeric'
                ),
                html.Div(id='heatmap')
            ]),
        ]),
    ]),
], fluid=True)


@app.callback(
    Output('my-treemap', 'figure'),
    [Input('slct_year', 'value'),
     Input('input-county', 'value')]
)
def update_treemap(selected_years, input_county):
    # Filter your dataframe based on the selected_years
    filtered_df = df[df["27 - Year Built"].between(
        selected_years[0], selected_years[1])]
    # filtered_df = df[(df['27 - Year Built'] >= selected_years[0]) & (df['27 - Year Built'] <= selected_years[1])]

    # If a county name was entered in the search box, filter the dataframe by that county
    if input_county is not None:
        filtered_df = filtered_df[filtered_df['3 - County Name'].str.contains(
            input_county, case=False, na=False)]

    # Fill NaN values in relevant columns
    filtered_df = filtered_df.fillna({'1 - State Name': 'Unknown', '3 - County Name': 'Unknown',
                                      'CAT10 - Bridge Condition': 'Unknown',
                                      '43B - Main Span Design': 'Unknown', '43A - Main Span Material': 'Unknown'})

    # Generate a new treemap with the filtered data
    fig = px.treemap(filtered_df,
                     path=['1 - State Name', '3 - County Name', 'CAT10 - Bridge Condition',
                           '43B - Main Span Design', '43A - Main Span Material'],
                     color='CAT10 - Bridge Condition',
                     color_continuous_scale='RdBu',
                     title="Treemap of Bridge Data")
    fig.update_layout(
        autosize=True,
        margin=dict(l=30, r=30, b=30, t=50),
    )

    return fig


@app.callback(
    Output(component_id='line_chart', component_property='figure'),
    [Input(component_id='slct_year', component_property='value'),
     Input(component_id='county_dropdown', component_property='value')]
)
def update_line_chart(selected_years, selected_counties):
    # Filter data by selected years
    dff = df[df["27 - Year Built"].between(
        selected_years[0], selected_years[1])]

    # Filter data by selected counties
    if selected_counties:
        dff = dff[dff['3 - County Name'].isin(selected_counties)]

    # Group by county name and year built, and calculate the mean
    dff = dff.groupby(['3 - County Name', '27 - Year Built']).agg(
        {'64 - Operating Rating (US tons)': 'mean',
         '66 - Inventory Rating (US tons)': 'mean'}).reset_index()

    # Create a figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces for Operating Rating and Inventory Rating
    for county in selected_counties:
        county_data = dff[dff['3 - County Name'] == county]
        fig.add_trace(
            go.Scatter(x=county_data["27 - Year Built"],
                       y=county_data['64 - Operating Rating (US tons)'],
                       name=f'{county} - Operating Rating'),
            secondary_y=True
        )
        fig.add_trace(
            go.Scatter(x=county_data["27 - Year Built"],
                       y=county_data['66 - Inventory Rating (US tons)'],
                       name=f'{county} - Inventory Rating'),
            secondary_y=True
        )

    # Set y-axes titles
    fig.update_yaxes(title_text="Original Unit (US tons)", secondary_y=True)

    # Additional layout customization
    fig.update_layout(
        title_text="Operating and Inventory Rating over Time by County",
        autosize=True,
        margin=dict(l=30, r=30, b=30, t=50),
    )

    return fig


@app.callback(
    Output(component_id='second_line_chart', component_property='figure'),
    [Input(component_id='slct_year', component_property='value'),
     Input(component_id='county_dropdown', component_property='value')]
)
def update_second_line_chart(selected_years, selected_counties):
    # Create a figure
    fig = go.Figure()

    # If no counties are selected, return an empty figure
    if not selected_counties:
        return fig

    # First filter data based on selected year range
    dff = df[df["27 - Year Built"].between(
        selected_years[0], selected_years[1])]

    # Specify the numeric columns you want to calculate the mean for
    numeric_cols = ['Average Relative Humidity',
                    'Average Temperature']

    # Group data by '27 - Year Built' and '3 - County Name', and calculate the mean only for numeric columns
    dff = dff.groupby(["27 - Year Built", "3 - County Name"]
                      )[numeric_cols].mean().reset_index()

    # Filter data by selected counties
    dff = dff[dff['3 - County Name'].isin(selected_counties)]

    # Loop through the selected counties and add a trace for each
    for county in selected_counties:
        county_data = dff[dff['3 - County Name'] == county]
        fig.add_trace(go.Scatter(
            x=county_data["27 - Year Built"], y=county_data['Average Relative Humidity'], name=county + ' - Average Relative Humidity'))
        fig.add_trace(go.Scatter(
            x=county_data["27 - Year Built"], y=county_data['Average Temperature'], name=county + ' - Average Temperature'))

    # Additional layout customization
    fig.update_layout(
        title_text="Weather Data over Time by County",
        autosize=True,
        margin=dict(l=30, r=30, b=30, t=50),
        showlegend=True  # Show the legend
    )

    return fig


@app.callback(
    Output(component_id='third_line_chart', component_property='figure'),
    [Input(component_id='slct_year', component_property='value'),
     Input(component_id='county_dropdown', component_property='value')]
)
def update_third_line_chart(selected_years, selected_counties):
    # First filter data based on selected year range
    dff = df[df["27 - Year Built"].between(
        selected_years[0], selected_years[1])]

    # Specify the numeric columns you want to calculate the mean for
    numeric_cols = ['29 - Average Daily Traffic']

    # Group data by '27 - Year Built' and '3 - County Name', and calculate the mean only for numeric columns
    dff = dff.groupby(["27 - Year Built", "3 - County Name"]
                      )[numeric_cols].mean().reset_index()

    # Filter data by selected counties
    if selected_counties:
        dff = dff[dff['3 - County Name'].isin(selected_counties)]

    # Create a figure
    fig = go.Figure()

    # Loop through the selected counties and add a trace for each
    for county in selected_counties:
        county_data = dff[dff['3 - County Name'] == county]
        fig.add_trace(go.Scatter(
            x=county_data["27 - Year Built"], y=county_data['29 - Average Daily Traffic'], name=county))

    # Additional layout customization
    fig.update_layout(
        title_text="Traffic Data over Time by County",
        autosize=True,
        margin=dict(l=30, r=30, b=30, t=50),
    )

    return fig


@app.callback(
    Output(component_id='stacked_bar', component_property='children'),
    [Input(component_id='slct_year', component_property='value'),
     Input(component_id='slct_span', component_property='value')]
)
def update_stacked_bar(option_slctd, span_slctd):
    dff = df[df["27 - Year Built"].between(option_slctd[0], option_slctd[1])]
    # dff = filter_georgia_coordinates(df).copy()
    # dff = dff[(dff["27 - Year Built"] >= option_slctd[0]) & (dff["27 - Year Built"] <= option_slctd[1])]

    # Get unique values from selected span column
    unique_values = dff[span_slctd].unique()

    # Stacked bar chart logic
    fig = go.Figure()
    for val in unique_values:
        filtered_df = dff[dff[span_slctd] == val]
        counts = filtered_df.groupby(
            '27 - Year Built').size().reset_index(name='Count')
        fig.add_trace(go.Bar(
            name=val,
            x=counts['27 - Year Built'],
            y=counts['Count']
        ))

    fig.update_layout(
        title_text=f'Count of Bridge Types by Year Built ({span_slctd})',
        xaxis_title='Year Built',
        yaxis_title='Count',
        barmode='stack'
    )

    fig.update_layout(
        autosize=True,
        margin=dict(l=30, r=30, b=30, t=50),
    )

    return dcc.Graph(figure=fig)


@app.callback(
    Output(component_id='scatter', component_property='children'),
    Input(component_id='slct_year', component_property='value'),
    Input('xaxis', 'value'),
    Input('yaxis', 'value'),
    Input('color', 'value'),
    Input('size', 'value')
)
def update_scatter(option_slctd, x, y, color, size):
    dff = df[df["27 - Year Built"].between(option_slctd[0], option_slctd[1])]
    filter_georgia_coordinates_inplace(dff)

    # Check if size column is numeric and not NaN
    if np.issubdtype(dff[size].dtype, np.number):
        dff = dff[np.isfinite(dff[size])]  # exclude NaN values

        fig = px.scatter(
            data_frame=dff,
            x=x,
            y=y,
            color=color,
            size=size,
            hover_name='43B - Main Span Design',
            color_continuous_scale='Viridis',
        )
    else:
        fig = px.scatter(
            data_frame=dff,
            x=x,
            y=y,
            color=color,
            hover_name='43B - Main Span Design',
            color_continuous_scale='Viridis',
        )

    fig.update_layout(
        title_text='Geographical Distribution of Bridges with Bridge Condition Rating',
        xaxis_title=x,
        yaxis_title=y,
    )

    fig.update_layout(
        autosize=True,
        height=800,
    )

    return dcc.Graph(figure=fig)


@app.callback(
    Output(component_id='density_map', component_property='children'),
    [Input(component_id='slct_year', component_property='value'),
     Input(component_id='dropdown', component_property='value')]
)
def update_density_map(option_slctd, dropdown_value):
    dff = df[df["27 - Year Built"].between(option_slctd[0], option_slctd[1])]
    # dff = filter_georgia_coordinates(df).copy()
    # dff = dff[(dff["27 - Year Built"] >= option_slctd[0]) &
    #           (dff["27 - Year Built"] <= option_slctd[1])]

    # Density map logic
    fig = go.Figure(go.Densitymapbox(
        lat=dff['16 - Latitude (decimal)'],
        lon=dff['17 - Longitude (decimal)'],
        z=dff[dropdown_value],
        radius=10,
        hovertemplate=(
            "<b>Structure Number:</b> %{customdata[0]}<br>"
            "<b>Year Built:</b> %{customdata[1]}<br>"
            "<b>County Name:</b> %{customdata[2]}<br>"
            "<b>Longitude:</b> %{customdata[3]}<br>"
            "<b>Latitude:</b> %{customdata[4]}<br>"
        ),
        customdata=dff[['8 - Structure Number', '27 - Year Built', '3 - County Name',
                        '17 - Longitude (decimal)', '16 - Latitude (decimal)']].values,
    ))

    fig.update_layout(
        title_text='Density Map of Bridge ' + dropdown_value,
        # mapbox_style='open-street-map',
        mapbox_style="carto-positron",
        mapbox_center_lat=33,
        mapbox_center_lon=-83,
        mapbox_zoom=6
    )

    fig.update_layout(
        autosize=True,
        height=800,
    )

    return dcc.Graph(figure=fig)


@app.callback(
    Output(component_id='heatmap', component_property='children'),
    [Input(component_id='slct_year', component_property='value'),
     Input(component_id='heatmap_dropdown', component_property='value')]
)
def update_heatmap(option_slctd, dropdown_value):
    # Ensure dropdown_value is a string
    if not isinstance(dropdown_value, str):
        dropdown_value = dropdown_value[0]

    dff = df[df["27 - Year Built"].between(option_slctd[0], option_slctd[1])]

    # dff = filter_georgia_coordinates(df).copy()
    # dff = dff[(dff["27 - Year Built"] >= option_slctd[0]) &
    #           (dff["27 - Year Built"] <= option_slctd[1])]

    # lat_bins = np.arange(dff['16 - Latitude (decimal)'].min(), dff['16 - Latitude (decimal)'].max(), 0.1)
    # lon_bins = np.arange(dff['17 - Longitude (decimal)'].min(), dff['17 - Longitude (decimal)'].max(), 0.1)

    lat_bins = np.linspace(
        georgia_coordinates['lat_min'], georgia_coordinates['lat_max'], 100)
    lon_bins = np.linspace(
        georgia_coordinates['lon_min'], georgia_coordinates['lon_max'], 100)

    dff['lat_bin'] = pd.cut(dff['16 - Latitude (decimal)'],
                            bins=lat_bins, include_lowest=True, right=True)
    dff['lon_bin'] = pd.cut(dff['17 - Longitude (decimal)'],
                            bins=lon_bins, include_lowest=True, right=True)

    # Aggregate values based on the bins
    z_values = dff.groupby(['lat_bin', 'lon_bin'])[
        dropdown_value].mean().unstack().fillna(0).values

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=lon_bins,
        y=lat_bins,
        colorscale='Viridis'))

    fig.update_layout(title_text='Heatmap of selected feature')

    fig.update_layout(
        autosize=True,
        height=800,
    )

    return dcc.Graph(figure=fig)


if __name__ == '__main__':
    app.run_server(debug=True)  # use_reloader=False
