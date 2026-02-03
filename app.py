import io
import ssl
import urllib.request

import certifi
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Bridge Dash", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/cynthialmy/NBI_Data/main/BridgesExport_AllYear.csv"


@st.cache_data
def load_and_process_data():
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(DATA_URL, context=ctx) as resp:
        raw = resp.read()
    df = pd.read_csv(
        io.BytesIO(raw),
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

    df.loc[:, 'CAT10 - Bridge Condition Numeric'] = df.loc[:,
                                                           'CAT10 - Bridge Condition'].apply(condition_mapping)
    df.loc[:, 'Bridge Age (yr)'] = 2023 - df.loc[:, '27 - Year Built']

    return df


df = load_and_process_data()

# Option lists for dropdowns (same as Dash)
X_OPTIONS = [
    '17 - Longitude (decimal)', '29 - Average Daily Traffic', '27 - Year Built', 'Average Temperature',
    '114 - Future Average Daily Traffic', 'Bridge Age (yr)', 'CAT29 - Deck Area (sq. ft.)',
    '106 - Year Reconstructed', '34 - Skew Angle (degrees)', '48 - Length of Maximum Span (ft.)',
    '51 - Bridge Roadway Width Curb to Curb (ft.)', 'Computed - Average Daily Truck Traffic (Volume)',
    'Average Relative Humidity', 'Average Temperature', 'Maximum Temperature', 'Minimum Temperature'
]
Y_OPTIONS = [
    '16 - Latitude (decimal)', '96 - Total Project Cost', '58 - Deck Condition Rating',
    '91 - Designated Inspection Frequency', '59 - Superstructure Condition Rating',
    '60 - Substructure Condition Rating', '64 - Operating Rating (US tons)', '66 - Inventory Rating (US tons)'
]
COLOR_OPTIONS = [
    'CAT10 - Bridge Condition', '43A - Main Span Material', '43B - Main Span Design', '3 - County Name',
    '59 - Superstructure Condition Rating', '49 - Structure Length (ft.)',
    '91 - Designated Inspection Frequency', '96 - Total Project Cost'
]
SIZE_OPTIONS = [
    '49 - Structure Length (ft.)', '29 - Average Daily Traffic',
    '51 - Bridge Roadway Width Curb to Curb (ft.)',
    '96 - Total Project Cost', '91 - Designated Inspection Frequency'
]
DENSITY_HEATMAP_OPTIONS = [
    'CAT10 - Bridge Condition Numeric',
    '64 - Operating Rating (US tons)',
    'Bridge Age (yr)'
]
# Only include options that exist in df
DENSITY_HEATMAP_OPTIONS = [c for c in DENSITY_HEATMAP_OPTIONS if c in df.columns]

georgia_coordinates = {
    'lat_min': 30.3556,
    'lat_max': 35.0000,
    'lon_min': -85.6052,
    'lon_max': -80.7514
}


def filter_georgia_coordinates(dataframe):
    mask = ((dataframe['16 - Latitude (decimal)'] >= georgia_coordinates['lat_min']) &
            (dataframe['16 - Latitude (decimal)'] <= georgia_coordinates['lat_max']) &
            (dataframe['17 - Longitude (decimal)'] >= georgia_coordinates['lon_min']) &
            (dataframe['17 - Longitude (decimal)'] <= georgia_coordinates['lon_max']))
    return dataframe.loc[mask].copy()


# ---- Sidebar: filters ----
with st.sidebar:
    st.header("Filters")
    year_min = int(df["27 - Year Built"].min())
    year_max = int(df["27 - Year Built"].max())
    selected_years = st.slider(
        "Select Year Built",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
        step=1
    )
    input_county = st.text_input("Input County Name", placeholder="Enter county name")
    county_options = [i if pd.notna(i) else 'Unknown' for i in df['3 - County Name'].unique()]
    selected_counties = st.multiselect("Select Counties", options=county_options, default=[])
    span_slctd = st.selectbox(
        "Select Main Span Category",
        options=['43A - Main Span Material', '43B - Main Span Design'],
        index=0
    )

# Base filtered df by year (and optional county name for treemap)
dff = df[df["27 - Year Built"].between(selected_years[0], selected_years[1])]
if input_county:
    dff_treemap = dff[dff['3 - County Name'].str.contains(input_county, case=False, na=False)].copy()
else:
    dff_treemap = dff.copy()

# ---- Main ----
st.title("Bridge Dash")
st.markdown(
    "This dashboard is a visual representation of bridge data. You can view and interact with various charts and graphs, and filter the data as per your needs."
)
st.markdown(
    "You can select data for different years, enter county names, choose main span categories and much more. Each selection will dynamically update the charts below."
)
st.caption("You can export charts as PNG from the hover menu on each chart (camera icon).")

# Treemap
treemap_df = dff_treemap.fillna({
    '1 - State Name': 'Unknown', '3 - County Name': 'Unknown',
    'CAT10 - Bridge Condition': 'Unknown',
    '43B - Main Span Design': 'Unknown', '43A - Main Span Material': 'Unknown'
})
fig_treemap = px.treemap(
    treemap_df,
    path=['1 - State Name', '3 - County Name', 'CAT10 - Bridge Condition',
          '43B - Main Span Design', '43A - Main Span Material'],
    color='CAT10 - Bridge Condition',
    color_continuous_scale='RdBu',
    title="Treemap of Bridge Data"
)
fig_treemap.update_layout(autosize=True, margin=dict(l=30, r=30, b=30, t=50))
st.plotly_chart(fig_treemap, use_container_width=True)

# Line charts (depend on selected counties)
st.subheader("Operating and Inventory Rating over Time by County")
dff_line = dff.copy()
if selected_counties:
    dff_line = dff_line[dff_line['3 - County Name'].isin(selected_counties)]
dff_line = dff_line.groupby(['3 - County Name', '27 - Year Built']).agg(
    {'64 - Operating Rating (US tons)': 'mean', '66 - Inventory Rating (US tons)': 'mean'}).reset_index()

fig_line = make_subplots(specs=[[{"secondary_y": True}]])
for county in selected_counties:
    county_data = dff_line[dff_line['3 - County Name'] == county]
    fig_line.add_trace(
        go.Scatter(x=county_data["27 - Year Built"],
                   y=county_data['64 - Operating Rating (US tons)'],
                   name=f'{county} - Operating Rating'),
        secondary_y=True
    )
    fig_line.add_trace(
        go.Scatter(x=county_data["27 - Year Built"],
                   y=county_data['66 - Inventory Rating (US tons)'],
                   name=f'{county} - Inventory Rating'),
        secondary_y=True
    )
fig_line.update_yaxes(title_text="Original Unit (US tons)", secondary_y=True)
fig_line.update_layout(
    title_text="Operating and Inventory Rating over Time by County",
    autosize=True, margin=dict(l=30, r=30, b=30, t=50),
)
st.plotly_chart(fig_line, use_container_width=True)

# Two-column: Weather and Traffic line charts
col1, col2 = st.columns(2)
with col1:
    st.subheader("Weather Data over Time by County")
    if not selected_counties:
        st.info("Select counties to see charts.")
    else:
        dff_weather = dff.groupby(["27 - Year Built", "3 - County Name"])[
            ['Average Relative Humidity', 'Average Temperature']].mean().reset_index()
        dff_weather = dff_weather[dff_weather['3 - County Name'].isin(selected_counties)]
        fig_weather = go.Figure()
        for county in selected_counties:
            county_data = dff_weather[dff_weather['3 - County Name'] == county]
            fig_weather.add_trace(go.Scatter(
                x=county_data["27 - Year Built"], y=county_data['Average Relative Humidity'],
                name=county + ' - Average Relative Humidity'))
            fig_weather.add_trace(go.Scatter(
                x=county_data["27 - Year Built"], y=county_data['Average Temperature'],
                name=county + ' - Average Temperature'))
        fig_weather.update_layout(
            title_text="Weather Data over Time by County",
            autosize=True, margin=dict(l=30, r=30, b=30, t=50), showlegend=True
        )
        st.plotly_chart(fig_weather, use_container_width=True)

with col2:
    st.subheader("Traffic Data over Time by County")
    if not selected_counties:
        st.info("Select counties to see charts.")
    else:
        dff_traffic = dff.groupby(["27 - Year Built", "3 - County Name"])['29 - Average Daily Traffic'].mean().reset_index()
        dff_traffic = dff_traffic[dff_traffic['3 - County Name'].isin(selected_counties)]
        fig_traffic = go.Figure()
        for county in selected_counties:
            county_data = dff_traffic[dff_traffic['3 - County Name'] == county]
            fig_traffic.add_trace(go.Scatter(
                x=county_data["27 - Year Built"], y=county_data['29 - Average Daily Traffic'], name=county))
        fig_traffic.update_layout(
            title_text="Traffic Data over Time by County",
            autosize=True, margin=dict(l=30, r=30, b=30, t=50),
        )
        st.plotly_chart(fig_traffic, use_container_width=True)

# Stacked bar
dff_bar = dff.copy()
unique_values = dff_bar[span_slctd].unique()
fig_bar = go.Figure()
for val in unique_values:
    filtered_bar = dff_bar[dff_bar[span_slctd] == val]
    counts = filtered_bar.groupby('27 - Year Built').size().reset_index(name='Count')
    fig_bar.add_trace(go.Bar(name=val, x=counts['27 - Year Built'], y=counts['Count']))
fig_bar.update_layout(
    title_text=f'Count of Bridge Types by Year Built ({span_slctd})',
    xaxis_title='Year Built', yaxis_title='Count', barmode='stack',
    autosize=True, margin=dict(l=30, r=30, b=30, t=50),
)
st.plotly_chart(fig_bar, use_container_width=True)

# Scatter: x, y, color, size
st.subheader("Scatter: Geographical Distribution")
x_axis = st.selectbox("X", options=X_OPTIONS, index=0, key="x_axis")
y_axis = st.selectbox("Y", options=Y_OPTIONS, index=0, key="y_axis")
color_axis = st.selectbox("Color", options=COLOR_OPTIONS, index=0, key="color_axis")
size_axis = st.selectbox("Size", options=SIZE_OPTIONS, index=0, key="size_axis")

dff_scatter = dff.copy()
dff_scatter = filter_georgia_coordinates(dff_scatter)
if size_axis in dff_scatter.columns and np.issubdtype(dff_scatter[size_axis].dtype, np.number):
    dff_scatter = dff_scatter[np.isfinite(dff_scatter[size_axis])]
    fig_scatter = px.scatter(
        data_frame=dff_scatter, x=x_axis, y=y_axis, color=color_axis, size=size_axis,
        hover_name='43B - Main Span Design', color_continuous_scale='Viridis',
    )
else:
    fig_scatter = px.scatter(
        data_frame=dff_scatter, x=x_axis, y=y_axis, color=color_axis,
        hover_name='43B - Main Span Design', color_continuous_scale='Viridis',
    )
fig_scatter.update_layout(
    title_text='Geographical Distribution of Bridges with Bridge Condition Rating',
    xaxis_title=x_axis, yaxis_title=y_axis, autosize=True, height=800,
)
st.plotly_chart(fig_scatter, use_container_width=True)

# Density map and Heatmap in two columns
density_metric = st.selectbox(
    "Density map metric",
    options=DENSITY_HEATMAP_OPTIONS,
    index=0,
    key="density_metric"
)
fig_density = go.Figure(go.Densitymapbox(
    lat=dff['16 - Latitude (decimal)'],
    lon=dff['17 - Longitude (decimal)'],
    z=dff[density_metric],
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
fig_density.update_layout(
    title_text='Density Map of Bridge ' + density_metric,
    mapbox_style="carto-positron",
    mapbox_center_lat=33, mapbox_center_lon=-83, mapbox_zoom=6,
    autosize=True, height=800,
)
st.plotly_chart(fig_density, use_container_width=True)

heatmap_metric = st.selectbox(
    "Heatmap metric",
    options=DENSITY_HEATMAP_OPTIONS,
    index=0,
    key="heatmap_metric"
)
if not isinstance(heatmap_metric, str):
    heatmap_metric = heatmap_metric[0] if heatmap_metric else DENSITY_HEATMAP_OPTIONS[0]
dff_heat = dff.copy()
lat_bins = np.linspace(georgia_coordinates['lat_min'], georgia_coordinates['lat_max'], 100)
lon_bins = np.linspace(georgia_coordinates['lon_min'], georgia_coordinates['lon_max'], 100)
dff_heat['lat_bin'] = pd.cut(dff_heat['16 - Latitude (decimal)'],
                              bins=lat_bins, include_lowest=True, right=True)
dff_heat['lon_bin'] = pd.cut(dff_heat['17 - Longitude (decimal)'],
                              bins=lon_bins, include_lowest=True, right=True)
z_values = dff_heat.groupby(['lat_bin', 'lon_bin'])[heatmap_metric].mean().unstack().fillna(0).values
fig_heat = go.Figure(data=go.Heatmap(z=z_values, x=lon_bins, y=lat_bins, colorscale='Viridis'))
fig_heat.update_layout(title_text='Heatmap of selected feature', autosize=True, height=800)
st.plotly_chart(fig_heat, use_container_width=True)
