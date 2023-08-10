# Import necessary libraries
from bokeh.io import curdoc
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, LinearColorMapper, BasicTicker, ColorBar, HoverTool
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.palettes import Viridis256, Spectral6, Category10
from bokeh.models import FactorRange, Legend, LegendItem
from bokeh.tile_providers import get_provider, Vendors

import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('BridgesExport_AllYear.csv')

# Remove zero or NaN values
df = df.replace(0, np.nan)
df = df.dropna(subset=['17 - Longitude (decimal)', '16 - Latitude (decimal)'])
df['Bridge Age (yr)'].fillna(df['Bridge Age (yr)'].mean(), inplace=True)  # Fill with mean value
df['Bridge Age (yr)'] = df['Bridge Age (yr)'].astype(int)

# Remove outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Prepare data for scatter plot
source_scatter = ColumnDataSource(df)

scatter_plot = figure(title="Year Built vs Operating Rating",
           x_axis_label='Year Built',
           y_axis_label='Operating Rating (US tons)')

scatter_plot.circle('27 - Year Built',
                    '64 - Operating Rating (US tons)',
                    source=source_scatter,
                    color=factor_cmap('43A - Main Span Material', palette=Spectral6, factors=df['43A - Main Span Material'].unique()),
                    fill_alpha=0.6,
                    legend_field='43A - Main Span Material')

scatter_plot.legend.click_policy="hide"
scatter_plot.legend.location = "top_right"
scatter_plot.legend.orientation = "vertical"
scatter_plot.legend.label_text_font_size = "8pt"
scatter_plot.legend.spacing = 2
scatter_plot.legend.margin = 10
scatter_plot.legend.padding = 10

# Scatter plot hover tool
scatter_hover = HoverTool(tooltips=[("Year Built", "@{27 - Year Built}"),
                                    ("Operating Rating", "@{64 - Operating Rating (US tons)}"),
                                    ("Material", "@{43A - Main Span Material}")])  # Add material to hover tool
scatter_plot.add_tools(scatter_hover)

# Prepare data for bar chart
material_counts = df['43A - Main Span Material'].value_counts()
materials = material_counts.index.tolist()
counts = material_counts.values.tolist()

source_bar = ColumnDataSource(data=dict(materials=materials, counts=counts))

bar_plot = figure(x_range=materials, title="Main Span Material counts",
                  toolbar_location=None)

bar_plot.vbar(x='materials', top='counts', width=0.9, source=source_bar,
              legend_field="materials",
              fill_color=factor_cmap('materials', palette=Spectral6, factors=materials))

bar_plot.xaxis.major_label_orientation = 1.2

# Bar plot hover tool
bar_hover = HoverTool(tooltips=[("Material", "@materials"),
                                ("Counts", "@counts")])
bar_plot.add_tools(bar_hover)

# Group data by 'Year' and '43A - Main Span Material'
df_grouped = df.groupby(['Year', '43A - Main Span Material'])['29 - Average Daily Traffic'].mean().reset_index()

line_plot = figure(title="Average Daily Traffic Over Years",
                   x_axis_label='Year',
                   y_axis_label='Average Daily Traffic')

# Get a list of unique materials
materials = df['43A - Main Span Material'].unique()

# Add a line to the plot for each material
for i, material in enumerate(materials):
    df_material = df_grouped[df_grouped['43A - Main Span Material'] == material]
    source_line = ColumnDataSource(df_material)
    line_plot.line('Year', '29 - Average Daily Traffic', source=source_line, legend_label=material, color=Category10[10][i % 10])

# Line plot hover tool
line_hover = HoverTool(tooltips=[("Year", "@Year"),
                                 ("Average Daily Traffic", "@{29 - Average Daily Traffic}"),
                                 ("Material", "@{43A - Main Span Material}")])  # Add material to hover tool
line_plot.add_tools(line_hover)

# Show legend
line_plot.legend.click_policy="hide"
line_plot.legend.location = "top_right"
line_plot.legend.orientation = "vertical"
line_plot.legend.label_text_font_size = "8pt"
line_plot.legend.spacing = 2
line_plot.legend.margin = 10
line_plot.legend.padding = 10

from bokeh.palettes import Category10

# Group data by 'Year' and '43A - Main Span Material'
df_grouped_operating = df.groupby(['Year', '43A - Main Span Material'])[
    '64 - Operating Rating (US tons)'].mean().reset_index()
df_grouped_inventory = df.groupby(['Year', '43A - Main Span Material'])[
    '66 - Inventory Rating (US tons)'].mean().reset_index()

line_plot_2 = figure(title="Average Operating and Inventory Ratings Over Years",
                     x_axis_label='Year',
                     y_axis_label='Average Rating (US tons)')

# Get a list of unique materials
materials = df['43A - Main Span Material'].unique()

# Add a line to the plot for each material
for i, material in enumerate(materials):
    df_material_operating = df_grouped_operating[df_grouped_operating['43A - Main Span Material'] == material]
    df_material_inventory = df_grouped_inventory[df_grouped_inventory['43A - Main Span Material'] == material]

    source_line_operating = ColumnDataSource(df_material_operating)
    source_line_inventory = ColumnDataSource(df_material_inventory)

    line_plot_2.line('Year', '64 - Operating Rating (US tons)', source=source_line_operating,
                     legend_label=material + " (Operating)", color=Category10[10][i % 10])
    line_plot_2.line('Year', '66 - Inventory Rating (US tons)', source=source_line_inventory,
                     legend_label=material + " (Inventory)", color=Category10[10][i % 10], line_dash="dashed")

# Line plot hover tool
line_hover_2 = HoverTool(tooltips=[("Year", "@Year"),
                                   ("Operating Rating", "@{64 - Operating Rating (US tons)}"),
                                   ("Inventory Rating", "@{66 - Inventory Rating (US tons)}"),
                                   ("Material", "@{43A - Main Span Material}")])  # Add material to hover tool
line_plot_2.add_tools(line_hover_2)

# Show legend
line_plot_2.legend.click_policy="hide"
line_plot_2.legend.location = "top_right"
line_plot_2.legend.orientation = "vertical"
line_plot_2.legend.label_text_font_size = "8pt"
line_plot_2.legend.spacing = 2
line_plot_2.legend.margin = 10
line_plot_2.legend.padding = 10

# Replace nan values with 'Unknown'
df['CAT10 - Bridge Condition'].fillna('Unknown', inplace=True)

# Create a list of unique bridge conditions
bridge_conditions = df['CAT10 - Bridge Condition'].unique().tolist()

geo_source = ColumnDataSource(df)
geo_plot = figure(title="Geographical Distribution of Bridges",
                  x_axis_label='Longitude',
                  y_axis_label='Latitude',
                  tools="pan,wheel_zoom,reset")
geo_plot.add_tile(Vendors.CARTODBPOSITRON)  # Add a background map

# Map colors based on the bridge condition
geo_plot.circle(x='17 - Longitude (decimal)',
                y='16 - Latitude (decimal)',
                source=geo_source,
                color=factor_cmap('CAT10 - Bridge Condition', palette=Spectral6, factors=bridge_conditions),  # Color mapping
                fill_alpha=0.1,
                size=5)

geo_hover = HoverTool(tooltips=[("Longitude", "@{17 - Longitude (decimal)}"),
                                ("Latitude", "@{16 - Latitude (decimal)}"),
                                ("Structure Number", "@{8 - Structure Number}"),
                                ("State", "@{1 - State Name}"),
                                ("Bridge Condition", "@{CAT10 - Bridge Condition}")])  # Add bridge condition to hover tool
geo_plot.add_tools(geo_hover)

# Bridge Age and Bridge Condition Rating

df_grouped_age_condition = df.groupby(['Bridge Age (yr)', 'CAT10 - Bridge Condition']).size().reset_index(name='counts')
df_grouped_age_condition['Bridge Age (yr)'] = df_grouped_age_condition['Bridge Age (yr)'].astype(str)

df_grouped_age_condition['age_condition'] = df_grouped_age_condition[['Bridge Age (yr)', 'CAT10 - Bridge Condition']].apply(lambda x: ', '.join(x), axis=1)

source_age_condition = ColumnDataSource(df_grouped_age_condition)

color_mapper = LinearColorMapper(palette=Viridis256, low=min(df_grouped_age_condition['counts']), high=max(df_grouped_age_condition['counts']))

age_condition_plot = figure(x_range=FactorRange(*df_grouped_age_condition['age_condition'].unique().tolist()),
                            title="Bridge Age and Bridge Condition Rating",
                            x_axis_label='Age, Condition',
                            y_axis_label='Counts',
                            x_axis_location="above",
                            width=600,
                            height=400,
                            tools="hover",
                            toolbar_location=None,
                            tooltips=[('count', '@counts')])

age_condition_plot.rect(x='age_condition',
                        y='counts',
                        width=1,
                        height=1,
                        source=source_age_condition,
                        line_color=None,
                        fill_color={'field': 'counts', 'transform': color_mapper})

color_bar = ColorBar(color_mapper=color_mapper, location=(0, 0),
                     ticker=BasicTicker(desired_num_ticks=len(df_grouped_age_condition)))

age_condition_plot.add_layout(color_bar, 'right')

# Age-condition plot hover tool
age_condition_hover = HoverTool(tooltips=[("Age, Condition", "@age_condition"),
                                          ("Counts", "@counts")])
age_condition_plot.add_tools(age_condition_hover)

# Create a grid layout for the plots
grid = gridplot([[scatter_plot, bar_plot],
                 [line_plot, line_plot_2],
                 [geo_plot]])

# Add the grid layout to the current document
curdoc().add_root(grid)
