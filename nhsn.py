import streamlit as st
import numpy as np
import pandas as pd
import datetime
from epiweeks import Week

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
import utils

st.title('NHSN: Influenza Weekly Data')

# PSI submission

path_to_weekly_submission = "/Users/michal/Dropbox/CSMB05/CDC2024-2025/new_weekly_submissions/"
submission_files = utils.list_csv_files(path_to_weekly_submission)
file_dates = [file[:10] for file in submission_files]

# Find the file with the latest date
latest_file = max(submission_files, key=lambda file: file[:10])

path_to_psi_file = os.path.join(path_to_weekly_submission, latest_file)

@st.cache_data
def load_psi_submission(path_to_psi_file):
    data = pd.read_csv(path_to_psi_file)
    data['reference_date'] = pd.to_datetime(data['reference_date'])
    data['target_end_date'] = pd.to_datetime(data['target_end_date'])
    data['output_type_id'] = pd.to_numeric(data['output_type_id'], errors='coerce')
    data = data.loc[~data['location'].isin(['US'])]
    data = data[data['target'] == 'wk inc flu hosp']
    data['MMWR_week'] = data['target_end_date'].apply(lambda x: Week.fromdate(x).week)
    data['MMWR_year'] = data['target_end_date'].apply(lambda x: Week.fromdate(x).year)
    data = data.loc[:, data.columns.isin(['reference_date', 'target_end_date', 'MMWR_year', 'MMWR_week',
        'location', 'output_type_id', 'value'])]

    return data

psi_data = load_psi_submission(path_to_psi_file)

fips_df = utils.create_fips_mapping()

psi_data = psi_data.merge(fips_df, left_on='location', right_on='state_code', how='left')

# just test subsetting 
psi_data_locs = psi_data['state_abbr'].unique()

# for loc in psi_data_locs:
#     psi_subset = psi_data[psi_data['state_abbr'] == loc]

# today = datetime.date.today()
# # Get the day of the week
# day_of_week = today.strftime("%A")

# we will check both URLs and if both have the same last date take the more stable one 
# which is the url_main (The Friday URL)

url_dev  = "https://data.cdc.gov/resource/mpgq-jmmr.csv"
url_main = url = "https://data.cdc.gov/resource/ua7e-t2fy.csv"

# Read the CSV directly into a pandas DataFrame

data_load_state = st.text('Loading and Plotting data...')

# this is the path to historic data 

path_to_data = 'data/HHS_weekly-hosp_state.csv'

DATE_COLUMN = 'weekendingdate'
@st.cache_data
def load_historic_data(path_to_data):
    data = pd.read_csv(path_to_data)
    data['weekendingdate'] = pd.to_datetime(data['weekendingdate'])
    data = data.loc[~data['jurisdiction'].isin(['AS', 'VI', 'GU', 'MP', 'USA'])]
    return data

@st.cache_data
def preprocess_data(data):
    # Process data as needed
    data['weekendingdate'] = pd.to_datetime(data['weekendingdate'])
    data = data.loc[~data['jurisdiction'].isin(['AS', 'VI', 'GU', 'MP', 'USA'])]
    return data

historic_data = load_historic_data(path_to_data)
data_main = preprocess_data(pd.read_csv(url_main))
data_dev = preprocess_data(pd.read_csv(url_dev))

max_date_in_data_dev = data_dev['weekendingdate'].max()
max_date_in_data_main= data_main['weekendingdate'].max()

if (max_date_in_data_main >= max_date_in_data_dev):
    data = data_main
else:
    data = data_dev


data_flu = data.loc[:, data.columns.isin(['weekendingdate', 'jurisdiction', 'totalconfflunewadm'])]

data_rep = data.loc[:, data.columns.isin(['weekendingdate', 'jurisdiction', 'totalconfflunewadmperchosprep'])] 

data_flu['totalconfflunewadm'] = pd.to_numeric(data_flu['totalconfflunewadm'])
data_rep['totalconfflunewadmperchosprep'] = pd.to_numeric(data_rep['totalconfflunewadmperchosprep'])


hst_data_flu = historic_data.loc[:, historic_data.columns.isin(['weekendingdate', 'jurisdiction', 'totalconfflunewadm'])]

hst_data_rep = historic_data.loc[:, historic_data.columns.isin(['weekendingdate', 'jurisdiction', 'totalconfflunewadmperchosprep'])] 

# Find the maximum date in hst_data_flu
max_date_in_hst_data= historic_data['weekendingdate'].max()

# Replace the last row if the date matches
if max_date_in_hst_data in data['weekendingdate'].values:
    # Get the row from `data` with the same max date
    replacement_row = data[data['weekendingdate'] == max_date_in_hst_data]
    
    # Drop the last row in `hst_data_flu` and `rep`
    hst_data_flu = hst_data_flu[hst_data_flu['weekendingdate'] != max_date_in_hst_data]
    hst_data_rep = hst_data_rep[hst_data_rep['weekendingdate'] != max_date_in_hst_data]
    # Append the replacement row
    hst_data_flu = pd.concat([hst_data_flu, replacement_row], ignore_index=True)
    hst_data_rep = pd.concat([hst_data_rep, replacement_row], ignore_index=True)

# Filter rows from 'data_flu' with dates beyond the max date in 'hst_data_flu'
new_rows = data_flu[data_flu['weekendingdate'] > max_date_in_hst_data]

# Append the new rows to 'hst_data_flu' and overwrite data_flu
data_flu = pd.concat([hst_data_flu, new_rows], ignore_index=True)

# repeat for data_rep
new_rows = data_rep[data_rep['weekendingdate'] > max_date_in_hst_data]
data_rep = pd.concat([hst_data_rep, new_rows], ignore_index=True)


# Add MMWR year and week columns
data_flu['MMWR_year'] = data_flu['weekendingdate'].apply(lambda x: Week.fromdate(x).year)
data_flu['MMWR_week'] = data_flu['weekendingdate'].apply(lambda x: Week.fromdate(x).week)

# Add MMWR year and week columns
data_rep['MMWR_year'] = data_rep['weekendingdate'].apply(lambda x: Week.fromdate(x).year)
data_rep['MMWR_week'] = data_rep['weekendingdate'].apply(lambda x: Week.fromdate(x).week)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Fraction Reporting", "Flu Data", "Individual Locations"])

data_rep = pd.DataFrame(data_rep)

# Convert weekendingdate to datetime
start_date = datetime.datetime(2024, 8, 16)

data_rep = data_rep[data_rep['weekendingdate'] >= pd.Timestamp(start_date)]
end_date = np.max(data_rep['weekendingdate'])


# 
# Colors for each season
season_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA'] # Plotly default colors
season_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Matplotlib default colors
linewidth = [1, 1, 1, 2]

frcst_color = 'black'

# Streamlit app

# Tab 1: Fraction Reporting
with tab1:
    st.header(f"Facilities Reporting as of {end_date.strftime('%m-%d-%y')}")
    # Grid of plots for Fraction Reporting (data_rep)
    jurisdictions = data_rep['jurisdiction'].unique()
    n = len(jurisdictions)
    cols = 3
    rows = int(np.ceil(n / cols))

    # Create a subplot grid
    fig_rep = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=jurisdictions,
        vertical_spacing=0.02,  # Adjust spacing (must be less than 1 / (rows - 1))
        horizontal_spacing=0.05
    )

    # Add a bar plot for each jurisdiction
    for i, jurisdiction in enumerate(jurisdictions):
        # Filter data for the jurisdiction
        data_jurisdiction = data_rep[data_rep['jurisdiction'] == jurisdiction]

        # Determine row and column position
        row = (i // cols) + 1
        col = (i % cols) + 1

        # Add the trace to the correct subplot
        fig_rep.add_trace(
            go.Bar(
                x=data_jurisdiction['weekendingdate'],
                y=data_jurisdiction['totalconfflunewadmperchosprep'],
                name=jurisdiction,
                # marker=dict(color='blue'),
                opacity=0.7
            ),
            row=row,
            col=col
        )

        # Customize axes visibility
        show_xaxis = (row == rows)  # Show x-axis labels only for the bottom row
        show_yaxis = (col == 1)     # Show y-axis labels only for the leftmost column

        # Set shared x-axis and y-axis labels

        fig_rep.update_xaxes(
                title_text="Week Ending Date" if show_xaxis else None,
                tickangle=45,
                showticklabels=show_xaxis,
                row=row,
                col=col)

        fig_rep.update_yaxes(
                title_text="Fraction Reporting" if show_yaxis else None,
                showticklabels=show_yaxis,
                range = [0,1],
                tickvals = [0, 0.25, 0.5, 0.75, 1.0],
                ticktext = [0, 0.25, 0.5, 0.75, 1.0],
                row=row,
                col=col
            )

    # Update layout for the subplot grid
    fig_rep.update_layout(
        title="Fraction of Reporting Facilities Across Jurisdictions",
        showlegend=False,  # Hide legend for individual subplots
        height=rows * 250,  # Adjust height dynamically based on the number of rows
        template="plotly_white"
    )

    # Display the subplot grid in Streamlit
    st.plotly_chart(fig_rep, use_container_width=True)



# Tab 2: Flu Data and forecast

EW_start = Week.fromdate(start_date).week
EW_end   = 20

year_start = [2021, 2022, 2023, 2024]
year_end = [year + 1 for year in year_start]

# Subset the DataFrame for each season
season_dfs = []
nw_max = []

@st.cache_data
def prepare_season_data(data_flu, year_start, year_end, EW_start, EW_end):
    season_dfs = []
    for i in range(len(year_start)):
        season_df = data_flu[
            ((data_flu['MMWR_year'] == year_start[i]) & (data_flu['MMWR_week'] >= EW_start)) |
            ((data_flu['MMWR_year'] == year_end[i]) & (data_flu['MMWR_week'] <= EW_end))
        ]
        season_dfs.append(season_df)
        nw_max.append(season_df.shape[0]/len(data_flu['jurisdiction'].unique()))
    return season_dfs, nw_max

season_dfs, nw_max = prepare_season_data(data_flu, year_start, year_end, EW_start, EW_end)

max_index = np.argmax(nw_max)
mmwr_label = season_dfs[max_index]['MMWR_week'].unique()
mmwr_index= np.arange(0, np.max(nw_max))

with tab2:
    st.header(f"Flu Admissions up to {end_date.strftime('%m-%d-%y')}")
    
    # Grid of plots for Flu Data (data_flu)
    jurisdictions = data_flu['jurisdiction'].unique()
    n = len(jurisdictions)
    cols = 3
    rows = int(np.ceil(n / cols))

    # Create a subplot grid for Plotly
    fig_flu = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=jurisdictions,
        vertical_spacing=0.02,  # Adjust spacing
        horizontal_spacing=0.05
    )



    # Add plots for each jurisdiction
    for i, jurisdiction in enumerate(jurisdictions):
        # Filter data for the jurisdiction
        for season_idx, season_df in enumerate(season_dfs):
            # Filter the current jurisdiction's data for this season
            data_jurisdiction = season_df[season_df['jurisdiction'] == jurisdiction]

            # Determine row and column position
            row = (i // cols) + 1
            col = (i % cols) + 1

            # Add a line trace for this jurisdiction and season
            fig_flu.add_trace(
                go.Scatter(
                    x=np.arange(0, len(data_jurisdiction)),
                    y=data_jurisdiction['totalconfflunewadm'],
                    mode='lines+markers',  # Line plot with markers
                    name=f"{jurisdiction} ({year_start[season_idx]}-{year_end[season_idx]})",
                    line=dict(
                        color=season_colors[season_idx],
                        width=linewidth[season_idx]
                    ),
                    opacity=0.8
                ),
                row=row,
                col=col
            )

        def get_psi_filtered_data(jurisdiction, psi_data):
            psi_data_jurisdiction = psi_data[psi_data['state_abbr'] == jurisdiction]
            psi_data_jurisdiction = psi_data_jurisdiction.loc[:, psi_data_jurisdiction.columns.isin(['target_end_date', 'MMWR_year', 'MMWR_week',
            'state_abbr', 'output_type_id', 'value'])]
            return psi_data_jurisdiction

        psi_data_jurisdiction = get_psi_filtered_data(jurisdiction, psi_data)

        median_subset = psi_data_jurisdiction[psi_data_jurisdiction['output_type_id'] == 0.5]
        lower_subset  = psi_data_jurisdiction[psi_data_jurisdiction['output_type_id'] == 0.1]
        upper_subset  = psi_data_jurisdiction[psi_data_jurisdiction['output_type_id'] == 0.9]

        median_labels = np.array(median_subset['MMWR_week'])

        # Find the start and end indices of the psi forecast in mmwr_label
        start_idx = np.where(mmwr_label == median_labels[0])[0][0]  # Index where other_array starts
        end_idx = np.where(mmwr_label == median_labels[-1])[0][0] 
     
        # Create a padded array of NaNs with the same length as mmwr_label
        padded_array = np.full(len(mmwr_label), np.nan)

        median_array = np.full(len(mmwr_label), np.nan)
        lower_array = np.full(len(mmwr_label), np.nan)
        upper_array = np.full(len(mmwr_label), np.nan)

        # Fill the appropriate slice of the padded array with values from other_array
        median_array[start_idx:end_idx + 1] = np.array(median_subset['value'])
        lower_array[start_idx:end_idx + 1]  = np.array(lower_subset['value'])
        upper_array[start_idx:end_idx + 1]  = np.array(upper_subset['value'])
        
        # Create a DataFrame for better visualization
        psi_df = pd.DataFrame({'mmwr_index': mmwr_index, 'Lower': lower_array,
            'Median': median_array, 'Upper': upper_array})

    
        # Add the shaded area (between Lower and Upper)
        fig_flu.add_trace(
            go.Scatter(
                x=psi_df['mmwr_index'],
                y=psi_df['Upper'],
                mode='lines',
                line=dict(width=0),  # No line for the upper bound
                fill=None,           # No fill here
                name='90% PI',
                showlegend=False     # Hide legend for this trace
            ),
            row=row,
            col=col
        )

        fig_flu.add_trace(
            go.Scatter(
                x=psi_df['mmwr_index'],
                y=psi_df['Lower'],
                mode='lines',
                line=dict(width=0),  # No line for the lower bound
                fill='tonexty',      # Fill the area between this and the previous trace
                fillcolor='rgba(0, 200, 255, 0.4)',  # Transparent blue
                name='90% PI',
                showlegend=False
            ),
            row=row,
            col=col
        )

        fig_flu.add_trace(
            go.Scatter(
                x=psi_df['mmwr_index'],
                y=psi_df['Median'],
                mode='lines+markers',
                name='Forecast',
                line=dict(
                    color=frcst_color,
                    width=1),
                 opacity=1.0),
                row=row,
                col=col
        )
        
        # Customize x-axis visibility
        show_xaxis = (row == rows)  # Show x-axis labels only for the bottom row
        show_yaxis = (col == 1)
        fig_flu.update_xaxes(
            title_text="Epidemic Week" if show_xaxis else None,
            tickvals=np.arange(0, len(mmwr_label))[::3],
            ticktext=mmwr_label[::3],
            tickmode="array",
            row=row,
            col=col
        )

        # Customize y-axis visibility
        fig_flu.update_yaxes(
            title_text="Admissions" if show_yaxis else None,
            showticklabels=True,  # Each panel has its own y-axis labels
            row=row,
            col=col
        )

    # Update the layout for the subplot grid
    fig_flu.update_layout(
        title="Flu Admissions Across Jurisdictions",
        showlegend=False,  # Suppress legends for individual subplots
        height=rows * 300,  # Dynamically adjust figure height based on the number of rows
        template="plotly_white"
    )

    # Display the interactive Plotly subplot grid
    st.plotly_chart(fig_flu, use_container_width=True)

with tab3:
    st.header(f"Data up to {end_date.strftime('%m-%d-%y')}")

    # Selectbox for choosing a jurisdiction
    default_jurisdiction = 'CA'
    selected_jurisdiction = st.selectbox(
        "Select a Jurisdiction",
        options=data_flu['jurisdiction'].unique(),
        index=list(data_flu['jurisdiction'].unique()).index(default_jurisdiction)
    )

    # Filtered data
    @st.cache_data
    def get_filtered_data(jurisdiction, data_rep, season_dfs):
        data_rep_jurisdiction = data_rep[data_rep['jurisdiction'] == jurisdiction]
        data_flu_season = [
            season_df[season_df['jurisdiction'] == jurisdiction]
            for season_df in season_dfs
        ]
        return data_rep_jurisdiction, data_flu_season

    data_rep_jurisdiction, data_flu_season = get_filtered_data(selected_jurisdiction, data_rep, season_dfs)

    @st.cache_data
    def get_psi_filtered_data(jurisdiction, psi_data):
        psi_data_jurisdiction = psi_data[psi_data['state_abbr'] == jurisdiction]
        psi_data_jurisdiction = psi_data_jurisdiction.loc[:, psi_data_jurisdiction.columns.isin(['target_end_date', 'MMWR_year', 'MMWR_week',
        'state_abbr', 'output_type_id', 'value'])]
        return psi_data_jurisdiction

    psi_data_jurisdiction = get_psi_filtered_data(selected_jurisdiction, psi_data)
    #
    median_subset = psi_data_jurisdiction[psi_data_jurisdiction['output_type_id'] == 0.5]
    lower_subset  = psi_data_jurisdiction[psi_data_jurisdiction['output_type_id'] == 0.1]
    upper_subset  = psi_data_jurisdiction[psi_data_jurisdiction['output_type_id'] == 0.9]

    median_labels = np.array(median_subset['MMWR_week'])
    
    # Find the start and end indices of the psi forecast in mmwr_label
    start_idx = np.where(mmwr_label == median_labels[0])[0][0]  # Index where other_array starts
    end_idx = np.where(mmwr_label == median_labels[-1])[0][0] 
 
    # Create a padded array of NaNs with the same length as mmwr_label
    padded_array = np.full(len(mmwr_label), np.nan)

    median_array = np.full(len(mmwr_label), np.nan)
    lower_array = np.full(len(mmwr_label), np.nan)
    upper_array = np.full(len(mmwr_label), np.nan)

    # Fill the appropriate slice of the padded array with values from other_array
    median_array[start_idx:end_idx + 1] = np.array(median_subset['value'])
    lower_array[start_idx:end_idx + 1]  = np.array(lower_subset['value'])
    upper_array[start_idx:end_idx + 1]  = np.array(upper_subset['value'])
    
    # Create a DataFrame for better visualization
    psi_df = pd.DataFrame({'mmwr_index': mmwr_index, 'Lower': lower_array,
        'Median': median_array, 'Upper': upper_array})

    
    # Plot 1: Fraction Reporting
    fig_rep = px.bar(
        data_rep_jurisdiction,
        x='weekendingdate',
        y='totalconfflunewadmperchosprep',
        title=f"Fraction Reporting - {selected_jurisdiction}",
        labels={'weekendingdate': 'Week Ending Date', 'totalconfflunewadmperchosprep': 'Fraction Reporting'}
    )
    st.plotly_chart(fig_rep)


    # Plot Flu Admissions

    # Create a Plotly figure
    fig_flu = go.Figure()

    x_min = 0  # Minimum value of x-axis index
    x_max = np.max(nw_max)-1 # Maximum value of x-axis index

    # Add a trace for each season
    for season_idx, season_df in enumerate(data_flu_season):
        xaxis_index = np.arange(0, len(season_df))  # Create x-axis indices
        fig_flu.add_trace(
            go.Scatter(
                x=xaxis_index,
                y=season_df['totalconfflunewadm'],
                mode='lines+markers',  # Line plot with markers
                name=f"{year_start[season_idx]}",  # Legend label
                line=dict(
                    color=season_colors[season_idx],
                    width=linewidth[season_idx]
                ),
                opacity=1.0  # Set transparency
            ))
    
        # Add the shaded area (between Lower and Upper)
        fig_flu.add_trace(
            go.Scatter(
                x=psi_df['mmwr_index'],
                y=psi_df['Upper'],
                mode='lines',
                line=dict(width=0),  # No line for the upper bound
                fill=None,           # No fill here
                name='90% PI',
                showlegend=False     # Hide legend for this trace
            )
        )

        fig_flu.add_trace(
            go.Scatter(
                x=psi_df['mmwr_index'],
                y=psi_df['Lower'],
                mode='lines',
                line=dict(width=0),  # No line for the lower bound
                fill='tonexty',      # Fill the area between this and the previous trace
                fillcolor='rgba(0, 200, 255, 0.1)',  # Transparent blue
                name='90% PI',
                showlegend=False
            )
        )
    fig_flu.add_trace(
        go.Scatter(
            x=psi_df['mmwr_index'],
            y=psi_df['Median'],
            mode='lines+markers',
            name='Forecast',
            line=dict(
                color=frcst_color,
                width=1),
            opacity=1.0))

    # Update layout for titles, labels, and styling
    fig_flu.update_layout(
        title="Flu Admissions Across Seasons",
        xaxis=dict(
            title="Epidemic Week",
            # range=[x_min,x_max],
            tickvals=np.arange(0, len(mmwr_label)),
            ticktext=mmwr_label,
            tickmode="array",
            tick0=0,  # Starting tick
            dtick=1   # Interval between ticks
        ),
        yaxis=dict(
            title="Flu Admissions"
        ),
        legend=dict(
            title="Seasons",
            x=0,  # Position to the right
            y=1   # Top
        ),
        template="plotly_white"
    )

    # Display the figure in Streamlit
    st.plotly_chart(fig_flu)





