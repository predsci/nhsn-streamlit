import streamlit as st
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import datetime
from epiweeks import Week

st.title('NHSN - Flu Weekly')

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

path_to_data = '~/Dropbox/CSMB05/data/HHS_weekly-hosp_state.csv'

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
# Streamlit app

# Tab 1: Fraction Reporting
with tab1:
    st.header(f"Fraction Reporting up to {end_date}")
    # Grid of plots for Fraction Reporting (data_rep)
    jurisdictions = data_rep['jurisdiction'].unique()
    n = len(jurisdictions)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(30, rows * 4), constrained_layout=True)
    axes = axes.flatten()

    for i, jurisdiction in enumerate(jurisdictions):
        data_jurisdiction = data_rep[data_rep['jurisdiction'] == jurisdiction]
        ax = axes[i]
        bars = ax.bar(data_jurisdiction['weekendingdate'], data_jurisdiction['totalconfflunewadmperchosprep'], 
                      color='blue', alpha=1.0, width = 3)
        ax.set_ylim([0, 1])
        ax.set_xticks(data_jurisdiction['weekendingdate'])
        ax.set_xticklabels(data_jurisdiction['MMWR_week'], rotation=45, fontsize=24)
        if i % cols == 0:
            ax.set_ylabel("Fraction Reporting", fontsize=24)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1.0], fontsize=24)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(None)
        ax.legend([f"{jurisdiction}"], loc="upper left", frameon=False, fontsize=28)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    for j in range(len(jurisdictions), len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

# Tab 2: Flu Data

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
    st.header(f"Admission Data tp to {end_date}")
    # Grid of plots for Flu Data (data_flu)
    jurisdictions = data_flu['jurisdiction'].unique()
    n_flu = len(jurisdictions)
    cols_flu = 4
    rows_flu = int(np.ceil(n_flu / cols_flu))
    fig_flu, axes_flu = plt.subplots(rows_flu, cols_flu, figsize=(30, rows_flu * 4), constrained_layout=True)
    axes_flu = axes_flu.flatten()

    # Colors for each season
    season_colors = ['blue', 'red', 'green', 'black']
    linewidth = [1, 1, 1, 2]

    for i, jurisdiction in enumerate(jurisdictions):
        ax = axes_flu[i]
    
        # Plot each season for the current jurisdiction
        for season_idx, season_df in enumerate(season_dfs):
            # Filter data for the current jurisdiction
            data_jurisdiction = season_df[season_df['jurisdiction'] == jurisdiction]
            xaxis_index = np.arange(0, data_jurisdiction.shape[0])
        # Plot data with a unique color for each season
            ax.plot(
                # data_jurisdiction['weekendingdate'],
                xaxis_index,
                data_jurisdiction['totalconfflunewadm'], 
                label=f"{year_start[season_idx]}",
                color=season_colors[season_idx],
                linewidth=linewidth[season_idx],
                alpha=1.0)
    
        # Set labels and title
        ax.set_title(f"{jurisdiction}", fontsize=24)
        # ax.set_xticks(data_jurisdiction['weekendingdate'])
        # ax.set_xticklabels(data_jurisdiction['MMWR_week'], rotation=45, fontsize=24)
        ax.set_xticks(mmwr_index[::3])
        ax.set_xticklabels(mmwr_label[::3], rotation=45, fontsize=24)        
        ax.tick_params(axis='y', labelsize=24)
        ax.set_xlabel(None)
        ax.legend(fontsize=20)
        # ax.legend([f"{jurisdiction}"], loc="upper left", frameon=False, fontsize=24)
 

    for j in range(len(jurisdictions), len(axes_flu)):
        axes_flu[j].set_visible(False)
    fig_flu.tight_layout()
    st.pyplot(fig_flu)

with tab3:
    st.header(f"Data up to {end_date}")

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

    # Plot Fraction Reporting
    fig_rep, ax_rep = plt.subplots(figsize=(12, 6))
    ax_rep.bar(
        data_rep_jurisdiction['weekendingdate'],
        data_rep_jurisdiction['totalconfflunewadmperchosprep'],
        label="Fraction Reporting",
        color="blue",
        alpha=0.7,
        width = 3
    )

    ax_rep.set_xticks(data_rep_jurisdiction['weekendingdate'][::1])
    ax_rep.set_xticklabels(data_rep_jurisdiction['MMWR_week'][::1], rotation=45, fontsize=16)
    ax_rep.set_ylim([0, 1])
    ax_rep.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_rep.set_yticklabels([0, 0.25, 0.5, 0.75, 1.0], fontsize=20)
    ax_rep.set_title(f"Fraction Reporting - {selected_jurisdiction}", fontsize=20)
    ax_rep.set_xlabel(None)
    ax_rep.set_ylabel("Fraction Reporting", fontsize=20)
    ax_rep.grid(axis='y', linestyle='--', alpha=0.7)
    
    st.pyplot(fig_rep)

    # Plot Flu Admissions
    fig_flu, ax_flu = plt.subplots(figsize=(12, 6))
    for season_idx, season_df in enumerate(data_flu_season):
        xaxis_index = np.arange(len(season_df))
        ax_flu.plot(
            xaxis_index,
            season_df['totalconfflunewadm'],
            label=f"{year_start[season_idx]}",
            color=season_colors[season_idx],
            linewidth=linewidth[season_idx],
            alpha=1.0
        )
    ax_flu.set_xlim([mmwr_index.min(), mmwr_index.max()])
    ax_flu.set_xticks(mmwr_index)
    ax_flu.set_xticklabels(mmwr_label, rotation=90, fontsize=16)
    ax_flu.set_title(f"Flu Admissions - {selected_jurisdiction}", fontsize=20)
    ax_flu.set_ylabel("Flu Admissions", fontsize=20)
    ax_flu.tick_params(axis='y', labelsize=20)
    ax_flu.grid(axis='both', linestyle='--', alpha=0.7)
    ax_flu.legend(fontsize=16)
    st.pyplot(fig_flu)



