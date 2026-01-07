import os
import pandas as pd
import re
import requests
from datetime import datetime

# from astropy.wcs.utils import pixel_to_skycoord

def list_csv_files(directory_path):
    """
    List all .fits files in the specified directory.

    Parameters:
    directory_path (str): Path to the directory to search for .fits files.

    Returns:
    list: A list of filenames ending with '.fits' in the specified directory.
    """
    return [f for f in os.listdir(directory_path) if f.endswith('.csv')]

import pandas as pd

def create_fips_mapping():
    # Define the state FIPS codes, abbreviations, and names
    state_fips_data = {
        "state_code": [
            "01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "15", "16",
            "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
            "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42",
            "44", "45", "46", "47", "48", "49", "50", "51", "53", "54", "55", "56"
        ],
        "state_abbr": [
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID",
            "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO",
            "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
            "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
        ],
        "state_name": [
            "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
            "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
            "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
            "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
            "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
            "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
            "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
        ]
    }

    # Create a DataFrame for states and DC
    state_fips = pd.DataFrame(state_fips_data)

    # Add Puerto Rico as a territory
    territories = pd.DataFrame({
        "state_code": ["72"],
        "state_abbr": ["PR"],
        "state_name": ["Puerto Rico"]
    })

    # Combine states and territories
    full_fips = pd.concat([state_fips, territories], ignore_index=True)

    return full_fips

def debug_read_data(output_path=None):
    if output_path:
        data=pd.read_csv(output_path)
        print(f"Reading data from {output_path}")
        return data
    else:
        print(f"Error reading data")

def download_all_cdc_data(url, output_path=None):
    """
    Download all data from the given CDC API URL and save it locally if a path is provided.

    Parameters:
        url (str): The base URL of the CDC dataset in CSV format.
        output_path (str): Optional file path to save the downloaded data.

    Returns:
        pd.DataFrame: The downloaded data as a pandas DataFrame.
    """
    try:
        # Append $limit parameter to fetch all rows
        full_url = f"{url}?$limit=1000000"
        print(f"Fetching data from {full_url}...")
        data = pd.read_csv(full_url)

        # Optionally save the data locally
        if output_path:
            data.to_csv(output_path, index=False)
            print(f"Data saved to {output_path}")

        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None
