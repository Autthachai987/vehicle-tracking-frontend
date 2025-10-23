import os
import pandas as pd
import sqlite3
import json

# Path to the SQLite database
db_path = 'bbdrentvehicle.db'

# Path to the JSON file that stores the last modified timestamps
timestamp_file = 'file_timestamps.json'

# Paths to the CSV files
csv_files = {
    'model_table': 'Vehicle_model_spec.csv'
}

# Function to load or create the timestamp tracking file
def load_timestamps():
    if os.path.exists(timestamp_file):
        with open(timestamp_file, 'r') as f:
            return json.load(f)
    return {}

# Function to save the timestamps
def save_timestamps(timestamps):
    with open(timestamp_file, 'w') as f:
        json.dump(timestamps, f)

# Function to update the table if the CSV file has changed
def update_table_from_csv(table_name, csv_path, conn):
    # Load CSV into DataFrame
    df = pd.read_csv(csv_path, encoding='utf-8')
    # Write DataFrame to SQLite table
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"Updated table '{table_name}' from '{csv_path}'.")

# Main function to check for changes and update tables
def check_and_update_tables():
    # Load the last modification timestamps
    last_timestamps = load_timestamps()

    # Connect to the SQLite database
    with sqlite3.connect(db_path) as conn:
        # Loop through each CSV file
        for table_name, csv_path in csv_files.items():
            # Get the last modified time of the CSV file
            last_modified = os.path.getmtime(csv_path)

            # Check if the file has been modified since the last recorded timestamp
            if table_name not in last_timestamps or last_timestamps[table_name] != last_modified:
                # Update the table from the CSV file
                update_table_from_csv(table_name, csv_path, conn)
                print(f"{table_name} has updated")
                # Update the timestamp in memory
                last_timestamps[table_name] = last_modified

        # Save the updated timestamps
        save_timestamps(last_timestamps)

# Run the update check
check_and_update_tables()
