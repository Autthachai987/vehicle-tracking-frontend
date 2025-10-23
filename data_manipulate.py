from typing import Union
from fastapi import HTTPException
from typing import Optional, Union
from pydantic import BaseModel
from datetime import datetime, timedelta, date
from collections import deque
from tabulate import tabulate
from fpdf import FPDF
import numpy as np
import math
import csv
import requests
import base64 
import pandas as pd
import os
import json
import time
import sqlite3

pd.set_option('display.max_columns', None)  # To display all columns
import os

username = os.getenv("API_USERNAME", "DREN00001")
token = os.getenv("API_TOKEN", "55a0808ab5ace830ebd55fe20462b64ebb13a132c1c6e50ab70a05e3b941e379")
dtc_token = os.getenv("DTC_TOKEN", "LUGW24VH1NVLTCP51287S9WJXY4EMSKAKRZYDDMC873BFQHZEA3TNFGP5JXQ966U")
auth = f"{username}:{token}"
auth_bytes = auth.encode('ascii')
base64_bytes = base64.b64encode(auth_bytes)
base64_auth = base64_bytes.decode('ascii')
#alarm threshold setting
fuel_threshold = 30
battery_threshold = 30
critical_threshold = 10
# moving avg setting
window_size = 3 # Define the size of the moving average window (e.g., 3)
voltage_buffer = {} # Initialize a deque (double-ended queue) with max size as the window size
db_path = 'bbdrentvehicle.db'
log_db_path = 'data_log.db'

global fdata
global started
global previous_ev 
global alarm_cache
#init global value
previous_ev = None 
alarm_cache = {}
started = []

class ItemData(BaseModel):
    model: str
    chassis_number: str
    manufacturer: str
    vehicle_number : str
    speed : Optional[int] = 0 # Default to 0 if None
    ignition: Optional[bool] = False  # Default to False if None
    idling: Optional[bool] = False  # Default to False if None
    fuel: Optional[int] = None  # Default to None if None
    fuellitre: Optional[int] = None  # Default to None if None
    battery: Optional[float] = None  # Battery can be null
    battery_percentage: Optional[int] = None  # Battery can be null
    location: Optional[str] = "Unknown"  # Default to "Unknown" if None
    workhour: Union[int, str, None] = None  # Workhour can be int, str, or None
    status: Optional[str] = "Unknown"  # Default to "Unknown" if None
class Startedlog(BaseModel):
    chassis_number: str
    started: bool
class Config:
    # Allow population of fields with None values
    allow_population_by_field_name = True
    # flag: bool
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Vehicle Report", ln=True, align="C")
        self.ln(5)

    def add_vehicle_section(self, date_str, license_plate, model, rows):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, f"Date: {date_str}", ln=True)
        self.cell(0, 10, f"{license_plate}, {model}", ln=True)

        self.set_font("Arial", "B", 10)
        self.cell(35, 8, "Timestamp", 1)
        self.cell(35, 8, "Type", 1)
        self.cell(35, 8, "Status", 1)
        self.cell(40, 8, "Energy %", 1)
        self.cell(40, 8, "Energy Unit", 1)
        self.ln()

        self.set_font("Arial", "", 10)
        for row in rows:
            ts, vtype, status, percent, energy = row
            self.cell(35, 8, ts, 1)
            self.cell(35, 8, vtype, 1)
            self.cell(35, 8, status, 1)
            self.cell(40, 8, str(percent), 1)
            self.cell(40, 8, str(energy), 1)
            self.ln()

        self.ln(5)

class PDF_DAILY(FPDF):
    def __init__(self, report_date=None):
        super().__init__()
        self.report_date = report_date  # Store the date for use in the header

    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Vehicle Report", ln=True, align="C")
        if self.report_date:
            self.set_font("Arial", "", 12)
            self.cell(0, 10, f"Date: {self.report_date}", ln=True, align="C")
        self.ln(5)

    def add_vehicle_section(self, license_plate, model, rows):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, f"{license_plate}, {model}", ln=True)

        self.set_font("Arial", "B", 10)
        self.cell(35, 8, "Timestamp", 1)
        self.cell(35, 8, "Type", 1)
        self.cell(35, 8, "Status", 1)
        self.cell(40, 8, "Energy %", 1)
        self.cell(40, 8, "Energy Unit", 1)
        self.ln()

        self.set_font("Arial", "", 10)
        for row in rows:
            ts, vtype, status, percent, energy = row
            self.cell(35, 8, ts, 1)
            self.cell(35, 8, vtype, 1)
            self.cell(35, 8, status, 1)
            self.cell(40, 8, str(percent), 1)
            self.cell(40, 8, str(energy), 1)
            self.ln()

        self.ln(5)

def fetch_data():
    # Your code to fetch data goes here
    global fdata
    global noticedata
    # reset_log_file_if_needed()
    fdata = get_data()
    # print("fdata has updated")
    voltage_to_percent()
    # print("voltage has converted to percentage")
    check_transport()
    # print ("checked transports status")
    data_logger()
    print("Data fetched")
    return fdata

def get_fuel_vechicle():
    # Your code to fetch data goes here
    global fdata
    gas_vehicles = [item for item in fdata["data"] if item["vehicletype"] == "Petrol"]
    gas_vehicles = clean_nan_values(gas_vehicles)
    return {"data": gas_vehicles}

def get_ev():
    # Your code to fetch data goes here
    global fdata
    electric_vehicles = [item for item in fdata["data"] if item["vehicletype"] == "Electric"]
    electric_vehicles = clean_nan_values(electric_vehicles)
    return {"data": electric_vehicles}

def get_machine():
    machine_list = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row  # Enables name-based access
        cursor = conn.execute("SELECT * FROM register_table")
        for row in cursor:
            machine_list.append({
                "id": row["id"],
                "vehicle_number": row["License plate"],
                "chassis_number": row["Chassis number"],
                "model": row["Model"]
            })
    return {"data":machine_list}

def get_reg_model():
    model_list = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row  # Enables name-based access
        cursor = conn.execute("SELECT * FROM model_table")
        for row in cursor: #Model_Q,Model,Brand,Batt_voltage,Type,Powered,Tank_capacity
            model_list.append({
                "id": row["id"],
                "model": row["Model"],
                "brand": row["Brand"],
                "battery": row["Batt_voltage"],
                "type": row["Type"],
                "power_source": row["Powered"],
                "tank_size": row["Tank_capacity"]
            })
    return {"data":model_list}

def data_logger():
    
    global fdata, alarm_cache
    
    try:
        with sqlite3.connect(log_db_path) as con:
            # Configure WAL mode
            con.execute("PRAGMA journal_mode = WAL;")
            con.execute("PRAGMA wal_autocheckpoint = 500;")
            
            # Create table if it doesn't exist
            con.execute("""
                CREATE TABLE IF NOT EXISTS alarm_log (
                    id               INTEGER PRIMARY KEY,
                    ts               DATETIME DEFAULT (datetime('now', 'localtime')),
                    "License plate"  TEXT,
                    model            TEXT,
                    vehicletype      TEXT,
                    event            TEXT NOT NULL,
                    energy_percent   REAL,
                    energy_unit      REAL
                );
            """)
            
            # Load cache from database
            load_alarm_cache_from_db()

            
            # Validate fdata
            if not fdata or "data" not in fdata:
                print("Warning: No valid fdata available")
                return
            
            # Filter alarm vehicles
            alarm_vehicle = [
                item for item in fdata["data"] 
                if item.get("status") in ["LOWOFFLINE", "LOWOPERATE", "CHARGING"]
                and item.get("vehicle_number")
            ]
            
            # Process alarm vehicles
            for item in alarm_vehicle:
                vnum = item["vehicle_number"]
                status = item["status"]
                
                # Skip if we already logged this same status for this vehicle
                if alarm_cache.get(vnum) == status:
                    continue
                
                # Get energy data
                if item.get("vehicletype") == "Electric":
                    energy = item.get("battery", 0)
                    percent = item.get("battery_percent", 0)
                else:
                    energy = item.get("fuellitre", 0)
                    percent = item.get("fuel", 0)
                
                # Insert alarm record
                con.execute(
                    """INSERT INTO alarm_log
                       ("License plate", model, vehicletype, event, energy_percent, energy_unit)
                       VALUES (?,?,?,?,?,?)""",
                    (vnum, item.get("model", ""), item.get("vehicletype", ""), 
                     status, percent, energy)
                )
                
                # Update cache
                alarm_cache[vnum] = status
            
            # Handle stale records
            current_ids = {item["vehicle_number"] for item in alarm_vehicle}
            stale_ids = set(alarm_cache.keys()) - current_ids
            
            for vid in stale_ids:
                # Find latest data for this vehicle from fdata
                item = next((x for x in fdata["data"] if x.get("vehicle_number") == vid), None)

                if item:
                    # Get energy data
                    if item.get("vehicletype") == "Electric":
                        energy = item.get("battery", 0)
                        percent = item.get("battery_percent", 0)
                    else:
                        energy = item.get("fuellitre", 0)
                        percent = item.get("fuel", 0)
                    
                    con.execute(
                        """INSERT INTO alarm_log
                        ("License plate", model, vehicletype, event, energy_percent, energy_unit)
                        VALUES (?,?,?,?,?,?)""",
                        (vid, item.get("model", ""), item.get("vehicletype", ""),
                        "solved", percent, energy)
                    )
                else:
                    print(f"Warning: No fdata entry found for {vid}, skipping solved insert.")
                del alarm_cache[vid]
            
            con.commit()
    except sqlite3.Error as e:
        print(f"Database error in data_logger: {e}")
    except Exception as e:
        print(f"Unexpected error in data_logger: {e}")

def delete_old_logs(db_path="data_log.db", table="alarm_log", date_column="ts", days=30):
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_str = cutoff.strftime('%Y-%m-%d')

    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute(f"""
            DELETE FROM {table}
            WHERE DATE({date_column}) < DATE(?)
        """, (cutoff_str,))
        con.commit()
        print(f"🧹 Deleted logs older than {cutoff_str}")

def delete_old_reports(folder_path="logs", days=14):
    now = time.time()
    cutoff_time = now - days * 86400  # 86400 seconds in a day

    if not os.path.exists(folder_path):
        print(f"❌ Folder {folder_path} does not exist.")
        return

    deleted_files = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            file_mtime = os.path.getmtime(file_path)
            if file_mtime < cutoff_time:
                os.remove(file_path)
                print(f"🗑️ Deleted: {file_path}")
                deleted_files += 1

    print(f"✅ Cleanup complete. {deleted_files} files removed.")
    
def initialize_cache():
    global alarm_cache
    alarm_cache = {}
    load_alarm_cache_from_db()

def load_alarm_cache_from_db():
    global alarm_cache
    
    try:
        with sqlite3.connect(log_db_path) as con:
            # Get the latest status for each vehicle that's not 'solved'
            cursor = con.execute("""
                SELECT "License plate", event
                FROM alarm_log a1
                WHERE ts = (
                    SELECT MAX(ts)
                    FROM alarm_log a2
                    WHERE a2."License plate" = a1."License plate"
                )
                AND event != 'solved'
                ORDER BY "License plate"
            """)
            
            alarm_cache = {}
            for license_plate, event in cursor.fetchall():
                alarm_cache[license_plate] = event
                
            print(f"Loaded {len(alarm_cache)} vehicles from database cache")
            
    except sqlite3.Error as e:
        print(f"Error loading cache from database: {e}")
        alarm_cache = {}

def check_transport():
    global fdata
    list_data = fdata['data']
    status_list = []  # To store the status for each row
    for entry in list_data :  # Assuming each entry is a dictionary
        if 'speed' in entry and entry['speed'] > 5:
            entry['status'] = 'TRANSPORTS'
        status_list.append(entry['status'])  # Add the status to the list
    fdata['data'] = list_data

def clear_started_flag():
    global started
    for item in started:
        item["start_flag"] = False

def notice_logger(vehicle_number: str) -> None:
    """Copy the most recent alarm row for *vehicle_number*
       and re insert it with event='noticed'."""
    with sqlite3.connect(log_db_path) as con:
        cur = con.cursor()

        #–– one‑time pragmas; harmless if they’ve run before ––#
        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("PRAGMA wal_autocheckpoint = 500;")

        # ❶ pick the latest row's primary‑key id (fast; indexed) -----------------
        row_id = cur.execute(
            """
            SELECT id
            FROM   alarm_log
            WHERE  "License plate" = ?
            ORDER  BY ts DESC
            LIMIT  1
            """,
            (vehicle_number,)
        ).fetchone()

        if not row_id:
            return                      # nothing to copy – exit silently

        # ❷ duplicate that row while overriding the event field -----------------
        cur.execute(
            """
            INSERT INTO alarm_log
                   ("License plate", model, vehicletype, event,
                    energy_percent, energy_unit)
            SELECT  "License plate", model, vehicletype,
                    'noticed'        AS event,
                    energy_percent, energy_unit
            FROM    alarm_log
            WHERE   id = ?
            """,
            row_id                      # row_id is already a 1‑tuple (e.g., (42,))
        )

def vec_started_update(data):
    global started
    # if started is none or the not same length create a new chassis
    if started is None or len(started) != len(data):
        started = [{"chassis_number": item["chassis_number"]} for item in data] 
        #print (started)
    # ensure that flag is set to 0 if it not already exist
    for entry in started:
        if 'start_flag' not in entry:
            entry['start_flag'] = False
    for entry in started:
    # Find the matching entry in fdata["data"]
        for item in data:
            # Check if chassis_number matches and status is 'OPERATE'
            if entry["chassis_number"] == item["chassis_number"] and (item["status"] == "OPERATE" or item["status"] == "LOWOPERATE"):
                # Update 'start_flag' to '1' if conditions are met
                entry["start_flag"] = True
                break  # Exit the inner loop once a match is found

def create_folder_if_not_exists(folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)
    except Exception as e:
        print(f"An error occurred while creating the folder: {e}")

def generate_report_pdf(report_date, output_dir="logs"):
    """
    Generate a PDF report for a specific date (format: 'YYYY-MM-DD').
    Output file will be named: vehicle_report_DDMMYY.pdf
    """
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.strptime(report_date, "%Y-%m-%d").strftime("%d%m%y")
    output_path = os.path.join(output_dir, f"vehicle_report_{date_str}.pdf")
    date_pdf = datetime.strptime(report_date, "%Y-%m-%d").strftime("%d/%m/%y")
    pdf = PDF_DAILY(report_date=date_pdf)
    pdf.add_page()

    with sqlite3.connect("data_log.db") as con:
        cursor = con.execute("""
            SELECT DATE(ts) as report_date, "License plate", model,
                   ts, vehicletype, event, energy_percent, energy_unit
            FROM alarm_log
            WHERE DATE(ts) = ?
            ORDER BY "License plate", ts
        """, (report_date,))
        rows = cursor.fetchall()

    if not rows:
        print(f"No data found for date {report_date}")
        return

    report_data = {}
    for row in rows:
        date, plate, model, ts, vtype, event, percent, energy = row
        key = (plate, model)
        report_data.setdefault(key, []).append((ts, vtype, event, percent, energy))

    for (plate, model), entries in report_data.items():
        pdf.add_vehicle_section(plate, model, entries)

    pdf.output(output_path)
    print(f"PDF report for {report_date} saved to {output_path}")

def generate_report_pdf_week(output_path="logs/vehicle_report_last7days.pdf"):
    pdf = PDF()
    pdf.add_page()

    seven_days_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    with sqlite3.connect("data_log.db") as con:
        cursor = con.execute("""
            SELECT DATE(ts) as report_date, "License plate", model,
                   ts, vehicletype, event, energy_percent, energy_unit
            FROM alarm_log
            WHERE DATE(ts) >= DATE(?)
            ORDER BY "License plate", ts
        """, (seven_days_ago,))
        rows = cursor.fetchall()

    if not rows:
        print("📭 No data found in the last 7 days.")
        return

    report_data = {}
    for row in rows:
        _, plate, model, ts, vtype, event, percent, energy = row
        key = (plate, model)
        report_data.setdefault(key, []).append((ts, vtype, event, percent, energy))

    for (plate, model), entries in report_data.items():
        pdf.add_vehicle_section("Last 7 Days", plate, model, entries)

    pdf.output(output_path)
    print(f"📄 PDF report for the last 7 days saved to {output_path}")

def safe_int_conversion(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0
    
def get_data():
    global started
    reg_vehicle = get_registered_vehicle()
    # print("Fetched registered vehicles")
    vehicle_status = get_vehicle_status()
    # print("Fetched vehicles status")
    work_hour = get_vehicle_workhour()
    # print("Fetched vehicles work hour")
    get_vehicle_events()
    data = merge_by_chassis([reg_vehicle,vehicle_status,work_hour])
    # print("Fetched data merged")
    # print(data)
    for entry in data:
        if 'vehicletype' not in entry:
            entry['vehicletype'] = "UNKNOWN"
        fuel = entry.get('fuel')       # Safely get 'fuel', None if missing
        battery = entry.get('battery') # Safely get 'battery', None if missing
        
        if 'vehicletype' in entry and 'model' in entry:
            with sqlite3.connect(db_path) as conn:
                # Define the SQL query
                query = """
                SELECT Powered, Brand, Tank_capacity
                FROM model_table
                WHERE Model = ?
                """
                # Execute the query with the vehicle_number as a parameter
                cursor = conn.execute(query, (entry['model'],))
                result = cursor.fetchone()

            # Check if a result was found and print it
            if result:
                entry['vehicletype'] = result[0]
                entry['manufacturer'] = result[1]
                if result[2]!=0 and result[2]!=None and entry['fuel']!=None:
                    entry['fuellitre'] = int((entry['fuel']*result[2])/100)
            else:
                entry['vehicletype'] = 'UNKNOWN'
                entry['manufacturer'] = 'UNKNOWN'
            
        if 'workhour' not in entry:
            entry['workhour'] = 0
        if entry['vehicletype'] == "Petrol" and  fuel != None :
            if entry['ignition'] is False and entry['idling'] is False and fuel<=critical_threshold:
                entry['status'] = "LOWOFFLINE"
            elif entry['ignition'] is True and entry['idling'] is False and fuel<=critical_threshold:
                entry['status'] = "LOWOPERATE"
            elif entry['ignition']  is True and entry['idling']  is True and fuel<=critical_threshold:
                entry['status'] = "LOWOPERATE"
            elif entry['ignition'] is False and entry['idling'] is False:
                entry['status'] = "OFFLINE"
            elif entry['ignition'] is True and entry['idling'] is False:
                entry['status'] = "OPERATE"
            elif entry['ignition']  is True and entry['idling']  is True:
                entry['status'] = "OPERATE"
            else:
                 entry['status'] = "UNKNOWN"  # Handle unexpected combinations
        elif entry['vehicletype'] == "Petrol" and  fuel == None :
            if entry['ignition'] is False and entry['idling'] is False:
                entry['status'] = "OFFLINE"
            elif entry['ignition'] is True and entry['idling'] is False:
                entry['status'] = "OPERATE"
            elif entry['ignition']  is True and entry['idling']  is True:
                entry['status'] = "OPERATE"
            else:
                 entry['status'] = "UNKNOWN"  # Handle unexpected combinations

        elif entry['vehicletype'] == "Electric" and battery != None:
            if entry['charge_status'] == 'On':
                entry['status'] = "CHARGING"
            elif entry['ignition'] is False and entry['idling'] is False and battery<=critical_threshold:
                entry['status'] = "LOWOFFLINE"
            elif entry['ignition']  is False and entry['idling']  is True and battery<=critical_threshold:
                entry['status'] = "LOWOPERATE"
            elif entry['ignition'] is True and entry['idling'] is False and battery<=critical_threshold:
                entry['status'] = "LOWOPERATE"
            elif entry['ignition']  is True and entry['idling']  is True and battery<=critical_threshold:
                entry['status'] = "LOWOPERATE"
            elif entry['ignition'] is False and entry['idling'] is False:
                entry['status'] = "OFFLINE"
            elif entry['ignition']  is False and entry['idling']  is True:
                entry['status'] = "OPERATE"
            elif entry['ignition'] is True and entry['idling'] is False:
                entry['status'] = "OPERATE"
            elif entry['ignition']  is True and entry['idling']  is True:
                entry['status'] = "OPERATE"        
            else:
                entry['status'] = "UNKNOWN"  # Handle unexpected combinations
        else:
            entry['status'] = "UNKNOWN"  # Handle unexpected combinations
        
        if 'battery_percent' not in entry: # convert voltage to % battery
            entry['battery_percent'] = 0
    #print(data)        
    vec_started_update(data)
    data = merge_by_chassis([data,started])
    print("Return fetched data")
    #print(data)
    return {"data": data}

def clear_csv_file():
    with open('./logfile.csv', 'w', newline='') as file:
        headers = ["time", "manufacturer", "model", "chassis_number", "event", "status"]
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
    print(f"{datetime.now()}: Cleared the CSV file.")

def voltage_to_percent():
    current_ev = get_ev()
    list_edata = current_ev['data']
    df = pd.DataFrame(list_edata)
    global previous_ev
    if previous_ev is not None:
        p_df = pd.DataFrame(previous_ev)
        operate_mask = df["status"] == "OPERATE"
        # Step 2: Apply the moving average function only to rows where 'status' is 'OPERATE'
        df.loc[operate_mask, "moving_avg"] = df.loc[operate_mask].apply(
            lambda row: update_moving_average(row["chassis_number"], row["battery"]),
            axis=1
        )
        # Step 3: Create a mask for rows where battery should be updated
        df["moving_avg"] = df["moving_avg"].fillna(p_df["battery"])
        update_mask = (df["moving_avg"].notnull()) & \
                    (p_df["battery"]*0.99 <= df["moving_avg"])
        update_mask2 = (df["status"] == "OFFLINE")
        # Step 4: Update the 'battery' column only for those rows
        df.loc[update_mask&operate_mask, "battery"] = df.loc[update_mask, "moving_avg"]
        df.loc[update_mask2, "battery"] = df.loc[update_mask, "battery"]
        df.loc[~(update_mask | update_mask2), "battery"] = p_df.loc[~(update_mask | update_mask2), "battery"]
        # Update the 'battery' column where the condition is met
    else:
        previous_ev = list_edata
    # print("moving")
    # print(merged_df)  
        #update data wait for next fetch
     # Step 1: Extract unique models from merged_df
    unique_models = tuple(df['model'].unique())
    
    # Step 2: Fetch Battery Capacities for all models
    with sqlite3.connect(db_path) as conn:
        query = f"""
        SELECT Model, Batt_voltage
        FROM model_table
        WHERE Model IN ({','.join(['?'] * len(unique_models))})
        """
        cursor = conn.execute(query, unique_models)
        model_batt_data = cursor.fetchall()

    # Step 3: Create DataFrame for merging
    batt_df = pd.DataFrame(model_batt_data, columns=['model', 'Batt_voltage'])
    merged_df = df.merge(batt_df, on='model', how='left')

    # Step 4: Use the merged voltage data
    merged_df['battery_percent'] = merged_df.apply(
        lambda row: battery_profile_acid_lead(row['battery'], row['Batt_voltage']), axis=1
    )
    # print(merged_df[['vehicle_number','model','battery','battery_percent']])
    # print(merged_df)
    updf = merged_df.to_dict(orient="records")
    # print(updf)
    global fdata
    list_fdata = fdata['data']
    # Loop through fdata and update battery_percent where chassis_number matches
    for fdata_item in list_fdata:
        # Find the matching entry in updf by chassis_number
        for update_item in updf:
            if fdata_item['chassis_number'] == update_item['chassis_number']:
                # Update battery_percent in fdata with the value from updf
                fdata_item['battery_percent'] = update_item['battery_percent']
                break  # Move to the next fdata_item after updating

    # Assign the updated list back to fdata
    fdata = {"data":list_fdata}

def update_moving_average(car_id, new_value):
    # Add new value to the car's deque
    global voltage_buffer

    # Check if the car_id exists in the buffer, if not initialize a deque
    if car_id not in voltage_buffer:
        voltage_buffer[car_id] = deque(maxlen=window_size)

    # Add new value to the car's deque
    voltage_buffer[car_id].append(new_value)
    # print (voltage_buffer)
    # Calculate moving average if the deque is full
    if len(voltage_buffer[car_id]) == window_size:
        return sum(voltage_buffer[car_id]) / window_size
    else:
        return None  # Not enough data points for the moving average yet

def battery_profile_acid_lead(battery_value,battery_capacity):
    if pd.isna(battery_value) or pd.isna(battery_capacity):
        return None

        # Ensure battery_capacity is numeric
    try:
            battery_capacity = int(battery_capacity)
    except ValueError:
        return None
    
    if battery_capacity == 12:
        if battery_value > 12.76:
            return 100
        elif battery_value <= 11.34:
            return 0
        else:
            battcal = int(round(663-(battery_value * 173) + (10.1*(battery_value**2)), 0))
            if battcal<0 :
                battcal=0
            elif battcal>100:
                battcal=100
            return battcal    
    elif battery_capacity == 24:
        if battery_value > 25.441:
            return 100
        elif battery_value <= 22.57:
            return 0
        else:
            battcal = int(round(663-(battery_value * 86.7) + (2.54*(battery_value**2)), 0))
            if battcal<0 :
                battcal=0
            elif battcal>100:
                battcal=100
            return battcal      
    elif battery_capacity == 48:
        if battery_value > 51.06:
            return 100
        elif battery_value <= 45.442:
            return 0
        else:
            battcal = int(round(663-(battery_value * 43.4) + (0.634*(battery_value**2)), 0))
            if battcal<0 :
                battcal=0
            elif battcal>100:
                battcal=100
            return battcal   
    else :
        return None

def get_registered_vehicle(max_retries=4, backoff_factor=2):
    url = "https://fleetapi-th.cartrack.com/rest/vehicles"
    dtc_url = "https://gps.dtc.co.th:8099/getVehicleMaster"
    headers = {
        'Authorization': f'Basic {base64_auth}',
        'Content-Type': 'application/json'
    }
    body = {
        "api_token_key": dtc_token
    }
    reg_vehicle = []
    
    for attempt in range(max_retries):
    # Parse the JSON data from the response
        try:
            response = requests.request("GET", url, headers=headers, timeout=10)
            response.raise_for_status()
            data = json.loads(response.text)            
            for item in data['data']:
                reg_vehicle.append({
                    'model': item['model'],
                    'chassis_number': item['chassis_number'],
                    'manufacturer': item['manufacturer'],
                    'vehicle_number': item['client_vehicle_description']
                })
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"Error fetching data: {e}")
            time.sleep(backoff_factor * (2 ** attempt))
        try:
            response = requests.request("POST", dtc_url, json=body, timeout=10)
            response.raise_for_status()
            data = json.loads(response.text)            
            for item in data['data']:
                reg_vehicle.append({
                    'model': item['vehicle_name'].strip(), #strip to remove white space at front and back of string
                    'chassis_number': item['gps_id'],
                    'manufacturer': "undefine",
                    'vehicle_number': item['license_plate'].replace(" ", "") #remove all space from license plate
                })
            # print(reg_vehicle)
            return reg_vehicle
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"Error fetching data: {e}")
            time.sleep(backoff_factor * (2 ** attempt))

    return None  # Return None after max retries

def get_vehicle_status(max_retries=4, backoff_factor=2):
    url = "https://fleetapi-th.cartrack.com/rest/vehicles/status"
    dtc_url = "https://gps.dtc.co.th:8099/getRealtimeData"
    headers = {
        'Authorization': f'Basic {base64_auth}',
        'Content-Type': 'application/json'
    }
    body = {
        "api_token_key": dtc_token,
        "gps_list":"[]"
    }
    vehicle_stat = []
    
    for attempt in range(max_retries):
    # Parse the JSON data from the response
        try:
            response = requests.request("GET", url, headers=headers, timeout=10)
            response.raise_for_status()
            data = json.loads(response.text)          
            for item in data['data']:
                fuel_percentage = clean_float(item.get('fuel', {}).get('precentage_left'))
                fuel_level = clean_float(item.get('fuel', {}).get('level'))
                battery_percentage = clean_float(item.get('vext'))
                vehicle_stat.append({
                    'chassis_number': item['chassis_number'],
                    'speed' : item['speed'],
                    'ignition': item['ignition'],
                    'idling': item['idling'],
                    'fuel': fuel_percentage,
                    'fuellitre': fuel_level,
                    'battery': battery_percentage,
                    'location': item['location']['position_description'],
                    'charge_status' : "Place holder"
                })
            # print(vehicle_stat)
            # print("vehicle_stat end here")
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"Error fetching data: {e}")
            time.sleep(backoff_factor * (2 ** attempt))
        
        try:
            response = requests.request("POST", dtc_url, json=body, timeout=10)
            response.raise_for_status()
            data = json.loads(response.text)
            for item in data['data']:
                ignition_status = item.get('status_name_th') == "รถวิ่ง" or item.get('status_name_th') == "จอดไม่ดับเครื่อง" # <---------------- This should be fix
                idle_status = item.get('status_name_th') == "จอดไม่ดับเครื่อง" # <---------------- This should be fix
                location = item.get('sub_district_th')+", "+item.get('district_th')+", "+item.get('province_th')+", "+item.get('station_name')   
                fuel_percent = item.get("oil")      
                if fuel_percent:
                    fuel_percent = safe_int_conversion(fuel_percent.split('/')[0])  # Keep only the part before the '/'
                vehicle_stat.append({        
                    'chassis_number': item['gps_id'],
                    'speed' : item['gps_speed'],
                    'ignition': ignition_status,
                    'idling': idle_status,
                    'fuel': fuel_percent,
                    'fuellitre': None,
                    'battery': clean_float(item['car_volt']),
                    'location': location,
                    'workhour' : item['hourmeter'],
                    'charge_status' : item['pto01']
                })
            # print(vehicle_stat)
            return vehicle_stat
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"Error fetching data: {e}")
            time.sleep(backoff_factor * (2 ** attempt))
    return None

def clean_float(value):
    try:
        value = float(value) if value is not None else None
        return None if math.isnan(value) else value
    except (ValueError, TypeError):
        return None
    
def clean_nan_values(data):
    """Recursively replace NaN values with None in nested structures."""
    if isinstance(data, dict):
        return {k: clean_nan_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_nan_values(v) for v in data]
    elif isinstance(data, float) and (math.isnan(data) or data == float('inf') or data == float('-inf')):
        return None
    return data

def get_vehicle_workhour(max_retries=4, backoff_factor=2):
    today = date.today()
    formatted_date = today.strftime("%Y-%m-%d")
    url = f"https://fleetapi-th.cartrack.com/rest/vehicles/activity?filter[date]={formatted_date}"

    headers = {
        'Authorization': f'Basic {base64_auth}',
        'Content-Type': 'application/json'
    }
    vehicle_wh = []
    
    for attempt in range(max_retries):
    # Parse the JSON data from the response
        try:
            response = requests.request("GET", url, headers=headers, timeout=10)
            response.raise_for_status()
            data = json.loads(response.text)      
            for item in data['data']:
                vehicle_wh.append({
                    'chassis_number': item['chassis_number'],
                    'workhour': item['total_working_hours'],
                })
            
            return vehicle_wh
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"Error fetching data: {e}")
            time.sleep(backoff_factor * (2 ** attempt))
    return None

def get_vehicle_events(max_retries=4, backoff_factor=2):
    today = datetime.now()
    time_minus_60_seconds = today - timedelta(seconds=60)
    formatted_start_date = time_minus_60_seconds.strftime("%Y-%m-%d %H:%M:%S")
    formatted_stop_date = today.strftime("%Y-%m-%d %H:%M:%S")
    
    url = f"https://fleetapi-th.cartrack.com/rest/vehicles/events?start_timestamp={formatted_start_date}&end_timestamp={formatted_stop_date}"

    headers = { 
        'Authorization': f'Basic {base64_auth}',
        'Content-Type': 'application/json'
    }
    vehicle_events = []
    
    for attempt in range(max_retries):
    # Parse the JSON data from the response
        try:
            response = requests.request("GET", url, headers=headers, timeout=10)
            response.raise_for_status()
            data = json.loads(response.text)
            for item in data['data']:
                vehicle_events.append({
                    'chassis_number': item['chassis_number'],
                    'charging_voltage': item['analog_1'],
                })
            return vehicle_events
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"Error fetching data: {e}")
            time.sleep(backoff_factor * (2 ** attempt))
    return None

def merge_by_chassis(data_arrays):
    merged_data = {}

    for data_list in data_arrays:
        for item in data_list:
            chassis_number = item['chassis_number']
            merged_data.setdefault(chassis_number, {})
            merged_data[chassis_number].update(item)

    return list(merged_data.values())