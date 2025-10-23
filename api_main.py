from typing import List,Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field
from datetime import datetime,timedelta
from contextlib import asynccontextmanager
import pandas as pd
import os
import data_manipulate as dm
import database_function as db
import sqlite3
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

class StatusUpdate(BaseModel):
    noticeStatus: bool
# Model submit Form
class ModelForm(BaseModel):
    model: str
    combobrand: str
    dropdownbatt: int
    combotype: str
    dropdownenergy: str
    numberField: int = Field(ge=0)  # only allow positive integers

# List submit Form
class ListForm(BaseModel):
    license: str
    chassis: str
    dropdownModel: str

class ModelUpdate(BaseModel):
    model: str
    brand: Optional[str] = None
    battery: Optional[int] = None
    type: Optional[str] = None
    power_source: Optional[str] = None
    tank_size: Optional[int] = None

class VehicleUpdate(BaseModel):
    vehicle_number: str
    chassis_number: str
    model: str
# <-------------------------------- main function ------------------------------------>
# manager = ConnectionManager()
dm.initialize_cache()
scheduler = BackgroundScheduler()
noticedata = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global fdata
    try:
        print("⏳ Starting up...")
        fdata = dm.fetch_data()
        scheduler.add_job(update_fdata, IntervalTrigger(seconds=60))
        scheduler.add_job(dm.clear_started_flag, trigger='cron', hour=0, minute=0)
        scheduler.add_job(auto_daily_report, trigger='cron', hour=0, minute=1)
        scheduler.add_job(report_generator_and_clear, trigger='cron', day_of_week='sun', hour=23, minute=59)
        scheduler.start()
        dm.create_folder_if_not_exists('./logs')
        print("✅ Startup complete.")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
    yield  # app runs during this time
    print("🛑 Shutting down...")
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)
#assigned tasks order
def update_fdata():
    global fdata
    fdata = dm.fetch_data()
    dm.data_logger()

def report_generator_and_clear():
    dm.generate_report_pdf_week()
    dm.delete_old_logs()

def auto_daily_report():
    report_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')  # yesterday
    dm.generate_report_pdf(report_date)
    dm.delete_old_reports() #remove more than 14 day-old reports
# <---------------------------------- API ------------------------------------------->
#CORS allow
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["content-disposition"], 
)

# WebSocket endpoint
# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await manager.connect(websocket)
#     try:
#         while True:
#             data = await websocket.receive_text()
#             await manager.broadcast(data)
#     except WebSocketDisconnect:
#         manager.disconnect(websocket)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/testgetfunct")
def anyget():
    data = dm.get_data()
    return data

@app.get("/refreshdata")
def refresh():
    global fdata 
    fdata = dm.fetch_data()
    return fdata

@app.get("/logfiles", response_model=List[str])
async def list_files():
    log_dir = "./logs"  # Directory where your files are stored
    try:
        files = [f for f in os.listdir(log_dir) if f.endswith(".pdf")]
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"./logs/{filename}"
    return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')

@app.get("/downloadtoday")
async def download_file():
    report_date = datetime.now().strftime("%Y-%m-%d")  # Today's date
    dm.generate_report_pdf(report_date)  # This saves the PDF in logs/vehicle_report_DDMMYY.pdf

    # Match the filename used inside generate_report_pdf
    filename = f'vehicle_report_{datetime.now().strftime("%d%m%y")}.pdf'
    file_path = f"./logs/{filename}"
    
    return FileResponse(path=file_path, filename=filename, media_type='application/pdf')

@app.get("/generateweeklyreport")
def gen_w_report():
    # Generate full weekly report
    dm.generate_report_pdf_week()
    # Generate individual daily reports for past 7 days
    for i in range(7):
        report_date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        dm.generate_report_pdf(report_date)
    return {"message": "Weekly and daily reports for the last 7 days generated"}

@app.get("/fetchitems")
def read_item():
    alldata = dm.clean_nan_values(fdata)
    return alldata

@app.get("/fetchfuelvech")
def get_fuel_car():
    fuel=dm.get_fuel_vechicle()
    return fuel

@app.get("/fetchelecvech")
def get_elec_car():
    ev=dm.get_ev()
    return ev

@app.get("/fetchmch")
def get_mch_list():
    mch_list=dm.get_machine()
    return mch_list

@app.get("/fetchmodel")
def get_model_list():
    model_list=dm.get_reg_model()
    return model_list

@app.get("/getworkhour")
def read_workhour():
    data = dm.get_vehicle_workhour()
    return data

@app.get("/notices")
def get_notices():
    return {"data":noticedata}

@app.put("/statup/{item_id}/status")
def update_item_status(item_id: str, status_update: StatusUpdate):
    global noticedata
    # Check if the item already exists
    for item in noticedata:
        if item["vehicle_number"] == item_id:
            item["exclusive"] = status_update.noticeStatus
            break
    else:
        # If not found, append a new item
        noticedata.append({"vehicle_number": item_id, "exclusive": status_update.noticeStatus})
        # log the notice
    if status_update.noticeStatus:
        dm.notice_logger(item_id)

@app.post("/add_model")
def insert_model(data: ModelForm):
    try:
        conn = sqlite3.connect("bbdrentvehicle.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO model_table (Model, Brand, Batt_voltage, Type, Powered, Tank_capacity)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (data.model, data.combobrand, data.dropdownbatt, data.combotype, data.dropdownenergy, data.numberField)
        )
        conn.commit()
        conn.close()
        return {"message": "Model inserted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/add_registered")
def insert_machine(data: ListForm):
    model, _, brand = data.dropdownModel.partition(" - ")
    try:
        conn = sqlite3.connect("bbdrentvehicle.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO register_table ("License plate", "Chassis number", Model)
            VALUES (?, ?, ?)
            """,
            (data.license, data.chassis, model)
        )
        conn.commit()
        conn.close()
        return {"message": "Machine inserted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.put("/edit_model/{id}")
def edit_exist_model(id: int, data: ModelUpdate):
    print(f"Editing model_table, id={id}")
    print("Received update data:")
    print(data.dict())

    try:
        conn = sqlite3.connect('bbdrentvehicle.db')
        cursor = conn.cursor()

        # Check if the row exists
        cursor.execute("SELECT id FROM model_table WHERE id = ?", (id,))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Model not found")

        # Perform the update
        cursor.execute("""
            UPDATE model_table
            SET Model = ?, Brand = ?, Batt_voltage = ?, Type = ?, Powered = ?, Tank_capacity = ?
            WHERE id = ?
        """, (
            data.model,
            data.brand,
            data.battery,
            data.type,
            data.power_source,
            data.tank_size,
            id
        ))

        conn.commit()

    finally:
        conn.close()

    return {"message": "Model updated", "id": id, "data": data}


@app.put("/edit_register/{id}")
def edit_exist_registered_vehicle(id: int, data: VehicleUpdate):
    print(f"Editing mch table, id={id}")
    print(data)
    try:
        conn = sqlite3.connect('bbdrentvehicle.db')
        conn.row_factory = sqlite3.Row  # So we can access columns by name
        cursor = conn.cursor()

        # Look up the existing row
        cursor.execute("SELECT * FROM register_table WHERE id = ?", (id,))
        row = cursor.fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Model not found")

        # Perform the update
        cursor.execute("""
            UPDATE register_table
            SET Model = ?, "License plate" = ?, "Chassis number" = ?
            WHERE id = ?
        """, (
            data.model,
            data.vehicle_number,
            data.chassis_number,
            id
        ))

        conn.commit()
        

    finally:
        conn.close()
    return {"message": "Vehicle updated", "id": id, "data": data}

@app.delete("/delete_reg/{id}")
def delete_item(id: int):
    try:
        conn = sqlite3.connect('bbdrentvehicle.db')
        cursor = conn.cursor()

        # Check if item exists
        cursor.execute("SELECT * FROM register_table WHERE id = ?", (id,))
        if cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Item not found")

        # Delete item
        cursor.execute("DELETE FROM register_table WHERE id = ?", (id,))
        conn.commit()

    finally:
        conn.close()

    return {"message": "Item deleted", "id": id}

@app.delete("/delete_model/{id}")
def delete_item(id: int):
    try:
        conn = sqlite3.connect('bbdrentvehicle.db')
        cursor = conn.cursor()

        # Check if item exists
        cursor.execute("SELECT * FROM model_table WHERE id = ?", (id,))
        result = cursor.fetchone()
        if result is None:
            raise HTTPException(status_code=404, detail="Item not found")

        model_name = result[2]
        # Delete item
        cursor.execute("DELETE FROM model_table WHERE id = ?", (id,))
        # 3. Update register_table where Model == deleted model name
        cursor.execute("""
            UPDATE register_table
            SET Model = 'UNKNOWN'
            WHERE Model = ?
        """, (model_name,))

        conn.commit()       
    finally:
        conn.close()
    return {"message": f"Model '{model_name}' deleted and any related registrations updated to 'UNKNOWN'"}