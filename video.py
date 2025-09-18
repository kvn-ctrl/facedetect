import csv
from datetime import datetime

def mark_attendance(name):
    with open("attendance.csv", "r+", newline="") as f:
        data = f.readlines()
        name_list = [line.split(",")[0] for line in data]

        # Avoid duplicate attendance
        if name not in name_list:
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%d,%H:%M:%S")
            f.writelines(f"\n{name},{dt_string}")
