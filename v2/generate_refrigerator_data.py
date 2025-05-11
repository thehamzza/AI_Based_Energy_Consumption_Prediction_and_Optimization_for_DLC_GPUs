import json
import random
from datetime import datetime, timedelta

def generate_refrigerator_data(num_records=500):
    data = []
    current_time = datetime.now()

    for _ in range(num_records):
        timestamp = current_time.isoformat()
        door_open = random.choice([0, 1])
        door_open_duration = round(random.uniform(0, 10), 2) if door_open else 0
        ambient_temp = round(random.uniform(18.0, 28.0), 2)
        internal_temp = round(random.uniform(2.0, 8.0), 2)

        # Output rules
        power = round(60 + (door_open_duration * 1.5) + random.uniform(-3, 3), 2)
        internal_temp_out = round(internal_temp + 0.5 * door_open_duration + random.uniform(-1, 1), 2)
        duration = round(min(5.0, 1.0 + door_open_duration * 0.4), 2)

        data.append({
            "timestamp": timestamp,
            "doorOpen": door_open,
            "door_open_duration": door_open_duration,
            "ambientTemperature": ambient_temp,
            "internalTemperature": internal_temp,
            # Outputs
            "powerConsumption": power,
            "internal_temperature": internal_temp_out,
            "temperature_duration": duration
        })

        current_time += timedelta(minutes=5)

    with open("refrigerator_synthetic_data.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {num_records} refrigerator records with outputs.")

if __name__ == "__main__":
    generate_refrigerator_data()