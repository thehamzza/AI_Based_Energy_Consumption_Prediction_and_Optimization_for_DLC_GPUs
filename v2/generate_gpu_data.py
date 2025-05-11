import json
import random
from datetime import datetime, timedelta

def generate_gpu_data(num_records=500):
    data = []
    current_time = datetime.now()

    for _ in range(num_records):
        timestamp = current_time.isoformat()
        gpu_util = random.randint(20, 100)
        memory_util = random.randint(10, 100)
        temp_ambient = round(random.uniform(20.0, 30.0), 2)
        work_load = random.choice([0, 1])  # 0 = idle, 1 = running
        power_limit = random.randint(150, 300)
        fan_speed = random.randint(1000, 5000)

        # Output rules
        power = round(50 + 0.8 * gpu_util + random.uniform(-5, 5), 2)
        core_temp = round(50 + 0.2 * gpu_util + (5 if work_load else -3) + random.uniform(-2, 2), 2)
        duration = round(random.uniform(0.5, 3.5) + (0.5 if work_load else 0), 2)

        data.append({
            "timestamp": timestamp,
            "gpu_utilization": gpu_util,
            "memory_utilization": memory_util,
            "ambientTemperature": temp_ambient,
            "workLoadType": work_load,
            "powerLimit": power_limit,
            "fanSpeed": fan_speed,
            # Outputs
            "powerConsumption": power,
            "core_temperature": core_temp,
            "temperature_duration": duration
        })

        current_time += timedelta(minutes=5)

    with open("gpu_synthetic_data.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {num_records} GPU records with outputs.")

if __name__ == "__main__":
    generate_gpu_data()