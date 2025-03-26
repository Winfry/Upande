import serial
import time
import paho.mqtt.client as mqtt
import json

# MQTT Configuration (Optional)
BROKER = "mqtt.eclipseprojects.io"
PORT = 1883
TOPIC = "animal_movement/detection"

client = mqtt.Client("AnimalMovementMonitor")
client.connect(BROKER, PORT, 60)

# Serial Connection to SenseCAP A1101
SERIAL_PORT = "COM3"  # Update based on your system
BAUD_RATE = 115200
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# Threshold for movement detection (virtual line in the frame)
THRESHOLD = 320  # Assuming a 640px-wide image
entered_count = 0
exited_count = 0

# Object Tracker
object_tracker = {}  # Format: {object_id: {"prev_x": x, "current_x": x}}

def parse_data(raw_data):
    """
    Parses raw data from SenseCAP to extract object information.
    Format: "id:1,x:150,y:200,label:animal|id:2,x:400,y:300,label:animal"
    """
    parsed_objects = []
    objects = raw_data.split("|")
    for obj in objects:
        parts = obj.split(",")
        obj_id = int(parts[0].split(":")[1])
        x_center = int(parts[1].split(":")[1])
        y_center = int(parts[2].split(":")[1])
        label = parts[3].split(":")[1]
        parsed_objects.append((obj_id, x_center, y_center, label))
    return parsed_objects

def process_frame(frame_data):
    """
    Processes each frame's data to track animal movement and count entries/exits.
    """
    global entered_count, exited_count

    for obj in frame_data:
        obj_id, x_center, y_center, label = obj

        if label == "animal":  # Focus on detected animals
            if obj_id not in object_tracker:
                # Add new animal to tracker
                object_tracker[obj_id] = {"prev_x": x_center, "current_x": x_center}
            else:
                # Update existing tracker entry
                object_tracker[obj_id]["prev_x"] = object_tracker[obj_id]["current_x"]
                object_tracker[obj_id]["current_x"] = x_center

                # Check if the animal crossed the threshold
                prev_x = object_tracker[obj_id]["prev_x"]
                current_x = object_tracker[obj_id]["current_x"]

                if current_x < THRESHOLD and prev_x > THRESHOLD:
                    entered_count += 1
                    print(f"Animal {obj_id} entered. Total entered: {entered_count}")
                elif current_x > THRESHOLD and prev_x < THRESHOLD:
                    exited_count += 1
                    print(f"Animal {obj_id} exited. Total exited: {exited_count}")

def send_to_cloud():
    """
    Sends animal movement data to the MQTT broker.
    """
    payload = {
        "entered": entered_count,
        "exited": exited_count,
    }
    client.publish(TOPIC, json.dumps(payload))
    print(f"Sent to cloud: {payload}")

def read_from_device():
    """
    Reads data from SenseCAP A1101 and processes it frame by frame.
    """
    while True:
        try:
            raw_data = ser.readline().decode('utf-8').strip()
            if raw_data:
                frame_data = parse_data(raw_data)
                process_frame(frame_data)

                # Display results locally
                print(f"Entered: {entered_count}, Exited: {exited_count}")

                # Send data to the cloud (optional)
                send_to_cloud()
        except KeyboardInterrupt:
            print("Exiting...")
            break

# Start reading from device
print("Starting animal movement detection...")
read_from_device()
