import serial
import time

# Initialize SenseCAP
SERIAL_PORT = "COM3"
ser = serial.Serial(SERIAL_PORT, baudrate=115200, timeout=1)

def parse_data(raw_data):
    """
    Parses raw detection data from SenseCAP for hand inference.
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
    Processes detected hands in the current frame.
    """
    hand_count = sum(1 for obj in frame_data if obj[3] == "hand")
    print(f"Detected {hand_count} hands.")

def read_from_device():
    """
    Reads and processes frames from SenseCAP A1101.
    """
    while True:
        try:
            raw_data = ser.readline().decode('utf-8').strip()
            if raw_data:
                frame_data = parse_data(raw_data)
                process_frame(frame_data)
        except KeyboardInterrupt:
            print("Exiting...")
            break

# Run inference
read_from_device()
