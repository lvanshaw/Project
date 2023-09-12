from function import *

def set_upload_info(filename, info):
    uploads = load_uploads()
    for username in uploads:
        for file_info in uploads[username]:
            if file_info['filename'] == filename:
                file_info.update(info)
    with open(JSON_FILE, 'w') as f:
        json.dump(uploads, f, indent=2)

def update_upload(filename, new_data):
    uploads = load_uploads()

    for username, files in uploads.items():
        for file in files:
            if file["filename"] == filename:
                file.update(new_data)
                break

    with open(JSON_FILE, "w") as f:
        json.dump(uploads, f, indent=2)

# helper function to process speed
def calculate_speed(filename):
    # perform processing to calculate speed
    speed = '10 mph' # dummy result
    return speed

# helper function to process object detection
def detect_objects(filename):
    # perform object detection processing
    detection_result = 'Cars: 10, Bikes: 5, Pedestrians: 3' # dummy result
    return detection_result

