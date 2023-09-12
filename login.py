import json
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from flask import jsonify
#from plate import *
from login import *
JSON_FILE = 'file_info.json'
UPLOAD_DIR = 'data'

# Helper function to load users from JSON file
def load_users():
    with open('users.json') as f:
        return json.load(f)

# Helper function to save users to JSON file
def save_users(users):
    with open('users.json', 'w') as f:
        json.dump(users, f)

def load_uploads():
    if not os.path.exists(JSON_FILE):
        return {}
    with open(JSON_FILE) as f:
        return json.load(f)

def save_upload(username, filename, filetype, description, priority, processed=False, result=None, speed_clicked=False, detection_clicked=False):
    uploads = load_uploads()
    upload_data = {
        'username': username,
        'filename': username+datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'filetype': filetype,
        'description': description,
        'priority': priority,
        'processed': processed,
        'result': result,
        'speed_clicked': speed_clicked,
        'detection_clicked': detection_clicked,
        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    if username in uploads:
        uploads[username].append(upload_data)
    else:
        uploads[username] = [upload_data]

    with open(JSON_FILE, 'w') as f:
        json.dump(uploads, f, indent=2)
