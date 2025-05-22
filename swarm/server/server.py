import requests
from IPython.display import Image, display, clear_output
import time
from urllib.parse import urlencode
import io 
import torch 

# Function to set motors
def set_drive(jetbot_ip, forward, turning):
    params = {'forward': forward, 'turn': turning}
    url = f'http://{jetbot_ip}:8080/set_drive?{urlencode(params)}'
    response = requests.get(url)
    if response.status_code == 200:
        print("Drive set successfully")
    else:
        print("Failed to set drive")   

def set_motors(jetbot_ip, left_speed, right_speed):
    params = {'left': left_speed, 'right': right_speed}
    url = f'http://{jetbot_ip}:8080/set_motors?{urlencode(params)}'
    response = requests.get(url)
    if response.status_code == 200:
        print("Motors set successfully")
    else:
        print("Failed to set motors")   

# Function to execute left command
def move_left(jetbot_ip, speed):
    params = {'speed': speed}
    url = f'http://{jetbot_ip}:8080/left?{urlencode(params)}'
    response = requests.get(url)
    if response.status_code == 200:
        print("Left command executed successfully")
    else:
        print("Failed to execute left command")

# Function to execute right command
def move_right(jetbot_ip, speed):
    params = {'speed': speed}
    url = f'http://{jetbot_ip}:8080/right?{urlencode(params)}'
    response = requests.get(url)
    if response.status_code == 200:
        print("Right command executed successfully")
    else:
        print("Failed to execute right command")

# Function to execute forward command
def move_forward(jetbot_ip, speed):
    params = {'speed': speed}
    url = f'http://{jetbot_ip}:8080/forward?{urlencode(params)}'
    response = requests.get(url)
    if response.status_code == 200:
        print("Forward command executed successfully")
    else:
        print("Failed to execute forward command")

# Function to execute stop command
def stop_robot(jetbot_ip):
    url = f'http://{jetbot_ip}:8080/stop'
    response = requests.get(url)
    if response.status_code == 200:
        print("Stop command executed successfully")
    else:
        print("Failed to execute stop command")

import json
def get_state(jetbot_ip):
    url = f'http://{jetbot_ip}:8080/state'
    response = requests.get(url)
    string_response = response.content.decode("utf-8")
    dict_response = json.loads(string_response)
    return dict_response
    
def get_weights(jetbot_ip):
    url = f'http://{jetbot_ip}:8080/get_weights'
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        try:
            buffer = io.BytesIO(response.content)
            state_dict = torch.load(buffer, map_location='cpu')
            return state_dict
        except Exception as e:
            print("Failed to load torch weights:", str(e))
            with open("bad_response.bin", "wb") as f:
                f.write(response.content)
            raise
    else:
        raise ValueError(f"HTTP error: {response.status_code}")
    
def set_weights(jetbot_ip, weights):
        buffer = io.BytesIO()
        torch.save(weights, buffer)

        headers = {
            'Content-Type': 'application/octet-stream',
            'Content-Length': str(buffer.getbuffer().nbytes)
        }

        url = f'http://{jetbot_ip}:8080/set_weights'
        response = requests.post(url, data=buffer.getvalue(), headers=headers)

        if response.status_code == 200:
            print("Set Weights command executed successfully")
        else:
            print("Failed to execute Set Weights command:", response.status_code)

# Function to display continuous camera stream
def display_camera_stream(jetbot_ip):
    while True:
        response = requests.get(f'http://{jetbot_ip}:8080/camera')
        image = Image(response.content)
        clear_output(wait=True)
        display(image)
        time.sleep(0.1) # Adjust the sleep time to control the frame rate