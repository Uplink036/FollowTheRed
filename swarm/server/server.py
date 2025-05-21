import requests
from IPython.display import Image, display, clear_output
import time
from urllib.parse import urlencode

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
    response = requests.get(url)
    string_response = response.content.decode("utf-8")
    dict_response = json.loads(string_response)
    return dict_response
    
def set_weights(jetbot_ip, weights):
    string_response = response.content.encode("utf-8")
    params = {'weights': string_response}
    url = f'http://{jetbot_ip}:8080/set_weights?{urlencode(params)}'
    response = requests.get(url)
    dict_response = json.loads(string_response)
    return dict_response

# Function to display continuous camera stream
def display_camera_stream(jetbot_ip):
    while True:
        response = requests.get(f'http://{jetbot_ip}:8080/camera')
        image = Image(response.content)
        clear_output(wait=True)
        display(image)
        time.sleep(0.1) # Adjust the sleep time to control the frame rate