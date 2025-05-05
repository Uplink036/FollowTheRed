import requests
from IPython.display import Image, display, clear_output
import time
from urllib.parse import urlencode

# Function to set motors
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

# Function to display continuous camera stream
def display_camera_stream(jetbot_ip):
    while True:
        response = requests.get(f'http://{jetbot_ip}:8080/camera')
        image = Image(response.content)
        clear_output(wait=True)
        display(image)
        time.sleep(0.1) # Adjust the sleep time to control the frame rate

# Example usage
jetbot_ip = '194.47.156.22' # Replace with your Jetbot's IP address
#set_motors(jetbot_ip, 0.3, 0)
#move_left(jetbot_ip, 0.3)
#move_right(jetbot_ip, 0.3)
#move_forward(jetbot_ip, 0.3)
#stop_robot(jetbot_ip)
#display_camera_stream(jetbot_ip)