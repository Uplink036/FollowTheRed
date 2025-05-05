import cv2
import numpy as np
from jetbot import Camera, Robot
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import urllib.parse
# Function to convert BGR8 images to JPEG
def bgr8_to_jpeg(value):
    return cv2.imencode('.jpg', value)[1].tobytes()

# Initialize the camera and robot
camera = Camera.instance(width=224, height=224)
robot = Robot()

# HTTP request handler class
class CameraHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/camera'):
            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.end_headers()
            self.wfile.write(bgr8_to_jpeg(camera.value))
        else:
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            if self.path.startswith('/set_motors'):
                left_speed = float(params.get('left', [0])[0])
                right_speed = float(params.get('right', [0])[0])
                robot.set_motors(left_speed, right_speed)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'Motors set')
            elif self.path.startswith('/left'):
                speed = float(params.get('speed', [0])[0])
                robot.left(speed)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'Left command executed')
            elif self.path.startswith('/right'):
                speed = float(params.get('speed', [0])[0])
                robot.right(speed)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'Right command executed')
            elif self.path.startswith('/forward'):
                speed = float(params.get('speed', [0])[0])
                robot.forward(speed)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'Forward command executed')
            elif self.path.startswith('/stop'):
                robot.stop()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'Stop command executed')
                
# Function to run the HTTP server
def run_server():
    server = HTTPServer(('0.0.0.0', 8080), CameraHandler)
    server.serve_forever()
    # Start the server in a separate thread
    thread = threading.Thread(target=run_server)
    thread.start()