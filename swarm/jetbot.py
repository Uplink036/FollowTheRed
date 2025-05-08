import cv2
from jetbot import Camera, Robot
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import urllib.parse

from algorithm import BoxFollower


# Function to convert BGR8 images to JPEG
def bgr8_to_jpeg(value):
    return cv2.imencode('.jpg', value)[1].tobytes()


camera = Camera.instance(width=1920, height=1080)
robot = Robot()


# HTTP request handler class
class NetworkHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.running = True
        self.paused = False
        self.image = None

        self.box_follower_thread = threading.Thread(
            target=self.box_follower_handler, daemon=True)
        self.box_follower_thread.start()

        # Important: Call superclass constructor last
        super().__init__(*args, **kwargs)

    def box_follower_handler(self):
        bf = BoxFollower()
        while self.running:
            if not self.paused:
                self.image = camera.value
                decision = bf.step(self.image)
                print(f"Decision: {decision}")

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
            elif self.path.startswith('/run_algo'):
                self.paused = False
            elif self.path.startswith('/stop_algo'):
                self.paused = True


# Function to run the HTTP server
def run_server():
    server = HTTPServer(('0.0.0.0', 8080), NetworkHandler)
    server.serve_forever()

    # Start the server in a separate thread
    thread = threading.Thread(target=run_server)
    thread.start()

