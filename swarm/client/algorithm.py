import cv2
import numpy as np
import torch
import json
from model import get_model, get_image_transform


class BoxFollower():
    def __init__(self):
        self.gamma = 0.0001
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        self.gamma = 0.99
        self.epsilion = 0.0001
        
        self.turn_speed = 0
        self.forward_speed = 0
        self.direction = 1

        with open("settings.json") as json_file:
            self.settings = json.load(json_file)

        self.colours = [colour["colour"] for colour in self.settings["settings"]]

        self.model = get_model(len(self.colours))
        self.model = self.model.to(self.DEVICE)
        self.preprocess = get_image_transform()

        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.result = {}
        self.decision = []

    def find_color_center(self, frame, hsv_bounds_list, min_area=5000):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame = cv2.GaussianBlur(frame, (7, 7), 0)

        # Combine masks for all HSV ranges
        combined_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        for lower, upper in hsv_bounds_list:
            mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return (-1, -1, 0)

        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
        if not large_contours:
            return (-1, -1, 0)

        largest = max(large_contours, key=cv2.contourArea)
        M = cv2.moments(largest)

        if M["m00"] == 0:
            return (-1, -1, 0)

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        height, width = frame.shape[:2]

        return (cX / width, cY / height, 1)

    def get_bounds_from_setting(self, setting):
        bounds = []
        for bound in setting["bounds"]:
            min_h = bound["min"]["hue"]
            min_s = bound["min"]["sat"]
            min_v = bound["min"]["val"]

            max_h = bound["max"]["hue"]
            max_s = bound["max"]["sat"]
            max_v = bound["max"]["val"]

            lower_bound = (min_h, min_s, min_v)
            upper_bound = (max_h, max_s, max_v)

            bounds.append((lower_bound, upper_bound))

        return bounds


    def get_colour_centers(self, frame, settings):
        state = {}
        for setting in settings["settings"]:
            bounds = self.get_bounds_from_setting(setting)
            colour_name = setting["colour"]

            center_norm_x, center_norm_y, found = self.find_color_center(frame, bounds, min_area=50)

            state[colour_name] = (center_norm_x, center_norm_y)

            if not found:
                state[colour_name] = (-1, -1)

        return state


    def step(self, image) -> (float, float):
        if image is None:
            return (0, 0, "None")

        tensor_rgb = self.numpy_to_tensor(image)
        if np.random.random() < self.gamma:
            method = "Algorithm"
            self.result = self.get_colour_centers(image, self.settings)
            self.decision = []
            for colour in self.result:
                self.decision.append(self.result[colour][0])
                self.decision.append(self.result[colour][1])
                
            tensor_rgb = self.numpy_to_tensor(image)
            self.train_model(tensor_rgb, self.decision)
        else:
            method = "Model"
            predection = self.predict(tensor_rgb)
            self.decision = predection

            for i in range(len(self.decision)):
                self.result[self.colours[i]] = self.decision[2*i]
                self.result[self.colours[i]] = self.decision[2*i+1]

        tensor_rgb.detach()
        # img_data_list.append((CAMRGB, self.forward_speed, self.turn_speed))
        self.gamma = self.gamma*(1-self.epsilion)

        if self.DEVICE == "cuda":
            torch.cuda.empty_cache()

        return (self.decision, self.result, method)

    def predict(self, tensor_rgb):
        with torch.no_grad():
            self.model.eval()
            xy = self.model(self.preprocess(tensor_rgb))
        xy = xy[0]
        return xy  
    
    def get_weights(self):
        return self.model.state_dict()
        
    def save(self, PATH="weights.pth"):
        torch.save(self.model.state_dict(), PATH)
        
    def load(self, PATH="weights.pth"):
        self.set_weights(torch.load(PATH))
    
    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def train_model(self, tensor_rgb, target):
        self.model.train()
        self.optimizer.zero_grad()
        preprocess_rgb = self.preprocess(tensor_rgb)
        predection = self.model(preprocess_rgb)
        predection = predection.squeeze(0)
        target = torch.tensor(target, dtype=torch.float, device=self.DEVICE)
        output = self.loss(predection, target)
        output.backward()
        self.optimizer.step()
        target.detach()

    def numpy_to_tensor(self, image):
        tensor_rgb = torch.from_numpy(image[:, :, 0:3])
        tensor_rgb = tensor_rgb.to(self.DEVICE)
        tensor_rgb = tensor_rgb.permute(2, 0, 1)
        tensor_rgb = tensor_rgb.float()
        tensor_rgb = tensor_rgb / 255.0
        tensor_rgb = tensor_rgb.unsqueeze(0)
        return tensor_rgb

