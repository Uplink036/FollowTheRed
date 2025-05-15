
import numpy as np
import torch
from model import get_model, get_image_transform


class BoxFollower():
    def __init__(self):
        self.gamma = 0.0001
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = get_model()
        self.model = self.model.to(self.DEVICE)
        self.preprocess = get_image_transform()

        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.gamma = 0.99
        self.epsilion = 0.0001
        
        self.turn_speed = 0
        self.forward_speed = 0
        self.direction = 1

    def step(self, image) -> (float, float):
        if image is None:
            return (0, 0)

        red_areas = np.all([image[:, :, 0] > 200,
                            image[:, :, 1] < 100,
                            image[:, :, 2] < 100], axis=0)

        red_cols = np.argwhere(np.any(red_areas, axis=0))
        red_size = len(red_cols)

        if np.random.random() < self.gamma:
            image_half_width = image.shape[0]/2
            box_width = 999
            if red_size > 0:
                mean_loc = np.mean(red_cols)
                self.turn_speed = abs(mean_loc-image_half_width)/image_half_width
                if np.mean(red_cols) > image_half_width:
                    self.direction = -1
                elif np.mean(red_cols) < image_half_width:
                    self.direction = 1

                box_width = float(red_cols[-1]-red_cols[0])

            if box_width > image_half_width/2:
                self.forward_speed = 0
            else:
                self.forward_speed = 1-(box_width/image_half_width)

            self.optimizer.zero_grad()
            tensor_rgb = torch.from_numpy(image[:, :, 0:3])
            tensor_rgb = tensor_rgb.to(self.DEVICE)
            tensor_rgb = tensor_rgb.permute(2, 0, 1)
            tensor_rgb = tensor_rgb.float()
            tensor_rgb = tensor_rgb / 255.0
            tensor_rgb = tensor_rgb.unsqueeze(0)
            preprocess_rgb = self.preprocess(tensor_rgb)

            xy = self.model(preprocess_rgb)
            target = torch.tensor([self.forward_speed, self.direction*self.turn_speed],
                                  dtype=torch.float, device=self.DEVICE)
            output = self.loss(xy, target)
            output.backward()
            self.optimizer.step()
        else:
            tensor_rgb = torch.from_numpy(image[:, :, 0:3])
            tensor_rgb = tensor_rgb.to(self.DEVICE)
            tensor_rgb = tensor_rgb.permute(2, 0, 1)
            tensor_rgb = tensor_rgb.float()
            tensor_rgb = tensor_rgb / 255.0
            tensor_rgb = tensor_rgb.unsqueeze(0)
            xy = self.model(self.preprocess(tensor_rgb))
            xy = xy[0]
            self.forward_speed = float(xy[0])
            self.turn_speed = float(xy[1])
            self.direction = 1

        # img_data_list.append((CAMRGB, self.forward_speed, self.turn_speed))
        self.gamma = self.gamma*(1-self.epsilion)

        if self.DEVICE == "cuda":
            torch.cuda.empty_cache()

        return (self.forward_speed/2, self.direction*self.turn_speed*2*np.pi)
