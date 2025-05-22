import numpy as np
import cv2
import json
import requests

def find_color_center(frame, hsv_bounds_list, min_area=5000):
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

def get_bounds_from_setting(setting):
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


def get_colour_centers(frame, settings):
    state = {}
    for setting in settings["settings"]:
        bounds = get_bounds_from_setting(setting)
        colour_name = setting["colour"]

        center_norm_x, center_norm_y, found = find_color_center(frame, bounds, min_area=100)

        state[colour_name] = (center_norm_x, center_norm_y)

        if not found:
            state[colour_name] = (-1, -1)

    return state


def main():
    settings = {}
    with open("robot_settings.json") as json_settings:
        settings = json.load(json_settings)

    url ="http://194.47.156.142:8080/camera"

    try:
        while True:
            response = requests.get(url, timeout=5)
            response.raise_for_status()

            img_array = np.frombuffer(response.content, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            state = get_colour_centers(frame, settings)
            print(json.dumps(state, indent=4))

            for colour in state:
                height, width = frame.shape[:2]
                center = (int(state[colour][0]* width), int(state[colour][1]* height))

                cv2.circle(frame, center, 5, (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    colour,
                    (center[0] + 10, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )


            cv2.imshow("Image", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


