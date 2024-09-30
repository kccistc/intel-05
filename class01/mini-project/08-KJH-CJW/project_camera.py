# %%
import cv2
from ultralytics import YOLO
import time
import os

# Load YOLO model
model = YOLO("yolov8m.pt")
names = model.model.names

# Constants for distance calculation
KNOWN_DISTANCE = 76.2
KNOWN_WIDTH = 50.0
focal_length = 700
DISTANCE_SCALE = 0.7

def distance_finder(focal_length, real_width, width_in_frame):
    return (real_width * focal_length) / width_in_frame * DISTANCE_SCALE

# Define the directory to save captured images
save_directory = "./captured_img/"  # Change this to your desired path

# Create the directory if it doeqsn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

cap = cv2.VideoCapture(0)  # Change the index if necessary

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

first_person_detected = False
first_person_id = None
threshold_line_x = None
moved_right = False
moved_left = False
first_person_distance = None

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    height, width, _ = im0.shape
    threshold_line_x = width // 2

    results = model.track(im0, persist=True, show=False, verbose=False)
    persons = []

    for result in results:
        for track in result.boxes:
            if track is not None:
                cls = int(track.cls[0])
                x1, y1, x2, y2 = map(int, track.xyxy[0])
                if names[cls] == 'person':
                    persons.append((track.id, x1, y1, x2, y2))

    for (person_id, x1, y1, x2, y2) in persons:
        person_width = x2 - x1
        distance = distance_finder(focal_length, KNOWN_WIDTH, person_width)

        if not first_person_detected:
            first_person_detected = True
            first_person_id = person_id
            first_person_distance = distance
            color = (0, 0, 255)
        elif person_id == first_person_id:
            color = (0, 0, 255)
            first_person_distance = distance
        else:
            color = (255, 0, 0)

        cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
        label = "Owner" if person_id == first_person_id else "Person"
        cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if person_id == first_person_id:
            cv2.putText(im0, f"Distance: {round(distance, 2)} cm", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if first_person_detected and person_id == first_person_id:
            if x1 > threshold_line_x and x2 > threshold_line_x:
                moved_right = True
                moved_left = False
            elif x1 < threshold_line_x and x2 < threshold_line_x:
                moved_left = True
                moved_right = False
            elif x1 < threshold_line_x and x2 > threshold_line_x:
                moved_left = False
                moved_right = False

    if moved_right:
        cv2.putText(im0, "RIGHT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if moved_left:
        cv2.putText(im0, "LEFT", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if first_person_distance is not None and first_person_distance > 150:
        cv2.putText(im0, "Straight", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.line(im0, (threshold_line_x, 0), (threshold_line_x, height), (255, 255, 255), 1)

    cv2.imshow("Detection", im0)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        capture_path = os.path.join(save_directory, f"capture_{timestamp}.png")
        cv2.imwrite(capture_path, im0)
        print(f"Frame captured and saved as {capture_path}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# %% 방향

import cv2
from ultralytics import YOLO
import time
import os

# Load YOLO model
model = YOLO("yolov8m.pt")
names = model.model.names

# Constants for distance calculation
KNOWN_DISTANCE = 76.2  # cm, distance to reference image
KNOWN_WIDTH = 50.0     # cm, average width of a person
focal_length = 700     # Focal length in pixels, set according to your camera setup
DISTANCE_SCALE = 0.7   # Scale factor to adjust the perceived distance

def distance_finder(focal_length, real_width, width_in_frame):
    return (real_width * focal_length) / width_in_frame * DISTANCE_SCALE

# Define the directory to save captured images
save_directory = "./captured_img/"  # Change this to your desired path

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

cap = cv2.VideoCapture(2)  # Change the index if necessary

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

first_person_detected = False
first_person_id = None
threshold_line_x = None
moved_right = False
moved_left = False

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    height, width, _ = im0.shape
    threshold_line_x = width // 2

    results = model.track(im0, persist=True, show=False, verbose=False)
    persons = []

    for result in results:
        for track in result.boxes:
            if track is not None:
                cls = int(track.cls[0])
                x1, y1, x2, y2 = map(int, track.xyxy[0])
                if names[cls] == 'person':
                    persons.append((track.id, x1, y1, x2, y2))

    for (person_id, x1, y1, x2, y2) in persons:
        person_width = x2 - x1
        distance = distance_finder(focal_length, KNOWN_WIDTH, person_width)

        if not first_person_detected:
            first_person_detected = True
            first_person_id = person_id
            color = (0, 0, 255)  # 빨간색
        elif person_id == first_person_id:
            color = (0, 0, 255)  # 빨간색
        else:
            color = (255, 0, 0)  # 파란색

        cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
        label = "Owner" if person_id == first_person_id else "Person"
        cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 하얀색 기준선 그리기
    cv2.line(im0, (threshold_line_x, 0), (threshold_line_x, height), (255, 255, 255), 1)

    # 방향 표시 유지
    if moved_right:
        cv2.putText(im0, "RIGHT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if moved_left:
        cv2.putText(im0, "LEFT", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Detection", im0)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        capture_path = os.path.join(save_directory, f"capture_{timestamp}.png")
        cv2.imwrite(capture_path, im0)
        print(f"Frame captured and saved as {capture_path}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# %% 거리
