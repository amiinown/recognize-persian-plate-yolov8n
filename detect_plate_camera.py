from ultralytics import YOLO
import torch
import cv2 as cv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {mydevice}')

plate_model = YOLO('runs/all_plate_cars/train/weights/best.pt')
plate_model.to(mydevice)
char_model = YOLO('runs/plate_characters/train/weights/best.pt')
char_model.to(mydevice)

tracker = DeepSort(max_age=30, n_init=4, max_iou_distance=0.7, nn_budget=200)

font_path = 'font/BNAZNNBD.TTF'
font = ImageFont.truetype(font_path, 25)

char_mapping = {
    '0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴', '5': '۵',
    '6': '۶', '7': '۷', '8': '۸', '9': '۹',
    'B': 'ب', 'C': 'س', 'D': 'د', 'G': 'ق', 'H': 'ه',
    'J': 'ج', 'L': 'ل', 'M': 'م', 'N': 'ن', 'S': 'ص', 
    'T': 'ط', 'V': 'و', 'Y': 'ی'
}

cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    org_img_pil = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    plate_preds = plate_model(frame, conf=0.25)
    detections = []
    
    for plate in plate_preds:
        for box_plate in plate.boxes:
            x, y, x2, y2 = map(int, box_plate.xyxy[0])
            conf = float(box_plate.conf)
            class_id = int(box_plate.cls[0])
            detections.append([[x, y, x2, y2], conf, class_id])

            cropped_plate_img = frame[y:y2, x:x2]
            char_preds = char_model(cropped_plate_img)

            detected_chars_with_locx_boxes = []
            for char in char_preds:
                for box_char in char.boxes:
                    cls = int(box_char.cls[0])
                    x_ = box_char.xyxy[0][0].item()
                    detected_chars_with_locx_boxes.append((x_, char_model.names[cls]))

            sorted_chars = sorted(detected_chars_with_locx_boxes, key=lambda x: x[0])
            detected_chars = [char for x, char in sorted_chars]
            detected_chars = [char_mapping[char] for char in detected_chars if char in char_mapping]

            plate_characters = ''.join(detected_chars)
            draw = ImageDraw.Draw(org_img_pil) 
            draw.rectangle([(x, y), (x2, y2)], outline='red', width=2)
            if plate_characters and len(plate_characters)>=8:
                plate_characters = plate_characters[:8]
                plate_characters = plate_characters[:6] + '-' + plate_characters[6:]
            else:
                plate_characters = ':('
            draw.text((x, y - 25), plate_characters, font=font, fill='red')

    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        # cv.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    frame = cv.cvtColor(np.array(org_img_pil), cv.COLOR_RGB2BGR)
    cv.imshow('Online recognition plate', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
