from ultralytics import YOLO
import torch
import cv2 as cv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datetime import datetime

mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {mydevice}')

plate_model = YOLO('runs/all_plate_cars/train/weights/best.pt')
plate_model.to(mydevice)
char_model = YOLO('runs/plate_characters/train/weights/best.pt')
char_model.to(mydevice)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
input_path = 'input/' # Or your full directory
output_path = f'output/image/{timestamp}.jpg' # Or your full directory

org_img = cv.imread(input_path)
org_img_pil = Image.fromarray(cv.cvtColor(org_img, cv.COLOR_BGR2RGB))
char_mapping = {
    '0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴', '5': '۵',
    '6': '۶', '7': '۷', '8': '۸', '9': '۹',
    'B': 'ب', 'C': 'س', 'D': 'د', 'G': 'ق', 'H': 'ه',
    'J': 'ج', 'L': 'ل', 'M': 'م', 'N': 'ن', 'S': 'ص', 
    'T': 'ط', 'V': 'و', 'Y': 'ی'
}

font_path = 'font/BNAZNNBD.TTF'
font = ImageFont.truetype(font_path, 20)

plate_preds = plate_model(input_path, conf=0.25)

for plate in plate_preds:
    for box_plate in plate.boxes:
        conf = box_plate.conf
        x, y, x2, y2 = map(int, box_plate.xyxy[0])
        print(f'Plate Location: ({x}, {y}), ({x2}, {y2}) with confidence: {conf}')
        print(f'***********************************\nBox data:{plate.boxes}\n***********************************')

        cropped_plate_img = org_img[y:y2, x:x2]
        char_preds = char_model(cropped_plate_img)

        detected_chars = []
        detected_chars_with_locx_boxes = []
        for char in char_preds:
            for box_char in char.boxes:
                cls = int(box_char.cls[0])
                x_ = box_char.xyxy[0][0].item()
                detected_chars_with_locx_boxes.append((x_,char_model.names[cls]))

        sorted_chars = sorted(detected_chars_with_locx_boxes, key=lambda x: x[0])
        detected_chars = [char for x, char in sorted_chars]

        detected_chars = [char_mapping[char] for char in detected_chars if char in char_mapping]

        plate_characters = ''.join(detected_chars)
        print(f'detected plate Characters: {plate_characters}')

        draw = ImageDraw.Draw(org_img_pil)
        draw.rectangle([(x, y), (x2, y2)], outline='red', width=1)

        if plate_characters and len(plate_characters)>=8:
            plate_characters = plate_characters[:8]
            plate_characters = plate_characters[:6] + '-' + plate_characters[6:]
        else:
            plate_characters = ':('
        draw.text((x, y - 20), plate_characters, font=font, fill='red')

org_img = cv.cvtColor(np.array(org_img_pil), cv.COLOR_RGB2BGR)
cv.imwrite(output_path, org_img)
print(f'Output saved to {output_path}')