import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageFont

from lib.util import create_unverified_ssl_context


create_unverified_ssl_context()


def draw_boxes(image: Image, boxes: list[list[int]], predictions: list[str]):
    assert len(boxes) == len(predictions), f'Predictions list and boxes list need to be of same length'
    # width, height = image.size

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=28)

    for prediction, box in zip(predictions, boxes):
        rect = box[0] + box[2]
        print('rect:', rect)
        draw.rectangle(rect, outline='red', width=3)
        draw.text((box[0][0], box[1][1] - 32), text=prediction, fill='red', font=font)

    image.show()
    return image


image = Image.open('../../../../data/ocr/mnu/MN_1388_5.jpg').convert('RGB')

reader = easyocr.Reader(['de'])
result = reader.readtext(np.array(image), paragraph=False)
boxes = [item[0] for item in result]
predictions = [item[1] for item in result]
draw_boxes(image, boxes, predictions)
