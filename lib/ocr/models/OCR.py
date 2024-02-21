import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from transformers import LayoutLMv2Processor

from lib.ocr.clustering import Clustering
from lib.ocr.models.TrOCR import TrOCR
from lib.ocr.preprocessing import remove_stamp
from lib.util import create_unverified_ssl_context, Timer

BoundingBox = tuple[int, int, int, int]


class OCR:

    def __init__(self, layout_model: str, ocr_model: str):
        create_unverified_ssl_context()
        timer = Timer()
        print('loading layout model...', end=' ')
        timer.start()
        self.layout_model = LayoutLMv2Processor.from_pretrained(layout_model)
        print(timer.stop(as_string=True) + 's')
        print('loading ocr model...', end=' ')
        timer.start()
        self.ocr_model = TrOCR(ocr_model)
        print(timer.stop(as_string=True) + 's')
        self.knn_model = Clustering()

    def predict(self, path, overlap_threshold: float = 0.8):
        """Predict bounding boxes and recognize the text in them"""
        timer = Timer()

        print('finding boxes...', end=' ')
        timer.start()

        image = Image.open(path).convert('RGB')
        image = Image.fromarray(remove_stamp(image_data=np.asarray(image), stamp_path='lib/ocr/asla_stamp_cropped.jpg'))
        encoding = self.layout_model(image, return_tensors="pt")
        del encoding["image"]

        bounding_boxes = self.__get_bounding_boxes(encoding, image.size, overlap_threshold)
        ordered_boxes = self.__order_boxes(bounding_boxes)

        print(timer.stop(as_string=True) + 's')

        print('extracting text...', end=' ')
        timer.start()

        prediction = []
        for group in ordered_boxes:
            predictions = [self.__predict_text(box, image) for box in group]
            clean_predictions = list(filter(lambda p: p is not None, predictions))
            prediction.append(' '.join(clean_predictions))

        print(timer.stop(as_string=True) + 's')

        print('prediction:', prediction)

        annotated_image = self.__draw_boxes(image, bounding_boxes)
        annotated_image.show()
        return prediction

    def replace_layout_model(self, new_model: str):
        print('loading layout model...')
        self.layout_model = LayoutLMv2Processor.from_pretrained(new_model)

    def replace_ocr_model(self, new_model: str):
        print('loading ocr model...')
        self.ocr_model = TrOCR(new_model)

    def __get_bounding_boxes(self, encoding, image_size: tuple[int, int], overlap_threshold: float) -> list[list[BoundingBox]]:
        """Normalize, combine and group bounding boxes"""
        image_width, image_height = image_size
        normalized_boxes = list(set([(box.numpy()[0], box.numpy()[1], box.numpy()[2], box.numpy()[3]) for box in encoding['bbox'][0]]))
        all_boxes = [self.__unnormalize_box(box, image_width, image_height) for box in normalized_boxes]
        overlapping_boxes = list(filter(lambda box: not self.__is_small_box(box, min_axis_length=16), all_boxes))
        ungrouped_boxes = self.__combine_overlapping_boxes(overlapping_boxes, overlap_threshold)
        if len(ungrouped_boxes) == 0:
            raise ValueError('ERROR: Layout model did not find any boxes')
        return self.knn_model.predict_groups(ungrouped_boxes)

    def __order_boxes(self, boxes_groups: list[list[BoundingBox]]) -> list[list[BoundingBox]]:
        """Bring all bounding boxes of a group into normal reading order (top to bottom, left to right)"""
        ordered_boxes = []
        for boxes in boxes_groups:
            rows: list[BoxRow] = []
            for box in boxes:
                self.__find_boxes_in_same_row(box, rows)

            sorted_rows = self.__sort_boxes(rows)
            ordered_boxes.append(sorted_rows)
        return ordered_boxes

    @staticmethod
    def __find_boxes_in_same_row(box, rows):
        """Combine boxes with a similar y value into a BoxRow"""
        if len(rows) == 0:
            rows.append(BoxRow(upper_bound=box[1], lower_bound=box[3], children=[box]))
            return

        for row in rows:
            box_center = box[1] + round((box[3] - box[1]) / 2)
            if row.upper_bound < box_center < row.lower_bound:
                row.add(box)
                return
        rows.append(BoxRow(upper_bound=box[1], lower_bound=box[3], children=[box]))
        return

    @staticmethod
    def __sort_boxes(rows: list[any]) -> list[BoundingBox]:
        """Sort all boxes in a row / line from left to right according to the left x value"""
        sorted_boxes = []
        # sort lines (rows) vertically
        vertically_sorted_rows = list(sorted(rows, key=lambda row: row.upper_bound))
        # sort words in line left to right
        for row in vertically_sorted_rows:
            sorted_row = list(sorted(row.children, key=lambda box: box[0]))
            sorted_boxes += sorted_row
        return sorted_boxes

    def __predict_text(self, box: BoundingBox, image: Image) -> str | None:
        resized_box = self.__resize_bounding_box(box, by=10, image_size=image.size)
        masked_image = image.crop(resized_box)
        return self.ocr_model.predict(masked_image)

    @staticmethod
    def __unnormalize_box(bbox, width: int, height: int) -> BoundingBox:
        """The LayoutLM model returns normalized (0-100) bounding boxes which need to be scaled back to normal"""
        return (
            round(width * (bbox[0] / 1000)),
            round(height * (bbox[1] / 1000)),
            round(width * (bbox[2] / 1000)),
            round(height * (bbox[3] / 1000)),
        )

    @staticmethod
    def __resize_bounding_box(bounding_box: BoundingBox, by: int, image_size: tuple[int, int]) -> BoundingBox:
        """Add a 'safety' margin to bounding boxes to handle slight prediction errors"""
        return (
            max(bounding_box[0] - by, 0),
            max(bounding_box[1] - by, 0),
            min(bounding_box[2] + by, image_size[0]),
            min(bounding_box[3] + by, image_size[1]),
        )

    def __combine_overlapping_boxes(self, boxes: list[BoundingBox], overlap_threshold: float) -> list[BoundingBox]:
        """Combine boxes that overlap a certain amount into one"""
        combined_boxes = []

        for i, box1 in enumerate(boxes):
            skip_box = False
            for j, box2 in enumerate(combined_boxes):
                if self.__iom(box1, box2) > overlap_threshold:
                    # Merge the boxes if the overlap is above the threshold
                    combined_boxes[j] = (
                        min(box1[0], box2[0]),
                        min(box1[1], box2[1]),
                        max(box1[2], box2[2]),
                        max(box1[3], box2[3]),
                    )
                    skip_box = True
                    break

            if not skip_box:
                combined_boxes.append(box1)

        return combined_boxes

    @staticmethod
    def __iom(box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate the Intersection over Minimum (IoM) of two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        min_area = min((box1[2] - box1[0]) * (box1[3] - box1[1]), (box2[2] - box2[0]) * (box2[3] - box2[1]))

        iom = intersection_area / min_area if min_area > 0 else 0
        return iom

    @staticmethod
    def __is_small_box(box: BoundingBox, min_axis_length: int = 10):
        x_axis = box[2] - box[0]
        y_axis = box[3] - box[1]
        return x_axis < min_axis_length or y_axis < min_axis_length

    @staticmethod
    def __draw_boxes(image: Image, grouped_boxes: list[list[BoundingBox]]):
        """draw predictions over the image"""
        draw = ImageDraw.Draw(image)
        group_colors = ['#FF3B30', '#FF9500', '#FFCC00', '#00C7BE', '#59ADC4',
                        '#007AFF', '#5856D6', '#AF52DE', '#FF2D55', '#A2845E']
        for index, group in enumerate(grouped_boxes):
            color = group_colors[index % len(group_colors)]
            for box in group:
                draw.rectangle(box, outline=color, width=3)
        return image


@dataclass
class BoxRow:
    """Used to find bounding boxes in the same row"""
    upper_bound: int
    lower_bound: int
    children: list[BoundingBox]

    def add(self, box: BoundingBox):
        self.children.append(box)
        self.upper_bound = min(box[1], self.upper_bound)
        self.lower_bound = max(box[3], self.lower_bound)

    def get_center(self) -> int:
        return round((self.lower_bound - self.upper_bound) / 2) + self.upper_bound
