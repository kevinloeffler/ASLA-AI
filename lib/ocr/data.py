import csv
import os.path

from PIL import Image


def load_images(csv_file: str, image_base_path: str, count: int = None) -> list[tuple[Image, dict]]:
    output: list[tuple[Image, dict]] = []
    with open(csv_file, 'r') as file:
        reader = list(csv.reader(file))
        if not count:
            count = len(reader)

        for row in reader:
            image_path = f'{image_base_path}{row[0].lower()}/{row[1]}.jpg'
            if count <= 0:
                return output
            if not os.path.exists(image_path) or row[1] == '':
                continue
            image = Image.open(image_path)
            metadata = {'head': row[2], 'mst': row[3], 'date': row[4]}
            count -= 1
            output.append((image, metadata))
    return output
