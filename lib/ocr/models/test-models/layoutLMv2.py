from transformers import LayoutLMv2Processor
from PIL import Image, ImageDraw, ImageFont

# image = Image.open('../../../data/ocr/mnu/MN_1388_3.jpg').convert("RGB")
# encoding = processor(image, return_tensors="pt")  # you can also add all tokenizer parameters here such as padding, truncation


def unnormalize_box(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    # bounding_box = bbox.numpy()
    return (
        round(width * (bbox[0] / 1000)),
        round(height * (bbox[1] / 1000)),
        round(width * (bbox[2] / 1000)),
        round(height * (bbox[3] / 1000)),
    )


def draw_boxes(image: Image, boxes):
    width, height = image.size
    normalizes_boxes = [unnormalize_box(box, width, height) for box in boxes]

    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=24)
    for box in normalizes_boxes:
        draw.rectangle(box, outline='red', width=3)
    return image


def run_inference(path, processor):
    print('start prediction...')
    # create layout_model input
    image = Image.open(path).convert('RGB')
    image_width, image_height = image.size
    encoding = processor(image, return_tensors="pt")
    del encoding["image"]
    bounding_boxes = list(set([(box.numpy()[0], box.numpy()[1], box.numpy()[2], box.numpy()[3]) for box in encoding['bbox'][0]]))
    print(bounding_boxes)
    annotated_image = draw_boxes(image, bounding_boxes)
    annotated_image.show()


def load_model(path):
    print('loading layout_model...')
    return LayoutLMv2Processor.from_pretrained(path)


# processor = load_model('microsoft/layoutlmv2-base-uncased')
# run_inference('../../../data/ocr/mnu/MN_1388_3.jpg', processor)
# run_inference('../../../data/ocr/test/MN_1388_1_edited.jpg', processor)
