from transformers import LayoutLMForTokenClassification, LayoutLMv2Processor
from PIL import Image, ImageDraw, ImageFont

Bounding_Box = tuple[int, int, int, int]

# load layout_model and processor from huggingface hub
model = LayoutLMForTokenClassification.from_pretrained('philschmid/layoutlm-funsd')
processor = LayoutLMv2Processor.from_pretrained('philschmid/layoutlm-funsd')


# helper function to unnormalize bboxes for drawing onto the image
def unnormalize_box(bbox: Bounding_Box, width: int, height: int) -> Bounding_Box:
    # bounding_box = bbox.numpy()
    return (
        round(width * (bbox[0] / 1000)),
        round(height * (bbox[1] / 1000)),
        round(width * (bbox[2] / 1000)),
        round(height * (bbox[3] / 1000)),
    )


label2color = {
    "B-HEADER": "blue",
    "B-QUESTION": "red",
    "B-ANSWER": "green",
    "I-HEADER": "blue",
    "I-QUESTION": "red",
    "I-ANSWER": "green",
}


# draw results onto the image
def draw_boxes(image: Image, boxes, predictions):
    width, height = image.size
    normalizes_boxes = [unnormalize_box(box, width, height) for box in boxes]

    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=24)
    for prediction, box in zip(predictions, normalizes_boxes):
        if prediction == "O":
            continue
        draw.rectangle(box, outline="black", width=3)
        draw.rectangle(box, outline=label2color[prediction], width=3)
        draw.text((box[0] + 10, box[1] - 10), text=prediction, fill=label2color[prediction], font=font)
    return image


# run inference
def run_inference(path, model=model, processor=processor, output_image=True):
    # create layout_model input
    image = Image.open(path).convert("RGB")
    image_width, image_height = image.size
    encoding = processor(image, return_tensors="pt")
    del encoding["image"]
    # run inference
    outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    # get labels
    labels = [model.config.id2label[prediction] for prediction in predictions]
    # bounding_boxes = [(box.numpy()[0], box.numpy()[1], box.numpy()[2], box.numpy()[3]) for box in encoding['bbox'][0]]
    bounding_boxes = list(set([(box.numpy()[0], box.numpy()[1], box.numpy()[2], box.numpy()[3]) for box in encoding['bbox'][0]]))
    # TODO: unnormalize boxes and convert from tensor to int tuple

    #ocr_model = TrOCR('microsoft/trocr-large-handwritten')

    #for bounding_box in bounding_boxes:
    #    box = unnormalize_box(bounding_box, image_width, image_height)
    #    box = resize_bounding_box(box, 5, image.size)
    #    if box[0] == box[2] or box[1] == box[3]:
    #        continue
    #    masked_image = image.crop(box)
    #    ocr_model.predict(masked_image, visualize=True)

    if output_image:
        annotated_image = draw_boxes(image, bounding_boxes, labels)
        annotated_image.show()
        return annotated_image
    else:
        return labels


def resize_bounding_box(bounding_box: Bounding_Box, by: int, image_size: tuple[int, int]) -> Bounding_Box:
    return (
        max(bounding_box[0] - by, 0),
        max(bounding_box[1] - by, 0),
        min(bounding_box[2] + by, image_size[0]),
        min(bounding_box[3] + by, image_size[1]),
    )


run_inference('../../../../data/ocr/mnu/MN_1388_3.jpg')



#def ocr(image, processor, layout_model):
#    pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
#    generated_ids = layout_model.generate(pixel_values)
#    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#    print('generated text:', generated_text)
#    return generated_text
