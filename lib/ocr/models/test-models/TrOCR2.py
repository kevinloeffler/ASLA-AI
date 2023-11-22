from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

from lib.ocr.data import load_images


class TrOCR_2:

    def __init__(self, config, model: str):
        self.config = config
        self.processor = TrOCRProcessor.from_pretrained(model)
        self.model = VisionEncoderDecoderModel.from_pretrained(model)

    def predict(self, image: Image):
        pixel_values = self.processor(image, return_tensors='pt').pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        print(generated_text)

