from typing import Any

import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch


class TrOCR:
    def __init__(self, model: str):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.processor = TrOCRProcessor.from_pretrained(model)
        self.model = VisionEncoderDecoderModel.from_pretrained(model).to(self.device)
        # model config
        self.model.config.max_new_tokens = 10
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4

    def predict(self, image: Image or str, visualize: bool = False) -> str | None:
        img = self.__load_image(image)
        pixel_values = self.processor(img, return_tensors='pt').pixel_values.to(self.device)
        # generated_ids = self.model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)

        '''
        # Approach with greedy tokenizer (faster but no prediction score)
        generation_result = self.model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)
        generated_ids = generation_result['sequences']

        scores = generation_result['scores'][0][0].numpy()
        max_score = np.max(scores)
        if max_score < 15:
            return None
        '''

        result = self.model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)
        generated_ids = result['sequences']
        score = result['sequences_scores'].numpy()[0]

        if score < -0.1:
            return None

        predicted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if visualize:
            self.__visualize_prediction(image=image, prediction=predicted_text)
        return predicted_text

    @staticmethod
    def __visualize_prediction(image: Image, prediction: str):
        canvas = Image.new('RGB', (image.size[0] + 100, image.size[1] + 100), (255, 255, 255))
        canvas.paste(image, (50, 66))

        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default(size=28)
        draw.text((50, 10), text=prediction, fill='black', font=font)

        canvas.show()

    @staticmethod
    def __load_image(image: Image or str) -> Image:
        if type(image) is Image.Image:
            return image
        return Image.open(image)

# Models:
# microsoft/trocr-base-printed
# microsoft/trocr-base-handwritten
# microsoft/trocr-large-handwritten
