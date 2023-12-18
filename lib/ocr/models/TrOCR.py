from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch


class TrOCR:
    def __init__(self, model: str):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.processor = TrOCRProcessor.from_pretrained(model)
        self.model = VisionEncoderDecoderModel.from_pretrained(model).to(self.device)

    def predict(self, image: Image or str, visualize: bool = False) -> str:
        img = self.__load_image(image)
        pixel_values = self.processor(img, return_tensors='pt').pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values)
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
