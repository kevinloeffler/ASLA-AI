import pytesseract
import numpy as np
from PIL import Image

image = Image.open('../../../../data/ocr/test/slices_edited/slice3.jpg').convert('RGB')
ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
ocr_df = ocr_df.dropna().reset_index(drop=True)
float_cols = ocr_df.select_dtypes('float').columns
ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
print(ocr_df)
words = ' '.join([word for word in ocr_df.text if str(word) != 'nan'])
print(words)
