import sys
import os
from PIL import Image
import numpy as np

sys.path.append("fedmed_model")
import predict

img_zeros = np.zeros((224, 224, 3), dtype=np.uint8)
img_zeros_path = "zeros.png"
Image.fromarray(img_zeros).save(img_zeros_path)

img_ones = np.ones((224, 224, 3), dtype=np.uint8) * 255
img_ones_path = "ones.png"
Image.fromarray(img_ones).save(img_ones_path)

print("Zeros:", predict.predict_single(img_zeros_path))
print("Ones:", predict.predict_single(img_ones_path))
