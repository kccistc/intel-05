# %%
from PIL import Image
import numpy as np

image_path = 'path_to_your_image.jpg' 
image = Image.open(image_path)

image_np = np.array(image)

image_np_expanded = np.expand_dims(image_np, axis=0)

image_np_transposed = np.transpose(image_np_expanded, (0, 3, 1, 2))

print("Original shape:", image_np.shape)
print("Expanded shape:", image_np_expanded.shape)
print("Transposed shape:", image_np_transposed.shape)
