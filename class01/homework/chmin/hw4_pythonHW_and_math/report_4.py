from PIL import Image
import numpy as np

image = Image.open('download.jpeg')

image_array = np.array(image)
print("Original:\t", image_array.shape)
expend_dim = np.expand_dims(image_array, axis = 0)

print("Expand_dim:\t", expend_dim.shape)
trans_pose = np.transpose(expend_dim,(0,3,1,2))

print("Transpose:\t",trans_pose.shape)