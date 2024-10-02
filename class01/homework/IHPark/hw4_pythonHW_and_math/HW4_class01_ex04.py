# 실습4
# 1. 임의의 이미지 파일을 불러온다.
# 2. Numpy의 expend_dims를 사용해서 
# 이미지 파일의 차원을 하나 더 늘려 (Height, Width, Channel)을 
# (Batch, Height, Width, Channel)로 확장한다. (이미지 출력 불필요)
# 3. Numpy의 transpose를 이용해서 차원의 순서를 (Batch, Width, Height, Channel)에서
# (Batch, Channel, Width, Height)로 변경한다.
# (이미지 출력 불필요)
# 
# 해당 결과는 imge.shape를 통해 결과를 확인한다.

import numpy as np
from PIL import Image

img = Image.open('flower.jpg')
img_array = np.array(img)

expanded_img_array = np.expand_dims(img_array, axis=0)
print("After expand_dims:", expanded_img_array.shape)

transposed_img_array = np.transpose(expanded_img_array, (0, 3, 1, 2))
print("After transpose:", transposed_img_array.shape)