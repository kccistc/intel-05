# %%
from pathlib import Path

import keras
import numpy as np
from keras.api import Model
from keras.api.datasets import mnist

saved_path: Path = Path.home() / "build" / "model"
saved_path.mkdir(parents=True, exist_ok=True)

(image_t, label_t), (image_test, label_test) = mnist.load_data()
# 이미지 데이터를 255로 나누어 각 픽셀의 값을 0과 1 사이로 정규화합니다. 이는 신경망 모델의 학습 성능을 향상시키는 일반적인 전처리 작업입니다.
image_t, image_test = image_t / 255.0, image_test / 255.0

image_train = image_t[:1000, :, :]
label_train = label_t[:1000]
image_val = image_t[1000:1200, :, :]
label_val = label_t[1000:1200]
val_dataset = (image_val, label_val)


# class_names = [str(object=i) for i in range(10)]
class_names = list(range(10))

num = 10000
model_relu: Model = keras.saving.load_model("mnist_relu.keras")
model_sigmoid: Model = keras.saving.load_model("mnist_sigmoid.keras")

predict_relu = model_relu.predict(image_test[:num])
predict_sigmoid = model_sigmoid.predict(image_test[:num])

print("input image = ", label_test[:num])
print("predict relu = ", np.argmax(predict_relu, axis=1))
print("predict sigmoid = ", np.argmax(predict_sigmoid, axis=1))
