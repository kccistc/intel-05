import tensorflow as tf, keras
import matplotlib.pyplot as plt
import pickle


# Model load: MNIST / Fashion MNIST Dataset
# 아래 2가지 중 택 1 하여 진행
# 1. mnist = tf.keras.datasets.mnist
# 2. fashion_mnist = tf.keras.datasets.fashion_mnist
#mnist = tf.keras.datasets.mnist
fashion_mnist = tf.keras.datasets.fashion_mnist


# 트레이닝 데이터랑 테스트 데이터를 나눠서 가져간다
# 위의 1, 2 선택에 따라 갈린다
#(image_t, label_t), (image_test, label_test) = mnist.load_data()
(image_t, label_t), (image_test, label_test) = fashion_mnist.load_data()

# normalized images , 입력 데이터들 정규화 진행 0~255 -> 0~1
image_t, image_test = image_t / 255.0, image_test / 255.0


image_train = image_t[:1000,:,:]
label_train = label_t[:1000]
image_val = image_t[1000:1200,:,:]
label_val = label_t[1000:1200]
val_dataset = (image_val, label_val)



# 레이블이 숫자로 저장되어 있어서 레이블과 우리가 인식할 클래스 이름을 매핑 해줘야 한다.
# 1. mnist 진행시 클래스네임에 숫자들 작성
# 2. fashion_mnist 진행시 클래스네임에 옷 이름 작성
# 어짜피 결과물에 크게 상관은 없음. 밑에 텍스트 표시가 불일치할 뿐임
#class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] # 1. mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankel boot'] # 2. fashion_mnist


plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_train[i])
    plt.xlabel(class_names[label_train[i]])
plt.show()
# 여기까지가 데이터 셋 준비가 잘 됐는지 본거임~


# ANN
model = keras.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='sigmoid'))
model.add(keras.layers.Dense(64, activation='sigmoid'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
history = model.fit(image_train, label_train, epochs=10, batch_size=10, validation_data=val_dataset)

with open('history_fashion_mnist_sigmoid', 'wb') as file_pi: # wb = write binary
    pickle.dump(history.history, file_pi)

model.summary()
model.save('fashion_mnist_sigmoid.h5')