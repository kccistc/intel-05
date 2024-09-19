# %%
from urllib.request import urlopen

import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# Fetch the image from the correct URL and decode it
req = urlopen("https://upload.wikimedia.org/wikipedia/commons/5/50/Vd-Orig.png")
img_array = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

# 커널 정의
kernels: dict[str, np.ndarray] = {
    "Original": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),  # Identity kernel
    "Edge Detection 1": np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
    "Edge Detection 2": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "Box Blur": np.ones((3, 3), np.float32) / 9,
    "Gaussian Blur 3x3": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
    "Gaussian Blur 5x5": np.array(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ]
    )
    / 256,
}

# 서브플롯 생성
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

# 이미지와 필터링된 결과를 서브플롯에 추가
axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("Original")
axes[0, 0].axis("off")

row, col = 0, 1
for title, kernel in kernels.items():
    output = cv2.filter2D(img, -1, kernel)
    axes[row, col].imshow(output, cmap="gray")
    axes[row, col].set_title(title)
    axes[row, col].axis("off")

    col += 1
    if col > 2:
        col = 0
        row += 1

# 그래프 출력
plt.tight_layout()
plt.show()
