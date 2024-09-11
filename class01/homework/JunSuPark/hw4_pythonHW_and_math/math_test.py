# %%
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
import numpy as np

## Homework 1
print("===== Homework 1 =====")
my_tuple = ("A", "B")
my_tuple = (*my_tuple, "C")
my_tuple

## Homework 2
print("===== Homework 2 =====")
arr = np.array(range(1, 17))
arr_2d = arr.reshape(4, 4)
arr_2d

## Homework 3
print("===== Homework 3 =====")
arr_flattend = arr_2d.flatten()
arr_flattend

## Homework 4
print("===== Homework 4 =====")
image_eg = np.random.randint(0, 255, size=(64, 64, 3))  # Height=64, Width=64, Channel=3
# insert "Batch" dimensions insert [0]
image_with_batch = np.expand_dims(image_eg, axis=0)
image_with_batch.shape
image_with_batch = np.transpose(image_with_batch, (0, 3, 1, 2))
image_with_batch.shape

## Homework 5
print("===== Homework 5 =====")
# Example image with RGBA channels
image_eg = np.random.randint(0, 255, size=(128, 128, 4))
image_eg.shape
# Kernel (Edge detection)
kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

pad_size = kernel.shape[0] // 2  # Compute the padding size based on the kernel size
height, width, channels = image_eg.shape  # Correct order: Height, Width, Channels

# Create new image to store the result (excluding boundary pixels)
new_image_eg = np.zeros((height - 2 * pad_size, width - 2 * pad_size, channels))

# Flip the kernel 180 degrees for proper convolution
flipped_kernel = kernel[::-1, ::-1]  # or      flipped_kernel_manual = np.flip(kernel)

# Perform manual convolution without padding
for c in range(channels):
    for y in range(pad_size, height - pad_size):
        for x in range(pad_size, width - pad_size):
            # Extract region of the image corresponding to the kernel size
            region = image_eg[
                y - pad_size : y + pad_size + 1, x - pad_size : x + pad_size + 1, c
            ]

            # Apply the flipped kernel to the region and store the result
            new_image_eg[y - pad_size, x - pad_size, c] = np.sum(
                region * flipped_kernel
            )

print("Resulting image shape:", new_image_eg.shape)
# cv2.imshow("edge", new_image_eg)
# cv2.waitKey(0)


# %%
import matplotlib.pyplot as plt
import numpy as np

# Define x and y values using linspace
x = np.linspace(-2, 2, 11)
y = np.linspace(-2, 2, 11)

# Print the x and y values
print(x)
print(y)

# Create a meshgrid for x and y
x, y = np.meshgrid(x, y)
print(x)
print(y)

# Define the function f
f = lambda x, y: (x - 1) ** 2 + (y - 1) ** 2
z = f(x, y)
print(z)

# Compute the gradient of f
grad_f_x = lambda x, y: 2 * (x - 1)
grad_f_y = lambda x, y: 2 * (y - 1)

dz_dx = grad_f_x(x, y)
dz_dy = grad_f_y(x, y)

# Plot the contour and gradient field
ax = plt.axes()
ax.contour(x, y, z, levels=np.linspace(0, 10, 20), cmap=plt.cm.jet)
ax.quiver(x, y, -dz_dx, -dz_dy)

# Set axis to equal and add grid
ax.axis("equal")
ax.grid()

# Label axes
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

# Show plot
plt.show()
