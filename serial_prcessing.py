import numpy as np
from PIL import Image
import time


def sobel_serial(image_path, output_path):
    # Load image
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    pixels = np.array(img)
    height, width = pixels.shape

    # Initialize output
    output = np.zeros((height, width))

    # Sobel kernels
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    start_time = time.time()

    # Apply Sobel operator
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            gx = np.sum(kernel_x * pixels[i - 1 : i + 2, j - 1 : j + 2])
            gy = np.sum(kernel_y * pixels[i - 1 : i + 2, j - 1 : j + 2])
            output[i, j] = np.sqrt(gx**2 + gy**2)

    # Normalize and save
    output = (output / output.max()) * 255
    edge_img = Image.fromarray(output.astype(np.uint8))
    edge_img.save(output_path)

    return time.time() - start_time
