from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import time


def sobel_threaded(image_path, output_path, num_threads=4):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    pixels = np.array(img)
    height, width = pixels.shape
    output = np.zeros((height, width))

    # Sobel kernels
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    start_time = time.time()

    def process_chunk(start_row, end_row):
        """Process a chunk of rows in the image"""
        for i in range(start_row, end_row):
            for j in range(1, width - 1):
                if i >= 1 and i < height - 1:  # Boundary check
                    gx = np.sum(kernel_x * pixels[i - 1 : i + 2, j - 1 : j + 2])
                    gy = np.sum(kernel_y * pixels[i - 1 : i + 2, j - 1 : j + 2])
                    output[i, j] = np.sqrt(gx**2 + gy**2)

    # Split work among threads using ThreadPoolExecutor
    chunk_size = height // num_threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for t in range(num_threads):
            start = t * chunk_size
            end = (t + 1) * chunk_size if t != num_threads - 1 else height
            executor.submit(process_chunk, start, end)

    # Normalize and save
    output = (output / output.max()) * 255
    edge_img = Image.fromarray(output.astype(np.uint8))
    edge_img.save(output_path)

    return time.time() - start_time
