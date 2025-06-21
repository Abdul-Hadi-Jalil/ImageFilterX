from mpi4py import MPI
from multiprocessing import Pool
from PIL import Image
import time
import numpy as np


def sobel_mpi_omp(image_path, output_path, num_processes=4):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Master process loads the image
        img = Image.open(image_path).convert("L")
        pixels = np.array(img)
        height, width = pixels.shape
    else:
        height, width, pixels = None, None, None

    # Broadcast image dimensions
    height = comm.bcast(height, root=0)
    width = comm.bcast(width, root=0)

    # Scatter rows of the image
    chunk_size = height // size
    local_pixels = np.zeros((chunk_size + 2, width))  # +2 for overlap
    comm.Scatterv([pixels, chunk_size * width, MPI.FLOAT], local_pixels[1:-1], root=0)

    # Share boundary rows
    if rank > 0:
        comm.Send(pixels[rank * chunk_size - 1], dest=rank - 1, tag=0)
    if rank < size - 1:
        comm.Recv(local_pixels[-1], source=rank + 1, tag=0)

    # Local processing with multiprocessing
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    start_time = time.time()

    def process_pixel(i, j):
        gx = np.sum(kernel_x * local_pixels[i - 1 : i + 2, j - 1 : j + 2])
        gy = np.sum(kernel_y * local_pixels[i - 1 : i + 2, j - 1 : j + 2])
        return np.sqrt(gx**2 + gy**2)

    local_output = np.zeros((chunk_size, width))
    with Pool(processes=num_processes) as pool:
        results = []
        for i in range(1, chunk_size + 1):
            for j in range(1, width - 1):
                results.append(pool.apply_async(process_pixel, (i, j)))

        idx = 0
        for i in range(1, chunk_size + 1):
            for j in range(1, width - 1):
                local_output[i - 1, j] = results[idx].get()
                idx += 1

    # Gather results
    if rank == 0:
        output = np.zeros((height, width))
    else:
        output = None

    comm.Gather(local_output, output, root=0)

    if rank == 0:
        output = (output / output.max()) * 255
        edge_img = Image.fromarray(output.astype(np.uint8))
        edge_img.save(output_path)
        return time.time() - start_time
    return None
