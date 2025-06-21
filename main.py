from mpi4py import MPI  # Add this import at the top
from threads import sobel_threaded
from parallel import sobel_mpi_omp
from serial_prcessing import sobel_serial

if __name__ == "__main__":
    image_path = "test_image.jpg"
    output_serial = "edges_serial.jpg"
    output_threaded = "edges_threaded.jpg"
    output_mpi_omp = "edges_mpi_omp.jpg"

    # Serial
    serial_time = sobel_serial(image_path, output_serial)
    print(f"Serial execution time: {serial_time:.2f} seconds")

    # Threaded (4 threads)
    threaded_time = sobel_threaded(image_path, output_threaded, 4)
    print(f"Threaded (4 threads) execution time: {threaded_time:.2f} seconds")
    print(f"Speedup: {serial_time / threaded_time:.2f}x")

    # MPI + OpenMP (run with: mpiexec -n 4 python script.py)
    if MPI.COMM_WORLD.Get_size() > 1:
        mpi_time = sobel_mpi_omp(image_path, output_mpi_omp, 2)
        if mpi_time is not None:
            print(f"MPI+OpenMP execution time: {mpi_time:.2f} seconds")
            print(f"Speedup: {serial_time / mpi_time:.2f}x")
