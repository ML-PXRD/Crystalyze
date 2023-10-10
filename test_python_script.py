import time
import torch

def time_on_cpu(matrix_size=100000):
    # Initialize matrices with random numbers
    A = torch.rand(matrix_size, matrix_size)
    B = torch.rand(matrix_size, matrix_size)
    
    # Record start time
    start_time = time.time()
    
    # Perform matrix multiplication on CPU
    C = torch.matmul(A, B)
    
    # Record end time
    end_time = time.time()
    
    # Compute time taken and return
    time_taken = end_time - start_time
    return time_taken

def time_on_gpu(matrix_size=100000):
    # Initialize matrices with random numbers on GPU
    A = torch.rand(matrix_size, matrix_size, device='cuda')
    B = torch.rand(matrix_size, matrix_size, device='cuda')
    
    # Record start time
    start_time = time.time()
    
    # Perform matrix multiplication on GPU
    C = torch.matmul(A, B)
    
    # Wait for GPU to finish computation
    torch.cuda.synchronize()
    
    # Record end time
    end_time = time.time()
    
    # Compute time taken and return
    time_taken = end_time - start_time
    return time_taken

if __name__ == "__main__":
    matrix_size = 20000
    
    cpu_time = time_on_cpu(matrix_size)
    gpu_time = time_on_gpu(matrix_size)
    
    speedup = cpu_time / gpu_time
    
    print(f"Matrix size: {matrix_size}x{matrix_size}")
    print(f"Time on CPU: {cpu_time} seconds")
    print(f"Time on GPU: {gpu_time} seconds")
    print(f"Speedup: {speedup}x")
