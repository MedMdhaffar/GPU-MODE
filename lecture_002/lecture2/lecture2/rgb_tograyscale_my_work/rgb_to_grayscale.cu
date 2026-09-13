#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>


__global__
void grayscale_kernel(unsigned char* output, unsigned char* input, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // Base pixel index (CHW layout: channel * H * W + row * W + col)
        int pixel_idx = row * width + col;
        int plane     = height * width;          // size of one channel plane

        unsigned char r = input[pixel_idx];
        unsigned char g = input[pixel_idx +   plane];
        unsigned char b = input[pixel_idx + 2*plane];

        output[pixel_idx] = static_cast<unsigned char>(
            0.21f * r + 0.72f * g + 0.07f * b
        );
    }
}

// helper function for ceiling unsigned integer division
inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}


torch::Tensor rgb_to_grayscale(torch::Tensor image) {
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);

    const auto channels = image.size(0);
    const auto height = image.size(1);
    const auto width = image.size(2);

    auto result = torch::empty({1, height, width}, image.options());

    dim3 threads_per_block(16, 16, 3);
    dim3 number_of_blocks(
        cdiv(width, threads_per_block.x),
        cdiv(height, threads_per_block.y)
    );

    grayscale_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width,
        height
    );

    // check CUDA error status (calls cudaGetLastError())
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}
