#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../include/memory.h"

class MBTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess) << "Failed to set CUDA device";
    }

    void TearDown() override {
        cudaDeviceReset();
    }
};


/********** Resource allocation and deallocation **********/
TEST_F(MBTest, AllocDeallocOnCPU) {
    uint32_t size = 1024;
    cuda::MemoryBuffer buffer(size, cuda::DeviceType::CPU);

    ASSERT_NE(buffer.data(), nullptr);
    ASSERT_EQ(buffer.get_size(), size);
    ASSERT_EQ(buffer.get_device_type(), cuda::DeviceType::CPU);
}

TEST_F(MBTest, AllocDeallocOnGPU) {
    uint32_t size = 1024;
    cuda::MemoryBuffer buffer(size, cuda::DeviceType::GPU);

    ASSERT_NE(buffer.data(), nullptr);
    ASSERT_EQ(buffer.get_size(), size);
    ASSERT_EQ(buffer.get_device_type(), cuda::DeviceType::GPU);
}


/********** Data copying between different devices **********/
TEST_F(MBTest, CopyFromCpuToGpu) {
    uint32_t size = 1024;
    cuda::MemoryBuffer cpu_buffer(size, cuda::DeviceType::CPU);
    cuda::MemoryBuffer gpu_buffer(size, cuda::DeviceType::GPU);

    uint8_t* cpu_data = static_cast<uint8_t*>(cpu_buffer.data());
    for (int i = 0; i < size; ++i) {
        cpu_data[i] = static_cast<uint8_t>(i % 256);
    }

    gpu_buffer.copy_from(cpu_buffer);

    cuda::MemoryBuffer cpu_buffer_verify(size, cuda::DeviceType::CPU);
    cpu_buffer_verify.copy_from(gpu_buffer);

    uint8_t* verify_data = static_cast<uint8_t*>(cpu_buffer_verify.data());
    for (int i = 0; i < size; ++i) {
        ASSERT_EQ(verify_data[i], cpu_data[i]);
    }
}

TEST_F(MBTest, CopyFromGpuToCpu) {
    uint32_t size = 1024;
    cuda::MemoryBuffer gpu_buffer(size, cuda::DeviceType::GPU);
    cuda::MemoryBuffer cpu_buffer(size, cuda::DeviceType::CPU);

    uint8_t* temp_data = new uint8_t[size];
    for (int i = 0; i < size; ++i) {
        temp_data[i] = static_cast<uint8_t>((i * 2) % 256);
    }
    cudaMemcpy(gpu_buffer.data(), temp_data, size, cudaMemcpyHostToDevice);
    delete[] temp_data;

    cpu_buffer.copy_from(gpu_buffer);
    uint8_t* cpu_data = static_cast<uint8_t*>(cpu_buffer.data());
    for (int i = 0; i < size; ++i) {
        ASSERT_EQ(cpu_data[i], static_cast<uint8_t>((i * 2) % 256));
    }
}

TEST_F(MBTest, CopyBetweenGPUs) {
    uint32_t size = 1024;
    cuda::MemoryBuffer gpu_buffer_src(size, cuda::DeviceType::GPU);
    cuda::MemoryBuffer gpu_buffer_dst(size, cuda::DeviceType::GPU);

    uint8_t* temp_data = new uint8_t[size];
    for (int i = 0; i < size; ++i) {
        temp_data[i] = static_cast<uint8_t>((i * 3) % 256);
    }
    cudaMemcpy(gpu_buffer_src.data(), temp_data, size, cudaMemcpyHostToDevice);
    delete[] temp_data;

    gpu_buffer_dst.copy_from(gpu_buffer_src);
    cuda::MemoryBuffer cpu_buffer(size, cuda::DeviceType::CPU);

    cpu_buffer.copy_from(gpu_buffer_dst);
    uint8_t* cpu_data = static_cast<uint8_t*>(cpu_buffer.data());
    for (int i = 0; i < size; ++i) {
        ASSERT_EQ(cpu_data[i], static_cast<uint8_t>((i * 3) % 256));
    }
}

TEST_F(MBTest, CopyBetweenCPUs) {
    uint32_t size = 1024;
    cuda::MemoryBuffer cpu_buffer_src(size, cuda::DeviceType::CPU);
    cuda::MemoryBuffer cpu_buffer_dst(size, cuda::DeviceType::CPU);

    uint8_t* src_data = static_cast<uint8_t*>(cpu_buffer_src.data());
    for (int i = 0; i < size; ++i) {
        src_data[i] = static_cast<uint8_t>((i * 4) % 256);
    }

    cpu_buffer_dst.copy_from(cpu_buffer_src);
    uint8_t* dst_data = static_cast<uint8_t*>(cpu_buffer_dst.data());
    for (int i = 0; i < size; ++i) {
        ASSERT_EQ(dst_data[i], src_data[i]);
    }
}


/********** Exceptions should do their jobs **********/
TEST_F(MBTest, InvalidDeviceAllocation) {
    uint32_t size = 1024;
    EXPECT_THROW({
        cuda::MemoryBuffer buffer_a(size, cuda::DeviceType::UNKNOWN);
    }, std::invalid_argument);

    EXPECT_THROW({
        cuda::MemoryBuffer buffer_b(size, static_cast<cuda::DeviceType>(42));
    }, std::invalid_argument);
}

TEST_F(MBTest, ZeroSizeAllocation) {
    // gpu_buffer.ptr_ will be nullptr which will trigger the exception.
    cuda::MemoryBuffer cpu_buffer(0, cuda::DeviceType::CPU);
    ASSERT_NE(cpu_buffer.data(), nullptr);
    ASSERT_EQ(cpu_buffer.get_size(), 0);

    cuda::MemoryBuffer gpu_buffer(0, cuda::DeviceType::GPU);
    ASSERT_EQ(gpu_buffer.data(), nullptr);
    ASSERT_EQ(gpu_buffer.get_size(), 0);

    EXPECT_THROW( {cpu_buffer.copy_from(gpu_buffer);}, std::runtime_error);
}