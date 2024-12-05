// mnist_dataset.cc

#include "mnist_dataset.h"
#include <fstream>
#include <vector>

namespace snnf {

namespace dataset {

template <typename T>
MNISTDataset<T>::MNISTDataset() : in_features(784), out_features(10) {}

template <typename T>
void MNISTDataset<T>::load_data(const std::string& image_file, const std::string& label_file) {
    parse_images(image_file);
    parse_labels(label_file);

    if (images_.size() != labels_.size()) {
        LOG_ERROR(std::runtime_error, "MNISTDataset::load_data : Number of images and labels do not match.");
    }
}

template <typename T>
void MNISTDataset<T>::parse_images(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR(std::runtime_error, "MNISTDataset::parse_imaegs : Failed to open image file.");
    }

    uint32_t magic_number = 0;
    uint32_t num_images = 0;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;

    // Mnist uses big-end byte sequence. Use __builtin_bswap for covertion.
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    magic_number = __builtin_bswap32(magic_number);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    num_images = __builtin_bswap32(num_images);
    file.read(reinterpret_cast<char*>(&num_rows), 4);
    num_rows = __builtin_bswap32(num_rows);
    file.read(reinterpret_cast<char*>(&num_cols), 4);
    num_cols = __builtin_bswap32(num_cols);

    size_t image_size = num_rows * num_cols;
    if (image_size != in_features) {
        LOG_ERROR(std::runtime_error, "MNISTDataset::parse_images : Got a image that has wrong size.");
    }

    for (uint32_t i = 0; i < num_images; ++i) {
        std::vector<unsigned char> buffer(image_size);
        file.read(reinterpret_cast<char*>(buffer.data()), image_size);

        // The min and max value for MNIST dataset is known to be 0 and 255.
        std::vector<T> image_data(image_size);
        for (size_t j = 0; j < image_size; ++j) {
            image_data[j] = static_cast<T>(buffer[j]) / static_cast<T>(255.0);
        }

        images_.emplace_back(std::vector<uint32_t>{static_cast<uint32_t>(image_size)}, image_data);
    }

    file.close();
}

template <typename T>
void MNISTDataset<T>::parse_labels(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR(std::runtime_error, "MNISTDataset::parse_labels : Failed to open label file.");
    }

    uint32_t magic_number = 0;
    uint32_t num_labels = 0;

    // Image data
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    magic_number = __builtin_bswap32(magic_number);
    file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = __builtin_bswap32(num_labels);

    // Label data
    for (uint32_t i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);
        std::vector<T> label_data(10, static_cast<T>(0));
        label_data[static_cast<size_t>(label)] = static_cast<T>(1);
        labels_.emplace_back(std::vector<uint32_t>{out_features}, label_data);
    }

    file.close();
}

template <typename T>
size_t MNISTDataset<T>::size() const {
    return images_.size();
}

template <typename T>
std::pair<Tensor<T>, Tensor<T>> MNISTDataset<T>::get_item(size_t index) const {
    return { images_[index], labels_[index] };
}

template <typename T>
void MNISTDataset<T>::normalize(float mean, float std) {
    // Each element is already sdevided by 255 in parse_images.
    uint32_t image_size = images_.size();
    for (uint32_t i = 0; i < image_size-1; ++i) {
        for (uint32_t j = 0; j < in_features; ++j) {
            images_[i].at({j}) = ( images_[i].at({j}) - mean ) / std;
        }
    }
}

template class MNISTDataset<float>;
template class MNISTDataset<double>;

} // dataset

} // snnf