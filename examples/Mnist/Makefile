GCC = g++
CFLAG = -g -Wall -O2
INCLUDES = -I../../include/ -I../../include/dataset

TARGET = mnist

MNIST_DATA = data/train-images-idx3-ubyte \
			 data/train-labels-idx1-ubyte \
			 data/t10k-images-idx3-ubyte  \
			 data/t10k-labels-idx1-ubyte

SOURCE_FILES = mnist.cc ../../src/tensor.cc ../../src/linear.cc ../../src/activation.cc \
../../src/optimizer.cc ../../src/loss.cc ../../src/model.cc ../../src/dataset.cc \
../../src/dataset/mnist_dataset.cc

all: prepare mnist

prepare:
	@echo "\e[1;32m...Getting MNIST dataset...\e[0m"
	@missing=0; \
	for file in $(MNIST_DATA); do \
		if [ ! -e $$file ]; then \
			missing=1; \
			./get_mnist.sh; \
		fi; \
	done; \
	if [ $$missing -eq 1 ]; then \
		./get_mnist.sh; \
	fi
	@echo "\e[1;32mdone\e[0m"

mnist: $(SOURCE_FILES) $(MNIST_DATA)
	$(GCC) $(INCLUDES) $(CFLAG) $(SOURCE_FILES) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: mnist