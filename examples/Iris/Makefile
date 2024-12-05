GCC = g++
CFLAG = -g -Wall -O2
INCLUDES = -I../../include/ -I../../include/dataset

TARGET = iris

IRIS_DATA = data/iris.txt

SOURCE_FILES = iris.cc ../../src/tensor.cc ../../src/layer/linear.cc ../../src/layer/activation.cc \
../../src/layer/softmax.cc ../../src/optimizer.cc ../../src/loss.cc ../../src/model.cc ../../src/dataset/dataset.cc \
../../src/dataset/iris_dataset.cc

all: prepare iris

prepare:
	@echo "\e[1;32m...Getting Iris dataset...\e[0m"
	@if [ ! -e $(IRIS_DATA) ]; then \
		./get_iris.sh; \
	fi
	@echo "\e[1;32mdone\e[0m"

iris: $(SOURCE_FILES) $(IRIS_DATA)
	$(GCC) $(INCLUDES) $(CFLAG) $(SOURCE_FILES) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: iris