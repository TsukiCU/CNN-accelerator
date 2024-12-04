GCC = g++
CFLAG = -g -Wall -O2
INCLUDES = -I../../include/ -I../../include/dataset

TARGET = iris

IRIS_DATA = data/iris.txt

SOURCE_FILES = iris.cc ../../src/tensor.cc ../../src/linear.cc ../../src/activation.cc \
../../src/optimizer.cc ../../src/loss.cc ../../src/model.cc ../../src/dataset.cc \
../../src/dataset/iris_dataset.cc

all: prepare iris

prepare:
	@echo "Getting Iris dataset..."
	@if [ ! -e $(IRIS_DATA) ]; then \
		./get_iris.sh; \
	fi

iris: $(SOURCE_FILES) $(IRIS_DATA)
	$(GCC) $(INCLUDES) $(CFLAG) $(SOURCE_FILES) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: iris