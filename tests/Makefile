GCC = g++
CFLAG = -g -Wall -O2
LIBS = -lgtest -lgtest_main -lpthread

TARGET = test_tensor test_layer

# source files.
SRC = 
TENSOR_SRC = test_tensor.cc
LAYER_SRC = test_layer.cc

all: $(TARGET)

test_tensor : $(TENSOR_SRC) $(SRC)
	$(GCC) $(CFLAG) $(TENSOR_SRC) -o $@ $(LIBS)

test_layer : $(LAYER_SRC) $(SRC)
	$(GCC) $(CFLAG) $(LAYER_SRC) -o $@ $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean