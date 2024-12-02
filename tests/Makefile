GCC = g++
CFLAG = -g -Wall -O2
LIBS = -lgtest -lgtest_main -lpthread
INCLUDES = -I../include/

TARGET = test_tensor test_layer test_loss test_optimizer

# source files.
TENSOR_SRC = test_tensor.cc ../src/tensor.cc
LAYER_SRC  = test_layer.cc ../src/tensor.cc ../src/linear.cc ../src/activation.cc
LOSS_SRC   = test_loss.cc ../src/tensor.cc ../src/loss.cc
OPTIMIZER_SRC = test_optimizer.cc ../src/tensor.cc ../src/linear.cc ../src/optimizer.cc

all: $(TARGET)

test_tensor : $(TENSOR_SRC)
	$(GCC) $(INCLUDES) $(CFLAG) $(TENSOR_SRC) -o $@ $(LIBS)

test_layer : $(LAYER_SRC)
	$(GCC) $(INCLUDES) $(CFLAG) $(LAYER_SRC) -o $@ $(LIBS)

test_loss : $(LOSS_SRC)
	$(GCC) $(INCLUDES) $(CFLAG) $(LOSS_SRC) -o $@ $(LIBS)

test_optimizer : $(OPTIMIZER_SRC)
	$(GCC) $(INCLUDES) $(CFLAG) $(OPTIMIZER_SRC) -o $@ $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean