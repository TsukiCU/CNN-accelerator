GCC = g++

TARGET = test_tensor
TEST_SRC = test_tensor.cc
SRC = ../src/memory.cc ../src/common.cc ../src/tensor.cc

LIBS = -lgtest -lgtest_main -lpthread -lfmt
LDFLAGS = -L/usr/local/lib
INCLUDES = -I/usr/local/include

all: $(TARGET)

$(TARGET): $(TEST_SRC) $(SRC)
	$(GCC) -o $(TARGET) $(TEST_SRC) $(SRC) $(LIBS) $(LDFLAGS) $(INCLUDES)

clean:
	rm -f $(TARGET)

.PHONY: all clean