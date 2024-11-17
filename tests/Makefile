NVCC = g++

TARGET = test_memory
TEST_SRC = test_memory.cu
SRC = ../src/memory.cu ../src/common.cu

LIBS = -lgtest -lgtest_main -lpthread # -lcudart -lcuda
LDFLAGS = -L/usr/local/lib
INCLUDES = -I/usr/local/include

all: $(TARGET)

$(TARGET): $(TEST_SRC) $(SRC)
	$(NVCC) -o $(TARGET) $(TEST_SRC) $(SRC) $(LIBS) $(LDFLAGS) $(INCLUDES)

clean:
	rm -f $(TARGET)

.PHONY: all clean