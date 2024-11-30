GCC = g++

TARGET = test_tensor
TEST_SRC = test_tensor.cc
SRC = 

LIBS = -lgtest -lgtest_main -lpthread -lfmt

all: $(TARGET)

$(TARGET): $(TEST_SRC) $(SRC)
	$(GCC) -o $(TARGET) $(TEST_SRC) $(SRC) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean