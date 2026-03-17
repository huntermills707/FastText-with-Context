# Makefile
CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=native
TARGET = fasttext_context
SRC = fasttext_context.cpp main_context.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)
