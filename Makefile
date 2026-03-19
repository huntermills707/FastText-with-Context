CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=native -fopenmp
LDFLAGS = -fopenmp
TARGET = fasttext_context
SRC = fasttext_context.cpp main.cpp

all: $(TARGET)

$(TARGET): $(SRC) fasttext_context.h
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)
