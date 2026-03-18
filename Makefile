# Makefile
CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=native
TARGET = fasttext_contex
SRC = fasttext_context.cpp main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)


# Makefile
CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=native
TARGET = fasttext_context
SRC = fasttext_context.cpp main.cpp

all: $(TARGET)

$(TARGET): $(SRC) fasttext_context.h
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	@echo "Running FastText with context..."
	./$(TARGET)




