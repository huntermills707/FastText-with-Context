CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=native -fopenmp
LDFLAGS = -fopenmp

TARGET_TRAIN = train
TARGET_QUERY = query
SRC_COMMON = fasttext_context.cpp
SRC_TRAIN = train.cpp
SRC_QUERY = query.cpp
HEADER = fasttext_context.h

all: $(TARGET_TRAIN) $(TARGET_QUERY)

$(TARGET_TRAIN): $(SRC_COMMON) $(SRC_TRAIN) $(HEADER)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET_TRAIN) $(SRC_COMMON) $(SRC_TRAIN)

$(TARGET_QUERY): $(SRC_COMMON) $(SRC_QUERY) $(HEADER)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET_QUERY) $(SRC_COMMON) $(SRC_QUERY)

clean:
	rm -f $(TARGET_TRAIN) $(TARGET_QUERY)

run-train: $(TARGET_TRAIN)
	./$(TARGET_TRAIN) training_data_with_context.txt model.bin

run-query: $(TARGET_QUERY)
	@echo "Usage: ./$(TARGET_QUERY) model.bin <word> [k]"
	@echo "Example: ./$(TARGET_QUERY) model.bin machine 10"

help:
	@echo "Targets:"
	@echo "  make all          - Build both train and query binaries"
	@echo "  make clean        - Remove binaries"
	@echo "  make run-train    - Train a model (requires training_data_with_context.txt)"
	@echo "  make run-query    - Instructions for querying"
