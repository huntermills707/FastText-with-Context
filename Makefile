CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=native -fopenmp
LDFLAGS = -fopenmp

TARGET_TRAIN = train
TARGET_QUERY = query
TARGET_COMPARE = compare

# Source files
SRC_CORE = fasttext_context.cpp vocabulary.cpp trainer.cpp inference.cpp
SRC_TRAIN = train.cpp
SRC_QUERY = query.cpp
SRC_COMPARE = compare.cpp

HEADER = fasttext_context.h

.PHONY: all clean

all: $(TARGET_TRAIN) $(TARGET_QUERY) $(TARGET_COMPARE)

$(TARGET_TRAIN): $(SRC_CORE) $(SRC_TRAIN) $(HEADER)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET_TRAIN) $(SRC_CORE) $(SRC_TRAIN)

$(TARGET_QUERY): $(SRC_CORE) $(SRC_QUERY) $(HEADER)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET_QUERY) $(SRC_CORE) $(SRC_QUERY)

$(TARGET_COMPARE): $(SRC_CORE) $(SRC_COMPARE) $(HEADER)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET_COMPARE) $(SRC_CORE) $(SRC_COMPARE)

clean:
	rm -f $(TARGET_TRAIN) $(TARGET_QUERY) $(TARGET_COMPARE)
