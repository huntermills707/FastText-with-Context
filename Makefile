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
