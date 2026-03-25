#ifndef FASTTEXT_MATRIX_H
#define FASTTEXT_MATRIX_H

#include <vector>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <cstring>
#include <iostream>

namespace fasttext {

class Matrix {
public:
    Matrix() : rows_(0), cols_(0) {}
    Matrix(int64_t rows, int64_t cols) : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {}
    
    void resize(int64_t rows, int64_t cols) {
        rows_ = rows;
        cols_ = cols;
        data_.resize(rows * cols, 0.0f);
    }
    
    void zero() {
        std::fill(data_.begin(), data_.end(), 0.0f);
    }
    
    inline float& at(int64_t row, int64_t col) {
        return data_[row * cols_ + col];
    }
    
    inline const float& at(int64_t row, int64_t col) const {
        return data_[row * cols_ + col];
    }
    
    inline float* row(int64_t r) {
        return data_.data() + r * cols_;
    }
    
    inline const float* row(int64_t r) const {
        return data_.data() + r * cols_;
    }
    
    inline float* data() { return data_.data(); }
    inline const float* data() const { return data_.data(); }
    inline int64_t rows() const { return rows_; }
    inline int64_t cols() const { return cols_; }
    inline int64_t size() const { return data_.size(); }
    
    template<typename RNG>
    void randomInit(RNG& rng, float scale) {
        std::normal_distribution<float> dist(0.0, 1.0);
        for (auto& val : data_) {
            val = dist(rng) * scale;
        }
    }
    
    void save(std::ostream& out) const {
        out.write(reinterpret_cast<const char*>(data_.data()), data_.size() * sizeof(float));
    }
    
    void load(std::istream& in) {
        in.read(reinterpret_cast<char*>(data_.data()), data_.size() * sizeof(float));
    }

private:
    int64_t rows_;
    int64_t cols_;
    std::vector<float> data_;
};

} // namespace fasttext

#endif
