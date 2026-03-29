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
    inline int64_t size() const { return static_cast<int64_t>(data_.size()); }

    template<typename RNG>
    void randomInit(RNG& rng, float scale) {
        std::normal_distribution<float> dist(0.0, 1.0);
        for (auto& val : data_) {
            val = dist(rng) * scale;
        }
    }

    // y = A * x  (A is this matrix, rows x cols; x is cols-length; y is rows-length)
    void mulVec(const float* x, float* y) const {
        for (int64_t i = 0; i < rows_; ++i) {
            float dot = 0.0f;
            const float* r = data_.data() + i * cols_;
            for (int64_t j = 0; j < cols_; ++j)
                dot += r[j] * x[j];
            y[i] = dot;
        }
    }

    // y = A^T * x  (A is this matrix, rows x cols; x is rows-length; y is cols-length)
    void mulVecTranspose(const float* x, float* y) const {
        std::fill(y, y + cols_, 0.0f);
        for (int64_t i = 0; i < rows_; ++i) {
            const float* r = data_.data() + i * cols_;
            float xi = x[i];
            for (int64_t j = 0; j < cols_; ++j)
                y[j] += r[j] * xi;
        }
    }

    // A += alpha * (x * y^T)  (rank-1 update; x is rows-length, y is cols-length)
    void addOuterProduct(const float* x, const float* y, float alpha) {
        for (int64_t i = 0; i < rows_; ++i) {
            float* r = data_.data() + i * cols_;
            float ax = alpha * x[i];
            for (int64_t j = 0; j < cols_; ++j)
                r[j] += ax * y[j];
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
