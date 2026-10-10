#pragma once

#include <random>

namespace chemical_sim {

class Rng {
public:
    explicit Rng(unsigned seed) : gen_(seed) {}

    void seed(unsigned s) { gen_.seed(s); }

    double uniform01() { return dist01_(gen_); }

    int uniform_int(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(gen_); }

    bool bernoulli(double p) { return uniform01() <= p; }

    template <typename A>
    A choice2(const A& a, const A& b) {
        return uniform01() < 0.5 ? a : b;
    }

private:
    std::mt19937 gen_;
    std::uniform_real_distribution<double> dist01_{0.0, 1.0};
};

}  // пространство имён chemical_sim
