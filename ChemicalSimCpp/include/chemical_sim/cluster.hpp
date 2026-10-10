#pragma once

#include "cell.hpp"
#include "constants.hpp"

#include <atomic>
#include <string>
#include <vector>

namespace chemical_sim {

class Cluster {
public:
    explicit Cluster(int limiter);

    int number() const { return number_; }
    Status status() const { return status_; }
    void set_status(Status s) { status_ = s; }

    int adjoined() const { return adjoined_; }

    const std::vector<Cell>& atoms() const { return atoms_; }

    int size() const { return static_cast<int>(atoms_.size()); }
    int speed() const { return size(); }

    Cell border_right() const { return max_; }
    Cell border_left() const { return min_; }

    bool is_colored() const { return is_colored_; }
    void set_colored(bool v) { is_colored_ = v; }

    bool can_leaving() const;

    void add_atom(const Cell& atom);
    void leave();
    void transition();
    void merger(Cluster& other);

    void define_rect_border();

    std::string image() const;

private:
    char symbol_at_relative(int rel_i, int rel_j) const;

    std::vector<Cell> atoms_;
    Status status_;
    int number_;
    int adjoined_ = 0;
    Cell min_;
    Cell max_;
    int leave_limit_ = 0;
    int limiter_;
    bool is_colored_ = false;

    static std::atomic<int> index_cluster_;
};

}  // пространство имён chemical_sim
