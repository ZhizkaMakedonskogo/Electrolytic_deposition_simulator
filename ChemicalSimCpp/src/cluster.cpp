#include "chemical_sim/cluster.hpp"

#include <algorithm>
#include <sstream>

namespace chemical_sim {

std::atomic<int> Cluster::index_cluster_{0};

Cluster::Cluster(int limiter)
    : status_(Status::OnSurface),
      number_(index_cluster_.fetch_add(1) + 1),
      adjoined_(0),
      min_{0, 0},
      max_{0, 0},
      leave_limit_(0),
      limiter_(limiter),
      is_colored_(false) {}

bool Cluster::can_leaving() const {
    if (border_right().y >= leave_limit_) {
        return false;
    }
    return true;
}

void Cluster::define_rect_border() {
    if (atoms_.empty()) {
        return;
    }
    adjoined_ = 0;
    for (const Cell& i : atoms_) {
        if (i.y == 0) {
            ++adjoined_;
        }
    }
    int min_x = atoms_.front().x;
    int max_x = atoms_.front().x;
    int min_y = atoms_.front().y;
    int max_y = atoms_.front().y;
    for (const Cell& a : atoms_) {
        min_x = std::min(min_x, a.x);
        max_x = std::max(max_x, a.x);
        min_y = std::min(min_y, a.y);
        max_y = std::max(max_y, a.y);
    }
    max_.x = max_x;
    max_.y = max_y;
    min_.x = min_x;
    min_.y = min_y;

    if (border_left().y == 0 && status_ != Status::Leaving) {
        status_ = Status::OnSurface;
    }

    if (border_right().x < 0 || border_left().x >= limiter_) {
        status_ = Status::OffSurface;
    }

    const double half = static_cast<double>(size()) / 2.0;
    if (half < static_cast<double>(limiter_)) {
        leave_limit_ = static_cast<int>(half);
    } else {
        leave_limit_ = limiter_ - 1;
    }
}

void Cluster::add_atom(const Cell& atom) {
    atoms_.push_back(atom);
    max_.x = std::max(atom.x, max_.x);
    max_.y = std::max(atom.y, max_.y);
    min_.x = std::min(atom.x, min_.x);
    min_.y = std::min(atom.y, min_.y);
}

void Cluster::leave() {
    for (Cell& a : atoms_) {
        a.y += 1;
    }
    status_ = Status::Leaving;
    define_rect_border();
}

void Cluster::transition() {
    for (Cell& a : atoms_) {
        if (status_ == Status::DownAlongSurface) {
            a.x += 1;
        } else {
            a.x -= 1;
        }
    }
    define_rect_border();
}

void Cluster::merger(Cluster& other) {
    for (const Cell& atom : other.atoms_) {
        add_atom(atom);
    }
    define_rect_border();
    if (other.status_ == Status::Leaving) {
        status_ = Status::Leaving;
    }
}

char Cluster::symbol_at_relative(int rel_i, int rel_j) const {
    const Cell left = border_left();
    for (int k = 0; k < size(); ++k) {
        if (atoms_[k].x - left.x == rel_i && atoms_[k].y - left.y == rel_j) {
            return '*';
        }
        if (k == size() - 1) {
            return ' ';
        }
    }
    return ' ';
}

std::string Cluster::image() const {
    std::ostringstream out;
    for (int i = 0; i <= border_right().x - border_left().x; ++i) {
        for (int j = 0; j <= border_right().y - border_left().y; ++j) {
            out << symbol_at_relative(i, j);
        }
        out << '\n';
    }
    return out.str();
}

}  
