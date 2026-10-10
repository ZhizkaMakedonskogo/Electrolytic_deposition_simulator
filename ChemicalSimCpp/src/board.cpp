#include "chemical_sim/board.hpp"

#include <algorithm>

namespace chemical_sim {

Board::Board(
    const int rows,
    const double addprob,
    const double transitprob,
    const double mergeprob,
    const Mode mode,
    Rng& rng
)
    : rows_(rows),
      columns_(kResizeStep),
      place_(static_cast<size_t>(rows_), std::vector<int>(static_cast<size_t>(columns_), -1)),
      mode_(mode),
      create_atom_prob_(addprob),
      transit_atom_prob_(transitprob),
      merge_cluster_prob_(mergeprob),
      rng_(rng) {}

void Board::resize_place() {
    for (int r = 0; r < rows_; ++r) {
        place_[static_cast<size_t>(r)].resize(static_cast<size_t>(columns_ + kResizeStep), -1);
    }
    columns_ += kResizeStep;
}

Cell Board::add_atom(const int row) {
    int j = -1;
    for (int c = 0; c < columns_; ++c) {
        if (place_[static_cast<size_t>(row)][static_cast<size_t>(c)] == -1) {
            j = c;
            break;
        }
    }
    if (j < 0) {
        j = columns_;
    }
    if (j >= columns_) {
        resize_place();
    }
    place_[static_cast<size_t>(row)][static_cast<size_t>(j)] = 0;
    return Cell{row, j};
}

Cell Board::atom_transition(Cell current) {
    const bool is_up = rng_.choice2(true, false);
    place_[static_cast<size_t>(current.x)][static_cast<size_t>(current.y)] = -1;

    if (is_up && current.x - 1 >= 0) {
        current = add_atom(current.x - 1);
    } else if (!is_up && current.x + 1 < rows_) {
        current = add_atom(current.x + 1);
    }

    return current;
}

bool Board::is_interior_with_all_neighbors_equal(const int x, const int y, const int cur_val) const {
    if (!(x - 1 >= 0 && x + 1 < rows_ && y - 1 >= 0 && y + 1 < columns_)) {
        return false;
    }
    const auto& p = place_;
    return p[static_cast<size_t>(x - 1)][static_cast<size_t>(y)] == cur_val &&
           p[static_cast<size_t>(x + 1)][static_cast<size_t>(y)] == cur_val &&
           p[static_cast<size_t>(x)][static_cast<size_t>(y - 1)] == cur_val &&
           p[static_cast<size_t>(x)][static_cast<size_t>(y + 1)] == cur_val &&
           p[static_cast<size_t>(x + 1)][static_cast<size_t>(y + 1)] == cur_val &&
           p[static_cast<size_t>(x - 1)][static_cast<size_t>(y + 1)] == cur_val &&
           p[static_cast<size_t>(x + 1)][static_cast<size_t>(y - 1)] == cur_val &&
           p[static_cast<size_t>(x - 1)][static_cast<size_t>(y - 1)] == cur_val;
}

void Board::check_cluster(
    const Cell& current,
    std::vector<Cell>& candidates,
    std::vector<std::vector<int>>& temp_place
) {
    const int x = current.x;
    const int y = current.y;

    if (place_[static_cast<size_t>(x)][static_cast<size_t>(y)] == -1) {
        temp_place[static_cast<size_t>(x)][static_cast<size_t>(y)] = 1;
        return;
    }
    if (temp_place[static_cast<size_t>(x)][static_cast<size_t>(y)] == 1) {
        return;
    }

    temp_place[static_cast<size_t>(x)][static_cast<size_t>(y)] = 1;

    const int cur_val = place_[static_cast<size_t>(x)][static_cast<size_t>(y)];

    if (is_interior_with_all_neighbors_equal(x, y, cur_val)) {
        return;
    }

    candidates.push_back(Cell{x, y});

    if (x - 1 >= 0) {
        check_cluster(Cell{x - 1, y}, candidates, temp_place);
    }
    if (x + 1 < rows_) {
        check_cluster(Cell{x + 1, y}, candidates, temp_place);
    }
    if (y - 1 >= 0) {
        check_cluster(Cell{x, y - 1}, candidates, temp_place);
    }
    if (y + 1 < columns_) {
        check_cluster(Cell{x, y + 1}, candidates, temp_place);
    }
    if (x - 1 >= 0 && y - 1 >= 0) {
        check_cluster(Cell{x - 1, y - 1}, candidates, temp_place);
    }
    if (x + 1 < rows_ && y - 1 >= 0) {
        check_cluster(Cell{x + 1, y - 1}, candidates, temp_place);
    }
    if (x - 1 >= 0 && y + 1 < columns_) {
        check_cluster(Cell{x - 1, y + 1}, candidates, temp_place);
    }
    if (x + 1 < rows_ && y + 1 < columns_) {
        check_cluster(Cell{x + 1, y + 1}, candidates, temp_place);
    }
}

int Board::neighbor_cluster_id_for_free_atom(const Cell& atom) const {
    int new_found = -1;
    const int x = atom.x;
    const int y = atom.y;
    const auto& p = place_;
    const int r = rows_;
    const int c = columns_;

    auto consider = [&](const int nx, const int ny) {
        if (p[static_cast<size_t>(nx)][static_cast<size_t>(ny)] > 0) {
            new_found = p[static_cast<size_t>(nx)][static_cast<size_t>(ny)];
        }
    };

    if (x - 1 >= 0) {
        consider(x - 1, y);
    }
    if (x + 1 < r) {
        consider(x + 1, y);
    }
    if (y - 1 >= 0) {
        consider(x, y - 1);
    }
    if (y + 1 < c) {
        consider(x, y + 1);
    }
    if (x - 1 >= 0 && y - 1 >= 0) {
        consider(x - 1, y - 1);
    }
    if (x + 1 < r && y - 1 >= 0) {
        consider(x + 1, y - 1);
    }
    if (x - 1 >= 0 && y + 1 < c) {
        consider(x - 1, y + 1);
    }
    if (x + 1 < r && y + 1 < c) {
        consider(x + 1, y + 1);
    }
    return new_found;
}

void Board::cluster_uncoloring(Cluster& cluster) {
    for (const Cell& item : cluster.atoms()) {
        if (item.x >= 0 && item.x < rows_ && item.y >= 0 && item.y < columns_) {
            place_[static_cast<size_t>(item.x)][static_cast<size_t>(item.y)] = -1;
        }
    }
    cluster.set_colored(false);
}

void Board::cluster_coloring(Cluster& cluster) {
    for (const Cell& item : cluster.atoms()) {
        if (item.y >= columns_) {
            resize_place();
        }
        if (item.x >= 0 && item.x < rows_ && item.y >= 0 && item.y < columns_) {
            place_[static_cast<size_t>(item.x)][static_cast<size_t>(item.y)] = cluster.number();
        }
    }
    cluster.set_colored(true);
}

bool Board::has_adjacent_occupied_cell_excluding_cluster(
    const int x,
    const int y,
    const int cluster_number
) const {
    const auto& p = place_;
    const int r = rows_;
    const int c = columns_;

    auto foreign = [&](const int nx, const int ny) -> bool {
        const int v = p[static_cast<size_t>(nx)][static_cast<size_t>(ny)];
        return v != cluster_number && v != -1;
    };

    return (x - 1 >= 0 && foreign(x - 1, y)) || (x + 1 < r && foreign(x + 1, y)) ||
           (y - 1 >= 0 && foreign(x, y - 1)) || (y + 1 < c && foreign(x, y + 1)) ||
           (x - 1 >= 0 && y - 1 >= 0 && foreign(x - 1, y - 1)) ||
           (x - 1 >= 0 && y + 1 < c && foreign(x - 1, y + 1)) ||
           (x + 1 < r && y - 1 >= 0 && foreign(x + 1, y - 1)) ||
           (x + 1 < r && y + 1 < c && foreign(x + 1, y + 1));
}

void Board::cluster_merger(std::vector<int> candidates) {
    if (candidates.size() > 2) {
        const int target = candidates.back();
        for (size_t i = 0; i + 1 < candidates.size(); ++i) {
            clusters_.at(target).merger(clusters_.at(candidates[i]));
            clusters_.at(candidates[i]).set_status(Status::Merging);
            cluster_uncoloring(clusters_.at(candidates[i]));
        }
        cluster_coloring(clusters_.at(target));
        return;
    }
    if (candidates.size() < 2) {
        return;
    }

    const double randmerge = rng_.uniform01();
    Cluster& c0 = clusters_.at(candidates[0]);
    Cluster& c1 = clusters_.at(candidates[1]);
    if (randmerge <= merge_cluster_prob_ || c0.status() == Status::OnSurface || c1.status() == Status::OnSurface) {
        c1.merger(c0);
        cluster_uncoloring(c0);
        cluster_coloring(c1);
        c0.set_status(Status::Merging);
    } else {
        int repulsion_index = candidates[1];
        int stay_index = candidates[0];
        if (c0.size() >= c1.size()) {
            repulsion_index = candidates[1];
            stay_index = candidates[0];
        } else {
            repulsion_index = candidates[0];
            stay_index = candidates[1];
        }

        Cluster& repulsion = clusters_.at(repulsion_index);
        Cluster& stay = clusters_.at(stay_index);

        if (repulsion.border_left().x >= stay.border_right().x) {
            
            if (repulsion.status() == Status::UpAlongSurface) {
                repulsion.set_status(Status::DownAlongSurface);
            }
        } else {
            if (repulsion.status() != Status::UpAlongSurface) {
                repulsion.set_status(Status::UpAlongSurface);
            }
        }

        cluster_uncoloring(repulsion);
        repulsion.transition();
        cluster_coloring(repulsion);
    }
}

void Board::create_cluster(const Cell& current) {
    std::vector<std::vector<int>> temp_place(static_cast<size_t>(rows_), std::vector<int>(static_cast<size_t>(columns_), 0));
    std::vector<Cell> candidates;
    check_cluster(current, candidates, temp_place);

    int min_clusternumber = 0;
    for (const Cell& atom : candidates) {
        const int v = place_[static_cast<size_t>(atom.x)][static_cast<size_t>(atom.y)];
        if (v != 0) {
            if (min_clusternumber == 0) {
                min_clusternumber = v;
            } else {
                min_clusternumber = std::min(min_clusternumber, v);
            }
        }
    }

    if (min_clusternumber == 0) {
        if (static_cast<int>(candidates.size()) >= 4) {
            Cluster new_cluster(rows_);
            for (const Cell& atom : candidates) {
                new_cluster.add_atom(atom);
            }
            const int new_id = new_cluster.number();
            clusters_.insert({new_id, std::move(new_cluster)});
            cluster_coloring(clusters_.at(new_id));
        }
    } else {
        std::vector<int> merging_ids;
        for (const Cell& atom : candidates) {
            const int v = place_[static_cast<size_t>(atom.x)][static_cast<size_t>(atom.y)];
            if (v == min_clusternumber) {
                continue;
            }
            if (v != 0) {
                if (std::find(merging_ids.begin(), merging_ids.end(), v) == merging_ids.end()) {
                    merging_ids.push_back(v);
                }
            } else {
                const int new_found = neighbor_cluster_id_for_free_atom(atom);
                if (new_found != -1) {
                    clusters_.at(new_found).add_atom(atom);
                    cluster_coloring(clusters_.at(new_found));
                }
            }
        }
        if (!merging_ids.empty()) {
            std::vector<int> merger_list = merging_ids;
            merger_list.push_back(min_clusternumber);
            cluster_merger(merger_list);
        }
    }
}

void Board::check_cluster_for_clusters(const int found) {
    Cluster& cl = clusters_.at(found);
    const int cluster_number = cl.number();
    for (int i = 0; i < cl.size(); ++i) {
        const int x = cl.atoms()[static_cast<size_t>(i)].x;
        const int y = cl.atoms()[static_cast<size_t>(i)].y;
        if (x >= 0 && x < rows_ && y >= 0 && y < columns_) {
            if (has_adjacent_occupied_cell_excluding_cluster(x, y, cluster_number)) {
                create_cluster(cl.atoms()[static_cast<size_t>(i)]);
                return;
            }
        }
    }
}

void Board::queue_transit() {
    std::vector<int> ids;
    ids.reserve(clusters_.size());
    for (const auto& kv : clusters_) {
        ids.push_back(kv.first);
    }

    for (const int cid : ids) {
        auto it = clusters_.find(cid);
        if (it == clusters_.end()) {
            continue;
        }
        Cluster& cluster = it->second;

        if (cluster.status() == Status::OffSurface && cluster.is_colored()) {
            cluster.set_colored(!cluster.is_colored());
            cluster_uncoloring(cluster);
        }
        if (cluster.status() == Status::Merging || cluster.status() == Status::OnSurface ||
            cluster.status() == Status::OffSurface) {
            continue;
        }

        if (cluster.status() == Status::DownAlongSurface || cluster.status() == Status::UpAlongSurface) {
            int speed = cluster.speed();
            if (speed >= rows_) {
                speed -= 1;
            }
            bool exit_loop = false;
            int j = speed;
            while (j > 0 && !exit_loop) {
                check_cluster_for_clusters(cluster.number());
                if (cluster.status() == Status::Merging || cluster.status() == Status::OnSurface ||
                    cluster.status() == Status::OffSurface) {
                    exit_loop = true;
                    continue;
                }
                cluster_uncoloring(clusters_.at(cluster.number()));
                cluster.transition();
                cluster_coloring(clusters_.at(cluster.number()));
                --j;
            }
        }

        if (cluster.status() == Status::Leaving) {
            if (cluster.can_leaving()) {
                check_cluster_for_clusters(cluster.number());
                if (cluster.status() == Status::Merging || cluster.status() == Status::OnSurface ||
                    cluster.status() == Status::OffSurface) {
                    continue;
                }
                cluster_uncoloring(clusters_.at(cluster.number()));
                cluster.leave();
                cluster_coloring(clusters_.at(cluster.number()));
            } else if (cluster.border_left().y != 0) {
                cluster.set_status(rng_.choice2(Status::UpAlongSurface, Status::DownAlongSurface));
            }
        }
    }
}

void Board::run() {
    double create_prob = create_atom_prob_;
    if (mode_ == Mode::Var) {
        create_prob = rng_.uniform01();
    }

    if (rng_.bernoulli(create_prob)) {
        ++atoms_count_;
        const int position = rng_.uniform_int(0, rows_ - 1);
        Cell current = add_atom(position);
        create_cluster(current);
        if (place_[static_cast<size_t>(current.x)][static_cast<size_t>(current.y)] == 0) {
            if (rng_.bernoulli(transit_atom_prob_)) {
                current = atom_transition(current);
                create_cluster(current);
            }
        }
        if (place_[static_cast<size_t>(current.x)][static_cast<size_t>(current.y)] > 0) {
            const double rand_leaving = rng_.uniform01();
            const int cluster_number = place_[static_cast<size_t>(current.x)][static_cast<size_t>(current.y)];
            Cluster& cl = clusters_.at(cluster_number);
            const double leaving_prob =
                1.0 - static_cast<double>(cl.adjoined()) / static_cast<double>(std::max(1, cl.size()));
            if (rand_leaving < leaving_prob) {
                cl.set_status(Status::Leaving);
            }
        }
    }
    queue_transit();
}

std::map<int, int> Board::create_bar() const {
    std::map<int, int> bar;
    for (const auto& kv : clusters_) {
        const Cluster& cluster = kv.second;
        const int sz = cluster.size();
        bar[sz] = bar.count(sz) ? bar.at(sz) + 1 : 1;
    }
    return bar;
}

}  
