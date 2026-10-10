#pragma once

#include "cell.hpp"
#include "cluster.hpp"
#include "constants.hpp"
#include "rng.hpp"

#include <map>
#include <vector>

namespace chemical_sim {

class Board {
public:
    Board(int rows, double addprob, double transitprob, double mergeprob, Mode mode, Rng& rng);

    int rows() const { return rows_; }

    void run();

    std::map<int, int> create_bar() const;

private:
    void resize_place();
    Cell add_atom(int row);
    Cell atom_transition(Cell current);
    void check_cluster(const Cell& current, std::vector<Cell>& candidates, std::vector<std::vector<int>>& temp_place);
    bool is_interior_with_all_neighbors_equal(int x, int y, int cur_val) const;
    int neighbor_cluster_id_for_free_atom(const Cell& atom) const;
    void create_cluster(const Cell& current);
    void cluster_uncoloring(Cluster& cluster);
    void cluster_coloring(Cluster& cluster);
    void cluster_merger(std::vector<int> candidates);
    bool has_adjacent_occupied_cell_excluding_cluster(int x, int y, int cluster_number) const;
    void check_cluster_for_clusters(int found);
    void queue_transit();

    int rows_;
    int columns_;
    std::vector<std::vector<int>> place_;
    std::map<int, Cluster> clusters_;
    Mode mode_;
    double create_atom_prob_;
    double transit_atom_prob_;
    double merge_cluster_prob_;
    int atoms_count_ = 0;
    Rng& rng_;
};

}  // пространство имён chemical_sim
