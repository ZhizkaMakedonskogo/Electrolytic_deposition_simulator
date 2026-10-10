#pragma once

namespace chemical_sim {

inline constexpr int kResizeStep = 3;

enum class Mode { Const, Var };

enum class Status {
    OnSurface,
    Leaving,
    UpAlongSurface,
    DownAlongSurface,
    Merging,
    OffSurface,
};

}  // пространство имён chemical_sim
