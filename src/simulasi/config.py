from dataclasses import dataclass, field


@dataclass
class GridConfig:
    width: int = 10
    height: int = 10
    cell_size_km: float = 1.0
    initial_palm_ratio: float = 0.6  # 60% sawit awal


@dataclass
class HydroConfig:
    base_infiltration_mm_per_hr: float = 5.0
    rainfall_mm_per_hr: float = 15.0
    rainfall_mm_per_hour: float = 15.0  # alias
    drainage_efficiency: float = 0.2  # 0-1 fraction removed per step
    flood_threshold_mm: float = 50.0


@dataclass
class EconomicConfig:
    palm_profit_per_cell: float = 100.0
    flood_penalty_per_cell: float = 200.0
    biodiversity_bonus_forest: float = 20.0
    palm_revenue_per_ha: float = 25.0
    flood_damage_per_ha: float = 10.0


@dataclass
class SimulationConfig:
    grid: GridConfig = field(default_factory=GridConfig)
    hydro: HydroConfig = field(default_factory=HydroConfig)
    economic: EconomicConfig = field(default_factory=EconomicConfig)
    max_steps: int = 50
