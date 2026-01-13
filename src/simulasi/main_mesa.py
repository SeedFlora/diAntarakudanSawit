from .config import SimulationConfig
from .mesa_model import PalmFloodMesaModel


def run():
    cfg = SimulationConfig()
    model = PalmFloodMesaModel(cfg)
    for _ in range(cfg.max_steps):
        metrics = model.step()
        print(metrics)


if __name__ == "__main__":
    run()
