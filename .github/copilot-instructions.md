# Simulasi Sawit & Banjir Sumatera

## Project Overview
Prototipe simulasi berbasis Python untuk mengeksplorasi dampak perubahan tata guna lahan (sawit vs hutan) terhadap risiko banjir di wilayah Sumatera.

## Tech Stack
- Python 3.11
- Mesa (Agent-based modeling)
- SimPy (Discrete-event simulation)
- Gymnasium + stable-baselines3 (Reinforcement Learning)
- Streamlit (Interactive dashboard)
- matplotlib, plotly, numpy, pandas

## Quick Commands

### Run Streamlit Dashboard
```bash
# Activate virtual environment
.venv\Scripts\activate

# Set PYTHONPATH
$env:PYTHONPATH="src"

# Launch dashboard
streamlit run app.py
```

### Run Simulations
```bash
python -m simulasi.main_mesa       # Agent-based (Mesa)
python -m simulasi.main_simpy      # Discrete-event (SimPy)
python -m simulasi.visualization   # Matplotlib visualization
python -m simulasi.rl_training     # Reinforcement Learning (DQN)
```

### Generate Documentation Figures
```bash
python scripts/generate_figures.py
```

## Project Structure
```
├── app.py                  # Streamlit dashboard
├── src/simulasi/           # Core simulation code
│   ├── config.py           # Configuration dataclasses
│   ├── environment.py      # Base hydrology environment
│   ├── mesa_model.py       # Mesa agent-based model
│   ├── simpy_model.py      # SimPy discrete-event model
│   ├── gym_env.py          # Gymnasium RL wrapper
│   ├── rl_training.py      # DQN/PPO/A2C training
│   └── visualization.py    # Matplotlib UI
├── docs/
│   ├── background.md       # Research background (20 references)
│   └── images/             # Generated figures
└── scripts/
    └── generate_figures.py # Figure generation script
```

## Features
1. **Agent-Based Modeling** - Land cells as adaptive agents
2. **Discrete-Event Simulation** - Stochastic rain events
3. **Reinforcement Learning** - DQN for policy optimization
4. **Interactive Dashboard** - Real-time parameter tuning
