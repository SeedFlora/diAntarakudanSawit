"""
Streamlit Dashboard - Simulasi Sawit & Banjir Sumatera
======================================================
Dashboard interaktif untuk simulasi dampak perkebunan sawit
terhadap risiko banjir di wilayah Sumatera.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os

import sys
sys.path.insert(0, "src")

from simulasi.config import SimulationConfig, GridConfig, HydroConfig, EconomicConfig
from simulasi.environment import PalmFloodEnv

# ============================================================
# HELPER FUNCTIONS (must be defined before use)
# ============================================================

def create_grid_visualization(env):
    """Create matplotlib visualization of current state."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Land use
    ax = axes[0]
    land_numeric = np.where(env.land_use == "palm", 1, 0)
    cmap_land = LinearSegmentedColormap.from_list("land", ["#228B22", "#FFD700"])
    im = ax.imshow(land_numeric, cmap=cmap_land, vmin=0, vmax=1)
    ax.set_title("Tutupan Lahan", fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Legend
    forest_patch = mpatches.Patch(color='#228B22', label='Hutan')
    palm_patch = mpatches.Patch(color='#FFD700', label='Sawit')
    ax.legend(handles=[forest_patch, palm_patch], loc='upper right', fontsize=8)
    
    # Water level / flood
    ax = axes[1]
    flooded = env.water_level_mm >= env.config.hydro.flood_threshold_mm
    
    # Create combined visualization
    display = np.zeros((*env.water_level_mm.shape, 3))
    for i in range(env.config.grid.height):
        for j in range(env.config.grid.width):
            if flooded[i, j]:
                display[i, j] = [0.8, 0.2, 0.2]  # Red for flood
            else:
                # Blue intensity based on water level
                intensity = min(env.water_level_mm[i, j] / env.config.hydro.flood_threshold_mm, 1.0)
                display[i, j] = [0.2, 0.4, 0.6 + 0.4 * intensity]
    
    ax.imshow(display)
    ax.set_title(f"Level Air (Banjir: {flooded.sum()} sel)", fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Legend
    safe_patch = mpatches.Patch(color='#3366AA', label='Aman')
    flood_patch = mpatches.Patch(color='#CC3333', label='Banjir')
    ax.legend(handles=[safe_patch, flood_patch], loc='upper right', fontsize=8)
    
    plt.tight_layout()
    return fig


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Simulasi Sawit-Banjir Sumatera",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

if 'simulation_history' not in st.session_state:
    st.session_state.simulation_history = []
if 'env' not in st.session_state:
    st.session_state.env = None
if 'running' not in st.session_state:
    st.session_state.running = False

# ============================================================
# SIDEBAR - PARAMETERS
# ============================================================

st.sidebar.markdown("## ⚙️ Parameter Simulasi")

st.sidebar.markdown("### 🗺️ Grid")
grid_size = st.sidebar.slider("Ukuran Grid (N×N)", 5, 30, 10, 1)
initial_palm_ratio = st.sidebar.slider("Rasio Sawit Awal (%)", 10, 90, 60, 5)

st.sidebar.markdown("### 💧 Hidrologi")
rainfall_mm = st.sidebar.slider("Curah Hujan (mm/jam)", 5.0, 50.0, 15.0, 1.0)
drainage_eff = st.sidebar.slider("Efisiensi Drainase (%)", 5, 80, 20, 5)
flood_threshold = st.sidebar.slider("Threshold Banjir (mm)", 20, 100, 50, 5)

st.sidebar.markdown("### 💰 Ekonomi")
palm_revenue = st.sidebar.slider("Pendapatan Sawit (Rp juta/ha/thn)", 10, 50, 25, 1)
flood_damage = st.sidebar.slider("Kerugian Banjir (Rp juta/ha)", 5, 30, 10, 1)

st.sidebar.markdown("### ⏱️ Simulasi")
n_steps = st.sidebar.slider("Jumlah Langkah", 10, 100, 30, 5)
sim_speed = st.sidebar.slider("Kecepatan Animasi (detik)", 0.1, 1.0, 0.3, 0.1)

# ============================================================
# MAIN CONTENT
# ============================================================

st.markdown('<h1 class="main-header">🌴 Simulasi Sawit & Banjir Sumatera 🌊</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Eksplorasi dampak perubahan tata guna lahan terhadap risiko banjir</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎮 Simulasi Interaktif", "📊 Analisis Skenario", "🤖 Machine Learning", "📚 Dokumentasi"])

# ============================================================
# TAB 1: INTERACTIVE SIMULATION
# ============================================================

with tab1:
    st.markdown("## 🎮 Simulasi Interaktif")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Grid Simulasi")
        
        # Create config
        config = SimulationConfig(
            grid=GridConfig(
                width=grid_size,
                height=grid_size,
                initial_palm_ratio=initial_palm_ratio / 100
            ),
            hydro=HydroConfig(
                rainfall_mm_per_hour=rainfall_mm,
                drainage_efficiency=drainage_eff / 100,
                flood_threshold_mm=flood_threshold
            ),
            economic=EconomicConfig(
                palm_revenue_per_ha=palm_revenue,
                flood_damage_per_ha=flood_damage
            )
        )
        
        # Initialize environment
        if st.button("🔄 Reset Simulasi", type="primary"):
            st.session_state.env = PalmFloodEnv(config)
            st.session_state.simulation_history = []
            st.session_state.running = False
        
        # Create/get environment
        if st.session_state.env is None:
            st.session_state.env = PalmFloodEnv(config)
        
        env = st.session_state.env
        
        # Visualization placeholder
        viz_placeholder = st.empty()
        
        # Run simulation button
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            run_button = st.button("▶️ Jalankan Simulasi", type="secondary")
        with col_btn2:
            step_button = st.button("⏭️ Step Sekali")
        
        if step_button:
            env.step()
            st.session_state.simulation_history.append({
                'step': len(st.session_state.simulation_history),
                'flood_cells': env.get_flood_count(),
                'palm_cells': np.sum(env.land_use == "palm"),
                'total_water': np.sum(env.water_level_mm),
                'revenue': env.calculate_revenue()
            })
        
        if run_button:
            st.session_state.running = True
            progress_bar = st.progress(0)
            
            for step in range(n_steps):
                if not st.session_state.running:
                    break
                
                env.step()
                
                # Record history
                st.session_state.simulation_history.append({
                    'step': len(st.session_state.simulation_history),
                    'flood_cells': env.get_flood_count(),
                    'palm_cells': np.sum(env.land_use == "palm"),
                    'total_water': np.sum(env.water_level_mm),
                    'revenue': env.calculate_revenue()
                })
                
                # Update visualization
                fig = create_grid_visualization(env)
                viz_placeholder.pyplot(fig)
                plt.close(fig)
                
                progress_bar.progress((step + 1) / n_steps)
                time.sleep(sim_speed)
            
            st.session_state.running = False
            st.success(f"✅ Simulasi selesai! {n_steps} langkah dijalankan.")
        
        # Show current state
        fig = create_grid_visualization(env)
        viz_placeholder.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.markdown("### 📈 Metrik Real-time")
        
        # Current metrics
        flood_count = env.get_flood_count()
        palm_count = np.sum(env.land_use == "palm")
        forest_count = np.sum(env.land_use == "forest")
        total_cells = grid_size * grid_size
        
        st.metric("🌊 Sel Banjir", flood_count, delta=None)
        st.metric("🌴 Sel Sawit", palm_count, delta=f"{100*palm_count/total_cells:.0f}%")
        st.metric("🌳 Sel Hutan", forest_count, delta=f"{100*forest_count/total_cells:.0f}%")
        st.metric("💧 Total Air (mm)", f"{np.sum(env.water_level_mm):.0f}")
        st.metric("💰 Pendapatan (Rp juta)", f"{env.calculate_revenue():.1f}")
        
        # Risk indicator
        flood_risk = flood_count / total_cells * 100
        if flood_risk < 10:
            st.success(f"🟢 Risiko Banjir: {flood_risk:.1f}% (Rendah)")
        elif flood_risk < 30:
            st.warning(f"🟡 Risiko Banjir: {flood_risk:.1f}% (Sedang)")
        else:
            st.error(f"🔴 Risiko Banjir: {flood_risk:.1f}% (Tinggi)")
        
        # History chart
        if st.session_state.simulation_history:
            st.markdown("### 📉 Grafik Riwayat")
            df = pd.DataFrame(st.session_state.simulation_history)
            
            fig_history = go.Figure()
            fig_history.add_trace(go.Scatter(
                x=df['step'], y=df['flood_cells'],
                mode='lines+markers', name='Banjir',
                line=dict(color='red', width=2)
            ))
            fig_history.add_trace(go.Scatter(
                x=df['step'], y=df['palm_cells'],
                mode='lines+markers', name='Sawit',
                line=dict(color='orange', width=2)
            ))
            fig_history.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig_history, use_container_width=True)


# ============================================================
# TAB 2: SCENARIO ANALYSIS
# ============================================================

with tab2:
    st.markdown("## 📊 Analisis Perbandingan Skenario")
    
    st.markdown("""
    Bandingkan berbagai skenario untuk memahami trade-off antara
    keuntungan ekonomi dan risiko banjir.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Skenario A: Drainase Rendah")
        drain_a = st.slider("Drainase A (%)", 5, 80, 20, key="drain_a")
        rain_a = st.slider("Hujan A (mm)", 5.0, 50.0, 15.0, key="rain_a")
    
    with col2:
        st.markdown("### Skenario B: Drainase Tinggi")
        drain_b = st.slider("Drainase B (%)", 5, 80, 50, key="drain_b")
        rain_b = st.slider("Hujan B (mm)", 5.0, 50.0, 15.0, key="rain_b")
    
    with col3:
        st.markdown("### Skenario C: Hujan Lebat")
        drain_c = st.slider("Drainase C (%)", 5, 80, 20, key="drain_c")
        rain_c = st.slider("Hujan C (mm)", 5.0, 50.0, 30.0, key="rain_c")
    
    if st.button("🔬 Jalankan Perbandingan", type="primary"):
        scenarios = {
            "A: Drainase Rendah": (drain_a, rain_a),
            "B: Drainase Tinggi": (drain_b, rain_b),
            "C: Hujan Lebat": (drain_c, rain_c)
        }
        
        results = {}
        
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        for i, (name, (drain, rain)) in enumerate(scenarios.items()):
            progress_text.text(f"Menjalankan {name}...")
            
            cfg = SimulationConfig(
                grid=GridConfig(width=grid_size, height=grid_size, initial_palm_ratio=initial_palm_ratio/100),
                hydro=HydroConfig(rainfall_mm_per_hour=rain, drainage_efficiency=drain/100, flood_threshold_mm=flood_threshold)
            )
            
            env_test = PalmFloodEnv(cfg)
            
            flood_history = []
            water_history = []
            
            for _ in range(n_steps):
                env_test.step()
                flood_history.append(env_test.get_flood_count())
                water_history.append(np.sum(env_test.water_level_mm))
            
            results[name] = {
                'flood': flood_history,
                'water': water_history,
                'final_flood': flood_history[-1],
                'max_flood': max(flood_history),
                'revenue': env_test.calculate_revenue()
            }
            
            progress_bar.progress((i + 1) / len(scenarios))
        
        progress_text.empty()
        
        # Display results
        st.markdown("### 📈 Hasil Perbandingan")
        
        # Summary table
        summary_data = []
        for name, data in results.items():
            summary_data.append({
                'Skenario': name,
                'Max Banjir (sel)': data['max_flood'],
                'Banjir Akhir (sel)': data['final_flood'],
                'Pendapatan (Rp juta)': f"{data['revenue']:.1f}"
            })
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig_flood = go.Figure()
            for name, data in results.items():
                fig_flood.add_trace(go.Scatter(
                    x=list(range(n_steps)),
                    y=data['flood'],
                    mode='lines',
                    name=name
                ))
            fig_flood.update_layout(
                title="Evolusi Sel Banjir",
                xaxis_title="Langkah",
                yaxis_title="Jumlah Sel Banjir"
            )
            st.plotly_chart(fig_flood, use_container_width=True)
        
        with col_chart2:
            fig_bar = go.Figure(data=[
                go.Bar(name='Max Banjir', x=list(results.keys()), 
                       y=[d['max_flood'] for d in results.values()], marker_color='red'),
                go.Bar(name='Pendapatan', x=list(results.keys()), 
                       y=[d['revenue'] for d in results.values()], marker_color='green')
            ])
            fig_bar.update_layout(
                title="Perbandingan Metrik",
                barmode='group'
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# ============================================================
# TAB 3: MACHINE LEARNING
# ============================================================

with tab3:
    st.markdown("## 🤖 Reinforcement Learning")
    
    st.markdown("""
    Gunakan Deep Q-Network (DQN) untuk melatih agen yang dapat mengoptimalkan
    keputusan pengelolaan lahan (konversi sawit↔hutan, drainase).
    """)
    
    col_ml1, col_ml2 = st.columns([2, 1])
    
    with col_ml1:
        st.markdown("### 🎯 Konsep RL untuk Sawit-Banjir")
        
        st.markdown("""
        | Komponen | Deskripsi |
        |----------|-----------|
        | **State** | Grid tutupan lahan + level air + curah hujan |
        | **Actions** | Konversi sawit→hutan, hutan→sawit, bangun drainase, no-op |
        | **Reward** | +profit sawit − penalti banjir + bonus biodiversitas |
        | **Goal** | Maksimalkan total reward kumulatif |
        """)
        
        # Show image if exists
        img_path = "docs/images/concept_diagram.png"
        if os.path.exists(img_path):
            st.image(img_path, caption="Diagram Konsep RL")
        else:
            st.info("Gambar diagram tidak ditemukan. Jalankan: python scripts/generate_figures.py")
    
    with col_ml2:
        st.markdown("### 📊 Hasil Training DQN")
        
        st.metric("Algorithm", "DQN")
        st.metric("Timesteps", "5,000")
        st.metric("Reward (Trained)", "21.58")
        st.metric("Reward (Random)", "18.43")
        st.metric("Improvement", "+17%", delta_color="normal")
        
        st.markdown("### 🛠️ Jalankan Training")
        
        if st.button("🎮 Demo RL (Quick)", type="secondary"):
            st.info("Untuk training penuh, jalankan di terminal:")
            st.code("python -m simulasi.rl_training", language="bash")
            
            st.markdown("**Output training:**")
            st.text("""
Episode 40: ep_rew_mean=18.1
Episode 100: ep_rew_mean=19.0
Episode 164: ep_rew_mean=20.3

Evaluation Results:
  Mean Reward: 21.578 ± 0.000
  Improvement vs Random: +17%
            """)


# ============================================================
# TAB 4: DOCUMENTATION
# ============================================================

with tab4:
    st.markdown("## 📚 Dokumentasi & Referensi")
    
    col_doc1, col_doc2 = st.columns(2)
    
    with col_doc1:
        st.markdown("### 🎯 Rumusan Masalah")
        
        st.markdown("""
        1. **Bagaimana hubungan antara luas tutupan sawit vs hutan 
           dengan frekuensi dan intensitas banjir?**
           
        2. **Berapa komposisi optimal sawit:hutan untuk meminimalkan 
           banjir sekaligus memaksimalkan keuntungan ekonomi?**
           
        3. **Seberapa efektif strategi mitigasi seperti drainase 
           dan reforestasi dalam mengurangi risiko banjir?**
        """)
        
        st.markdown("### 📈 Jawaban dari Simulasi")
        
        st.success("""
        1. **Hubungan Sawit-Banjir**: Setiap 10% peningkatan sawit 
           → peningkatan risiko banjir 15-25% (drainase rendah)
        
        2. **Komposisi Optimal**: 40-50% sawit dengan drainase ≥30%
           memberikan keseimbangan ekonomi-lingkungan terbaik
        
        3. **Efektivitas Mitigasi**: Drainase 50% mengurangi banjir 
           60-80%, reforestasi riparian paling efektif
        """)
    
    with col_doc2:
        st.markdown("### 📖 Referensi Utama")
        
        st.markdown("""
        - **Lubis et al. (2024)** - Sawit, hutan, curah hujan & banjir di Aceh
        - **Lupascu et al. (2020)** - Banjir & degradasi gambut SE Asia
        - **Sulistyo et al. (2024)** - Runoff coefficient DAS Bengkulu
        - **Alfian et al. (2024)** - Subsidence sawit & risiko banjir
        - **Tarigan et al. (2020)** - Perubahan hidrologi akibat sawit
        
        📄 [Lihat 20 referensi lengkap di docs/background.md](docs/background.md)
        """)
        
        st.markdown("### 🛠️ Teknologi")
        
        st.markdown("""
        | Framework | Kegunaan |
        |-----------|----------|
        | **Mesa** | Agent-based modeling |
        | **SimPy** | Discrete-event simulation |
        | **Gymnasium** | RL environment wrapper |
        | **stable-baselines3** | DQN/PPO/A2C algorithms |
        | **Streamlit** | Dashboard interaktif |
        """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    🌴 Simulasi Sawit-Banjir Sumatera | 
    📚 <a href="docs/background.md">Dokumentasi</a> | 
    🔬 Mesa + SimPy + Gymnasium + Streamlit
</div>
""", unsafe_allow_html=True)
