"""
Generate dan simpan gambar hasil simulasi untuk dokumentasi
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

from simulasi.config import SimulationConfig, GridConfig, HydroConfig
from simulasi.visualization import SimulationVisualizer, run_scenario_comparison


def generate_all_figures(output_dir: str = "docs/images"):
    """Generate semua gambar untuk dokumentasi."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("GENERATING DOCUMENTATION FIGURES")
    print("=" * 60)
    
    # 1. Run scenario comparison
    print("\n[1/4] Running scenario comparison...")
    results, fig_compare = run_scenario_comparison()
    fig_compare.savefig(f"{output_dir}/scenario_comparison.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/scenario_comparison.png")
    plt.close(fig_compare)
    
    # 2. Generate individual scenario results
    print("\n[2/4] Generating individual scenario figures...")
    
    for name, data in results.items():
        fig = data['viz'].plot_final_results()
        fig.savefig(f"{output_dir}/scenario_{name}.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_dir}/scenario_{name}.png")
        plt.close(fig)
    
    # 3. Generate land use evolution figure
    print("\n[3/4] Generating land use evolution...")
    fig_evolution = generate_evolution_figure(results['low_drain']['viz'])
    fig_evolution.savefig(f"{output_dir}/land_use_evolution.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/land_use_evolution.png")
    plt.close(fig_evolution)
    
    # 4. Generate concept diagram
    print("\n[4/4] Generating concept diagram...")
    fig_concept = generate_concept_diagram()
    fig_concept.savefig(f"{output_dir}/concept_diagram.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/concept_diagram.png")
    plt.close(fig_concept)
    
    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    return output_dir


def generate_evolution_figure(viz):
    """Generate figure showing land use evolution over time."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Evolusi Tutupan Lahan & Genangan", fontsize=14, fontweight='bold')
    
    # Select timesteps to show
    n_steps = len(viz.land_use_history)
    indices = [0, n_steps//4, n_steps//2, n_steps-1]
    
    for i, idx in enumerate(indices):
        # Land use
        ax = axes[0, i]
        land_map = np.where(viz.land_use_history[idx] == "palm", 1, 0)
        ax.imshow(land_map, cmap='RdYlGn_r', vmin=0, vmax=1)
        ax.set_title(f"Tutupan Lahan (t={idx})")
        ax.axis('off')
        
        # Water level
        ax = axes[1, i]
        water = viz.water_history[idx]
        flooded = water >= viz.config.hydro.flood_threshold_mm
        ax.imshow(flooded, cmap='RdBu_r', vmin=0, vmax=1)
        ax.set_title(f"Banjir: {flooded.sum()} sel")
        ax.axis('off')
    
    # Add legend
    forest_patch = mpatches.Patch(color='green', label='Hutan')
    palm_patch = mpatches.Patch(color='#FFFF00', label='Sawit')
    safe_patch = mpatches.Patch(color='blue', label='Aman')
    flood_patch = mpatches.Patch(color='red', label='Banjir')
    
    fig.legend(handles=[forest_patch, palm_patch, safe_patch, flood_patch],
               loc='lower center', ncol=4, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    return fig


def generate_concept_diagram():
    """Generate conceptual diagram of the simulation."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title("Konsep Simulasi Sawit-Banjir", fontsize=16, fontweight='bold', pad=20)
    
    # State box
    state_box = mpatches.FancyBboxPatch((0.5, 6), 4, 3, boxstyle="round,pad=0.1",
                                         facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(state_box)
    ax.text(2.5, 8.5, "STATE", fontsize=12, fontweight='bold', ha='center')
    ax.text(2.5, 7.8, "• Grid tutupan lahan", fontsize=9, ha='center')
    ax.text(2.5, 7.3, "• Kedalaman air (mm)", fontsize=9, ha='center')
    ax.text(2.5, 6.8, "• Curah hujan", fontsize=9, ha='center')
    ax.text(2.5, 6.3, "• Step count", fontsize=9, ha='center')
    
    # Action box
    action_box = mpatches.FancyBboxPatch((5, 6), 4, 3, boxstyle="round,pad=0.1",
                                          facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax.add_patch(action_box)
    ax.text(7, 8.5, "ACTIONS", fontsize=12, fontweight='bold', ha='center')
    ax.text(7, 7.8, "• Konversi sawit ↔ hutan", fontsize=9, ha='center')
    ax.text(7, 7.3, "• Bangun drainase", fontsize=9, ha='center')
    ax.text(7, 6.8, "• Reforestasi", fontsize=9, ha='center')
    ax.text(7, 6.3, "• No-op", fontsize=9, ha='center')
    
    # Reward box
    reward_box = mpatches.FancyBboxPatch((9.5, 6), 4, 3, boxstyle="round,pad=0.1",
                                          facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax.add_patch(reward_box)
    ax.text(11.5, 8.5, "REWARD", fontsize=12, fontweight='bold', ha='center')
    ax.text(11.5, 7.8, "+ Profit sawit", fontsize=9, ha='center')
    ax.text(11.5, 7.3, "− Penalti banjir", fontsize=9, ha='center')
    ax.text(11.5, 6.8, "+ Biodiversitas hutan", fontsize=9, ha='center')
    ax.text(11.5, 6.3, "= Trade-off optimal", fontsize=9, ha='center')
    
    # Environment box
    env_box = mpatches.FancyBboxPatch((3, 1), 8, 3.5, boxstyle="round,pad=0.1",
                                       facecolor='lavender', edgecolor='black', linewidth=2)
    ax.add_patch(env_box)
    ax.text(7, 4, "ENVIRONMENT DYNAMICS", fontsize=12, fontweight='bold', ha='center')
    ax.text(7, 3.3, "1. Rainfall → +water_mm per cell", fontsize=9, ha='center')
    ax.text(7, 2.7, "2. Infiltration → −water_mm (forest > palm)", fontsize=9, ha='center')
    ax.text(7, 2.1, "3. Drainage → −water_mm * efficiency", fontsize=9, ha='center')
    ax.text(7, 1.5, "4. Flood = water_mm ≥ threshold", fontsize=9, ha='center')
    
    # Arrows
    ax.annotate('', xy=(5, 7.5), xytext=(4.5, 7.5),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(9.5, 7.5), xytext=(9, 7.5),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(7, 6), xytext=(7, 4.5),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(2.5, 6), xytext=(4, 4.5),
                arrowprops=dict(arrowstyle='->', lw=2, connectionstyle="arc3,rad=-0.3"))
    
    # ML/RL box
    ml_box = mpatches.FancyBboxPatch((0.5, 1), 2, 2, boxstyle="round,pad=0.1",
                                      facecolor='mistyrose', edgecolor='black', linewidth=2)
    ax.add_patch(ml_box)
    ax.text(1.5, 2.5, "ML/RL", fontsize=11, fontweight='bold', ha='center')
    ax.text(1.5, 2, "DQN", fontsize=9, ha='center')
    ax.text(1.5, 1.5, "PPO/A2C", fontsize=9, ha='center')
    
    # Framework box
    fw_box = mpatches.FancyBboxPatch((11.5, 1), 2, 2, boxstyle="round,pad=0.1",
                                      facecolor='wheat', edgecolor='black', linewidth=2)
    ax.add_patch(fw_box)
    ax.text(12.5, 2.5, "TOOLS", fontsize=11, fontweight='bold', ha='center')
    ax.text(12.5, 2, "Mesa/SimPy", fontsize=9, ha='center')
    ax.text(12.5, 1.5, "Gymnasium", fontsize=9, ha='center')
    
    return fig


if __name__ == "__main__":
    generate_all_figures()
