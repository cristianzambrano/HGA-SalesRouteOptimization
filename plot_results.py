import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_comparative_results2(results, save_to_file=False):
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('ggplot')
    
    variants = full_history['Variant'].unique()
    fig = plt.figure(figsize=(18, 12))
    
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    
    
    palette = {'Standard GA': '#1f77b4','GA-AM': '#ff7f0e', 'GAAM-TS': '#2ca02c' }
    linestyles = {'Standard GA': '--', 'GA-AM': '-.', 'GAAM-TS': '-'}
    
    for variant in variants:
        mask = full_history['Variant'] == variant
        ax1.plot(full_history[mask]['generation'],
                    full_history[mask]['best_fitness'],
                    label=variant,
                    color=palette[variant],
                    linestyle=linestyles[variant],
                    linewidth=2)
    ax1.set_title('Comparative Fitness Evolution', fontsize=12, pad=20)
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Best Fitness', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
        
    for variant in variants:
        mask = full_history['Variant'] == variant
        ax2.plot(full_history[mask]['generation'],
                    full_history[mask]['best_distance'],
                    label=variant,
                    color=palette[variant],
                    linestyle=linestyles[variant],
                    linewidth=2)
    ax2.set_title('Comparative Distance Evolution', fontsize=12, pad=20)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Distance (km)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    for variant in variants:
        mask = full_history['Variant'] == variant
        ax3.plot(full_history[mask]['generation'],
                    full_history[mask]['best_time'],
                    label=variant,
                    color=palette[variant],
                    linestyle=linestyles[variant],
                    linewidth=2)
    ax3.set_title('Comparative Time Evolution', fontsize=12, pad=20)
    ax3.set_xlabel('Generation', fontsize=12)
    ax3.set_ylabel('Time (min)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    for variant in variants:
        mask = full_history['Variant'] == variant
        ax4.plot(full_history[mask]['generation'],
                    full_history[mask]['diversity'],
                    label=variant,
                    color=palette[variant],
                    linestyle=linestyles[variant],
                    linewidth=2)
    ax4.set_title('Comparative Diversity Evolution', fontsize=12, pad=20)
    ax4.set_xlabel('Generation', fontsize=12)
    ax4.set_ylabel('Diversity', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    
    plt.show()



def plot_comparative_results(folder_output, full_history):
       
    os.makedirs(folder_output, exist_ok=True)
    
    variants = full_history['Variant'].unique()
    palette = {'Standard GA': '#1f77b4', 'GA-AM': '#ff7f0e', 'GAAM-TS': '#2ca02c'}
    linestyles = {'Standard GA': '--', 'GA-AM': '-.', 'GAAM-TS': '-'}
    
    metrics = {
        'best_fitness': ('Comparative Fitness Evolution', 'Best Fitness'),
        'best_distance': ('Comparative Distance Evolution', 'Distance (km)'),
        'best_time': ('Comparative Time Evolution', 'Time (min)'),
        'diversity': ('Comparative Diversity Evolution', 'Diversity')
    }
    
    for metric, (title, ylabel) in metrics.items():
        plt.figure(figsize=(6, 4))
        
        for variant in variants:
            mask = full_history['Variant'] == variant
            data = full_history[mask]
            
            plt.plot(data['generation'], 
                    data[metric],
                    label=variant,
                    color=palette[variant],
                    linestyle=linestyles[variant],
                    linewidth=2)
        
        plt.title(title, fontsize=14, pad=15)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(False)
        plt.legend()
        plt.tight_layout()
        
        filename = f"{folder_output}/{metric}_results.jpg"
        plt.savefig(filename, format='jpg', dpi=300, bbox_inches='tight')
        plt.close()

folder = "results25"
    
ga_history = pd.read_csv(f"{folder}//history_Standard_GA.csv")
gaam_history = pd.read_csv(f"{folder}//history_GA-AM.csv")
gaamts_history = pd.read_csv(f"{folder}//history_GAAM-TS.csv")
full_history = pd.concat([ga_history, gaam_history, gaamts_history])
plot_comparative_results(folder,full_history)
print(full_history.head(10))


summary = pd.read_csv(f"{folder}/comparative_report.csv")
print("\nComparative Results:")
print(summary)