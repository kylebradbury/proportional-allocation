import matplotlib.pyplot as plt
import numpy as np

def plot_one_d_samples(samples, title):
    """
    Plot one-dimensional samples.
    """
    fig, ax = plt.subplots(2,1,figsize=(10, 2), gridspec_kw={'height_ratios': [1, 1]})
    
    # Add a histogram to visualize the distribution of samples
    ax[0].hist(samples, bins=20, density=True, alpha=0.7, color='darkgrey')
    ax[0].set(
        title=title,
        xlabel='Sample Value',
        ylabel='Density',
        # xlim=(0, 1),
        xticks=[],
    )
    
    # Plot the samples as vertical lines
    ax[1].plot(samples, np.zeros_like(samples), 
               marker='|',
               color='black',
               markersize=5,
               linestyle='None',
               alpha=0.4)
    ax[1].set(
        yticks=[],
        # xlim=(0, 1),
        xlabel='Sample Value'
    )
    
    plt.subplots_adjust(hspace=0)
    
def plot_two_d_samples(samples, title):
    """
    Plot two-dimensional samples.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(samples[:, 0], samples[:, 1], marker='.', facecolor='black', edgecolor='black')
    ax.set(
        title=title,
        xlabel='X-axis',
        ylabel='Y-axis',
    )
    plt.show()

    
def plot_simulation_results(dg, dp, results):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].plot(dg, results['mean_estimate_centroid'], label='Centroid Allocation', color='blue')
    ax[0].plot(dg, results['mean_estimate_proportional'], label='Proportional Allocation', color='orange')
    ax[0].set_title('Mean Allocation Error')
    ax[0].set_xlabel(f'Grid Size (Polygon Size Fixed at {dp[0]})')
    ax[0].set_ylabel('Mean Error')
    ax[0].legend()
    ax[1].plot(dg, results['var_estimate_centroid'], label='Centroid Allocation', color='blue')
    ax[1].plot(dg, results['var_estimate_proportional'], label='Proportional Allocation', color='orange')
    ax[1].set_title('Variance of Allocation Error')
    ax[1].set_xlabel(f'Grid Size (Polygon Size Fixed at {dp[0]})')
    ax[1].set_ylabel('Variance Error')
    ax[1].legend()
    ax[2].plot(dg, results['mean_mape_centroid'], label='Centroid Allocation', color='blue')
    ax[2].plot(dg, results['mean_mape_proportional'], label='Proportional Allocation', color='orange')
    ax[2].set_title('Allocation MAPE')
    ax[2].set_xlabel(f'Grid Size (Polygon Size Fixed at {dp[0]})')
    ax[2].set_ylabel('MAPE')
    ax[2].legend()
    plt.tight_layout()
    plt.show()

def plot_simulation_results_ratio(dg, dp, results):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].semilogx(dp/dg, results['mean_estimate_centroid'], label='Centroid Allocation', color='blue')
    ax[0].semilogx(dp/dg, results['mean_estimate_proportional'], label='Proportional Allocation', color='orange')
    ax[0].set_title('Mean Allocation Error')
    ax[0].set_xlabel('Polygon:Grid Size Ratio')
    ax[0].set_ylabel('Mean Error')
    ax[0].legend()
    ax[1].semilogx(dp/dg, results['var_estimate_centroid'], label='Centroid Allocation', color='blue')
    ax[1].semilogx(dp/dg, results['var_estimate_proportional'], label='Proportional Allocation', color='orange')
    ax[1].set_title('Variance of Allocation Error')
    ax[1].set_xlabel('Polygon:Grid Size Ratio')
    ax[1].set_ylabel('Variance Error')
    ax[1].legend()
    ax[2].semilogx(dp/dg, results['mean_mape_centroid'], label='Centroid Allocation', color='blue')
    ax[2].semilogx(dp/dg, results['mean_mape_proportional'], label='Proportional Allocation', color='orange')
    ax[2].set_title('Allocation MAPE')
    ax[2].set_xlabel('Polygon:Grid Size Ratio')
    ax[2].set_ylabel('MAPE')
    ax[2].legend()
    plt.tight_layout()
    plt.show()

def plot_simulation_error_ratios(dg, dp, results):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.loglog(dp/dg, np.array(results['mean_mape_centroid']) / np.array(results['mean_mape_proportional']), label='MAPE Ratio - Centroid:Proportional', color='blue')
    ax.loglog(dp/dg, np.ones(len(dg)), label='Equal Performance', color='darkgray', linestyle='--')
    ax.set_title('Allocation MAPE Ratio - Centroid:Proportional Allocation')
    ax.set_xlabel('Polygon:Grid Size Ratio')
    ax.set_ylabel('MAPE')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_simulation_error_ratios_multiple(dg, dp, results, names):
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, result in enumerate(results):
        ax.loglog(dp/dg, np.array(result['mean_mape_centroid']) / np.array(result['mean_mape_proportional']), 
                  label=f'MAPE Ratio - {names[i]}', 
                  linewidth=0.5)
    ax.loglog(dp/dg, np.ones(len(dg)), label='Equal Performance', color='darkgray', linestyle='--')
    ax.set_title('Allocation MAPE Ratio - Centroid:Proportional Allocation')
    ax.set_xlabel('Polygon:Grid Size Ratio')
    ax.set_ylabel('MAPE')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
def plot_simulation_error_ratios_multiple_2d(dg, dp, results, names):
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, result in enumerate(results):
        ax.loglog((dp**2)/(dg**2), np.array(result['mean_mape_centroid']) / np.array(result['mean_mape_proportional']), 
                  label=f'MAPE Ratio - {names[i]}', 
                  linewidth=0.5)
    ax.loglog((dp**2)/(dg**2), np.ones(len(dg)), label='Equal Performance', color='darkgray', linestyle='--')
    ax.set_title('Allocation MAPE Ratio - Centroid:Proportional Allocation')
    ax.set_xlabel('Polygon:Grid Size Ratio (By Area)')
    ax.set_ylabel('MAPE')
    ax.legend()
    plt.tight_layout()
    plt.show()