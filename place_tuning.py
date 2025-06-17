"""Visualize hidden unit activations as place maps."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


def get_hidden_activations(agent, state):
    """Extract hidden layer activations for a given state."""
    state_t = torch.as_tensor(
        state, dtype=torch.float32, device=agent.device
    ).unsqueeze(0)
    
    # Forward pass through the first layer only
    with torch.no_grad():
        # Access the first linear layer and ReLU
        x = agent.policy_net.net[0](state_t)  # Linear layer
        hidden = agent.policy_net.net[1](x)    # ReLU activation
    
    return hidden.squeeze(0).cpu().numpy()


def generate_place_maps(agent, env, include_walls=True):
    """Generate activation maps for all hidden units across all grid positions."""
    width, height = env.width, env.height
    hidden_size = agent.policy_net.net[0].out_features
    
    # Initialize place maps
    place_maps = np.zeros((hidden_size, height, width))
    
    # Test each grid position
    for y in range(height):
        for x in range(width):
            # Skip if it's a wall
            if (x, y) in env.walls and include_walls:
                place_maps[:, y, x] = np.nan
            else:
                # Create normalized state for this position
                state = np.array([x / (width - 1), y / (height - 1)], dtype=np.float32)
                # Get activations
                activations = get_hidden_activations(agent, state)
                place_maps[:, y, x] = activations
    
    return place_maps


def visualize_place_maps(place_maps, env, num_units_to_show=16, figsize=(12, 12)):
    """Visualize place maps as heatmaps for selected hidden units."""
    n_units = min(num_units_to_show, place_maps.shape[0])
    n_cols = int(np.sqrt(n_units))
    n_rows = int(np.ceil(n_units / n_cols))
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)
    
    # Create colormap that handles NaN (walls) properly
    cmap = plt.cm.hot
    cmap.set_bad(color='gray')
    
    for idx in range(n_units):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        
        # Plot the activation map
        im = ax.imshow(place_maps[idx], cmap=cmap, origin='upper', aspect='equal')
        
        # Add grid lines
        for x in range(env.width + 1):
            ax.axvline(x - 0.5, color='white', linewidth=0.5, alpha=0.3)
        for y in range(env.height + 1):
            ax.axhline(y - 0.5, color='white', linewidth=0.5, alpha=0.3)
        
        # Mark start and goal positions
        start_rect = patches.Rectangle((env.start_pos[0] - 0.4, env.start_pos[1] - 0.4), 
                                     0.8, 0.8, linewidth=2, edgecolor='green', 
                                     facecolor='none')
        goal_rect = patches.Rectangle((env.goal_pos[0] - 0.4, env.goal_pos[1] - 0.4), 
                                    0.8, 0.8, linewidth=2, edgecolor='blue', 
                                    facecolor='none')
        ax.add_patch(start_rect)
        ax.add_patch(goal_rect)
        
        ax.set_title(f'Unit {idx}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar for each subplot
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
    
    plt.suptitle('Hidden Unit Place Maps\n(Green=Start, Blue=Goal, Gray=Walls)', 
                 fontsize=14)
    
    return fig


def visualize_all_units_grid(place_maps, env, units_per_fig=16):
    """Generate multiple figures if there are many hidden units."""
    n_units = place_maps.shape[0]
    n_figs = int(np.ceil(n_units / units_per_fig))
    
    figures = []
    for fig_idx in range(n_figs):
        start_idx = fig_idx * units_per_fig
        end_idx = min(start_idx + units_per_fig, n_units)
        
        # Extract subset of place maps
        subset_maps = place_maps[start_idx:end_idx]
        
        fig = visualize_place_maps(subset_maps, env, num_units_to_show=subset_maps.shape[0])
        figures.append(fig)
    
    return figures


def get_activation_statistics(place_maps):
    """Compute statistics about the place maps."""
    stats = {}
    
    # Remove NaN values (walls) for statistics
    valid_activations = place_maps[~np.isnan(place_maps)]
    
    if len(valid_activations) > 0:
        stats['mean_activation'] = np.mean(valid_activations)
        stats['std_activation'] = np.std(valid_activations)
        stats['max_activation'] = np.max(valid_activations)
        stats['sparsity'] = np.mean(valid_activations == 0)
        
        # Compute spatial selectivity for each unit
        n_units = place_maps.shape[0]
        selectivity = []
        for unit_idx in range(n_units):
            unit_map = place_maps[unit_idx]
            valid_unit = unit_map[~np.isnan(unit_map)]
            if len(valid_unit) > 0 and np.max(valid_unit) > 0:
                # Normalized activation variance as selectivity measure
                norm_map = valid_unit / (np.max(valid_unit) + 1e-8)
                selectivity.append(np.std(norm_map))
        
        stats['mean_selectivity'] = np.mean(selectivity) if selectivity else 0
        stats['n_active_units'] = len([s for s in selectivity if s > 0.1])
    
    return stats


def create_activation_trajectory(agent, env, trajectory, fig_size=(15, 5)):
    """Visualize activations along a specific trajectory."""
    hidden_size = agent.policy_net.net[0].out_features
    n_steps = len(trajectory)
    
    # Collect activations along trajectory
    activations = np.zeros((n_steps, hidden_size))
    for t, (x, y) in enumerate(trajectory):
        state = np.array([x / (env.width - 1), y / (env.height - 1)], dtype=np.float32)
        activations[t] = get_hidden_activations(agent, state)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
    
    # Plot trajectory on grid
    ax1.set_xlim(-0.5, env.width - 0.5)
    ax1.set_ylim(-0.5, env.height - 0.5)
    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    
    # Draw walls
    for wall in env.walls:
        rect = patches.Rectangle((wall[0] - 0.5, wall[1] - 0.5), 1, 1, 
                               facecolor='gray', edgecolor='black')
        ax1.add_patch(rect)
    
    # Draw trajectory
    traj_x = [pos[0] for pos in trajectory]
    traj_y = [pos[1] for pos in trajectory]
    ax1.plot(traj_x, traj_y, 'b-', linewidth=2, marker='o', markersize=6)
    ax1.plot(traj_x[0], traj_y[0], 'go', markersize=10, label='Start')
    ax1.plot(traj_x[-1], traj_y[-1], 'ro', markersize=10, label='End')
    
    ax1.set_title('Trajectory')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot activation patterns
    im = ax2.imshow(activations.T, aspect='auto', cmap='hot', origin='lower')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Hidden Unit')
    ax2.set_title('Hidden Unit Activations Along Trajectory')
    plt.colorbar(im, ax=ax2)
    
    return fig