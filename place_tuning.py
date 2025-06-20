"""Visualize hidden unit activations as place maps."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


def get_bottleneck_activations(agent, state):
    """Extract bottleneck layer activations (after first ReLU)."""
    state_t = torch.as_tensor(
        state, dtype=torch.float32, device=agent.device
    ).unsqueeze(0)
    
    with torch.no_grad():
        # Forward through layers 0-1
        bottleneck = agent.policy_net.net[:2](state_t)
    
    return bottleneck.squeeze(0).cpu().numpy()


def get_hidden_activations(agent, state):
    """Extract hidden layer activations (after second ReLU)."""
    state_t = torch.as_tensor(
        state, dtype=torch.float32, device=agent.device
    ).unsqueeze(0)
    
    with torch.no_grad():
        # Forward through layers 0-3 (up to and including second ReLU)
        hidden = agent.policy_net.net[:4](state_t)
    
    return hidden.squeeze(0).cpu().numpy()


def generate_place_maps(agent, env, layer='hidden', include_walls=True):
    """Generate activation maps for all hidden units across all grid positions."""
    width, height = env.width, env.height
    
    if layer == 'bottleneck':
        hidden_size = 2  # Bottleneck dimension
        get_activations = get_bottleneck_activations
    else:  # 'hidden'
        hidden_size = agent.policy_net.net[2].out_features  # Hidden layer size
        get_activations = get_hidden_activations
    
    # Initialize place maps
    place_maps = np.zeros((hidden_size, height, width))
    
    # Test each grid position
    for y in range(height):
        for x in range(width):
            # Skip if it's a wall
            if (x, y) in env.walls and include_walls:
                place_maps[:, y, x] = np.nan
            else:
                # Create one-hot state for this position
                state = np.zeros(width*height)
                state[x + y*height] = 1
                # Get activations
                activations = get_activations(agent, state)
                place_maps[:, y, x] = activations
    
    return place_maps


def visualize_place_maps(place_maps, env, num_units_to_show=None, figsize=(12, 12), title='Hidden Unit Place Maps'):
    """Visualize place maps as heatmaps for selected hidden units."""
    if num_units_to_show is None:
        num_units_to_show = place_maps.shape[0]
    
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
    
    plt.suptitle(f'{title}\n(Green=Start, Blue=Goal, Gray=Walls)', 
                 fontsize=14)
    
    return fig