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
        # x = agent.policy_net.net[2](x) #linear  
        # hidden = agent.policy_net.net[3](x) #relu
    
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
                state = np.zeros(width*height)
                state[x + y*height] = 1
                # Get activations
                activations = get_hidden_activations(agent, state)
                place_maps[:, y, x] = activations
    
    return place_maps


def visualize_place_maps(place_maps, env, num_units_to_show=1600, figsize=(12, 12)):
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