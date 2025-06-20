"""DQN agent + Q‑network."""

import random

import torch
import torch.nn as nn
import torch.optim as optim


from replay_buffer import ReplayBuffer


class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 20):
        super().__init__()
        # self.lin1 = nn.Linear(state_dim,hidden_size)
        # self.lin2 = nn.Linear(hidden_size,action_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 2),
            nn.ReLU(),
            nn.Linear(2,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
        
        # Freeze the second linear layer (index 2) from backprop
        for param in self.net[2].parameters():
            param.requires_grad = False
        
        # Store initial weights and initialize Oja component
        self.W_0 = self.net[2].weight.data.clone()  # Initial weights
        self.W_oja = torch.zeros_like(self.W_0).to(self.W_0.device)  # Oja component

    def forward(self, x):
        return self.net(x)
    
    def get_layer_activations(self, x):
        """Get activations before and after layer 2 for Oja update."""
        # Forward through layers 0-1 (up to but not including layer 2)
        layer2_input = self.net[:2](x)
        
        # Forward through layer 2
        layer2_output = self.net[2](layer2_input)
        
        return layer2_input, layer2_output


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        min_replay: int = 1_000,
        target_sync: int = 200,
        hidden_size: int = 20,
        buffer_capacity: int = 50_000,
        oja_lr: float = 0.01,  # Learning rate for Oja's rule
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_dim, action_dim, hidden_size).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Copy Oja components to target net
        self.target_net.W_0 = self.policy_net.W_0.clone()
        self.target_net.W_oja = self.policy_net.W_oja.clone()

        # Only optimize parameters that require gradients
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.policy_net.parameters()), 
            lr=lr
        )
        self.loss_fn = nn.SmoothL1Loss()

        self.replay = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.min_replay = min_replay
        self.gamma = gamma
        self.target_sync = target_sync
        self.total_steps = 0
        self.oja_lr = oja_lr

    # ϵ‑greedy
    def act(self, state, eps: float):
        if random.random() < eps:
            return random.randrange(self.policy_net.net[-1].out_features)
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            return int(self.policy_net(state_t).argmax())

    def remember(self, *args):
        self.replay.push(*args)
    
    def oja_update(self, states):
        """Update layer 2 weights using Oja's rule.
        Oja's rule: Δw_ij = η * y_i * (x_j - y_i * w_ij)
        """
        with torch.no_grad():
            # Get activations
            x, y = self.policy_net.get_layer_activations(states)
            
            # Current total weights are W_0 + W_oja
            W_total = self.policy_net.W_0 + self.policy_net.W_oja
            
            # Oja's rule update
            # For batch: average the updates
            batch_size = x.shape[0]
            
            # Compute outer product and update
            # y: (batch, hidden_size), x: (batch, 2)
            
            for i in range(batch_size):
                y_i = y[i:i+1].T  # (hidden_size, 1)
                x_i = x[i:i+1]    # (1, 2)
                
                # Oja update for this sample using total weights
                delta_W = self.oja_lr * (y_i @ x_i - y_i @ y_i.T @ W_total)
                self.policy_net.W_oja += delta_W / batch_size
            
            # Update the actual network weights
            self.policy_net.net[2].weight.data = self.policy_net.W_0 + self.policy_net.W_oja

    def learn(self):
        if len(self.replay) < max(self.batch_size, self.min_replay):
            return

        batch = self.replay.sample(self.batch_size)
        s = torch.tensor(batch.state, dtype=torch.float32, device=self.device)
        a = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(
            1
        )
        r = torch.tensor(
            batch.reward, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        ns = torch.tensor(batch.next_state, dtype=torch.float32, device=self.device)
        d = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(
            1
        )

        # Update layer 2 with Oja's rule before backprop
        self.oja_update(s)

        q_sa = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            next_q = self.target_net(ns).max(1, keepdim=True)[0]
            target = r + self.gamma * next_q * (1 - d)

        loss = self.loss_fn(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_steps % self.target_sync == 0:
            # Need to update target net's Oja components too
            self.target_net.W_0 = self.policy_net.W_0.clone()
            self.target_net.W_oja = self.policy_net.W_oja.clone()
            self.target_net.load_state_dict(self.policy_net.state_dict())
