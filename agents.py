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

    def forward(self, x):
        return self.net(x)
        # x = self.lin1(x)
        # x = nn.ReLU(x)
        # x = self.lin2(x)
        return


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
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_dim, action_dim, hidden_size).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.min_replay = min_replay
        self.gamma = gamma
        self.target_sync = target_sync
        self.total_steps = 0

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

        q_sa = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            next_q = self.target_net(ns).max(1, keepdim=True)[0]
            target = r + self.gamma * next_q * (1 - d)

        loss = self.loss_fn(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_steps % self.target_sync == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
