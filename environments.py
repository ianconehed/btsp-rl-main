"""Grid‑maze with optional RGB rendering via pygame."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# pygame is optional; only imported when render() is called
import pygame


class GridMazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        width: int = 6,
        height: int = 6,
        walls=None,
        max_steps: int = 50,
        cell_size: int = 32,
    ) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self.walls = set(tuple(w) for w in walls) if walls else set()
        self.max_steps = max_steps
        self.cell_size = cell_size

        # Gym spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # positions
        self.start_pos = (0, 0)
        self.goal_pos = (width - 1, height - 1)
        self.agent_pos = self.start_pos
        self.step_count = 0

        # colours
        self._bg_color = (255, 255, 255)
        self._wall_color = (0, 0, 0)
        self._agent_color = (255, 0, 0)
        self._goal_color = (0, 255, 0)

        self._screen = None
        self._clock = None

    # ────────────────────────────────────────────────────────────── helpers ──
    def _get_obs(self):
        x, y = self.agent_pos
        return np.array([x / (self.width - 1), y / (self.height - 1)], dtype=np.float32)

    # ─────────────────────────────────────────────────────────── gym API ──
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
        self.agent_pos = self.start_pos
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action: int):
        x, y = self.agent_pos
        if action == 0:
            new_pos = (x, y - 1)  # up
        elif action == 1:
            new_pos = (x, y + 1)  # down
        elif action == 2:
            new_pos = (x - 1, y)  # left
        elif action == 3:
            new_pos = (x + 1, y)  # right
        else:
            raise ValueError("Action must be 0‑3")

        nx, ny = new_pos
        if 0 <= nx < self.width and 0 <= ny < self.height and new_pos not in self.walls:
            self.agent_pos = new_pos

        self.step_count += 1
        terminated = self.agent_pos == self.goal_pos
        truncated = self.step_count >= self.max_steps
        reward = 1.0 if terminated else 0.0
        # reward = 10.0 if terminated else -1.0
        return self._get_obs(), reward, terminated, truncated, {}

    def _draw_surface(self) -> pygame.Surface:
        """Return a Surface with the current grid drawn but don’t show it."""
        surf = pygame.Surface(
            (self.width * self.cell_size, self.height * self.cell_size)
        )
        surf.fill(self._bg_color)

        # walls
        for wx, wy in self.walls:
            rect = pygame.Rect(
                wx * self.cell_size, wy * self.cell_size, self.cell_size, self.cell_size
            )
            pygame.draw.rect(surf, self._wall_color, rect)

        # goal
        gx, gy = self.goal_pos
        pygame.draw.rect(
            surf,
            self._goal_color,
            pygame.Rect(
                gx * self.cell_size, gy * self.cell_size, self.cell_size, self.cell_size
            ),
        )

        # agent
        ax, ay = self.agent_pos
        pygame.draw.rect(
            surf,
            self._agent_color,
            pygame.Rect(
                ax * self.cell_size, ay * self.cell_size, self.cell_size, self.cell_size
            ),
        )
        return surf

    def render(self, mode: str = "human"):
        """Render the grid. mode ∈ {'human', 'rgb_array'}."""
        if mode not in ("human", "rgb_array"):
            raise NotImplementedError
            # ensure pygame is ready, even in headless rgb mode
        if not pygame.get_init():
            pygame.init()
        # if mode == "rgb_array" and self._screen is None:
        #     # create a hidden display so blits land on a real surface
        #     pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)

        # Draw off-screen first
        frame_surf = self._draw_surface()

        if mode == "human":
            if self._screen is None:
                pygame.init()
                self._screen = pygame.display.set_mode(frame_surf.get_size())
                pygame.display.set_caption("Grid Maze")
                self._clock = pygame.time.Clock()

            self._screen.blit(frame_surf, (0, 0))
            pygame.display.flip()
            self._clock.tick(30)  # limit fps
            return  # nothing to return for human mode

        # rgb_array – no window, just return the pixels
        return pygame.surfarray.array3d(frame_surf).swapaxes(0, 1)

    def close(self):
        if self._screen:
            pygame.display.quit()
            pygame.quit()
            self._screen = None
