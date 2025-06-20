"""Entry‑point that wires everything together via Hydra config."""

from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
import wandb
import matplotlib.pyplot as plt

from agents import DQNAgent
from environments import GridMazeEnv
from utils import set_seed
import place_tuning as pt


def evaluate_agent(eval_env_ctor, agent, max_steps, record_video=False):
    """Run one greedy episode. Optionally capture RGB frames."""
    env = eval_env_ctor()
    state, _ = env.reset()
    frames = []
    total_r = 0.0
    for step in range(max_steps):
        if record_video:
            frames.append(env.render(mode="rgb_array"))
        action = agent.act(state, eps=0.0)  # greedy
        state, reward, terminated, truncated, _ = env.step(action)
        total_r += reward
        if terminated or truncated:
            break
    env.close()
    return total_r, frames


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.train.seed)

    # ── Environment ───────────────────────────────────────────────────────────
    train_env = GridMazeEnv(
        width=cfg.env.width,
        height=cfg.env.height,
        walls={tuple(w) for w in cfg.env.walls},
        max_steps=cfg.env.max_steps,
        cell_size=cfg.env.render_cell_size,
    )

    # eval is constructor so we ran new instance every time
    eval_env_ctor = lambda: GridMazeEnv(
        width=cfg.env.width,
        height=cfg.env.height,
        walls={tuple(w) for w in cfg.env.walls},
        max_steps=cfg.env.max_steps,
        cell_size=cfg.env.render_cell_size,
    )

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent = DQNAgent(
        state_dim=cfg.env.width*cfg.env.height,
        action_dim=train_env.action_space.n,
        gamma=cfg.agent.gamma,
        lr=cfg.agent.lr,
        batch_size=cfg.agent.batch_size,
        min_replay=cfg.agent.min_replay,
        target_sync=cfg.agent.target_sync,
        hidden_size=cfg.agent.hidden_size,
        buffer_capacity=cfg.agent.buffer_capacity,
    )

    # ── W&B ───────────────────────────────────────────────────────────────────
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.watch(agent.policy_net, log="all")

    # ── Training loop ─────────────────────────────────────────────────────────
    rewards, eps = [], cfg.train.eps_start
    for ep in range(1, cfg.train.episodes + 1):
        state, _ = train_env.reset(seed=None)
        ep_r = 0
        for _ in range(cfg.train.max_steps):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated

            if cfg.train.render:
                train_env.render("human")

            agent.remember(state, action, reward, next_state, float(done))
            agent.learn()
            agent.total_steps += 1

            state = next_state
            ep_r += reward
            if done:
                break

        rewards.append(ep_r)
        eps = max(cfg.train.eps_end, eps * cfg.train.eps_decay)

        # ── Periodic evaluation with video ───────────────────────────────────
        if (ep % cfg.train.eval_interval == 0) or (ep == 1):
            eval_reward, frames = evaluate_agent(
                eval_env_ctor, agent, max_steps=cfg.env.max_steps, record_video=True
            )
            if cfg.wandb.enabled:
                # Convert to NumPy array: (T, H, W, C) -> (T, C, H, W)
                video_array = np.stack(frames).transpose(0, 3, 1, 2)
                # Must be uint8 format
                video_array = video_array.astype(np.uint8)
                video = wandb.Video(video_array, fps=30, format="mp4")
                
                # Create eval environment for place field visualizations
                eval_env = eval_env_ctor()
                
                # Generate bottleneck place maps (2 units)
                bottleneck_place_maps = pt.generate_place_maps(
                    agent, eval_env, layer='bottleneck', include_walls=True
                )
                bottleneck_fig = pt.visualize_place_maps(
                    bottleneck_place_maps, eval_env, 
                    figsize=(8, 4),
                    title='Bottleneck Layer Place Maps'
                )
                
                # Generate hidden layer place maps (hidden_size units)
                hidden_place_maps = pt.generate_place_maps(
                    agent, eval_env, layer='hidden', include_walls=True
                )
                # Show up to 16 units for hidden layer
                hidden_fig = pt.visualize_place_maps(
                    hidden_place_maps, eval_env, 
                    num_units_to_show=min(16, cfg.agent.hidden_size),
                    figsize=(12, 12),
                    title='Hidden Layer Place Maps'
                )
                
                # Log everything
                wandb.log({
                    "episode": ep, 
                    "eval_reward": eval_reward, 
                    "eval_video": video,
                    "bottleneck_place_maps": wandb.Image(bottleneck_fig),
                    "hidden_place_maps": wandb.Image(hidden_fig),
                }, step=agent.total_steps)
                
                plt.close(bottleneck_fig)
                plt.close(hidden_fig)
                eval_env.close()
                
            print(f"[eval] episode {ep}: reward = {eval_reward:.2f}")
            

        # ── Logging ──────────────────────────────────────────────────────────
        if cfg.wandb.enabled:
            wandb.log({"episode": ep, "reward": ep_r, "epsilon": eps})
        if ep % 50 == 0:
            avg50 = np.mean(rewards[-50:])
            print(
                f"Episode {ep}/{cfg.train.episodes} | avgR₅₀: {avg50:.2f} | ε: {eps:.2f}"
            )
        

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
