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
        state_dim=2,
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
        if ep % cfg.train.eval_interval == 0:
            eval_reward, frames = evaluate_agent(
                eval_env_ctor, agent, max_steps=cfg.env.max_steps, record_video=True
            )
            if cfg.wandb.enabled:
                # Convert to NumPy array: (T, H, W, C) -> (T, C, H, W)
                video_array = np.stack(frames).transpose(0, 3, 1, 2)
                # Must be uint8 format
                video_array = video_array.astype(np.uint8)
                video = wandb.Video(video_array, fps=30, format="mp4")
                # video = wandb.Video(np.stack(frames), fps=30, format="gif")
                wandb.log(
                    {"episode": ep, "eval_reward": eval_reward, "eval_video": video},
                    step=agent.total_steps,
                )
            print(f"[eval] episode {ep}: reward = {eval_reward:.2f}")

            # activation = {}
            # def getActivation(name):
            # # the hook signature
            #     def hook(model, input, output):
            #         activation[name] = output.detach()
            #     return hook

            # # register forward hooks on the layers of choice
            # h1 = model.avgpool.register_forward_hook(getActivation('avgpool'))
            # h2 = model.maxpool.register_forward_hook(getActivation('maxpool'))
            # h3 = model.layer3[0].downsample[1].register_forward_hook(getActivation('comp'))

            # # forward pass -- getting the outputs
            # out = model(X)

            # print(activation)

            # # detach the hooks
            # h1.remove()
            # h2.remove()
            # h3.remove()
            

        # ── Logging ──────────────────────────────────────────────────────────
        if cfg.wandb.enabled:
            wandb.log({"episode": ep, "reward": ep_r, "epsilon": eps})
        if ep % 50 == 0:
            avg50 = np.mean(rewards[-50:])
            print(
                f"Episode {ep}/{cfg.train.episodes} | avgR₅₀: {avg50:.2f} | ε: {eps:.2f}"
            )


         # ── Final place map visualization ────────────────────────────────────────
        if cfg.wandb.enabled:
            final_env = eval_env_ctor()
            final_place_maps = pt.generate_place_maps(agent, final_env, include_walls=True)
            
            # Show all units if there are more than 16
            if cfg.agent.hidden_size > 16:
                all_figs = pt.visualize_all_units_grid(final_place_maps, final_env)
                for i, fig in enumerate(all_figs):
                    wandb.log({f"final_place_maps_batch_{i}": wandb.Image(fig)})
                    plt.close(fig)
            else:
                final_fig = pt.visualize_place_maps(final_place_maps, final_env)
                wandb.log({"final_place_maps": wandb.Image(final_fig)})
                plt.close(final_fig)
            
            final_env.close()
            # wandb.finish()    

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
