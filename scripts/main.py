import numpy as np
import os
from ml_collections import config_dict
os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['JAX_DEFAULT_MATMUL_PRECISION'] = 'highest'

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

from functools import partial
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
try:
    import wandb
except ImportError:
    wandb = None


import jax
# import mediapy as media # media import removed as it wasn't used after refactor
from randomize import domain_randomize
import balance # Import balance to access default_config
from datetime import datetime


from mujoco_playground import wrapper
from ml_collections import config_dict


np.set_printoptions(precision=3, suppress=True, linewidth=100)


def train_run(config=None, use_wandb=False):
    """Trains the G1 balance task with PPO, optionally using wandb."""
    run = None
    if use_wandb and wandb:
        run = wandb.init(config=config)
        cfg = wandb.config # Use wandb config if available
    else:
        # Use provided config dict or create an empty one if none provided
        cfg = config if config is not None else config_dict.ConfigDict()

    env_cfg = balance.default_config()



    env = balance.G1Env()
    eval_env = balance.G1Env()

    env_name = "g1_balance"
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    run_name = (run.name if run and hasattr(run, 'name') else timestamp) if use_wandb and wandb else timestamp
    exp_name = f"{env_name}-{run_name}"

    ckpt_path = os.path.abspath(os.path.join(".", "checkpoints", exp_name))
    os.makedirs(ckpt_path, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    # --- PPO Parameters ---
    # Use sweep config for hyperparameters if available, otherwise use defaults
    ppo_params = config_dict.create(
        num_timesteps=cfg.get('num_timesteps', 400_000_000), # Reverted to original default
        reward_scaling=cfg.get('reward_scaling', 0.1),
        episode_length=env_cfg.episode_length, # Use episode length from env_cfg
        normalize_observations=True,
        num_resets_per_eval=1,
        action_repeat=1,
        unroll_length=cfg.get('unroll_length', 32), # Reverted to original default
        num_minibatches=cfg.get('num_minibatches', 32), # Reverted to original default
        num_updates_per_batch=cfg.get('num_updates_per_batch', 5), # Reverted to original default
        discounting=cfg.get('discounting', 0.98),
        learning_rate=cfg.get('learning_rate', 1e-4), # Get LR from sweep config
        entropy_cost=cfg.get('entropy_cost', 0),
        num_envs=cfg.get('num_envs', 32768), # Reverted to original default
        batch_size=cfg.get('batch_size', 1024), # Reverted to original default
        num_evals=cfg.get('num_evals', 16), # Reverted to original default
        clipping_epsilon=cfg.get('clipping_epsilon', 0.2),
        log_training_metrics=True,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=cfg.get('policy_hidden_layer_sizes', (512, 256, 64)), # Reverted to original default
            value_hidden_layer_sizes=cfg.get('value_hidden_layer_sizes', (256, 256, 256, 256)), # Reverted to original default
            value_obs_key="privileged_state"
        )
    )
    # Log the actual PPO params being used if wandb is active
    if use_wandb and run:
        # Convert config_dict to a standard dict for wandb logging
        wandb.config.update(ppo_params.to_dict(), allow_val_change=True)

    # --- Progress Callback ---
    def progress_cli(num_steps, metrics):
        if use_wandb and run:
            wandb.log(metrics, step=num_steps)
        print(".", end="", flush=True)

    # --- Training Setup ---
    ppo_training_params = dict(ppo_params)
    network_factory_config = ppo_training_params.pop("network_factory")
    network_factory = partial(
        ppo_networks.make_ppo_networks,
        **network_factory_config
    )

    train_fn = partial(
        ppo.train, **ppo_training_params,
        network_factory=network_factory,
        progress_fn=progress_cli,
        randomization_fn=domain_randomize,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        save_checkpoint_path=ckpt_path
    )

    # --- Run Training ---
    print("Starting training...")
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=eval_env
    )
    print("\nTraining finished.")

    # --- Save Final Checkpoint as Wandb Artifact ---
    print(f"Checking for final checkpoint in {ckpt_path}...")
    try:
        # List all entries in the checkpoint directory
        entries = os.listdir(ckpt_path)
        # Filter for directories that are valid integers (potential timesteps)
        timestep_dirs = [d for d in entries if os.path.isdir(os.path.join(ckpt_path, d)) and d.isdigit()]

        if timestep_dirs:
            # Find the directory with the highest timestep number
            latest_timestep_dir = max(timestep_dirs, key=int)
            final_ckpt_path = os.path.join(ckpt_path, latest_timestep_dir)

            if use_wandb and run:
                print(f"Saving final checkpoint {final_ckpt_path} to wandb...")
                artifact = wandb.Artifact(f'{exp_name}-checkpoint', type='model')
                # Add the entire directory as the artifact
                artifact.add_dir(final_ckpt_path)
                run.log_artifact(artifact)
                print("Checkpoint saved to wandb.")
            else:
                print(f"Final checkpoint located at: {final_ckpt_path}")
        else:
            print(f"Warning: No valid checkpoint directories found in {ckpt_path}.")
    except Exception as e:
        print(f"Error during final checkpoint handling: {e}")


    # --- Evaluation & Logging ---
    print("Starting evaluation...")
    # Use a fresh eval env instance with the same overrides
    eval_env_2 = balance.G1Env()

    jit_reset = jax.jit(eval_env_2.reset)
    jit_step = jax.jit(eval_env_2.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    rng = jax.random.PRNGKey(cfg.get('eval_seed', 42))
    rollout = []
    n_episodes = 1

    for _ in range(n_episodes):
        state = jit_reset(rng)
        rollout.append(state)
        for i in range(env_cfg.episode_length):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            rollout.append(state)


    frames = eval_env_2.render(rollout, camera="track")
    frames_np = np.array(frames)
    frames_np_rearranged = np.transpose(frames_np, (0, 3, 1, 2))
    if use_wandb and run:
        wandb.log({"video": wandb.Video(frames_np_rearranged, fps=1.0 / env.dt, format="gif")})
    else:
        # Consider saving the video locally if wandb is not used
        print("Video generated. Consider saving it locally.") # Placeholder for local saving

    print("Evaluation finished.")
    if use_wandb and run:
        run.finish()

if __name__ == "__main__":
    # Example: Train without wandb by default
    # To enable wandb, you could parse command-line args here
    # For now, it defaults to False
    train_run(use_wandb=False)
