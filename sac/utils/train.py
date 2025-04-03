"""
train.py:
  - All training parameters are hard-coded in the script, no need to pass via command line.
  - Modify these parameters to debug the training process.
  - Call sac_modified.run_sac(args) to start training.
  - Use the command "tensorboard --logdir=./sac/utils/train_logs_{total_timesteps}" to view evaluation metrics.
"""
from sac.models import sac_modified


def main():
    # Empty object to store parameters
    args = type('Args', (), {})()

    # Mode settings
    args.evaluate = False  # Training mode
    args.checkpoint = None  # Load an existing model during training (Default: None)

    # Environment related parameters
    args.env_id = "PickCube-v1"  # Environment ID
    args.seed = 0  # Random seed
    args.cuda = True  # CPU/GPU mode (Default: True)
    args.control_mode = "pd_joint_delta_pos"  # Control mode
    args.capture_video = True  # Record training and evaluation videos

    # Parallel environment settings
    args.num_envs = 8  # Number of parallel environments used in training
    args.eval_num_envs = 1  # Number of parallel environments used in evaluation

    # Training hyperparameters
    args.total_timesteps = 50000  # Total number of environment interaction steps for training
    args.gamma = 0.99  # Discount factor
    args.alpha = 0.2  # Fixed entropy coefficient if autotune is not enabled
    args.autotune = False  # Enable automatic tuning of the entropy coefficient (Default: False)
    args.learning_rate = 3e-4  # Learning rate for both Actor and Critic
    args.batch_size = 256  # Batch size sampled from the Replay Buffer each update
    args.buffer_size = 200000  # Total capacity of the Replay Buffer
    args.start_steps = 5000  # Use random actions for the first n steps to collect data
    args.train_freq = 1000  # Start updating after collecting n environment steps per iteration
    args.updates_per_iter = 100  # Number of gradient updates per iteration
    args.tau = 0.005  # Soft update coefficient

    """ Evaluating parameters (No need to set in training mode but must be declared) """
    args.eval_interval = 5000  # Evaluate every n environment steps
    args.eval_steps = 50  # Number of steps to run in the environment during evaluation

    # Log directory
    args.log_dir = f"./run/train_logs_{args.total_timesteps}"

    # Training
    sac_modified.run_sac(args)


if __name__ == "__main__":
    main()
