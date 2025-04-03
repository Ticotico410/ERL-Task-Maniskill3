"""
evaluate.py:
  - All evaluation parameters are hard-coded in the script, no need to pass via command line.
  - Modify these parameters to debug the evaluation process.
  - Call sac_modified.run_sac(args) to start inference/evaluation.
"""
from sac.models import sac_modified


def main():
    # Empty parameter object
    args = type('Args', (), {})()

    # Mode settings
    args.evaluate = True  # Force evaluation mode
    args.checkpoint = "./run/train_logs_50000/PickCube-v1_seed0_20250402-153647/final_model.pt"

    # Environment related parameters
    args.env_id = "PickCube-v1"  # Environment ID to use
    args.seed = 0  # Random seed
    args.cuda = True  # Whether to use GPU
    args.control_mode = "pd_joint_delta_pos"  # Control mode
    args.capture_video = True  # Record video during evaluation

    # Parallel environment settings
    args.num_envs = 1  # Number of parallel environments used during evaluation
    args.eval_num_envs = 1  # Number of parallel environments used during evaluation
    args.eval_steps = 100  # Run n steps during evaluation

    """ Training parameters (No need to set in training mode but must be declared) """
    args.total_timesteps = 0
    args.gamma = 0.99
    args.alpha = 0.2
    args.autotune = False
    args.learning_rate = 3e-4
    args.batch_size = 256
    args.buffer_size = 200000
    args.start_steps = 1000
    args.train_freq = 1000
    args.updates_per_iter = 100
    args.tau = 0.005
    args.eval_interval = 5000

    # Log directory
    args.log_dir = f"./run/eval_logs_50000"

    # Evaluating
    sac_modified.run_sac(args)


if __name__ == "__main__":
    main()
