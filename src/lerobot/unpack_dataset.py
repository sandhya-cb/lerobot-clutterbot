import numpy as np
import torch as th
from gymnasium.spaces import Box
from torch.utils.data import DataLoader, Dataset

# Core Imitation and RL Libraries
from imitation.algorithms import adversarial
from imitation.data import rollout
from imitation.data.types import Trajectory, Transitions
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# LeRobot Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --- Isaac Lab/Gym Environment (Mocked/Placeholder) ---
# In a real setup, this would be a custom wrapper for your Isaac Lab environment
# that implements the Gymnasium VecEnv interface (e.g., using isaaclab.utils.wrappers)
class IsaacLabEnvPlaceholder(DummyVecEnv):
    def __init__(self, action_dim, obs_dim, n_envs=1):
        # Your robot's state and action space from the dataset
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        self.observation_dim = obs_dim
        self.action_dim = action_dim
        
        # Create a simple environment function for DummyVecEnv
        def make_env():
            from gymnasium import make
            # Use a generic simple environment if you cannot link to Isaac Lab here
            # For real usage, you'd replace this with your Isaac Lab environment registration
            return make("CartPole-v1") 

        # We need a proper Gym env for PPO/AIRL to work, a placeholder will suffice 
        # for a non-runnable script, but in real life, you must replace DummyVecEnv 
        # with your Isaac Lab VecEnv wrapper.
        super().__init__([make_env for _ in range(n_envs)]) 
        # Override the spaces to match your robot's (if BipedalWalker-v3 doesn't match)
        self.observation_space = self.envs[0].observation_space = self.single_observation_space = self.observation_space
        self.action_space = self.envs[0].action_space = self.single_action_space = self.action_space

    def step_async(self, actions):
        # Placeholder logic for step_async
        return self.envs[0].step_async(actions)
        
    def step_wait(self):
        # Placeholder logic for step_wait
        return self.envs[0].step_wait()
        
    def reset(self, **kwargs):
        # Placeholder logic for reset
        return self.envs[0].reset(**kwargs)
        
    def seed(self, seed=None):
        return [self.envs[0].seed(seed)]

# --- Custom LeRobot Trajectory Dataset Loader ---
class LeRobotTrajectoryDataset(Dataset):
    """
    A custom PyTorch Dataset to load the LeRobot data and convert it into 
    imitation's Trajectory format, using only state observations.
    """
    def __init__(self, repo_id: str, local_path: str = None):
        print(f"Loading LeRobot dataset from {repo_id}...")
        self.ds = LeRobotDataset(repo_id=repo_id, root=local_path)
        
        # Extract episode boundaries for Trajectory creation
        self.episode_starts = self.ds.episode_data_index["from"].tolist()
        self.episode_ends = self.ds.episode_data_index["to"].tolist()
        
    def __len__(self):
        # The number of expert trajectories (episodes)
        return len(self.episode_starts)
    def __getitem__(self, idx: int) -> Trajectory:
        start_idx = self.episode_starts[idx]
        end_idx = self.episode_ends[idx]
        
        # Fetch all frames for the episode
        # Frames contains data for T+1 states (s_0 to s_T) and T actions (a_0 to a_{T-1})
        frames = [self.ds[i] for i in range(start_idx, end_idx)]
        
        # Extract observations (s_0 to s_T)
        # If the episode has T steps, obs length = T + 1
        full_obs = th.stack([frame["observation.state"] for frame in frames]) 
        
        # Extract actions (a_0 to a_T). Since the last action a_T might be a dummy or 
        # unrelated to the dynamics learned in IRL, we must drop it.
        # If the episode has T steps, acts length = T + 1
        full_acts = th.stack([frame["action"] for frame in frames])
        
        # --- FIX ---
        
        # 1. Observations: Keep ALL states (s_0 to s_T)
        # The length will be N + 1 (e.g., 140)
        obs = full_obs 
        
        # 2. Actions: Keep all actions EXCEPT the very last one (a_0 to a_{T-1})
        # The length will be N (e.g., 139)
        acts = full_acts[:-1]
        
        # 3. Create 'dones' array: False for all steps except the last one
        # Dones must have the same length as acts (N)
        dones = np.zeros(len(acts), dtype=bool)
        dones[-1] = True # Last step is done=True
        
        # 4. Convert to numpy for Trajectory type
        return Trajectory(
            obs=obs.cpu().numpy(),
            acts=acts.cpu().numpy(),
            infos=None,  # No extra info needed for basic AIRL
            terminal=True, # This flag is optional but good practice
        )
        

def get_expert_transitions(dataset_path: str, n_trajectories: int = -1) -> Transitions:
    """
    Loads expert data from LeRobot dataset and converts it to Transitions for AIRL.
    """
    expert_dataset = LeRobotTrajectoryDataset(repo_id=dataset_path, local_path=dataset_path)
    
    trajectories = []
    # If -1, use all, otherwise limit the number of episodes
    num_to_use = len(expert_dataset) if n_trajectories == -1 else n_trajectories
    print (num_to_use)
    print (len(expert_dataset))
    for i in range(min(num_to_use, len(expert_dataset))):
        print(i)
        trajectories.append(expert_dataset[i])
        
    # Convert Trajectories to Transitions
    transitions = rollout.flatten_trajectories(trajectories)
    print("Retrieved trajectories")
    return transitions

# --- Configuration ---
# REPLACE THESE WITH YOUR ACTUAL VALUES
DATASET_PATH = "/home/sandhya/lerobot/lerobot/sandhyavs/dusty_450_eps" #"/path/to/your/local/lerobot/dataset"  # e.g., 'data/'
DATASET_REPO_ID = "sandhyavs/dusty_450_eps"  # If loading from HuggingFace Hub
LOG_DIR = "airl_dusty_irl_output"
N_EXPERT_TRAJECTORIES = 100 # How many of your 433 episodes to use

# Robot dimensions from your dataset info:
ACTION_DIM = 4 
STATE_DIM = 4 
# The full observation space includes images, but we use state for the discriminator input.

# AIRL/PPO Training parameters
TOTAL_TIMESTEPS = 1_000_000 # Total steps for PPO to train
IRL_TRAIN_STEPS = 500_000 # Total steps for the Discriminator to train
N_ENVS = 4 # Number of parallel environments in VecEnv
BATCH_SIZE = 1024 # Batch size for PPO/AIRL updates

# --- Main AIRL Training Script ---

def train_airl():
    print("--- 1. Setting up Environment ---")
    # This is the crucial Isaac Lab integration point. 
    # The Placeholder is used for a non-runnable script. 
    # In practice, use your Isaac Lab VecEnv wrapper here.
    venv = IsaacLabEnvPlaceholder(
        action_dim=ACTION_DIM, 
        obs_dim=STATE_DIM, 
        n_envs=N_ENVS
    )
    
    print("--- 2. Loading Expert Demonstrations ---")
    # Load and convert LeRobot data to imitation's Transitions format
    expert_transitions = get_expert_transitions(
        dataset_path=DATASET_REPO_ID, 
        n_trajectories=N_EXPERT_TRAJECTORIES
    )
    print(f"Loaded {len(expert_transitions)} expert steps.")

    print("--- 3. Setting up Discriminator/Reward Network ---")
    # AIRL uses a Discriminator (D) which acts as the learned reward function.
    # The architecture can be a simple MLP for state-only observations.
    reward_net = adversarial.BasicShapedRewardNet(
        venv.observation_space, 
        venv.action_space, 
        # Using a simple network architecture for the reward function
        hid_sizes=(64, 64) 
    )

    print("--- 4. Setting up Generator (PPO Agent) ---")
    # PPO is the default and recommended RL algorithm (Generator) for AIRL.
    # It learns a policy in the environment using the reward signal from the discriminator.
    learner = PPO(
        policy="MlpPolicy",
        env=venv,
        seed=0,
        batch_size=BATCH_SIZE,
        # Adjust PPO parameters as needed for your task
        n_steps=BATCH_SIZE,
        learning_rate=3e-4,
        ent_coef=0.0,
        verbose=1,
        device="cuda" if th.cuda.is_available() else "cpu"
    )

    print("--- 5. Initializing and Training AIRL ---")
    airl_trainer = adversarial.AIRL(
        demonstrations=expert_transitions,
        demo_batch_size=BATCH_SIZE, # Should be the same as PPO batch size
        gen_algo=learner,
        reward_net=reward_net,
        venv=venv,
        # Note: You can adjust the ratio of generator (PPO) to discriminator (AIRL) updates
        n_disc_updates_per_round=4, # A common ratio for good stability
        # Use Sacred/Tee_Logger for logging
        log_dir=LOG_DIR,
    )

    # Train the AIRL algorithm
    airl_trainer.train(
        total_timesteps=IRL_TRAIN_STEPS, 
    )

    print("\n--- 6. AIRL Training Complete ---")
    print(f"Learned Reward Network saved in: {LOG_DIR}")
    
    # You can now save the learned reward network
    th.save(reward_net.state_dict(), f"{LOG_DIR}/airl_reward_net.pt")
    
    # The final learned policy is in airl_trainer.gen_algo (the PPO agent)
    airl_trainer.gen_algo.save(f"{LOG_DIR}/airl_learned_policy.zip")

    print(f"Learned Policy saved in: {LOG_DIR}")

# Call the training function
if __name__ == "__main__":
    # Ensure your Isaac Lab environment is properly registered/initialized before this
    train_airl()