
from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset
# hub_api = HfApi()
# hub_api.create_tag("sandhyavs/dusty_3cam_copy", tag="v2.1", repo_type="dataset")
dataset = LeRobotDataset(
    "sandhyavs/dusty_3cam_copy"
    # cfg.dataset.repo_id,
    # root=cfg.dataset.root,
)
dataset.check_set()
# Path or name of your dataset

dataset.mean_episode_length()