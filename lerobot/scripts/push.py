from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/home/sandhya/.cache/huggingface/lerobot/sandhyavs/push_purple_block",
    repo_id="sandhyavs/push_purple_block_webcam",
    repo_type="dataset",
)
