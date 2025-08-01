from datasets import load_dataset

dataset = load_dataset("/home/sandhya/.cache/huggingface/lerobot/sandhyavs/eval_smolvla_finetune-lego-pickup-clean-100eps-default-200k_steps/")
dataset.push_to_hub("sandhyavs/eval_smolvla_finetune-lego-pickup-clean-100eps-default-200k_steps")