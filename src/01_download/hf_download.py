# see https://huggingface.co/docs/huggingface_hub/guides/download for more details
from huggingface_hub import snapshot_download

# MODEL = "google/gemma-3-270m-it"
MODEL = "Qwen/Qwen3.5-0.8B"

snapshot_download(repo_id=MODEL, repo_type="model")

print(f"Download of {MODEL} completed successfully.")