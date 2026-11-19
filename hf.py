from huggingface_hub import hf_hub_download

repo = "epinnock/polymarket-nba"

files = [
    "data/markets.parquet",
    "data/prices/prices-2025-26.parquet",
]

for filename in files:
    path = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=filename,
        local_dir="data/polymarket"
    )
    print("Downloaded:", path)