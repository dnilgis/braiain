"""One-off: publish hf/report as the Space braiain-lab/technical-report and add it to the brAIAIn collection."""
import json, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo = "braiain-lab/technical-report"
api.create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
api.upload_folder(repo_id=repo, repo_type="space", folder_path=os.path.join(root, "hf", "report"), commit_message="Technical report v1")
coll = json.load(open(os.path.join(root, "data", "announce.json"))).get("hf_collection")
if coll:
    api.add_collection_item(coll, item_id=repo, item_type="space", exists_ok=True, note="Technical report: brAIAIn, a language model trained exclusively on what it was paid to learn.")
print("published", repo)
