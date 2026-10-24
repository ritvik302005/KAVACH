"""
Publish the Kavach backend as a Hugging Face Docker Space.

Copies backend/ (without tests, logs, weights or the LFS video) and the Space's
Dockerfile and README from this folder into a staging folder, then uploads it.

Usage (from the repository root):
    pip install huggingface_hub
    $env:HF_TOKEN = "<write token>"          # PowerShell
    python deploy/huggingface/push_space.py <hf-username>/kavach-api
"""
import os
import shutil
import sys
import tempfile

from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, "..", "..", "backend")
SKIP = shutil.ignore_patterns("__pycache__", "venv", "tests", "logs", "*.pt", "*.mp4", ".dockerignore", "Dockerfile", "README.md")


def main():
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        raise SystemExit("usage: python deploy/huggingface/push_space.py <hf-username>/<space-name>")
    repo_id = sys.argv[1]
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN to a Hugging Face token with write access")

    with tempfile.TemporaryDirectory() as stage:
        shutil.copytree(BACKEND, stage, ignore=SKIP, dirs_exist_ok=True)
        os.makedirs(os.path.join(stage, "data", "videos"), exist_ok=True)
        for name in ("Dockerfile", "README.md"):
            shutil.copy(os.path.join(HERE, name), os.path.join(stage, name))

        api = HfApi(token=token)
        api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(folder_path=stage, repo_id=repo_id, repo_type="space",
                          commit_message="Deploy Kavach backend")
    user, name = repo_id.split("/")
    print(f"Space: https://huggingface.co/spaces/{repo_id}")
    print(f"API:   https://{user.lower()}-{name.lower().replace('_', '-')}.hf.space")


if __name__ == "__main__":
    main()
