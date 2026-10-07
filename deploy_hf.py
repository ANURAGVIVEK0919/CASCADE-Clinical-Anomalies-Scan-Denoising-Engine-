from huggingface_hub import HfApi

def deploy():
    print("Starting upload to Hugging Face...")
    api = HfApi() # It will use the token you just logged in with!
    api.upload_folder(
        folder_path=".",
        repo_id="ANURAG1909/cascade-denoising-engine",
        repo_type="space",
        ignore_patterns=["*.git*", "*__pycache__*", "*venv*", "*.env*", "deploy_hf.py"]
    )
    print("Upload complete!")

if __name__ == "__main__":
    deploy()
