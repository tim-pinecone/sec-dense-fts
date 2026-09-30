import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).parent
SPACE_FILES = ["README.md", "requirements.txt", "app.py", "search_core.py", "fts_queries.py"]
SECRETS = ["PINECONE_API_KEY", "OPENAI_API_KEY"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy the Gradio app to a Hugging Face Space.")
    parser.add_argument("space_id", help="e.g. Tim-Pinecone/sec-fts-search")
    parser.add_argument("--private", action="store_true", help="Create the Space as private (first deploy only).")
    parser.add_argument("--hardware", default="zero-a10g", help="Space hardware requested on creation (default: zero-a10g).")
    parser.add_argument("--set-secrets", action="store_true", help=f"Copy {', '.join(SECRETS)} from .env into Space secrets.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without contacting the Hub.")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    missing = [f for f in SPACE_FILES if not (ROOT / f).is_file()]
    if missing:
        print(f"Missing files: {', '.join(missing)}", file=sys.stderr)
        return 1

    print(f"Space:  {args.space_id}")
    print(f"Files:  {', '.join(SPACE_FILES)}")
    print(f"Secrets: {'will be set from .env' if args.set_secrets else 'not changed'}")
    if args.dry_run:
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set (add it to .env).", file=sys.stderr)
        return 1
    api = HfApi(token=token)

    if not api.repo_exists(args.space_id, repo_type="space"):
        print(f"Creating Space ({'private' if args.private else 'public'}, gradio, {args.hardware}) ...")
        api.create_repo(
            args.space_id,
            repo_type="space",
            space_sdk="gradio",
            space_hardware=args.hardware,
            private=args.private,
        )

    if args.set_secrets:
        for key in SECRETS:
            value = os.environ.get(key)
            if not value:
                print(f"{key} is not set in .env.", file=sys.stderr)
                return 1
            api.add_space_secret(args.space_id, key, value)
            print(f"Set secret {key}")

    commit = api.upload_folder(
        folder_path=str(ROOT),
        repo_id=args.space_id,
        repo_type="space",
        allow_patterns=SPACE_FILES,
        commit_message="Deploy from sec-dense-fts",
    )
    print(f"Uploaded: {commit.commit_url}")
    print(f"App:      https://huggingface.co/spaces/{args.space_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
