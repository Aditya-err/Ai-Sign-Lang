#!/usr/bin/env python3
"""
push_to_github.py
==================
Run this script once from inside your project folder to:
  1. Create a new public GitHub repo called 'signlens-ai'
  2. Push the local git history to it

Usage:
    cd "D:\project\Ai_sign language\files"
    python push_to_github.py --token YOUR_GITHUB_TOKEN

How to get a token:
    GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
    → Generate new token → check 'repo' scope → copy the token
"""

import subprocess
import sys
import argparse


def run(cmd: list, cwd=None):
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"ERROR running: {' '.join(cmd)}")
        print(result.stderr)
        sys.exit(1)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token")
    parser.add_argument("--repo",  default="signlens-ai", help="Repo name (default: signlens-ai)")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    args = parser.parse_args()

    try:
        from github import Github, GithubException
    except ImportError:
        print("Installing PyGithub…")
        subprocess.run([sys.executable, "-m", "pip", "install", "PyGithub", "-q"])
        from github import Github, GithubException

    g    = Github(args.token)
    user = g.get_user()
    print(f"Logged in as: {user.login}")

    # Create repo
    try:
        repo = user.create_repo(
            name        = args.repo,
            description = "Real-time AI sign language detector — MediaPipe + TensorFlow LSTM + PyWebView UI",
            private     = args.private,
            auto_init   = False,
        )
        print(f"Created repo: {repo.html_url}")
    except GithubException as e:
        if "already exists" in str(e):
            repo = user.get_repo(args.repo)
            print(f"Repo already exists: {repo.html_url}")
        else:
            print(f"GitHub error: {e}")
            sys.exit(1)

    remote_url = f"https://{args.token}@github.com/{user.login}/{args.repo}.git"

    # Set remote
    try:
        run(["git", "remote", "add", "origin", remote_url])
    except SystemExit:
        run(["git", "remote", "set-url", "origin", remote_url])

    # Push
    print("Pushing to GitHub…")
    run(["git", "push", "-u", "origin", "main"])

    # Print clean URL (without token)
    print()
    print("=" * 50)
    print(f"  SUCCESS!")
    print(f"  Repo: {repo.html_url}")
    print("=" * 50)


if __name__ == "__main__":
    main()
