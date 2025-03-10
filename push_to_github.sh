#!/bin/bash
# Script to push the Rephysco repository to GitHub

# Check if the GitHub URL is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <github_repo_url>"
    echo "Example: $0 https://github.com/yourusername/rephysco.git"
    exit 1
fi

GITHUB_URL=$1

# Add the remote repository
git remote add origin $GITHUB_URL

# Set the main branch
git branch -M main

# Push to GitHub
git push -u origin main

echo "Repository pushed to GitHub successfully!"
echo "You can now visit $GITHUB_URL to see your repository." 