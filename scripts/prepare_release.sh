#!/bin/bash
# Script to prepare the repository for a GitHub release

set -e

VERSION=${1:-"v1.0.0"}
echo "Preparing release: $VERSION"

# Check if git is initialized
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: Music Genre Classification Project"
fi

# Create a tag
echo "Creating git tag: $VERSION"
git tag -a "$VERSION" -m "Release $VERSION: Music Genre Classification"

echo ""
echo "Release preparation complete!"
echo ""
echo "To push to GitHub:"
echo "  1. Create a repository on GitHub"
echo "  2. Add remote: git remote add origin <your-repo-url>"
echo "  3. Push: git push -u origin main"
echo "  4. Push tags: git push origin $VERSION"
echo ""
echo "Or create a release on GitHub and attach the tag $VERSION"

