#!/bin/bash

echo "🚀 ChemML Suite - GitHub Upload Helper"
echo "====================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Git repository not found. Please run ./init_git.sh first."
    exit 1
fi

echo "📋 Current repository status:"
git status --short
echo ""

echo "📊 Repository summary:"
echo "   - Files committed: $(git ls-files | wc -l | tr -d ' ')"
echo "   - Last commit: $(git log -1 --format='%h - %s')"
echo "   - Branch: $(git branch --show-current)"
echo ""

echo "🌐 Next steps to upload to GitHub:"
echo ""
echo "1️⃣  Go to GitHub.com and create a new repository:"
echo "    - Repository name: chemml-suite (or your preferred name)"
echo "    - Description: iOS-style ChemML Suite for AutoML chemical prediction"
echo "    - Make it Public or Private (your choice)"
echo "    - DON'T initialize with README (we already have one)"
echo ""

echo "2️⃣  Copy the repository URL from GitHub (it will look like):"
echo "    https://github.com/YOUR_USERNAME/chemml-suite.git"
echo ""

echo "3️⃣  Run these commands in terminal:"
echo "    git remote add origin https://github.com/YOUR_USERNAME/chemml-suite.git"
echo "    git branch -M main"
echo "    git push -u origin main"
echo ""

echo "4️⃣  After upload, deploy on Render.com:"
echo "    - Go to render.com"
echo "    - New → Web Service"
echo "    - Connect your GitHub repository"
echo "    - Render will auto-detect configuration"
echo "    - Click 'Create Web Service'"
echo ""

echo "📁 Your repository includes:"
echo "   ✅ Complete ChemML Suite application"
echo "   ✅ All 4 ML modules (classification, regression, graph models)"
echo "   ✅ iOS-style responsive interface"
echo "   ✅ Render.com deployment configuration"
echo "   ✅ Documentation and setup guides"
echo ""

echo "🎯 Ready for deployment!"
echo ""

# Offer to open GitHub in browser
read -p "🌐 Would you like to open GitHub.com to create a new repository? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v open &> /dev/null; then
        open "https://github.com/new"
        echo "✅ Opening GitHub in browser..."
    elif command -v xdg-open &> /dev/null; then
        xdg-open "https://github.com/new"
        echo "✅ Opening GitHub in browser..."
    else
        echo "Please manually open: https://github.com/new"
    fi
fi

echo ""
echo "💡 Tip: After creating the repository on GitHub, copy the commands"
echo "     shown in the 'push an existing repository' section!"
