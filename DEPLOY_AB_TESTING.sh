#!/bin/bash
# Deployment script for A/B Testing integration
# Run this from the Portfolio directory

echo "🚀 Deploying A/B Testing Project to GitHub..."
echo ""

# Navigate to Portfolio directory
cd /Workspace/Users/goladosurahman@gmail.com/Portfolio

# Check git status
echo "📋 Checking current status..."
git status

echo ""
echo "📦 Files to be committed:"
echo "   - app.py (updated with A/B Testing page)"
echo "   - AB Testing/ (new folder with data and docs)"
echo ""

# Stage all changes
echo "➕ Staging changes..."
git add .

# Commit with descriptive message
echo "💾 Committing changes..."
git commit -m "Add A/B Testing Analysis project

- Integrated comprehensive A/B testing analysis into portfolio
- Added new dedicated page with 6 interactive sections
- Includes frequentist, Bayesian, and Monte Carlo analyses
- Demonstrates peeking problem (4.6x false positive inflation)
- Updated Projects Summary with A/B Testing showcase
- Added 290K visitor dataset and documentation
- Updated technical skills section"

# Push to main branch
echo "🚢 Pushing to GitHub..."
git push origin main

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🔗 Your Streamlit app will auto-deploy from GitHub"
echo "📱 Check your Streamlit dashboard for deployment status"
echo ""
echo "Next: Test the live app once deployed:"
echo "   1. Navigate to Home page"
echo "   2. Click 'A/B Testing Analysis' button"
echo "   3. Verify all 6 sections load correctly"
echo "   4. Check visualizations render properly"
echo ""
