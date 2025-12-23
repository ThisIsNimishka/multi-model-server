# How to Push to GitHub

## Prerequisites
- Git installed on your system
- GitHub account
- GitHub CLI (optional but recommended)

## Step 1: Create a GitHub Repository

### Option A: Using GitHub Web Interface
1. Go to https://github.com/new
2. Repository name: `multi-model-server`
3. Description: "Production-ready multi-model LLM server with OpenAI-compatible API"
4. Choose: Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we have these already)
6. Click "Create repository"

### Option B: Using GitHub CLI
```bash
gh repo create multi-model-server --public --description "Production-ready multi-model LLM server with OpenAI-compatible API"
```

## Step 2: Extract and Navigate to the Repository

```bash
# Extract the zip file
# Navigate to the extracted folder
cd multi-model-server
```

## Step 3: Configure Git (if not already done)

```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

## Step 4: Update Repository URLs

Replace `YOUR_USERNAME` in these files with your actual GitHub username:
- `README.md` (multiple occurrences)
- `pyproject.toml` (project.urls section)

You can use find and replace:
```bash
# Windows PowerShell
(Get-Content README.md) -replace 'YOUR_USERNAME', 'your-actual-username' | Set-Content README.md
(Get-Content pyproject.toml) -replace 'YOUR_USERNAME', 'your-actual-username' | Set-Content pyproject.toml

# Or manually edit the files
```

## Step 5: Add Remote and Push

```bash
# Add the remote repository
git remote add origin https://github.com/YOUR_USERNAME/multi-model-server.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 6: Verify

1. Go to `https://github.com/YOUR_USERNAME/multi-model-server`
2. Check that all files are uploaded
3. Verify the README renders correctly
4. Check that GitHub Actions CI runs (under Actions tab)

## Optional: Add Topics/Tags

On your GitHub repository page:
1. Click the gear icon next to "About"
2. Add topics: `llm`, `ai`, `vllm`, `openai-api`, `multi-gpu`, `python`, `machine-learning`

## Optional: Enable GitHub Pages for Documentation

1. Go to Settings > Pages
2. Source: Deploy from a branch
3. Branch: main, folder: / (root)
4. Save

## Troubleshooting

### Authentication Issues
```bash
# Use GitHub CLI for authentication
gh auth login

# Or use Personal Access Token
# Generate at: https://github.com/settings/tokens
```

### Push Rejected
```bash
# If push is rejected, pull first (only if repo was initialized with files)
git pull origin main --allow-unrelated-histories
git push origin main
```

### Large File Issues
If you accidentally added model files:
```bash
git rm --cached path/to/large/file
git commit -m "Remove large file"
git push
```

## Next Steps After Upload

1. ⭐ Star your own repo
2. 📝 Create a release (v1.0.0)
3. 🏷️ Add topics for discoverability
4. 📢 Share with the community!
