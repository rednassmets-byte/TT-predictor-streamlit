# 🔄 How to Update Your Streamlit App

## Quick Reference

**Every time you make changes to your code, just run:**

```bash
git add .
git commit -m "Description of changes"
git push origin master
```

Streamlit Cloud will automatically redeploy! ✨

---

## 📝 Step-by-Step Update Process

### 1. Make Your Changes
Edit any files you want:
- `app.py` - Main application
- `train_model.py` - Regular model training
- `train_model_filtered.py` - Youth model training
- Any other files

### 2. Test Locally (Optional but Recommended)
```bash
streamlit run app.py
```

### 3. Commit & Push

**Option A: Use the Update Script (Easiest)**
- Double-click: `update_app.bat`
- Enter description of changes
- Done!

**Option B: Manual Commands**
```bash
# Add all changed files
git add .

# Commit with a message
git commit -m "Your update description"

# Push to GitHub
git push origin master
```

### 4. Wait for Redeployment
- Streamlit Cloud detects changes automatically
- Redeployment takes 1-2 minutes
- Your app will be updated!

---

## 🎯 Common Update Scenarios

### Update the UI/App Code
```bash
# Edit app.py
git add app.py
git commit -m "Update UI: improved layout"
git push origin master
```

### Retrain and Update Models
```bash
# Retrain models
python train_model.py
python train_model_filtered.py

# Push new model files
git add *.pkl
git commit -m "Update models: improved accuracy"
git push origin master
```

### Update Dependencies
```bash
# Edit requirements.txt
git add requirements.txt
git commit -m "Update dependencies"
git push origin master
```

### Update Documentation
```bash
# Edit README.md or other docs
git add README.md
git commit -m "Update documentation"
git push origin master
```

---

## 🔍 Monitor Deployment

### Check Deployment Status:
1. Go to: https://share.streamlit.io
2. Click on your app
3. Click "Manage app"
4. View "Logs" to see deployment progress

### Deployment Logs Show:
```
Pulling latest changes...
Installing dependencies...
Starting app...
App is live!
```

---

## ⚡ Quick Update Examples

### Example 1: Fix a Bug
```bash
# Fix the bug in app.py
git add app.py
git commit -m "Fix: resolved prediction error for JUN category"
git push origin master
```

### Example 2: Add New Feature
```bash
# Add new feature to app.py
git add app.py
git commit -m "Feature: add export predictions to CSV"
git push origin master
```

### Example 3: Update Multiple Files
```bash
# Make changes to multiple files
git add .
git commit -m "Update: improved models and UI"
git push origin master
```

---

## 🚨 Troubleshooting Updates

### "Nothing to commit"
- You haven't made any changes
- Or changes are already committed

### "Push rejected"
- Someone else pushed changes
- Solution: `git pull origin master` then push again

### "Deployment failed"
- Check Streamlit Cloud logs
- Common issues:
  - Syntax error in code
  - Missing dependency in requirements.txt
  - Model file too large

### App not updating?
- Wait 2-3 minutes (deployment takes time)
- Check if push was successful: `git log`
- Verify on GitHub: https://github.com/rednassmets-byte/TT-predictor-streamlit

---

## 📊 Update Workflow

```
1. Make changes locally
   ↓
2. Test locally (optional)
   ↓
3. git add .
   ↓
4. git commit -m "message"
   ↓
5. git push origin master
   ↓
6. Streamlit Cloud auto-deploys
   ↓
7. App updated! ✅
```

---

## 🎯 Best Practices

### Good Commit Messages:
✅ "Fix: resolved NG over-prediction issue"
✅ "Feature: add junior leniency slider"
✅ "Update: retrained models with new data"
✅ "Docs: updated README with new features"

### Bad Commit Messages:
❌ "update"
❌ "fix"
❌ "changes"
❌ "asdf"

### Before Pushing:
- ✅ Test locally if possible
- ✅ Check for syntax errors
- ✅ Verify model files aren't corrupted
- ✅ Update documentation if needed

---

## 🔄 Rollback to Previous Version

If an update breaks something:

```bash
# See recent commits
git log --oneline

# Rollback to previous commit
git revert HEAD

# Push the rollback
git push origin master
```

Or restore a specific file:
```bash
git checkout HEAD~1 app.py
git commit -m "Rollback: restore previous app.py"
git push origin master
```

---

## 📞 Need Help?

- **Deployment issues**: Check Streamlit Cloud logs
- **Git issues**: Run `git status` to see what's happening
- **Code issues**: Test locally first with `streamlit run app.py`

---

## ✅ Quick Checklist

Before every update:
- [ ] Made changes to code
- [ ] Tested locally (optional)
- [ ] Committed changes with clear message
- [ ] Pushed to GitHub
- [ ] Waited for Streamlit Cloud to redeploy
- [ ] Verified app works

---

**That's it! Updating is as simple as: edit → commit → push → done! 🚀**
