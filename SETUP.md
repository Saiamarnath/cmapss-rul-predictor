# Setup guide — Kaggle → GitHub → Render

## One-time setup (do this before first run)

### 1. Create the GitHub repo

```bash
# On GitHub.com: New repository → name it e.g. cmapss-rul-predictor
# Then clone and push the app files (everything except models/):
git init
git add app.py src/pipeline.py requirements.txt Procfile .gitignore .streamlit/config.toml README.md
git commit -m "init: app + pipeline"
git remote add origin https://github.com/YOUR_USERNAME/cmapss-rul-predictor.git
git push -u origin main
```

### 2. Create a GitHub Personal Access Token

1. GitHub → Settings → Developer settings → Personal access tokens → **Tokens (classic)**
2. Generate new token → check **`repo`** scope → copy it (shown only once)

### 3. Add Kaggle Secrets

In your Kaggle notebook:  
Settings (right panel) → **Add-ons → Secrets → Add secret**

| Label | Value |
|---|---|
| `GITHUB_TOKEN` | your PAT from step 2 |
| `GITHUB_REPO` | `yourname/cmapss-rul-predictor` |

---

## Every training run

1. Open `kaggle_train_and_push.ipynb` on Kaggle
2. Attach the NASA CMAPSS dataset (`/kaggle/input/nasa-data`)
3. Enable **GPU T4 × 2** accelerator
4. **Run All** — takes ~5 min for all 4 FD subsets
5. Cell 6 automatically pushes `models/FD00{1-4}/` to GitHub
6. Render auto-deploys on the new commit (if auto-deploy is on)

---

## Deploy to Render (first time)

1. Go to [render.com](https://render.com) → New → **Web Service**
2. Connect your GitHub repo
3. Settings:
   - **Runtime**: Python 3
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: *(auto-detected from Procfile)*
   - **Instance type**: Free (512 MB RAM is enough for CPU inference)
4. Deploy → done.

After that, every `git push` from Kaggle Cell 6 triggers a re-deploy automatically.

---

## Repo structure after first training run

```
cmapss-rul-predictor/
├── app.py
├── src/
│   └── pipeline.py
├── models/                   ← pushed from Kaggle
│   ├── FD001/
│   │   ├── model.pt
│   │   ├── kmeans.pkl
│   │   ├── scalers.pkl
│   │   ├── features.pkl
│   │   ├── config.pkl
│   │   └── metrics.json
│   ├── FD002/ ...
│   ├── FD003/ ...
│   └── FD004/ ...
├── kaggle_train_and_push.ipynb
├── requirements.txt
├── Procfile
├── .gitignore
└── .streamlit/
    └── config.toml
```

---

## Inference CSV format

Same format as CMAPSS test files. Whitespace or comma separated, no header.  
26 columns: `id  cycle  op1  op2  op3  s1 … s21`  
(or 24 columns without `id` and `cycle` — the app handles both)  
Must have **≥ 50 rows** (one full window).

To use a CMAPSS test file directly: upload `test_FD001.txt` and pick FD001 as the model.
