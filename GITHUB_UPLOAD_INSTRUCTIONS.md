# Getting this onto GitHub and running it in the cloud

This folder is plain files (no `.git` history bundled in) — built this way
specifically so you can use GitHub's **Add file → Upload files** button in
the browser, no command line required.

## 1. Create the repo on GitHub

github.com → New repository → name it (e.g. `redrob-ranker`) → **public or
private, your choice** → leave "Initialize with README" **unchecked** →
Create repository.

## 2. Upload everything

On the new (empty) repo page: **Add file → Upload files**, then drag in
every file and folder from this zip (select-all and drag the whole
extracted folder works in Chrome/Edge; Safari may need folders added one at
a time). Scroll down, write a commit message like "Initial commit", and
click **Commit changes**.

Two things to check after uploading:
- `.gitignore` and `.github/` are dot-folders/files — make sure your file
  manager isn't hiding them before you drag-select everything (on
  Mac/Linux, enable "show hidden files"; on Windows they're not hidden by
  default).
- GitHub's web uploader has a ~25MB-per-file limit and works best under
  ~100 files per batch. Everything in this zip is well under that — the
  largest single file is the slide deck (`submission_deck/*.pptx`, ~6MB).

If you'd rather use the command line instead of the browser uploader:

```bash
cd redrob-ranker          # the extracted folder
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/redrob-ranker.git
git push -u origin main
```

## 3. Add the large dataset file (kept local, not uploaded to GitHub)

`data/candidates.jsonl` — the organizer's 100K-candidate pool, ~465MB — is
**not** included in this zip. It's the hackathon organizers' data, not
yours to redistribute, and it's far past GitHub's comfortable size limits
anyway. Drop the file the organizers gave you into `data/candidates.jsonl`
on whatever machine/cloud environment you actually run `rank.py` from.
`.gitignore` already excludes that path, so it's safe to have it sitting in
a local clone without risking an accidental commit.

## 4. Run it in the cloud

**Option A — GitHub Codespaces (zero setup, runs entirely in-browser):**
On your repo page: **Code → Codespaces → Create codespace on main**. Once
it boots, in the integrated terminal:
```bash
pip install -r requirements.txt
pip install pytest
python -m pytest tests/ -v          # 26 should pass
```
Upload `candidates.jsonl` into the Codespace (drag into the file explorer,
or `gh codespace cp` from your machine), then:
```bash
python src/rank.py --candidates ./data/candidates.jsonl --out ./artifacts/pranjal_redrob_submission.csv
python data/validate_submission.py ./artifacts/pranjal_redrob_submission.csv
```

**Option B — Deploy the Streamlit sandbox (small-sample demo, public URL):**
Go to [share.streamlit.io](https://share.streamlit.io) → New app → pick
this repo → set the main file to `sandbox/app.py` and the requirements
file to `sandbox/requirements.txt` → Deploy. This gives you a public
`<your-app>.streamlit.app` link that runs the bundled
`data/sample_candidates.json` demo (the full 100K-candidate run is meant
for Option A or your own machine, not the free-tier sandbox — see
`sandbox/app.py`'s docstring).

**Option C — GitHub Actions CI (already wired up):**
`.github/workflows/ci.yml` runs the test suite automatically on every push,
no setup needed — check the "Actions" tab on your repo after the first
upload/push to see it run.

## 5. Update two placeholders once the repo/sandbox exist

- `submission_metadata.yaml` — `github_repo:` and `sandbox_link:` fields,
  plus your email/phone
- `submission_deck/Pranjal_Redrob_Idea_Submission.pptx` — slide 10
  ("Submission Assets")

Everything else (code, 26 passing tests, CI workflow, README, the filled-in
slide deck, the final ranked submission CSV) is already complete and was
verified working from a clean copy before this zip was built.
