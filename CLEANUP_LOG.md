# CLEANUP LOG

## A: Secrets
* **.env Git Status:**
  * Tracked in git: **No** (`git ls-files .env` returned empty).
  * Ever committed: **No** (`git log --all --oneline -- .env` returned empty).

* **deploy_hf.py Uploads:**
  * Uploads the entire root directory (`folder_path="."`).
  * **Includes:** `data/` and `output/` (which could cause size/limit issues).
  * **Excludes:** `*.env*`, preventing `.env` from being uploaded.

* **create_kaggle_zip.py Zips:**
  * Only iterates over `data/separated_scans` and `data/unified_dataset`.
  * **Excludes:** `.env`, `output/`, and all other root files.

* **data.zip Contents:**
  * `tar -tf data.zip | Select-String '\.env$'` was empty.
  * `.env` is **NOT** included in `data.zip`.

* **Hardcoded Secrets Found (via grep):**
  * `.env`:9 - `AWS_ACCESS_KEY_ID=AKIA****`
  * `.env`:10 - `AWS_SECRET_ACCESS_KEY=****`
  * `AUDIT_REPORT.md`:134 - `AWS_ACCESS_KEY_ID=AKIA****` and `AWS_SECRET_ACCESS_KEY=****` (Note: I have manually masked these in the report).
