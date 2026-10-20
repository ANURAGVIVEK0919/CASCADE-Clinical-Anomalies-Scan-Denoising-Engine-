---
title: CASCADE - Clinical Anomalies & Scan Denoising Engine
emoji: 🩺
colorFrom: indigo
colorTo: purple
sdk: static
pinned: true
---

# CASCADE (Clinical Anomalies & Scan Denoising Engine) 🩺

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Platform](https://img.shields.io/badge/Architecture-ARM64%20%2F%20CPU-2496ED?style=for-the-badge&logo=arm&logoColor=white)](https://arm.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 📖 Dual-Tier Zero-Subscription Deployment Architecture

This Hugging Face Space operates as a **zero-subscription static frontend** using the **Hugging Face Static SDK** (`sdk: static`).

```
+-------------------------------------------------------------------------+
| Hugging Face Static Space (https://huggingface.co/spaces/ANURAG1909/cascade) |
| - Runs 100% free with zero GPU/ZeroGPU requirements                    |
| - Serves index.html with dark PACS theme and responsive secure iframe    |
+-------------------------------------------------------------------------+
                                    |
                                    | HTTPS (Iframe Embedding)
                                    v
+-------------------------------------------------------------------------+
| Self-Hosted CPU Backend (OCI Always Free Ampere A1 VM / Cloud Run)     |
| - Nginx Reverse Proxy with Let's Encrypt SSL/TLS                       |
| - Python 3.11 + Gunicorn (2 workers, 4 threads)                        |
| - CPU-Only Neural Inference (U-Net Denoising + ResNet50V2 Diagnostics) |
| - All 3 Neural Model Weights Verified Pre-Flight                        |
+-------------------------------------------------------------------------+
```

### Why This Architecture?
* **Zero Platform Hardware Costs**: Standard Docker Spaces previously triggered ZeroGPU configuration errors or CPU-Basic payment holds on this account. Migrating the Space to `sdk: static` permanently eliminates Hugging Face hardware fees while keeping the canonical Space URL (`ANURAG1909/cascade`) active and accessible.
* **Full Computational Power**: Heavy convolutional tensor processing and multi-agent clinical workflows execute on a dedicated Always Free ARM64 compute instance (up to 4 OCPU, 24 GB RAM on OCI Ampere A1).

---

## ⚙️ Configuration & Deployment Instructions

### Step 1: Deploy the CPU Backend
Follow the complete deployment steps in:
[`deploy/free_cpu_backend/DEPLOYMENT_GUIDE_OCI_ARM64.md`](../free_cpu_backend/DEPLOYMENT_GUIDE_OCI_ARM64.md)

1. Provision an **OCI Always Free Ampere A1 VM** (`VM.Standard.A1.Flex`, Ubuntu 22.04/24.04 ARM64).
2. Install dependencies using `deploy/free_cpu_backend/requirements-arm64.txt`.
3. Transfer the three verified local `.keras` artifacts to `/opt/cascade/`.
4. Configure Nginx with a free HTTPS subdomain (e.g. `https://cascade-backend.duckdns.org` via Let's Encrypt / Certbot).
5. Start the backend with the provided `cascade.service` systemd unit.

### Step 2: Configure the Static Frontend
In `index.html`, update the placeholder constant with your live backend domain:

```javascript
// deploy/hf_static_frontend/index.html
const BACKEND_URL = "https://cascade-backend.duckdns.org";
```

> [!CAUTION]
> **Do not publish the unconfigured placeholder to Hugging Face.** Browsers will display a configuration warning screen until a valid HTTPS backend domain is supplied.

### Step 3: Publish Frontend to Hugging Face Space
Once the backend is live and verified over HTTPS, deploy only the two frontend files (`README.md` and `index.html`) to the `ANURAG1909/cascade` Hugging Face Space repository:

```bash
git remote add hf https://huggingface.co/spaces/ANURAG1909/cascade
# (Deploy only the static frontend assets; do not upload backend code to the Space)
```

> [!IMPORTANT]
> **Preserve Existing Remote Model Files**: The Hugging Face Space repository already contains `medical_ct_denoiser_full.keras` and `medical_ct_diagnostic_model.keras` tracked via Git LFS. When committing `index.html` and `README.md`, do **not** run destructive cleanups, delete remote LFS files, or purge existing model artifacts. The static Space frontend will simply ignore these binary files and serve `index.html`.

---

## 🔒 Security & Privacy Guarantees

* **Framing Security Policy**: The backend Flask application enforces a scoped Content Security Policy:
  ```http
  Content-Security-Policy: frame-ancestors 'self' https://huggingface.co https://*.hf.space;
  ```
  This permits embedding exclusively within official Hugging Face origins while blocking unauthorized third-party framing or clickjacking.
* **Mixed-Content Protection**: Both the Hugging Face Space and backend domain operate strictly over HTTPS, ensuring no browser mixed-content blocks occur.
* **Ephemeral In-Memory Processing**: Uploaded CT scans are stored temporarily in `/tmp/cascade_uploads` and are deleted immediately in `try/finally` blocks upon completion or failure.
* **No Direct Patient Identifiers**: DICOM metadata headers are stripped of patient names, IDs, and dates of birth.

---

## ⚠️ Regulatory & Clinical Safety Notice

> [!IMPORTANT]
> **CASCADE is an exploratory engineering prototype and research benchmark platform.**
> - It is **not** an FDA-cleared, CE-marked, or certified medical device (SaMD).
> - It must **not** be used for primary diagnostic interpretation, treatment planning, or acute emergency triage.
> - Emergency triage status remains strictly `Not Assessed (No Certified Pathology Classifier)`.
