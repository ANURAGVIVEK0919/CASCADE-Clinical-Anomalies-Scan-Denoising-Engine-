# CASCADE — OCI Always Free ARM64 CPU Backend Deployment Guide

This guide documents the technical procedure for deploying the **CASCADE Clinical Anomalies & Scan Denoising Engine** backend on an **Oracle Cloud Infrastructure (OCI) Always Free Ampere A1 (ARM64)** virtual machine, paired with a zero-cost **Hugging Face Static Space** frontend.

---

## 1. Architecture & Cost Model Overview

```mermaid
graph LR
    User[Clinician / Researcher Browser] -->|1. Visits Space URL| HF[Hugging Face Static Space<br/>anurag1909-cascade.hf.space]
    HF -->|2. Renders Iframe| User
    User -->|3. HTTPS API / Image Uploads| OCI[OCI Ampere A1 ARM64 VM<br/>Nginx Reverse Proxy + Gunicorn]
    OCI -->|4. Strict Model Verification| Models[(Verified Local .keras Artifacts)]
    OCI -->|5. CPU-Only Denoising & Diagnostic Inference| User
```

* **Zero-Subscription Goal**: The Hugging Face Space runs on the **Static SDK** (`sdk: static`), which requires no Docker container builds, no GPU resources, and zero paid Hugging Face subscriptions. The computationally intensive neural inference executes on an OCI Always Free ARM64 compute instance.
* **Separation of Concerns**: User interactions occur at `https://huggingface.co/spaces/ANURAG1909/cascade`, while form submissions, file uploads, and model evaluations execute against the secure HTTPS backend domain.

---

## 2. OCI Free Tier Prerequisites & Operational Realities

### Account Requirements & Eligibility
1. **Oracle Cloud Account**: Register for an Oracle Cloud account at [cloud.oracle.com](https://cloud.oracle.com/).
2. **Credit Card Verification**: Oracle requires payment card details during registration for identity verification and fraud prevention. The registration process may initiate a temporary authorization charge or hold according to Oracle and issuing bank policies, but no specific authorization amount or reversal timeframe is guaranteed.
3. **Always Free-Only Tenancy Ceiling vs. Recommended Instance Shape**:
   * **Strict Always Free-Only Allowance**: For a strict Always Free-only account setup (without active Free Trial promotional credits or paid account upgrades), the official Oracle Always Free allowance provides up to **1,500 OCPU-hours and 9,000 GB-hours per month** for Ampere A1 Compute (`VM.Standard.A1.Flex`), which equates continuously to an allowance of **2 OCPUs and 12 GB RAM** (plus up to **200 GB** total boot and block storage).
   * **Distinction from Trial / Paid Tiers**: While promotional Free Trial credits or upgraded Pay-As-You-Go accounts may cite or access larger pooled allocations (up to 3,000 OCPU-hours and 18,000 GB-hours per month / 4 OCPUs and 24 GB RAM), a strict free-only account is not automatically entitled to a 4-OCPU / 24-GB instance and must remain within the verified 1,500 OCPU-hour / 9,000 GB-hour monthly free allowance.
   * **Verified Free-Only Single-Instance Recommendation**: We configure exactly **2 OCPUs and 12 GB RAM** for the CASCADE backend VM. This configuration stays strictly within the verified free-only Ampere A1 monthly allowance (1,500 OCPU-hours and 9,000 GB-hours per month). While keeping the compute allocation within Always Free limits is necessary to avoid compute charges, it does not guarantee that the overall account incurs zero charges if other resources (such as block volumes exceeding the 200 GB total allowance, paid load balancers, or excess outbound bandwidth) are provisioned. In addition, actual ARM64 model loading memory footprint, peak inference RAM, and latency have not yet been empirically profiled on target ARM64 hardware and require live measurement.
   * **Home Region Restriction**: Always Free compute instances can **only** be launched in your tenancy's designated **Home Region** (selected permanently at signup). They cannot be launched in secondary subscribed regions.

### Critical Operational Realities & Policy Constraints
* **Regional Capacity Constraints & No SLA**: In certain OCI Home Regions, Ampere A1 host capacity can experience shortages ("Out of host capacity" error during creation). Always Free resources carry no guaranteed Service Level Agreement (SLA). If encountered, attempt provisioning in alternate availability domains or retry during off-peak hours. **Never upgrade to a paid account or provision paid Intel/AMD shapes to overcome capacity shortages.**
* **OCI Idle Instance Reclamation Policy**: Oracle's terms explicitly permit the platform to automatically reclaim (stop/terminate) Always Free compute instances if metrics over a rolling 7-day period fall below:
  - CPU utilization (95th percentile): **< 20%**
  - Network utilization: **< 20%**
  - Memory utilization: **< 20%**
  *Policy Compliance*: **Do not attempt to artificially bypass or evade this policy with synthetic cron jobs, fake ping traffic, or artificial CPU loops.** If your instance is reclaimed due to prolonged inactivity, you can restart it from the OCI Console.
* **No Guaranteed 100% Uptime**: Always Free compute instances offer no enterprise Service Level Agreement (SLA). Host maintenance, capacity reclamation, or account tier reviews can cause intermittent downtime. Always Free hosting is ideal for open-source showcases, portfolio demonstrations, and research benchmarking, but is not designed for mission-critical clinical production.
* **Cost Avoidance Discipline**: Always Free compute instances avoid compute charges only when kept within allowance boundaries. Avoiding account charges requires strict administrative discipline: you must not provision paid compute shapes, exceed 200 GB total storage across all boot and block volumes, attach paid load balancers, or exceed outbound data transfer quotas. Account billing is governed strictly by Oracle Cloud metering policies.

---

## 3. Provisioning the OCI Ampere A1 VM

1. Log in to the **OCI Console**.
2. Navigate to **Compute** &rarr; **Instances** &rarr; **Create Instance**.
3. **Configuration**:
   * **Name**: `cascade-cpu-backend`
   * **Placement**: Select an Availability Domain (e.g., `AD-1`).
   * **Image**: **Canonical Ubuntu 22.04 LTS (aarch64)** or **Ubuntu 24.04 LTS (aarch64)**.
   * **Shape**: Click *Change Shape* &rarr; select **Ampere (ARM)** &rarr; **VM.Standard.A1.Flex**.
     - OCPUs: **2** (matching the 1,500 OCPU-hour/month Always Free allowance)
     - Memory: **12 GB** (matching the 9,000 GB-hour/month Always Free allowance)
   * **Networking**:
     - Create new Virtual Cloud Network (VCN) or select existing.
     - Ensure **Assign a public IPv4 address** is checked.
   * **Add SSH Keys**: Upload or paste your public SSH key (`id_ed25519.pub` or `id_rsa.pub`).
   * **Boot Volume**: Default (50 GB) is well within the 200 GB free tier.
4. Click **Create** and wait for the instance status to transition to `RUNNING`. Note its public IP address (`VM_PUBLIC_IP`).

---

## 4. Dual-Layer Firewall Configuration

Incoming traffic must be permitted through both OCI's Cloud Security Lists and the Ubuntu host firewall.

### Layer 1: OCI VCN Ingress Rules
1. In the OCI Console, navigate to **Networking** &rarr; **Virtual Cloud Networks**.
2. Select your VCN &rarr; click on **Security Lists** &rarr; select **Default Security List**.
3. Click **Add Ingress Rules**:
   * **Source CIDR**: `0.0.0.0/0`
   * **IP Protocol**: `TCP`
   * **Destination Port Range**: `80, 443`
   * **Description**: `HTTP and HTTPS public ingress for CASCADE backend`
4. Confirm SSH (`TCP 22`) is also permitted.

### Layer 2: Host OS Firewall (UFW)
SSH into the instance:
```bash
ssh -i /path/to/private_key ubuntu@VM_PUBLIC_IP
```
Configure and enable UFW:
```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp comment 'SSH'
sudo ufw allow 80/tcp comment 'HTTP'
sudo ufw allow 443/tcp comment 'HTTPS'
sudo ufw --force enable
sudo ufw status verbose
```

---

## 5. Host Setup & ARM64 Environment Preparation

Update system packages and install necessary system libraries:
```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    nginx \
    certbot \
    python3-certbot-nginx \
    libgl1 \
    libglib2.0-0 \
    htop
```

Clone the repository and set up a clean Python virtual environment:
```bash
sudo mkdir -p /opt/cascade
sudo chown -R ubuntu:ubuntu /opt/cascade
cd /opt/cascade

# Clone repository
git clone https://github.com/ANURAGVIVEK0919/CASCADE-Clinical-Anomalies-Scan-Denoising-Engine-.git .

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install verified ARM64 dependencies
pip install --upgrade pip
pip install -r deploy/free_cpu_backend/requirements-arm64.txt
```

> [!NOTE]
> `requirements-arm64.txt` installs `tensorflow==2.21.0` using official pre-compiled `manylinux_2_27_aarch64` wheels. No local compilation or C++ build tools are required.

---

## 6. Model Artifact Transfer & Verification

The GitHub repository ignores large `.keras` model checkpoints via `.gitignore`. You must securely copy model weights from your local development machine to the VM before starting services:
* **Mandatory Baseline Models**: `medical_ct_denoiser_full.keras` and `medical_ct_diagnostic_model.keras` are strictly required by the startup integrity gate (`verify_backend_models.py`). Application startup will fail if either baseline model is absent or fails checksum validation.
* **Optional Research Candidate Model**: `candidate_unet_residual_edge_v3.keras` is optional for initial service startup. If omitted, the server boots normally, but the `/api/research/denoise_dicom` research endpoint will return HTTP 503 `UNAVAILABLE` until this candidate checkpoint is transferred to its expected nested path.

### Transfer Commands (Run from Local Development Machine)
```powershell
# From local Windows development machine in PowerShell:
$VM_IP = "YOUR_OCI_VM_PUBLIC_IP"
$KEY = "C:\Users\vivek\.ssh\id_rsa" # Adjust path to your private key

# Create remote experiment directory
ssh -i $KEY ubuntu@$VM_IP "mkdir -p /opt/cascade/artifacts/experiments/unet_residual_edge_v3/run_001"

# Copy baseline models
scp -i $KEY medical_ct_denoiser_full.keras ubuntu@${VM_IP}:/opt/cascade/
scp -i $KEY medical_ct_diagnostic_model.keras ubuntu@${VM_IP}:/opt/cascade/

# Copy Residual U-Net v3 candidate model
scp -i $KEY artifacts/experiments/unet_residual_edge_v3/run_001/candidate_unet_residual_edge_v3.keras ubuntu@${VM_IP}:/opt/cascade/artifacts/experiments/unet_residual_edge_v3/run_001/
```

### Run Artifact Verification Gate on VM
```bash
cd /opt/cascade
source venv/bin/activate
python deploy/free_cpu_backend/verify_backend_models.py /opt/cascade
```

**Model Verification Rules & Expected Output**:
* **Mandatory Models for Service Startup**:
  - `medical_ct_denoiser_full.keras`: `23,388,275` bytes | `ef84c7d1d808000a6e52b36cb1fdcbd7c929621d061cafbe61f822c2cf767842` (Required for Gunicorn/systemd boot and web denoising)
  - `medical_ct_diagnostic_model.keras`: `256,877,707` bytes | `a68b61a14c6ac170ef37a43e7447edeb90fafb3439ff0e0652ec33e1f4b64d41` (Required for Gunicorn/systemd boot and organ classification)
  If either mandatory model is missing or fails its checksum, `verify_backend_models.py` exits with code 1 (`[FAIL]`), aborting service startup.
* **Optional Research Candidate Model**:
  - `candidate_unet_residual_edge_v3.keras`: `1,936,708` bytes | `bc396511dfbea8d1f853b2067ededd801b3405b11caeb25157445d581fb1fcea`
  Located at: `artifacts/experiments/unet_residual_edge_v3/run_001/candidate_unet_residual_edge_v3.keras`.
  Classified as optional for process startup (`mandatory: False`). If omitted, the verifier issues a warning (`[WARN] Optional research candidate NOT FOUND. Fallback mode will be active.`) and returns exit code 0 (`[SUCCESS]`), allowing the primary web application to launch. However, calling `/api/research/denoise_dicom` will return HTTP 503 `UNAVAILABLE` until this candidate file is uploaded.
* Result: `[SUCCESS] All verified model artifacts match expected digests.` (Only verifies artifacts that are present; mandatory models must pass).

---

## 7. Systemd Service Setup

Create a systemd unit file to manage the CASCADE backend process automatically with logging and restart-on-failure:

```bash
sudo nano /etc/systemd/system/cascade.service
```

Paste the following configuration:
```ini
[Unit]
Description=CASCADE Clinical Anomalies & Scan Denoising Engine (Gunicorn)
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/cascade
Environment="PATH=/opt/cascade/venv/bin:/usr/local/bin:/usr/bin"
Environment="CUDA_VISIBLE_DEVICES=-1"
Environment="PYTHONDONTWRITEBYTECODE=1"
Environment="PYTHONUNBUFFERED=1"
Environment="CASCADE_UPLOAD_FOLDER=/tmp/cascade_uploads"
ExecStartPre=/opt/cascade/venv/bin/python /opt/cascade/deploy/free_cpu_backend/verify_backend_models.py /opt/cascade
ExecStart=/opt/cascade/venv/bin/gunicorn \
    --bind 127.0.0.1:7860 \
    --workers 2 \
    --threads 4 \
    --timeout 180 \
    --access-logfile /var/log/cascade/access.log \
    --error-logfile /var/log/cascade/error.log \
    app:app

Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Create logging directory and start service:
```bash
sudo mkdir -p /var/log/cascade
sudo chown -R ubuntu:ubuntu /var/log/cascade

sudo systemctl daemon-reload
sudo systemctl enable cascade
sudo systemctl start cascade
sudo systemctl status cascade
```

### Alternative: Containerized Execution with Docker (`Dockerfile.arm64`)

If you prefer containerized deployment over native systemd:

1. **Build Context & Model Weight Consideration**:
   * Note that `*.keras` model weights are ignored by git (`.gitignore`) and are NOT pulled during `git clone`.
   * Because `Dockerfile.arm64` executes `COPY . /app`, you must ensure that the two mandatory baseline models (`medical_ct_denoiser_full.keras`, `medical_ct_diagnostic_model.keras`) and optionally the candidate artifact under `artifacts/experiments/unet_residual_edge_v3/run_001/candidate_unet_residual_edge_v3.keras` are copied into the repository working tree **before** running `docker build`.
   * Build the ARM64 image on the VM:
     ```bash
     docker build -f deploy/free_cpu_backend/Dockerfile.arm64 -t cascade-arm64:latest .
     ```
2. **Launch Container**:
   * **Option A: With Baked-in Weights**:
     ```bash
     docker run -d \
         --name cascade-backend \
         --restart unless-stopped \
         -p 127.0.0.1:7860:7860 \
         -e PORT=7860 \
         cascade-arm64:latest
     ```
   * **Option B: With Host Volume Mounts** (keeps image weight-free):
     ```bash
     docker run -d \
         --name cascade-backend \
         --restart unless-stopped \
         -p 127.0.0.1:7860:7860 \
         -v /opt/cascade/medical_ct_denoiser_full.keras:/app/medical_ct_denoiser_full.keras \
         -v /opt/cascade/medical_ct_diagnostic_model.keras:/app/medical_ct_diagnostic_model.keras \
         -v /opt/cascade/artifacts/experiments/unet_residual_edge_v3/run_001:/app/artifacts/experiments/unet_residual_edge_v3/run_001 \
         -e PORT=7860 \
         cascade-arm64:latest
     ```
3. **Container Lifecycle & Health Checks**:
   * **Health Check**: `curl -I http://127.0.0.1:7860/metrics`
   * **Live Logs**: `docker logs -f cascade-backend`
   * **Restart**: `docker restart cascade-backend`
   * **Graceful Shutdown**: `docker stop cascade-backend`

---

## 8. Free Subdomain, Nginx Reverse Proxy & HTTPS

To allow secure iframe embedding inside Hugging Face Spaces, the backend must be accessible over HTTPS with a valid SSL/TLS certificate.

### Free Subdomain Options
1. **DuckDNS** (Free indefinitely):
   * Create an account at [duckdns.org](https://www.duckdns.org/).
   * Create a domain: e.g. `cascade-backend.duckdns.org`.
   * Point it to your `VM_PUBLIC_IP`.
2. **Cloudflare Tunnel / Custom Domain**:
   * Alternatively, use your own domain or Cloudflare free tier.

### Nginx Reverse Proxy Configuration
Create an Nginx server block:
```bash
sudo nano /etc/nginx/sites-available/cascade
```

Paste the following configuration (replace `cascade-backend.duckdns.org` with your actual domain):
```nginx
# Rate-limiting zones: prevent CPU saturation from unauthenticated upload flood
limit_req_zone $binary_remote_addr zone=cascade_api_limit:10m rate=15r/m;
limit_conn_zone $binary_remote_addr zone=cascade_conn_limit:10m;

server {
    listen 80;
    server_name cascade-backend.duckdns.org;

    # Maximum payload size matches app 32 MB limit
    client_max_body_size 32M;

    # General endpoints and dark PACS dashboard viewer
    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_connect_timeout 180s;
        proxy_send_timeout 180s;
        proxy_read_timeout 180s;

        # IMPORTANT: Do NOT inject 'X-Frame-Options: SAMEORIGIN' or 'DENY'.
        # The CASCADE Flask app sends a scoped Content-Security-Policy:
        # frame-ancestors 'self' https://huggingface.co https://*.hf.space;
        # which permits embedding exclusively within official Hugging Face Spaces.
    }

    # Heavy neural inference endpoints: enforce abuse prevention & rate limiting
    location ~ ^/(process|api/research/denoise_dicom) {
        limit_req zone=cascade_api_limit burst=5 nodelay;
        limit_conn cascade_conn_limit 2;

        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_connect_timeout 180s;
        proxy_send_timeout 180s;
        proxy_read_timeout 180s;
    }
}
```

Enable site and test configuration:
```bash
sudo ln -sf /etc/nginx/sites-available/cascade /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

### Obtain Free Let's Encrypt SSL/TLS Certificate
```bash
sudo certbot --nginx -d cascade-backend.duckdns.org --non-interactive --agree-tos -m your-email@example.com
```

Verify certificate auto-renewal:
```bash
sudo certbot renew --dry-run
```

Test your backend over HTTPS:
```bash
curl -I https://cascade-backend.duckdns.org/metrics
```
Should return `HTTP/2 200 OK` with `Content-Security-Policy: frame-ancestors 'self' https://huggingface.co https://*.hf.space`.

---

## 9. Logging, Monitoring & Maintenance Commands

* **Live Gunicorn Logs**:
  ```bash
  sudo journalctl -u cascade -f
  ```
* **Nginx Access & Error Logs**:
  ```bash
  sudo tail -f /var/log/nginx/access.log
  sudo tail -f /var/log/nginx/error.log
  ```
* **Resource Utilization**:
  ```bash
  htop
  ```
* **Restarting the Service**:
  ```bash
  sudo systemctl restart cascade
  ```

---

## 10. Security & Non-Clinical Safety Disclaimers

* **Research-Only Prototype**: CASCADE is an exploratory engineering demonstration and research benchmark engine. It is **not** an FDA-cleared, CE-marked, or certified medical device (SaMD).
* **Non-Clinical Environment**: Do not upload un-deidentified patient health information (PHI). Real clinical diagnosis or emergency triage must never rely on this platform.
* **Ephemeral Data Policy**: Uploaded scans are written to `/tmp/cascade_uploads` and securely unlinked on both success and failure within Python `try/finally` blocks.
