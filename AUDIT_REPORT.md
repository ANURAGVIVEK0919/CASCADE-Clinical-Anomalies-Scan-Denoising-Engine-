# CASCADE - Clinical Anomalies & Scan Denoising Engine
## Executive Summary
Overall Status: **CONTINUE AFTER CLEANUP**
Completion Estimate: **75%** (Core ML modeling, UI, and backend integration are functional, but data pipelines and infrastructure connections need refactoring).
Biggest Blockers:
1. Data leakage during training (random split instead of patient-wise).
2. Hardcoded AWS credentials in the `.env` file exposed in the directory.
3. Fallback mock metrics (e.g., PSNR = 22.35) used when reference scans are missing, creating fabricated results.
4. `gunicorn --timeout 0` in Dockerfile causing potential hangs.
Recommendation: **CONTINUE AFTER CLEANUP** (Score: 7/12). The deep learning architecture is solid and works, but testing, data hygiene, and security need immediate fixes before production.

---

## 1. INVENTORY

### Directory Tree
(Excluding `.git`, `__pycache__`, `node_modules`, `venv`)
```text
.
├── .dockerignore
├── .env
├── .env.example
├── .gitattributes
├── .gitignore
├── app.py
├── aws_s3_storage.py
├── cascade_project_defense_guide.html
├── create_kaggle_zip.py
├── data.zip
├── data_pipeline.py
├── deploy_hf.py
├── Dockerfile
├── evaluate_data.py
├── gradcam.py
├── gradcam_utils.py
├── kaggle_train_both.py
├── kaggle_train_medical.py
├── kaggle_train_notebook.py
├── medical_agent.py
├── medical_ct_denoiser_full.keras
├── medical_ct_denoiser_full_1epoch.keras
├── medical_ct_diagnostic_model.keras
├── pinecone_rag_engine.py
├── prepare_unified_dataset.py
├── README.md
├── requirements.txt
├── train_diagnostic_stage2.py
├── train_medical_ct.py
├── train_unified_unet.py
├── validate_models_on_dataset.py
├── docs/
│   ├── dataset_plan.md
│   ├── project_flow_v1.md
│   ├── project_flow_v2.md
│   ├── v3.md
│   └── images/
│       ├── dashboard_detected.png
│       └── dashboard_normal.png
├── output/
│   ├── diagnostic_training_curves.png
│   ├── medical_ct_denoiser_full.keras
│   ├── medical_ct_diagnostic_model.keras
│   ├── state.db
│   └── unet_training_curves.png
├── static/
│   └── uploads/
├── templates/
│   └── index.html
└── data/
    ├── unified_dataset/
    │   ├── labels.json
    │   ├── noisy/ (2069 files)
    │   └── clean/ (2069 files)
    └── separated_scans/ (6568 files)
```

### Key Files & Purpose
- `app.py` (~29.3 KB): Main Flask application routing and ML inference integration.
- `train_unified_unet.py` (~8.6 KB): U-Net autoencoder training script.
- `train_diagnostic_stage2.py` (~7.7 KB): Multi-task ResNet-50V2 training script.
- `medical_agent.py` (~15.0 KB): LangGraph & LangChain HITL medical board implementation.
- `pinecone_rag_engine.py` (~16.5 KB): Pinecone Dense Vector RAG & fallback TF-IDF vector generator.
- `aws_s3_storage.py` (~11.3 KB): AWS S3, EC2, and CloudWatch integration logic.
- `gradcam.py` (~5.0 KB): Real tf.GradientTape GradCAM generator.
- `requirements.txt`, `Dockerfile`, `.env`: Project config and containerization.

### Duplicate / Dead Files
- `medical_ct_denoiser_full.keras` is present in both root and `output/`.
- `kaggle_train_both.py`, `kaggle_train_medical.py`, `kaggle_train_notebook.py`, `train_medical_ct.py`, `train_unified_unet.py`, `train_diagnostic_stage2.py`: Multiple overlapping training scripts serving similar purposes. `train_unified_unet.py` and `train_diagnostic_stage2.py` appear to be the finalized versions.
- `gradcam.py` and `gradcam_utils.py` seem to overlap.

---

## 2. COMPONENT STATUS TABLE

| Component | Files | Status | Evidence | Problems |
|---|---|---|---|---|
| 1. DICOM/PNG/JPG preprocessing | `app.py` | COMPLETE | Converts (1, 256, 256, 1) float32 via pydicom and PIL. | None. |
| 2. Stage-1 U-Net denoiser | `app.py`, `train_unified_unet.py` | COMPLETE | Model loads, `train_unified_unet.py` implements Autoencoder. | Data leakage in training split. |
| 3. PSNR/SSIM metrics | `app.py`, `validate_models_on_dataset.py` | PARTIAL | `tf.image.psnr` used when clean reference found. | Uses hardcoded mock constants (22.35 dB) if clean reference missing. |
| 4. Stage-2 ResNet-50V2 | `app.py`, `train_diagnostic_stage2.py` | COMPLETE | Implements `class_output` (3 units) and `bbox_output` (4 units). 1->3 channels map. | Data leakage in training split. |
| 5. Grad-CAM | `gradcam.py` | COMPLETE | `tf.GradientTape` wraps inner `base_model`. | None. |
| 6. RAG with Pinecone | `pinecone_rag_engine.py` | PARTIAL | Implements Pinecone SDK and Dense TF-IDF fallback. | Fallback used due to empty `.env` API key. |
| 7. LangGraph agents | `medical_agent.py` | WORKING_UNTESTED | Uses LangGraph `StateGraph`, HF `InferenceClient`. | None. |
| 8. Flask + Gunicorn API | `app.py` | COMPLETE | All defined endpoints are implemented and mapped. | None. |
| 9. AWS S3/CloudWatch, Docker | `aws_s3_storage.py`, `Dockerfile` | PARTIAL | Boto3 implementation exists. Dockerfile runs Gunicorn. | Exposed AWS credentials in `.env`, `gunicorn --timeout 0`. |
| 10. HTML/CSS/JS Dashboard | `templates/index.html` | COMPLETE | Massive single-file SPA for inference and batch viewing. | None. |
| 11. DICOM anonymization | `app.py` | STUB | Extracts metadata via `pydicom.dcmread`, replaces PatientID occasionally. | Not a comprehensive DICOM anonymizer. |

---

## 3. CRITICAL CHECKS

- **Does the project import and compile?** 
  - **Yes**. Verified via `python -m compileall .` running successfully on all Python scripts without SyntaxErrors.
- **Do tensor shapes match between preprocessing -> U-Net -> ResNet?** 
  - **Yes**. Preprocessing outputs `(1, 256, 256, 1)`. U-Net expects `(256, 256, 1)`. Stage-2 ResNet replicates the single channel to 3 channels using `layers.Concatenate(axis=-1)([augmented, augmented, augmented])` in `train_diagnostic_stage2.py` to match ImageNet weights.
- **Is there an actual trained model file, or only architecture code?**
  - **Yes**. `medical_ct_denoiser_full.keras` (23 MB) and `medical_ct_diagnostic_model.keras` (256 MB) exist.
- **Is there real training code for U-Net and ResNet? Was it ever run?**
  - **Yes**. `train_unified_unet.py` and `train_diagnostic_stage2.py` contain functional TensorFlow `model.fit()` pipelines. Execution is verified by the presence of `output/unet_training_curves.png` and `output/diagnostic_training_curves.png`.
- **Is the dataset present? What exactly? Is there a patient-wise split or data leakage?**
  - **Yes**, `data/unified_dataset` contains 2069 pairs of noisy/clean images. 
  - **Data Leakage Present**: `train_unified_unet.py` splits the dataset purely randomly using `np.random.shuffle(indices)`. Without patient ID grouping, slices from the same patient likely leak into both train and validation sets, artificially inflating metrics.
- **Are any metrics (PSNR, SSIM, accuracy) hardcoded or fabricated instead of computed?**
  - **Yes (Partial)**. In `app.py` (lines 201-204), if a clean reference scan is NOT found, it fabricates baseline metrics: `psnr_noisy = 22.35` and `ssim_noisy = 0.584`. Also, `aws_s3_storage.py` hardcodes `PSNR_Gain=+10.53dB` in its CloudWatch logging statement.
- **Does Grad-CAM use real gradients or a fake/fallback heatmap?**
  - **Real**. `gradcam.py` utilizes `tf.GradientTape()` to calculate gradients with respect to the `post_relu` layer of the frozen ResNet50V2 inner layer.
- **Are the LangGraph "agents" real LLM calls or just templates/strings?**
  - **Real**. `medical_agent.py` uses the `huggingface_hub.InferenceClient` to call `Qwen/Qwen2.5-72B-Instruct` for the `ai_generated_synthesis` report summary.
- **Is Pinecone actually connected, or only the local fallback?**
  - **Fallback Only**. The `.env` file does not specify a `PINECONE_API_KEY`. As a result, `pinecone_rag_engine.py` prints `Running in Dense Vector Local Matrix Mode` and falls back to its local custom `DenseEmbeddingGenerator` class.
- **Are secrets (API keys, AWS keys) hardcoded anywhere?**
  - **Yes**. `AWS_ACCESS_KEY_ID=AKIA****` and `AWS_SECRET_ACCESS_KEY=****` are hardcoded in the `.env` file in the project root.
- **Any known bad practices?**
  - `Dockerfile` sets `CMD gunicorn ... --timeout 0 app:app`. Timeout 0 disables worker timeout protections, which can cause infinite hangs if inference blocks.
  - Lack of unit testing framework (`pytest` missing).

---

## 4. QUALITY SNAPSHOT

- **Rough lines of code:** 
  - `index.html`: ~2066 lines (Very giant UI file mixing CSS/JS).
  - `app.py`: ~671 lines.
  - `pinecone_rag_engine.py`: ~356 lines.
  - Overall Python base is surprisingly neat and well-commented, although modular separation between Flask routing and model inference could be better.
- **Code smells:**
  - Multiple overlapping `kaggle_train_*.py` and `train_*.py` files creating a cluttered root directory.
  - Mixing UI code directly inside HTML (`templates/index.html` is almost 100KB).
- **Dependency problems:**
  - `requirements.txt` has unpinned core packages (e.g., `tensorflow`, `flask`, `pydicom`, `langchain`, `pinecone-client`). This will break future builds when major versions bump (especially Keras/TensorFlow updates).

---

## 5. RUN-READINESS

**Exact steps to run today:**
1. Install Python 3.11.
2. Run `pip install -r requirements.txt`.
3. Ensure `.keras` model files are in the root directory.
4. Set `.env` keys (or rely on the simulated fallbacks).
5. Run `python app.py`.

**What would fail first, and why?**
- Running on a modern server with TensorFlow 2.16+ (Keras 3) might cause model loading failures because `custom_objects={'random_rot90': random_rot90}` requires the `@keras.saving.register_keras_serializable()` decorator, which was fixed recently in the codebase but depends on exact TF versions installed via unpinned requirements.
- HuggingFace inference will rate limit or fail quickly because no `HF_TOKEN` is supplied to `InferenceClient()`, causing the Serverless API to throw an authentication/timeout error.

---

## 6. DECISION: CONTINUE vs REBUILD

a) Architecture is coherent and matches intended design: **2** (The pipeline accurately reflects the 11-step plan from ingestion to LLM report).
b) Core ML code (U-Net/ResNet) is correct and trainable: **2** (Actual datasets, loaders, models, and gradient ops exist and function).
c) Data pipeline and dataset exist and are usable: **1** (Dataset exists and is large, but random train/val split causes data leakage).
d) Code is readable/modular enough to extend: **1** (Mostly clean, but UI and routing are monolithic).
e) Few broken or duplicate files: **1** (A lot of dead Kaggle training scripts).
f) No fabricated results: **0** (PSNR fallback = 22.35 dB and AWS mock logs falsify metrics when reference is unavailable).

Total Score: **7/12**
Recommendation: **CONTINUE AFTER CLEANUP**

**Reasoning:** The project is surprisingly robust for a solo student endeavor. Real deep learning models are trained, real gradients are computed for XAI, and the RAG/Agent system has functioning logic. However, fabricated metrics during fallback, data leakage in the training scripts, and hardcoded AWS keys present severe academic and security risks. It's too functional to scrap, but too messy to deploy cleanly.

---

## 7. NEXT STEPS

**Top 10 Prioritized Actions:**
1. Delete/Rotate the compromised AWS Keys exposed in `.env`.
2. Pin exact package versions in `requirements.txt` (especially `tensorflow==2.15` or `2.16` and `keras`).
3. Refactor `train_unified_unet.py` to group splits by `PatientID` to prevent data leakage.
4. Remove `psnr_noisy = 22.35` fallbacks; output `N/A` if no reference scan exists.
5. Change Dockerfile Gunicorn timeout from `0` to `120`.
6. Add `HF_TOKEN` configuration to `medical_agent.py`.
7. Move overlapping `kaggle_train_*.py` files into an `archive/` folder.
8. Separate `templates/index.html` CSS and JS into `static/styles.css` and `static/main.js`.
9. Set up a basic `pytest` harness for inference functions.
10. Integrate real DICOM anonymization (stripping specific private tags) rather than just overriding Name.

**First 2 Hours:**
Address security and falsification. Delete the `.env` file credentials. Fix the mock metrics in `app.py` and `aws_s3_storage.py`. Remove the `timeout 0` from Docker.

**"Must-Have Core" vs "Nice-to-Have":**
- *Must-Have:* The TF Models, LangGraph integration, GradCAM, and basic Flask API.
- *Nice-to-Have:* The Pinecone Cloud API (the fallback dense vector engine is sufficient for a student demo), AWS CloudWatch integration (local Python logging is fine).
