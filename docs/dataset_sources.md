# CASCADE Official Dataset Sources & Data Plan

This document details the official sources, licenses, access constraints, tasks, and storage specifications for clinical datasets utilized or targeted by the CASCADE platform.

---

## 1. Official Dataset Specifications

| Dataset Name | Official Source / Host | Intended Task & Model | Format & Size | Terms / License | Access Method |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mayo Clinic / TCIA Low Dose CT** (LDCT-and-Projection-Data) | [TCIA LDCT Collection](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/) | U-Net Denoising Engine (`medical_ct_denoiser_full.keras`) | Paired Full-Dose / Quarter-Dose DICOM CT (~1.2 TB full, ~500 MB targeted subset) | TCIA Limited Access License / CC BY 4.0 with attribution | NBIA Data Retriever / TCIA REST API |
| **NIH DeepLesion** | [NIH Clinical Center Box Repository](https://nihcc.app.box.com/v/DeepLesion) | Focal Lesion Localization Head & Lesion Classifier | 32,735 16-bit PNG slices + `DL_info.csv` (~250 GB full, 50-100 MB subset) | Open access under NIH scientific fair use; attribution required | NIH Box direct download / CSV manifest |
| **LIDC-IDRI** (Lung Image Database Consortium) | [NCI Cancer Imaging Data Commons](https://portal.imaging.datacommons.cancer.gov/collections/lidc_idri/) | Thoracic Anatomical Classification (Chest) | 1,018 thoracic CT scans (DICOM / NIfTI) | Creative Commons Attribution 3.0 Unported (CC BY 3.0) | IDC Portal / NBIA REST API |
| **CQ500 / RSNA Head CT** | [Centre for Advanced Studies / RSNA](http://headctstudy.qure.ai/) | Cranial Anatomical Classification (Brain/Head) | 491 cranial CT studies (CQ500) / RSNA challenge subset | Research non-commercial educational use | Open research registration / direct download |
| **AMOS22 / CT-ORG** | [Grand Challenge AMOS22](https://amos22.grand-challenge.org/) / [TCIA CT-ORG](https://www.cancerimagingarchive.net/collection/ct-org/) | Abdominal CT Region & Multi-Organ Segmentation | 500 multi-organ abdominal CT volumes (NIfTI) | CC BY-NC-SA 4.0 | Challenge download / TCIA REST API |

---

## 2. Model-Specific Dataset Plans

### A. U-Net CT Denoising Engine
* **Target Collection**: Mayo Clinic Low Dose CT (LDCT-and-Projection-Data).
* **Storage Footprint**: The entire TCIA LDCT collection is over 1.2 TB. Downloading the entire collection on this machine (78.9 GB free on D:) is strictly impossible and unwarranted.
* **Targeted Cohort**:
  * Selective download of single paired clinical cases (e.g. Chest `C002` or Liver `L067` containing matching Full-Dose `FD` and Quarter-Dose `QD` series).
  * Pair-matching criteria: Identical patient identifier, identical study date, identical axial slice location / z-coordinate, and matching slice thickness.
* **DICOM Preprocessing Contract**:
  * Rescale Slope and Intercept: $HU = \text{PixelValue} \times \text{RescaleSlope} + \text{RescaleIntercept}$.
  * Windowing: Soft tissue window (Width: 400, Level: 40) or normalized to $[0.0, 1.0]$.
  * Resolution: Standardized to $256 \times 256 \times 1$.
* **Controlled Synthetic Noise Cohort**:
  * Retained as a secondary experimental baseline for algorithm stress-testing.
  * Explicitly documented in manifests as synthetic noise ($\sigma=0.08$ Gaussian + Poisson), never conflated with physical low-dose acquisitions.

---

### B. Focal Lesion Localization (NIH DeepLesion)
* **Target Collection**: NIH DeepLesion (Yan et al., Journal of Medical Imaging 2018).
* **Local Inventory**: The authoritative metadata file `data/DL_info.csv` (8.48 MB, 32,735 records, 4,427 unique patients) is already verified locally in `data/`.
* **Annotation Schema**:
  * `Bounding_boxes`: RECIST lesion bookmark coordinates $[x_1, y_1, x_2, y_2]$ in $512 \times 512$ space.
  * Standardized normalized coordinates: $[ymin, xmin, ymax, xmax]$ in $[0.0, 1.0]$.
  * `Coarse_lesion_type`: 8 official categories:
    1: Bone (1,231), 2: Abdomen (4,198), 3: Mediastinum (3,478), 4: Liver (4,077), 5: Lung (2,394), 6: Kidney (1,190), 7: Soft tissue (1,749), 8: Pelvis (867).
* **Strict Scientific Safeguards**:
  * `Coarse_lesion_type == 8` is **Pelvis**. It is strictly preserved as Pelvis and never relabeled as Brain.
  * DeepLesion contains thoracic, abdominal, and pelvic CT scans, with **0 brain scans**.
* **Split Methodology**:
  * Enforces deterministic patient-level disjoint partitioning via `PatientGroupedSplitter`.
  * Slices from the same patient index never appear across train and test sets.
* **Target Metrics**: Bounding box Mean Intersection-over-Union (IoU), Precision@IoU $\ge 0.50$, and localization recall.

---

### C. 3-Region Anatomical Scan Classification (Head/Brain, Chest, Abdomen)
* **Task Definition**: Coarse anatomical scan orientation (identifying whether a scan slice depicts the cranial vault, thorax, or abdomen).
* **Dataset Challenge & Shortcut Warning**:
  * Sourcing brain scans from RSNA/CQ500, chest scans from LIDC-IDRI, and abdomen scans from AMOS22 introduces **scanner and dataset acquisition bias** (the neural network learns scanner pixel distributions, reconstruction kernels, or field-of-view margins rather than true anatomical morphology).
  * Consequently, cross-dataset multi-class classifiers require rigorous intensity normalization and separate-source external test sets.
* **Status**: **BLOCKED from certified clinical diagnosis**. Evaluated solely as an orientational pre-filter for PACS workflow routing.

---

### D. Clinical RAG Corpus
* **Existing Files**:
  * `knowledge_base/clinical_sources/fleischner_society_pulmonary_nodules.md`
  * `knowledge_base/clinical_sources/acr_appropriateness_head_trauma.md`
  * `knowledge_base/clinical_sources/lirads_liver_imaging_reporting.md`
* **Status**: Authored clinical summaries under fair use. Production validation requires owner-provided authorized institutional manuals or open-access guideline PDFs.
