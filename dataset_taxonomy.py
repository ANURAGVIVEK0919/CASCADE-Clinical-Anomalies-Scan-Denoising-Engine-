"""
CASCADE Dataset Taxonomy, DeepLesion Metadata Schema & Split Manager

This module establishes strict scientific safeguards for clinical CT datasets:
1. Preserves DeepLesion's authentic Coarse_lesion_type taxonomy (1: Bone to 8: Pelvis).
2. Explicitly rejects invalid heuristic remappings (e.g. Pelvis -> Brain).
3. Enforces deterministic patient-grouped / study-grouped splitting (preventing patient slice leakage).
4. Marks organ and pathology classification tasks as BLOCKED when ground-truth labels are absent.
"""

import os
import hashlib
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

# Official NIH DeepLesion Coarse Lesion Type Taxonomy (Yan et al., JMI 2018)
DEEPLESION_COARSE_LESION_MAP: Dict[int, str] = {
    1: "Bone",
    2: "Abdomen",
    3: "Mediastinum",
    4: "Liver",
    5: "Lung",
    6: "Kidney",
    7: "Soft tissue",
    8: "Pelvis"
}

# Official NIH DeepLesion Partition Assignments
DEEPLESION_SPLIT_MAP: Dict[int, str] = {
    1: "train",
    2: "val",
    3: "test"
}

# Supported Anatomical Scan Regions
VALID_ORGAN_LABELS: List[str] = ["Chest", "Brain", "Abdomen"]


class DatasetTaxonomyValidator:
    """Validates label semantics, detects invalid mappings, and enforces patient-grouped splits."""

    @staticmethod
    def validate_coarse_lesion_type(lesion_type: int) -> Optional[str]:
        """Returns official label name or None if unannotated/invalid (-1)."""
        return DEEPLESION_COARSE_LESION_MAP.get(int(lesion_type))

    @staticmethod
    def check_organ_classification_viability(dataset_name: str) -> Dict[str, Any]:
        """
        Evaluates whether a dataset can validly supervise a 3-class Chest/Brain/Abdomen organ classifier.
        DeepLesion contains thoracic, abdominal, and pelvic CT scans, but NO cranial/brain CT scans.
        """
        if "deeplesion" in dataset_name.lower():
            return {
                "dataset": "NIH DeepLesion",
                "can_supervise_organ_classifier": False,
                "reason": (
                    "DeepLesion contains thoracic, abdominal, and pelvic CT scans. "
                    "It contains 0 brain CT scans. Coarse_lesion_type denotes lesion categories "
                    "(e.g., 8=Pelvis), NOT anatomical organ scans. Pelvis must not be remapped to Brain."
                ),
                "status": "BLOCKED_PENDING_EXTERNAL_DATASET",
                "valid_alternative_task": "Coarse Lesion Type Classification (8 classes) or Universal Lesion Detection"
            }
        return {
            "dataset": dataset_name,
            "can_supervise_organ_classifier": True,
            "status": "READY"
        }

    @staticmethod
    def parse_bounding_boxes(bbox_str: str) -> List[List[float]]:
        """
        Parses DeepLesion Bounding_boxes string.
        Format in DeepLesion: xmin, ymin, xmax, ymax (512x512 space).
        Supports multiple bounding boxes separated by semicolons or commas.
        """
        if not bbox_str or pd.isna(bbox_str):
            return []

        boxes = []
        # Check if multiple boxes separated by semicolon
        box_strings = str(bbox_str).split(";")
        for b_str in box_strings:
            parts = [float(x.strip()) for x in b_str.split(",") if x.strip()]
            if len(parts) == 4:
                xmin, ymin, xmax, ymax = parts
                boxes.append([xmin, ymin, xmax, ymax])
        return boxes

    @staticmethod
    def normalize_box(coords: List[float], img_size: int = 512) -> List[float]:
        """
        Converts [xmin, ymin, xmax, ymax] to normalized [ymin, xmin, ymax, xmax] in [0.0, 1.0].
        Standardizes to TensorFlow / Keras detection coordinate order.
        """
        xmin, ymin, xmax, ymax = coords
        norm_ymin = max(0.0, min(1.0, ymin / float(img_size)))
        norm_xmin = max(0.0, min(1.0, xmin / float(img_size)))
        norm_ymax = max(0.0, min(1.0, ymax / float(img_size)))
        norm_xmax = max(0.0, min(1.0, xmax / float(img_size)))
        return [norm_ymin, norm_xmin, norm_ymax, norm_xmax]


class PatientGroupedSplitter:
    """
    Ensures slices and augmentations from the same patient never leak across training, validation, and test sets.
    """

    @staticmethod
    def get_official_deeplesion_split(train_val_test_code: int) -> str:
        """Returns official split: 1->train, 2->val, 3->test."""
        return DEEPLESION_SPLIT_MAP.get(int(train_val_test_code), "unassigned")

    @staticmethod
    def deterministic_patient_split(patient_ids: List[str],
                                     train_ratio: float = 0.70,
                                     val_ratio: float = 0.15,
                                     seed: int = 42) -> Dict[str, str]:
        """
        Deterministically assigns unique patients to train/val/test splits using patient ID hashing.
        Guarantees that all slices from a given patient map to the exact same split.
        """
        unique_patients = sorted(list(set(str(pid) for pid in patient_ids)))
        assignment = {}

        for pid in unique_patients:
            # Hash patient ID with salt for deterministic partition
            h = int(hashlib.sha256(f"{seed}_{pid}".encode("utf-8")).hexdigest(), 16) % 10000
            threshold_train = int(train_ratio * 10000)
            threshold_val = int((train_ratio + val_ratio) * 10000)

            if h < threshold_train:
                assignment[pid] = "train"
            elif h < threshold_val:
                assignment[pid] = "val"
            else:
                assignment[pid] = "test"

        return assignment


def audit_deeplesion_csv(csv_path: str) -> Dict[str, Any]:
    """Inspects DeepLesion metadata and reports official class counts and split distributions."""
    if not os.path.exists(csv_path):
        return {"error": f"Metadata file not found: {csv_path}"}

    df = pd.read_csv(csv_path)
    total_rows = len(df)
    unique_patients = df["Patient_index"].nunique() if "Patient_index" in df.columns else 0
    unique_studies = df["Study_index"].nunique() if "Study_index" in df.columns else 0

    # Coarse lesion counts
    coarse_counts = {}
    if "Coarse_lesion_type" in df.columns:
        counts = df["Coarse_lesion_type"].value_counts().to_dict()
        for k, v in counts.items():
            label_name = DEEPLESION_COARSE_LESION_MAP.get(int(k), f"Unlabeled ({k})")
            coarse_counts[label_name] = int(v)

    # Official splits
    split_counts = {}
    if "Train_Val_Test" in df.columns:
        splits = df["Train_Val_Test"].value_counts().to_dict()
        for k, v in splits.items():
            split_name = DEEPLESION_SPLIT_MAP.get(int(k), f"Unknown ({k})")
            split_counts[split_name] = int(v)

    return {
        "total_records": total_rows,
        "unique_patients": unique_patients,
        "unique_studies": unique_studies,
        "coarse_lesion_distribution": coarse_counts,
        "official_splits": split_counts,
        "organ_classification_viable": False,
        "viability_note": "DeepLesion cannot be used for Brain CT classification. Pelvis CT must not be remapped to Brain."
    }
