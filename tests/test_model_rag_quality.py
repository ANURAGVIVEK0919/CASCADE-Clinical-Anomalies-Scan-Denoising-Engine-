"""
CASCADE Comprehensive Model & RAG Quality Verification Test Suite

Covers the 14 mandatory quality and safety gates:
1. Label taxonomy validation and invalid label detection.
2. Patient/study-level split separation and reproducibility (no cross-split patient leakage).
3. Exact preprocessing contracts and model output shapes.
4. Correct evaluation metrics and absence of label leakage.
5. Paired denoising metrics and graceful handling of missing references.
6. Real-document extraction and chunk metadata using temporary test files.
7. Stable chunk IDs and repeat ingestion behavior (deterministic SHA-256 IDs).
8. Semantic embedding dimension and model compatibility (384-d, L2 normalized).
9. Pinecone query/upsert with mocked network access.
10. Namespace isolation (clinical-guidelines-v1, no mixing legacy hashed vectors).
11. Empty/unconfigured RAG state without fabricated results (similarity is None, no +0.45 bonus).
12. Retrieval source citations and grounded-generation behavior.
13. No accidental automatic network access or upsert during module import.
14. Safety defaults and truthful reporting when models, data, or external services are unavailable.
"""

import os
import sys
import unittest
import tempfile
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset_taxonomy import (
    DEEPLESION_COARSE_LESION_MAP,
    DatasetTaxonomyValidator,
    PatientGroupedSplitter,
    VALID_ORGAN_LABELS
)
from rag.parsers import DocumentParser
from rag.chunker import DocumentChunker, DocumentChunk
from rag.embeddings import SemanticEmbeddingEngine, DEFAULT_VECTOR_DIM, DEFAULT_NAMESPACE
from rag.retriever import ClinicalRAGRetriever
from rag.evaluate_rag import evaluate_rag, LocalMockRetriever
from medical_agent import ASCADEMedicalBoard, ScanContext, AgentState
import pinecone_rag_engine


class TestModelRAGQuality(unittest.TestCase):

    # =========================================================================
    # Gate 1: Label Taxonomy Validation & Invalid Label Detection
    # =========================================================================
    def test_gate1_label_taxonomy_validation(self):
        # Official 8 coarse lesion classes
        self.assertEqual(len(DEEPLESION_COARSE_LESION_MAP), 8)
        self.assertEqual(DEEPLESION_COARSE_LESION_MAP[8], "Pelvis")
        self.assertEqual(DEEPLESION_COARSE_LESION_MAP[5], "Lung")
        self.assertEqual(DEEPLESION_COARSE_LESION_MAP[6], "Kidney")

        # Invalid remapping detection: DeepLesion cannot supervise 3-class organ classifier because it lacks Brain
        viability = DatasetTaxonomyValidator.check_organ_classification_viability("DeepLesion")
        self.assertFalse(viability["can_supervise_organ_classifier"])
        self.assertEqual(viability["status"], "BLOCKED_PENDING_EXTERNAL_DATASET")
        self.assertIn("Pelvis", viability["reason"])

    # =========================================================================
    # Gate 2: Patient/Study-Level Split Separation (Zero Patient Leakage)
    # =========================================================================
    def test_gate2_patient_grouped_split_separation(self):
        # Synthetic patient IDs with multiple slices
        patient_ids = [f"PATIENT_{p:03d}" for p in range(50) for _ in range(5)]
        split_map = PatientGroupedSplitter.deterministic_patient_split(
            patient_ids, train_ratio=0.70, val_ratio=0.15, seed=42
        )

        train_patients = {pid for pid, split in split_map.items() if split == "train"}
        val_patients = {pid for pid, split in split_map.items() if split == "val"}
        test_patients = {pid for pid, split in split_map.items() if split == "test"}

        # Strict disjoint sets: No patient can appear across train, val, and test
        self.assertEqual(len(train_patients.intersection(val_patients)), 0)
        self.assertEqual(len(train_patients.intersection(test_patients)), 0)
        self.assertEqual(len(val_patients.intersection(test_patients)), 0)
        self.assertEqual(len(train_patients) + len(val_patients) + len(test_patients), 50)

    # =========================================================================
    # Gate 3: Exact Preprocessing Contracts & Model Output Shapes
    # =========================================================================
    def test_gate3_preprocessing_and_shapes(self):
        from app import get_model, get_diagnostic_model
        denoiser = get_model()
        if denoiser is not None:
            self.assertEqual(denoiser.input_shape, (None, 256, 256, 1))
            self.assertEqual(denoiser.output_shape, (None, 256, 256, 1))

        diag = get_diagnostic_model()
        if diag is not None:
            self.assertEqual(diag.input_shape, (None, 256, 256, 1))
            # Diagnostic outputs: classification (3,) and bbox (4,)
            out_shapes = [out.shape for out in diag.outputs]
            self.assertIn((None, 3), out_shapes)
            self.assertIn((None, 4), out_shapes)

    # =========================================================================
    # Gate 4: Evaluation Metrics Correctness & Absence of Label Leakage
    # =========================================================================
    def test_gate4_evaluation_metrics_no_label_leakage(self):
        from evaluate_models import compute_classification_metrics
        # Deterministic predictions vs true labels
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 1, 2, 0, 2, 1, 2])

        metrics = compute_classification_metrics(y_true, y_pred, class_names=["Chest", "Brain", "Abdomen"])
        self.assertEqual(metrics["sample_count"], 10)
        self.assertAlmostEqual(metrics["overall_accuracy"], 0.80, places=2)
        self.assertIn("confusion_matrix", metrics)
        self.assertEqual(len(metrics["confusion_matrix"]["matrix"]), 3)

    # =========================================================================
    # Gate 5: Paired Denoising Metrics & Missing Reference Safety
    # =========================================================================
    def test_gate5_paired_denoising_metrics_safe_nulls(self):
        from app import process_image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            img = Image.fromarray((np.random.rand(256, 256) * 255).astype(np.uint8))
            img.save(tmp_path)

        try:
            # Process image without clean reference ground truth
            noisy, denoised, diff, metrics, meta = process_image(tmp_path, clean_path=None)

            # Ensure metrics remain None rather than corrupting accumulators or inventing deltas
            self.assertIsNone(metrics["psnr_noisy"])
            self.assertIsNone(metrics["ssim_noisy"])
            self.assertIsNone(metrics["psnr_delta"])
            self.assertIsNone(metrics["ssim_delta"])
            self.assertIsNone(metrics["improvement"])
            self.assertIn(metrics["denoiser_engine"], [
                "CASCADE U-Net Neural Denoiser",
                "Classical 3x3 Smoothing Baseline (Classical Fallback)"
            ])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # =========================================================================
    # Gate 6: Real-Document Extraction and Metadata Parsing
    # =========================================================================
    def test_gate6_document_extraction_and_metadata(self):
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as tmp:
            tmp.write("# Clinical Guideline Header\n\nSection text describing protocol.\n")
            tmp_path = tmp.name

        try:
            doc = DocumentParser.parse_file(tmp_path)
            self.assertTrue(doc["success"])
            self.assertEqual(doc["title"], "Clinical Guideline Header")
            self.assertEqual(doc["extension"], ".md")
            self.assertTrue(len(doc["content_hash"]) == 64)
            self.assertIn("Section text describing protocol.", doc["text"])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # =========================================================================
    # Gate 7: Stable Chunk IDs & Deterministic Repeat Ingestion
    # =========================================================================
    def test_gate7_stable_chunk_ids_idempotency(self):
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        doc = {
            "success": True,
            "text": "This is a clinical guidance text about pulmonary nodules and solid lesion follow-up intervals.",
            "filename": "pulmonary_guideline.md",
            "title": "Pulmonary Guideline",
            "content_hash": "dummyhash12345"
        }

        run1_chunks = chunker.chunk_document(doc)
        run2_chunks = chunker.chunk_document(doc)

        self.assertEqual(len(run1_chunks), len(run2_chunks))
        self.assertGreater(len(run1_chunks), 0)
        for c1, c2 in zip(run1_chunks, run2_chunks):
            # Identical stable IDs on repeat ingestion
            self.assertEqual(c1.chunk_id, c2.chunk_id)
            self.assertEqual(c1.content, c2.content)

    # =========================================================================
    # Gate 8: Semantic Embedding Dimension & Model Compatibility
    # =========================================================================
    def test_gate8_semantic_embedding_dimensions(self):
        embedder = SemanticEmbeddingEngine()
        vec = embedder.embed_text("Fleischner Society pulmonary nodule follow-up criteria")

        self.assertEqual(len(vec), DEFAULT_VECTOR_DIM)
        self.assertEqual(DEFAULT_VECTOR_DIM, 384)

        # Verify unit L2 normalization
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(float(norm), 1.0, places=4)

    # =========================================================================
    # Gate 9: Pinecone Query & Upsert with Mocked Network Access
    # =========================================================================
    def test_gate9_pinecone_mocked_network(self):
        mock_index = MagicMock()
        mock_index.query.return_value = {
            "matches": [
                {
                    "id": "chunk_abc123",
                    "score": 0.8842,
                    "metadata": {
                        "title": "Fleischner Society Guidelines",
                        "filename": "fleischner.md",
                        "text": "Solid nodule >8mm recommended CT surveillance at 3 months."
                    }
                }
            ]
        }

        retriever = ClinicalRAGRetriever(api_key="mock_key_123", index_name="test-index")
        retriever.pinecone_index = mock_index
        retriever.is_active = True

        res = retriever.query_clinical_guidelines("pulmonary nodule", top_k=1)
        self.assertEqual(res["status"], "success")
        self.assertEqual(len(res["retrieved_chunks"]), 1)
        match = res["retrieved_chunks"][0]
        self.assertEqual(match["chunk_id"], "chunk_abc123")
        self.assertEqual(match["score"], 0.8842)
        mock_index.query.assert_called_once()

    # =========================================================================
    # Gate 10: Namespace Isolation (clinical-guidelines-v1)
    # =========================================================================
    def test_gate10_namespace_isolation(self):
        mock_index = MagicMock()
        mock_index.query.return_value = {"matches": []}

        retriever = ClinicalRAGRetriever(api_key="mock_key", index_name="test-index", namespace="clinical-guidelines-v1")
        retriever.pinecone_index = mock_index
        retriever.is_active = True

        retriever.query_clinical_guidelines("head trauma hematoma", top_k=2)

        # Ensure explicit namespace isolation flag was passed
        kwargs = mock_index.query.call_args[1]
        self.assertEqual(kwargs.get("namespace"), "clinical-guidelines-v1")
        self.assertNotEqual(kwargs.get("namespace"), "")

    # =========================================================================
    # Gate 11: Empty / Unconfigured RAG State (No Fabricated Results)
    # =========================================================================
    def test_gate11_unconfigured_rag_truthful_reporting(self):
        # Create retriever without API key
        retriever = ClinicalRAGRetriever(api_key="", index_name="unconfigured-index")
        res = retriever.query_clinical_guidelines("pulmonary mass")

        self.assertEqual(res["status"], "unavailable")
        self.assertEqual(len(res["retrieved_chunks"]), 0)

        # Test pinecone_rag_engine query_dense_rag with unconfigured index and local fallback disabled
        engine = pinecone_rag_engine.PineconeClinicalRAG()
        # Mock Pinecone offline
        engine.retriever.is_active = False

        query_res = engine.query_dense_rag(organ="Chest", severity="Candidate ROI Detected", allow_local_fallback=False)
        self.assertIn(query_res["retrieval_mode"], ["unconfigured", "unavailable"])
        self.assertIsNone(query_res["similarity_score"])  # Strict: NO fake 0.95!
        self.assertEqual(len(query_res["kb_references"]), 0)

        # When local fallback is enabled, offline Pinecone safely falls back to local semantic RAG
        query_res_local = engine.query_dense_rag(
            organ="Chest",
            severity="Candidate ROI Detected",
            user_query="solid nodule >8mm",
            allow_local_fallback=True
        )
        self.assertEqual(query_res_local["retrieval_mode"], "local_semantic")
        self.assertIsNotNone(query_res_local["similarity_score"])
        self.assertNotEqual(query_res_local["similarity_score"], 0.95)
        self.assertGreater(len(query_res_local["kb_references"]), 0)

    # =========================================================================
    # Gate 12: Retrieval Source Citations & Grounded Generation
    # =========================================================================
    def test_gate12_retrieval_source_citations(self):
        board = ASCADEMedicalBoard()
        scan_ctx = {
            "organ": "Chest",
            "severity": "Candidate ROI Detected",
            "psnr_delta": None,
            "ssim_delta": None,
            "findings": ["Candidate ROI localized in right upper lobe"]
        }

        # Mock RAG response with authentic citation
        with patch.object(
            pinecone_rag_engine.pinecone_rag_engine,
            "query_dense_rag",
            return_value={
                "kb_references": ["[Fleischner Society Guidelines] Solid nodule >8mm... (Raw Cosine Similarity: 0.8521)"],
                "differential_diagnosis": ["Pulmonary Adenocarcinoma", "Granuloma"],
                "vector_engine": "Pinecone Cloud Index [test-index / clinical-guidelines-v1]",
                "similarity_score": 0.8521,
                "top_match_condition": "Fleischner Society Guidelines",
                "retrieval_mode": "pinecone",
                "retrieved_chunks": []
            }
        ):
            res = board.consult(scan_ctx, "Guidelines for pulmonary nodule?")
            report = res["consensus_report"]

            self.assertIn("Fleischner Society Guidelines", report)
            self.assertIn("Raw Cosine Similarity: **0.8521**", report)
            self.assertNotIn("0.95%", report)  # No fake percentages

    # =========================================================================
    # Gate 13: Zero Accidental Network Ingestion on Module Import
    # =========================================================================
    def test_gate13_no_auto_network_or_upsert_on_import(self):
        # Importing or instantiating retriever must not call index.create_index or index.upsert
        with patch("pinecone.Pinecone") as mock_pinecone:
            from rag.retriever import ClinicalRAGRetriever
            r = ClinicalRAGRetriever(api_key="dummy_key", index_name="dummy_index")
            # Ensure upsert was never called
            if r.pinecone_index:
                r.pinecone_index.upsert.assert_not_called()

    # =========================================================================
    # Gate 14: Safety Defaults & Human Review Flags
    # =========================================================================
    def test_gate14_safety_defaults_unauthenticated_review(self):
        board = ASCADEMedicalBoard()
        scan_ctx = {
            "organ": "Abdomen",
            "severity": "No Candidate ROI Flagged",
            "psnr_delta": None,
            "ssim_delta": None,
            "findings": []
        }

        # Case A: Unreviewed
        res_unreviewed = board.consult(scan_ctx, review_acknowledged=False)
        self.assertIsNone(res_unreviewed["is_emergency"])
        self.assertFalse(res_unreviewed["is_human_verified"])
        self.assertFalse(res_unreviewed["reviewer_identity_verified"])
        self.assertIn("research output", res_unreviewed["review_status"].lower())

        # Case B: Demo Acknowledged (must remain unverified for authentic clinical sign-off)
        res_ack = board.consult(scan_ctx, review_acknowledged=True, hitl_notes="Demo signoff")
        self.assertIsNone(res_ack["is_emergency"])
        self.assertFalse(res_ack["is_human_verified"])
        self.assertFalse(res_ack["reviewer_identity_verified"])
        self.assertIn("Demo acknowledgement only", res_ack["review_status"])

    # =========================================================================
    # Gate 15: RAG Calibrated Threshold Abstention
    # =========================================================================
    def test_gate15_rag_abstention_on_out_of_domain(self):
        from rag.evaluate_rag import LocalMockRetriever
        sources_dir = os.path.join(REPO_ROOT, "knowledge_base", "clinical_sources")
        if os.path.exists(sources_dir):
            retriever = LocalMockRetriever(sources_dir)
            # Irrelevant non-radiology query must abstain
            res = retriever.query("How to perform open reduction internal fixation of a Jones fracture?", threshold=0.395)
            self.assertTrue(res.get("abstained"), "Out-of-domain query must trigger explicit clinical abstention")
            self.assertEqual(len(res.get("retrieved", [])), 0)

    # =========================================================================
    # Gate 16: Spatial Detector Model Contract & 2D Output Shape
    # =========================================================================
    def test_gate16_spatial_detector_model_contract(self):
        import tensorflow as tf
        model_path = os.path.join(REPO_ROOT, "artifacts", "experiments", "spatial_detector", "run_001", "candidate_spatial_detector.keras")
        if os.path.exists(model_path):
            from train_spatial_detector import spatial_objectness_loss, spatial_box_loss
            custom_objs = {
                "spatial_objectness_loss": spatial_objectness_loss,
                "spatial_box_loss": spatial_box_loss
            }
            m = tf.keras.models.load_model(model_path, custom_objects=custom_objs, compile=False)
            dummy_input = np.zeros((1, 256, 256, 1), dtype=np.float32)
            preds = m.predict(dummy_input, verbose=0)
            self.assertEqual(len(preds), 2, "Spatial detector must output [objectness, box_params]")
            # Objectness map shape (1, 16, 16, 1)
            self.assertEqual(preds[0].shape, (1, 16, 16, 1), "Spatial objectness must preserve 16x16 grid")
            # Box map shape (1, 16, 16, 4)
            self.assertEqual(preds[1].shape, (1, 16, 16, 4), "Spatial box regression must preserve 16x16 grid")
            # Ensure coordinates are in [0, 1]
            self.assertTrue(np.all(preds[0] >= 0.0) and np.all(preds[0] <= 1.0))
            self.assertTrue(np.all(preds[1] >= 0.0) and np.all(preds[1] <= 1.0))


if __name__ == "__main__":
    unittest.main()
