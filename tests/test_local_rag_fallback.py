"""
CASCADE Comprehensive Local Semantic RAG Fallback Verification Suite

Tests the 8 core requirements of the genuine offline clinical RAG fallback:
1. Missing Pinecone credentials automatically selects local retrieval.
2. Relevant clinical queries return excerpts from the matching tracked source documents
   (ACR Head Trauma, Fleischner Pulmonary Nodules, LI-RADS Liver Imaging).
3. Unrelated / out-of-domain queries abstain rather than return fabricated guidance.
4. Similarity scores are actual computed cosine values, not fabricated constants (no 0.95 or +0.45).
5. Source filenames (.md) and guideline titles are preserved in the consensus report.
6. Pinecone continues to take priority when a valid Pinecone index is active.
7. Missing embedding dependencies result in a clear, safe unavailable state.
8. LLM synthesis can remain unavailable (HF_TOKEN unset) without suppressing successful local RAG.
9. In-memory embeddings cache reuses precomputed vectors without re-indexing per query.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rag.local_retriever import (
    LocalClinicalRAGRetriever,
    DEFAULT_CLINICAL_SOURCES_DIR,
    DEFAULT_RELEVANCE_THRESHOLD
)
from pinecone_rag_engine import PineconeClinicalRAG
from medical_agent import ASCADEMedicalBoard


class TestLocalRAGFallback(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.retriever = LocalClinicalRAGRetriever(DEFAULT_CLINICAL_SOURCES_DIR)

    # -------------------------------------------------------------------------
    # 1. Missing Pinecone credentials selects local retrieval
    # -------------------------------------------------------------------------
    def test_missing_pinecone_credentials_selects_local_retrieval(self):
        engine = PineconeClinicalRAG()
        # Ensure Pinecone is offline/unconfigured
        engine.retriever.is_active = False

        res = engine.query_dense_rag(
            organ="Chest",
            severity="Candidate ROI Detected",
            user_query="pulmonary nodule management"
        )
        self.assertEqual(res["retrieval_mode"], "local_semantic")
        self.assertIn("Local Semantic RAG", res["vector_engine"])
        self.assertGreater(res["matches_count"], 0)
        self.assertIsNotNone(res["similarity_score"])

    # -------------------------------------------------------------------------
    # 2. Relevant queries return excerpts from matching tracked source documents
    # -------------------------------------------------------------------------
    def test_relevant_queries_return_matching_sources(self):
        # A. Chest query -> Fleischner
        res_chest = self.retriever.query_clinical_guidelines("Chest - solid pulmonary nodule >8mm", top_k=2)
        self.assertEqual(res_chest["status"], "success")
        self.assertEqual(res_chest["retrieval_mode"], "local_semantic")
        top_chunk_chest = res_chest["retrieved_chunks"][0]
        self.assertEqual(top_chunk_chest["source_file"], "fleischner_society_pulmonary_nodules.md")
        self.assertIn("Fleischner", top_chunk_chest["title"])

        # B. Brain query -> ACR Head Trauma
        res_brain = self.retriever.query_clinical_guidelines("Brain - acute head injury subdural hematoma", top_k=2)
        self.assertEqual(res_brain["status"], "success")
        top_chunk_brain = res_brain["retrieved_chunks"][0]
        self.assertEqual(top_chunk_brain["source_file"], "acr_appropriateness_head_trauma.md")
        self.assertIn("Head Trauma", top_chunk_brain["title"])

        # C. Abdomen query -> LI-RADS
        res_abdo = self.retriever.query_clinical_guidelines("Abdomen - liver lesion hepatocellular carcinoma LI-RADS", top_k=2)
        self.assertEqual(res_abdo["status"], "success")
        top_chunk_abdo = res_abdo["retrieved_chunks"][0]
        self.assertEqual(top_chunk_abdo["source_file"], "lirads_liver_imaging_reporting.md")
        self.assertIn("LI-RADS", top_chunk_abdo["title"])

    # -------------------------------------------------------------------------
    # 2b. Regression Test: Fleischner >8mm query prioritizes management recommendations
    # -------------------------------------------------------------------------
    def test_fleischner_solid_nodule_greater_than_8mm_prioritizes_management(self):
        query = "What do the Fleischner Society guidelines say about solid pulmonary nodules larger than 8 mm?"
        res = self.retriever.query_clinical_guidelines(query, top_k=2)

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["retrieval_mode"], "local_semantic")
        self.assertGreater(len(res["retrieved_chunks"]), 0)

        top_chunk = res["retrieved_chunks"][0]
        # Top chunk must be the specific solid nodule management recommendation, NOT title/scope
        self.assertNotIn("Scope & Patient Population", top_chunk["title"])
        self.assertIn("Nodules > 8 mm", top_chunk["title"])
        self.assertEqual(top_chunk["source_file"], "fleischner_society_pulmonary_nodules.md")

        # Must contain genuine cosine similarity > 0.70
        self.assertGreater(top_chunk["score"], 0.70)

        # Snippet/references must surface the actual clinical management guidance
        top_ref = res["kb_references"][0]
        self.assertIn("Consider follow-up CT at 3 months, PET-CT, or tissue biopsy", top_ref)
        self.assertIn("Nodules > 8 mm", top_ref)

        # Confirm through the full Medical Board consult
        board = ASCADEMedicalBoard()
        scan_ctx = {
            "organ": "Chest",
            "severity": "Candidate ROI Detected",
            "psnr_delta": None,
            "ssim_delta": None,
            "findings": ["Research query: solid pulmonary nodule larger than 8 mm"]
        }
        board_res = board.consult(scan_ctx, user_query=query)
        report = board_res["consensus_report"]

        self.assertIn("Clinical Specialist Vector RAG (Local Semantic RAG)", report)
        self.assertIn("Consider follow-up CT at 3 months, PET-CT, or tissue biopsy", report)
        self.assertIn("Nodules > 8 mm", report)
        self.assertIn("fleischner_society_pulmonary_nodules.md", report)

    # -------------------------------------------------------------------------
    # 3. Unrelated queries abstain rather than return unrelated clinical guidance
    # -------------------------------------------------------------------------
    def test_unrelated_queries_abstain(self):
        unrelated_queries = [
            "How to bake a chocolate cake with cream cheese frosting",
            "Quantum computing qubit superconducting coherence time",
            "Auto insurance liability coverage minimum limits"
        ]
        for query in unrelated_queries:
            res = self.retriever.query_clinical_guidelines(query, top_k=2)
            self.assertEqual(res["status"], "abstained")
            self.assertEqual(res["retrieval_mode"], "abstained")
            self.assertEqual(res["matches_count"], 0)
            self.assertEqual(len(res["retrieved_chunks"]), 0)
            self.assertEqual(len(res["kb_references"]), 0)
            self.assertIn("below clinical relevance threshold", res["notice"])

    # -------------------------------------------------------------------------
    # 4. Similarity scores are actual computed values and not fabricated
    # -------------------------------------------------------------------------
    def test_similarity_scores_are_computed_not_fabricated(self):
        res = self.retriever.query_clinical_guidelines("Chest - solid pulmonary nodule", top_k=2)
        score = res["top_score"]

        # Must be a float strictly between 0.0 and 1.0
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.40)
        self.assertLess(score, 1.0)

        # Must NOT equal legacy fabricated values (0.95 or 0.85 default bonuses)
        self.assertNotEqual(score, 0.95)
        self.assertNotEqual(score, 0.85)

        # Score in retrieved chunk must match top_score
        chunk_score = res["retrieved_chunks"][0]["score"]
        self.assertEqual(round(score, 4), round(chunk_score, 4))

    # -------------------------------------------------------------------------
    # 5. Source filenames/titles are preserved in the consensus report
    # -------------------------------------------------------------------------
    def test_source_filenames_and_titles_preserved_in_report(self):
        board = ASCADEMedicalBoard()
        scan_ctx = {
            "organ": "Chest",
            "severity": "Candidate ROI Detected",
            "psnr_delta": None,
            "ssim_delta": None,
            "findings": ["Spiculated pulmonary nodule measuring 7mm"]
        }
        res = board.consult(scan_ctx, user_query="Recommended CT follow-up for pulmonary nodule?")
        report = res["consensus_report"]

        # Must identify Local Semantic RAG
        self.assertIn("Clinical Specialist Vector RAG (Local Semantic RAG)", report)
        self.assertIn("Local Semantic RAG (all-MiniLM-L6-v2 / 384-d in-memory)", report)

        # Must cite the source filename and guideline title
        self.assertIn("fleischner_society_pulmonary_nodules.md", report)
        self.assertIn("Fleischner Society Guidelines", report)

        # Must contain raw cosine similarity without claiming diagnostic confidence
        self.assertIn("Raw Cosine Similarity:", report)
        self.assertIn("vector relevance score, not diagnostic confidence", report)

    # -------------------------------------------------------------------------
    # 6. Pinecone continues to be used when a valid mocked Pinecone succeeds
    # -------------------------------------------------------------------------
    def test_pinecone_continues_to_be_used_when_mock_succeeds(self):
        engine = PineconeClinicalRAG()
        # Mock active Pinecone retriever
        engine.retriever.is_active = True
        mock_chunks = [{
            "chunk_id": "pinecone-mock-001",
            "score": 0.8842,
            "title": "Mock Cloud ACR Guideline",
            "source_file": "cloud_acr.pdf",
            "content": "Pinecone cloud guideline text content",
            "metadata": {"differential": ["Cloud Traumatic Injury"]}
        }]
        engine.retriever.query_clinical_guidelines = MagicMock(return_value={
            "status": "success",
            "retrieved_chunks": mock_chunks
        })

        res = engine.query_dense_rag(organ="Brain", user_query="Head trauma protocol")
        self.assertEqual(res["retrieval_mode"], "pinecone")
        self.assertIn("Pinecone Cloud Index", res["vector_engine"])
        self.assertEqual(res["similarity_score"], 0.8842)
        self.assertIn("Mock Cloud ACR Guideline", res["top_match_condition"])

    # -------------------------------------------------------------------------
    # 7. Missing embedding dependencies result in a clear, safe unavailable state
    # -------------------------------------------------------------------------
    def test_missing_embedding_dependencies_safe_unavailable(self):
        broken_retriever = LocalClinicalRAGRetriever()
        broken_retriever.embedder = MagicMock()
        broken_retriever.embedder.embed_batch.side_effect = RuntimeError("sentence-transformers not installed")
        broken_retriever.embedder.embed_text.side_effect = RuntimeError("sentence-transformers not installed")

        # Invalidate cache for temporary test dir
        test_dir = os.path.join(REPO_ROOT, "knowledge_base", "clinical_sources")
        if test_dir in LocalClinicalRAGRetriever._shared_cache:
            del LocalClinicalRAGRetriever._shared_cache[test_dir]

        res = broken_retriever.query_clinical_guidelines("Chest - nodule")
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["retrieval_mode"], "dependency_error")
        self.assertIn("Offline RAG dependency error", res["error"])
        self.assertEqual(res["matches_count"], 0)
        self.assertIsNone(res["top_score"])

    # -------------------------------------------------------------------------
    # 8. LLM synthesis can remain unavailable without suppressing local RAG
    # -------------------------------------------------------------------------
    def test_llm_synthesis_unavailable_without_suppressing_local_rag(self):
        board = ASCADEMedicalBoard()
        scan_ctx = {
            "organ": "Brain",
            "severity": "High Risk",
            "psnr_delta": None,
            "ssim_delta": None,
            "findings": ["Extra-axial hyperdensity with mass effect"]
        }

        # Explicitly remove HF_TOKEN to test unconfigured LLM state
        with patch.dict(os.environ, {"HF_TOKEN": ""}, clear=False):
            res = board.consult(scan_ctx, user_query="Acute subdural hematoma protocol")
            report = res["consensus_report"]

            # Section 2: Local RAG evidence must be present
            self.assertIn("Clinical Specialist Vector RAG (Local Semantic RAG)", report)
            self.assertIn("acr_appropriateness_head_trauma.md", report)
            self.assertIn("Raw Cosine Similarity:", report)

            # Section 3: Honest reporting that LLM is unavailable
            self.assertIn("LLM synthesis service unavailable (HF_TOKEN not configured)", report)
            self.assertNotIn("Fabricated LLM diagnosis", report)

    # -------------------------------------------------------------------------
    # 9. In-memory embeddings cache reuses precomputed vectors
    # -------------------------------------------------------------------------
    def test_in_memory_embeddings_cache_reuses_vectors(self):
        retriever1 = LocalClinicalRAGRetriever()
        retriever1.query_clinical_guidelines("Chest nodule")

        # Inspect cache
        cache_key = retriever1.sources_dir
        self.assertIn(cache_key, LocalClinicalRAGRetriever._shared_cache)
        cached_vectors = LocalClinicalRAGRetriever._shared_cache[cache_key]["vectors"]
        self.assertIsNotNone(cached_vectors)
        self.assertEqual(cached_vectors.shape[1], 384)

        # Mock embed_batch to verify it is NOT called again on subsequent queries
        with patch.object(retriever1.embedder, "embed_batch") as mock_embed_batch:
            retriever2 = LocalClinicalRAGRetriever()
            retriever2.query_clinical_guidelines("Brain trauma")
            mock_embed_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
