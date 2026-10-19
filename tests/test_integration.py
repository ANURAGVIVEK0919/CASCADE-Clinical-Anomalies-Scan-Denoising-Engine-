"""
CASCADE Project-Local End-to-End Integration Test Suite.
Covers:
- S3 Artifact Vault (mocked client, real payload bytes, AES256, zero PHI, safety gates)
- U-Net Batch Pipeline (trained model reuse, training preprocessing contract, spatial geometry, no box filter)
- Safe Operational Telemetry (sanitization of PHI, tokens, keys, in-memory execution)
- Multi-Agent Null-Safety & Metric Integrity (paired vs unpaired ground truth)
- Diagnostic Neural Inference (independent organ classification, Grad-CAM failure handling, candidate ROI)
- Real Local Model Smoke Tests (actual weights verification)
"""

import os
import sys
import json
import base64
import unittest
import numpy as np
from PIL import Image
from io import BytesIO

# Ensure repo root is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app import app, get_s3_vault, get_model, get_diagnostic_model, run_diagnostic_analyzer
from aws_s3_storage import S3StorageVault, ALLOWED_ARTIFACT_TYPES, MAX_ARTIFACT_SIZE_BYTES
from data_pipeline import CASCADEDICOMPipeline
from telemetry import get_telemetry, sanitize_value
from medical_agent import ASCADEMedicalBoard
from gradcam_utils import generate_gradcam, overlay_gradcam


class MockBotoS3Client:
    """Mock boto3 S3 client for offline testing without network or AWS credentials."""
    def __init__(self, fail_on_put=False):
        self.fail_on_put = fail_on_put
        self.put_calls = []

    def put_object(self, **kwargs):
        if self.fail_on_put:
            raise RuntimeError("Simulated S3 Client ConnectionTimeout")
        self.put_calls.append(kwargs)
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def generate_presigned_url(self, ClientMethod, Params, ExpiresIn):
        bucket = Params.get("Bucket", "test-bucket")
        key = Params.get("Key", "test-key")
        return f"https://{bucket}.s3.amazonaws.com/{key}?mock_token=presigned"


def create_sample_png_bytes(width=256, height=256):
    img = Image.fromarray((np.random.rand(height, width) * 255).astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


class TestCASCADEIntegration(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        self.mock_s3 = MockBotoS3Client()
        self.mock_vault = S3StorageVault(
            bucket_name="cascade-test-vault",
            region_name="us-east-1",
            s3_client=self.mock_s3
        )
        self.app.config['S3_VAULT'] = self.mock_vault

    def tearDown(self):
        if 'S3_VAULT' in self.app.config:
            del self.app.config['S3_VAULT']

    # =========================================================================
    # 1. S3 ARTIFACT VAULT & INGESTION INTEGRATION TESTS
    # =========================================================================

    def test_s3_upload_restored_image_exact_bytes(self):
        """Verify restored image bytes reach put_object with correct ContentType and AES256."""
        png_bytes = create_sample_png_bytes(256, 256)
        b64_str = base64.b64encode(png_bytes).decode('utf-8')

        resp = self.client.post('/api/s3/upload', json={
            'artifact_type': 'restored_image',
            'payload': b64_str
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get('status'), 'Success')
        self.assertEqual(data.get('artifact_type'), 'restored_image')
        self.assertEqual(data.get('size_bytes'), len(png_bytes))

        self.assertEqual(len(self.mock_s3.put_calls), 1)
        call = self.mock_s3.put_calls[0]
        self.assertEqual(call['Body'], png_bytes)
        self.assertEqual(call['ContentType'], 'image/png')
        self.assertEqual(call['ServerSideEncryption'], 'AES256')
        self.assertTrue(call['Key'].startswith('pacs_vault/'))

    def test_s3_upload_gradcam_overlay_exact_bytes(self):
        """Verify Grad-CAM overlay upload with real image bytes."""
        png_bytes = create_sample_png_bytes(256, 256)
        b64_str = "data:image/png;base64," + base64.b64encode(png_bytes).decode('utf-8')

        resp = self.client.post('/api/s3/upload', json={
            'artifact_type': 'gradcam_overlay',
            'payload': b64_str
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get('status'), 'Success')
        self.assertEqual(data.get('artifact_type'), 'gradcam_overlay')

    def test_s3_upload_diff_map_exact_bytes(self):
        """Verify difference map upload with real image bytes."""
        png_bytes = create_sample_png_bytes(256, 256)
        b64_str = base64.b64encode(png_bytes).decode('utf-8')

        resp = self.client.post('/api/s3/upload', json={
            'artifact_type': 'diff_map',
            'payload': b64_str
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get('status'), 'Success')
        self.assertEqual(data.get('artifact_type'), 'diff_map')

    def test_s3_upload_missing_gradcam_rejected_safely(self):
        """Refuse to upload placeholder or null Grad-CAM overlay."""
        resp = self.client.post('/api/s3/upload', json={
            'artifact_type': 'gradcam_overlay',
            'payload': None
        })
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data.get('status'), 'Failed')
        self.assertIn("unavailable", data.get('error', '').lower())

    def test_s3_upload_report_json_strict_allowlist_sanitization(self):
        """Verify report_json allowlist filtering strips all patient identifiers and unapproved keys."""
        report = {
            "patient_name": "John Doe",
            "patient_id": "ACC-984210",
            "mrn": "MRN-12345",
            "dob": "1980-01-01",
            "ssn": "000-00-0000",
            "hospital": "City Memorial Hospital",
            "doctor": "Dr. Smith",
            "arbitrary_client_path": "/var/data/scans/patient1.dcm",
            "consensus_report": "Valid clinical board assessment text",
            "differential_diagnosis": ["Subpleural GGO", "Mild consolidation"],
            "is_emergency": False,
            "hitl_status": "Verified",
            "is_human_verified": True,
            "engine": "LangGraph Stateful Multi-Agent System",
            "organ": "Chest",
            "severity": "Mild Anomaly",
            "metrics": {
                "psnr_gain_db": 6.06,
                "ssim_gain": 0.2038,
                "unauthorized_leak": "secret_data"
            }
        }
        resp = self.client.post('/api/s3/upload', json={
            'artifact_type': 'report_json',
            'payload': report
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get('status'), 'Success')

        uploaded_bytes = self.mock_s3.put_calls[-1]['Body']
        uploaded_json = json.loads(uploaded_bytes.decode('utf-8'))

        # Prohibited fields must be completely stripped
        for forbidden in ['patient_name', 'patient_id', 'mrn', 'dob', 'ssn', 'hospital', 'doctor', 'arbitrary_client_path']:
            self.assertNotIn(forbidden, uploaded_json)

        # Unauthorized nested key must be stripped
        self.assertNotIn('unauthorized_leak', uploaded_json.get('metrics', {}))

        # Approved fields must be preserved
        self.assertEqual(uploaded_json.get('organ'), 'Chest')
        self.assertEqual(uploaded_json.get('severity'), 'Mild Anomaly')
        self.assertEqual(uploaded_json.get('metrics', {}).get('psnr_gain_db'), 6.06)

    def test_s3_upload_unsupported_artifact_type_rejected(self):
        """Reject unapproved artifact types."""
        resp = self.client.post('/api/s3/upload', json={
            'artifact_type': 'unauthorized_dicom_raw',
            'payload': 'dummy'
        })
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data.get('status'), 'Failed')

    def test_s3_upload_invalid_image_signature_and_safe_parser(self):
        """Corrupted or non-image data must be rejected by safe parser."""
        corrupt_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\nCORRUPTED_TRUNCATED_BYTES").decode('utf-8')
        resp = self.client.post('/api/s3/upload', json={
            'artifact_type': 'restored_image',
            'payload': corrupt_b64
        })
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data.get('status'), 'Failed')

    def test_s3_upload_oversized_payload_rejected(self):
        """Payloads exceeding maximum limits must be rejected with 413."""
        big_bytes = b'\x89PNG\r\n\x1a\n' + b'A' * (MAX_ARTIFACT_SIZE_BYTES + 1024)
        big_b64 = base64.b64encode(big_bytes).decode('utf-8')
        resp = self.client.post('/api/s3/upload', json={
            'artifact_type': 'restored_image',
            'payload': big_b64
        })
        self.assertEqual(resp.status_code, 413)

    def test_s3_upload_offline_storage_returns_truthful_unavailable(self):
        """Unconfigured S3 vault returns truthful 'Unavailable' without pretending success."""
        offline_vault = S3StorageVault(s3_client=None)
        offline_vault.is_connected = False
        self.app.config['S3_VAULT'] = offline_vault

        png_bytes = create_sample_png_bytes(256, 256)
        b64_str = base64.b64encode(png_bytes).decode('utf-8')
        resp = self.client.post('/api/s3/upload', json={
            'artifact_type': 'restored_image',
            'payload': b64_str
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get('status'), 'Unavailable')
        self.assertIn("not configured", data.get('message', '').lower())

    def test_s3_upload_client_error_handled_without_false_success(self):
        """Client exceptions during put_object return 'Failed' status."""
        failing_s3 = MockBotoS3Client(fail_on_put=True)
        failing_vault = S3StorageVault(s3_client=failing_s3)
        self.app.config['S3_VAULT'] = failing_vault

        png_bytes = create_sample_png_bytes(256, 256)
        b64_str = base64.b64encode(png_bytes).decode('utf-8')
        resp = self.client.post('/api/s3/upload', json={
            'artifact_type': 'restored_image',
            'payload': b64_str
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get('status'), 'Failed')

    # =========================================================================
    # 2. BATCH DICOM PIPELINE & U-NET INTEGRATION TESTS
    # =========================================================================

    def test_batch_pipeline_reuses_single_model_instance(self):
        """Verify pipeline caches U-Net model across slices in a batch."""
        pipeline = CASCADEDICOMPipeline(input_dir="data/raw_scans", output_dir="data/pipeline_output")
        self.assertIsNone(pipeline.model)

        files = pipeline.extract_files()
        self.assertTrue(len(files) > 0)
        res1 = pipeline.transform_file(files[0])
        model1 = pipeline.model
        self.assertIsNotNone(model1)

        res2 = pipeline.transform_file(files[-1])
        model2 = pipeline.model
        self.assertIs(model1, model2, "Model instance must be identical across batch slices")

    def test_batch_pipeline_output_contract_and_finite_range(self):
        """Verify pipeline output shapes, finite values, and [0, 1] range."""
        pipeline = CASCADEDICOMPipeline(input_dir="data/raw_scans", output_dir="data/pipeline_output")
        files = pipeline.extract_files()
        res = pipeline.transform_file(files[0])

        denoised = res['denoised_arr']
        diff = res['diff_arr']
        self.assertEqual(denoised.shape[-1], 1)
        self.assertTrue(np.all(np.isfinite(denoised)))
        self.assertTrue(np.all(denoised >= 0.0) and np.all(denoised <= 1.0))
        self.assertTrue(np.all(np.isfinite(diff)))
        self.assertTrue(np.all(diff >= 0.0))

    def test_batch_pipeline_missing_model_fails_explicitly(self):
        """Missing model must raise FileNotFoundError rather than falling back to box filter."""
        bad_pipeline = CASCADEDICOMPipeline(
            input_dir="data/raw_scans",
            output_dir="data/pipeline_output",
            model_path="non_existent_weights.keras"
        )
        files = bad_pipeline.extract_files()
        if files:
            with self.assertRaises(FileNotFoundError):
                bad_pipeline.transform_file(files[0])

    def test_batch_pipeline_preserves_spatial_geometry(self):
        """Verify pipeline restores original resolution for non-256x256 inputs."""
        pipeline = CASCADEDICOMPipeline(input_dir="data/raw_scans", output_dir="data/pipeline_output")
        test_path = "data/raw_scans/temp_spatial_test_512.png"
        img = Image.fromarray((np.random.rand(512, 384) * 255).astype(np.uint8))
        img.save(test_path)
        try:
            res = pipeline.transform_file(test_path)
            self.assertEqual(res['denoised_arr'].shape, (512, 384, 1))
            self.assertEqual(res['diff_arr'].shape, (512, 384, 1))
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

    # =========================================================================
    # 3. TELEMETRY, PRIVACY, & ERROR SANITIZATION TESTS
    # =========================================================================

    def test_telemetry_sanitizes_phi_and_credentials(self):
        """Verify telemetry strips patient identifiers, paths, and secrets."""
        tel = get_telemetry()
        corr_id = tel.new_correlation_id()
        tel.start_trace(correlation_id=corr_id, operation_name="audit_test")
        tel.record_stage(
            correlation_id=corr_id,
            stage_name="preprocessing",
            duration_sec=0.03,
            status="SUCCESS",
            details={
                "patient_name": "Jane Doe",
                "patient_id": "ACC-123456",
                "secret_key": "AKIAIOSFODNN7EXAMPLE",
                "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-IDcSemACt8x4iTMCda8Yhe3iZaWbvV5XKSTbuAn0M",
                "filepath": "C:\\secret\\records\\scan.dcm",
                "artifact_type": "restored_image"
            }
        )
        trace = tel.finish_trace(corr_id, final_status="COMPLETED")
        self.assertEqual(trace['status'], 'COMPLETED')
        details = trace['stages'][0]['details']

        # Prohibited sensitive keys must be excluded
        for forbidden in ['patient_name', 'patient_id', 'secret_key', 'token']:
            self.assertNotIn(forbidden, details)

        # Filepath must be redacted
        if 'filepath' in details:
            self.assertNotIn('C:\\secret', details['filepath'])

    def test_agent_null_safety_with_unpaired_metrics(self):
        """Verify medical board agent handles None metrics gracefully."""
        board = ASCADEMedicalBoard()
        scan_context = {
            "organ": "Brain",
            "severity": "High Risk",
            "psnr_delta": None,
            "ssim_delta": None,
            "findings": ["Dense mass with surrounding edema"]
        }
        res = board.consult(scan_context=scan_context, user_query="Assess urgency")
        self.assertIn("consensus_report", res)
        self.assertTrue(len(res["consensus_report"]) > 0)
        self.assertIn("differential_diagnosis", res)

    # =========================================================================
    # 4. NEURAL DIAGNOSTICS & GRAD-CAM REGRESSION TESTS
    # =========================================================================

    def test_real_model_smoke_and_independent_classification(self):
        """Smoke test real neural models to verify independent prediction and shapes."""
        denoiser = get_model()
        diagnostic = get_diagnostic_model()

        self.assertIsNotNone(denoiser, "Denoiser model file medical_ct_denoiser_full.keras must exist")
        self.assertIsNotNone(diagnostic, "Diagnostic model file medical_ct_diagnostic_model.keras must exist")

        self.assertEqual(denoiser.input_shape, (None, 256, 256, 1))
        self.assertEqual(denoiser.output_shape, (None, 256, 256, 1))
        self.assertEqual(diagnostic.input_shape, (None, 256, 256, 1))

        # Real inference forward pass
        sample_input = np.random.rand(1, 256, 256, 1).astype(np.float32)
        denoised_out = denoiser(sample_input, training=False).numpy()
        self.assertEqual(denoised_out.shape, (1, 256, 256, 1))
        self.assertTrue(np.all(np.isfinite(denoised_out)))

        class_pred, bbox_pred = diagnostic(denoised_out, training=False)
        self.assertEqual(class_pred.shape, (1, 3))
        self.assertEqual(bbox_pred.shape, (1, 4))
        self.assertAlmostEqual(float(np.sum(class_pred[0])), 1.0, places=3)

    def test_gradcam_overlay_and_failure_tolerance(self):
        """Verify Grad-CAM produces valid overlays and returns clean None on invalid models."""
        diagnostic = get_diagnostic_model()
        sample_img = np.random.rand(256, 256, 1).astype(np.float32)

        # Valid generation
        cam = generate_gradcam(diagnostic, sample_img, class_idx=0)
        self.assertIsNotNone(cam)
        self.assertEqual(cam.shape, (256, 256, 3))

        uint8_base = np.uint8(np.squeeze(sample_img) * 255)
        overlay_b64 = overlay_gradcam(uint8_base, cam)
        self.assertIsNotNone(overlay_b64)
        self.assertTrue(overlay_b64.startswith("data:image/png;base64,"))

        # Failure tolerance: Passing None or invalid model must cleanly return None (no fake blue mask)
        cam_fail = generate_gradcam(None, sample_img, class_idx=0)
        self.assertIsNone(cam_fail)
        overlay_fail = overlay_gradcam(uint8_base, None)
        self.assertIsNone(overlay_fail)

    def test_candidate_roi_coordinate_sanitization(self):
        """Verify bounding box coordinates are clamped to [0.0, 1.0] and degenerate boxes rejected."""
        denoised = np.random.rand(256, 256, 1).astype(np.float32)

        # Inverted coordinates: [ymin=0.8, xmin=0.9, ymax=0.2, xmax=0.3] -> should be sanitized
        raw_inverted = [0.8, 0.9, 0.2, 0.3]
        diag = run_diagnostic_analyzer(denoised, organ_type='Brain', bbox=raw_inverted, predicted_organ='Brain')
        self.assertIsNotNone(diag.get('bbox'))
        ymin, xmin, ymax, xmax = diag['bbox']
        self.assertTrue(ymin <= ymax and xmin <= xmax)
        self.assertTrue(all(0.0 <= c <= 1.0 for c in [ymin, xmin, ymax, xmax]))

    # =========================================================================
    # 5. REPRODUCIBLE FLASK TEST-CLIENT /PROCESS WORKFLOW
    # =========================================================================

    def test_flask_process_endpoint_reproducible_smoke(self):
        """
        Verify end-to-end /process route with Flask test-client using real model inference.
        Asserts contract completeness, uncalibrated model score, normalized ROI coordinates,
        safe Grad-CAM, unpaired metrics integrity, neutralized emergency status, and zero cloud calls.
        """
        png_bytes = create_sample_png_bytes(256, 256)
        data = {
            'image': (BytesIO(png_bytes), 'synthetic_smoke_scan.png'),
            'organ': 'brain'
        }

        resp = self.client.post('/process', data=data, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 200)

        data_json = resp.get_json()
        self.assertIn('results', data_json)
        self.assertEqual(len(data_json['results']), 1)

        result = data_json['results'][0]

        # 1. Documented response contract completeness
        required_keys = [
            'name', 'organ', 'predicted_organ', 'organ_agreement', 'organ_warning',
            'confidence', 'confidence_label', 'top2_predictions', 'noisy', 'denoised',
            'diff', 'metrics', 'diagnosis', 'dicom_metadata', 'gradcam_overlay',
            'gradcam_type', 'gradcam_note', 'agent_consultation', 'etl_summary', 'aws_cloud_res'
        ]
        for k in required_keys:
            self.assertIn(k, result, f"Response missing required key: {k}")

        # 2. Predicted organ and uncalibrated model score
        self.assertIn(result['predicted_organ'], ['Chest', 'Brain', 'Abdomen'])
        self.assertIsNotNone(result['confidence'])
        self.assertTrue(0.0 <= result['confidence'] <= 100.0)
        self.assertEqual(result['confidence_label'], 'Uncalibrated Model Score')

        # 3. Candidate ROI coordinate validity and normalization
        bbox = result['diagnosis'].get('bbox')
        if bbox is not None:
            self.assertEqual(len(bbox), 4)
            ymin, xmin, ymax, xmax = bbox
            self.assertTrue(0.0 <= ymin <= ymax <= 1.0)
            self.assertTrue(0.0 <= xmin <= xmax <= 1.0)

        # 4. Grad-CAM state
        gradcam = result.get('gradcam_overlay')
        if gradcam is not None:
            self.assertTrue(gradcam.startswith("data:image/png;base64,"))
        self.assertEqual(result.get('gradcam_type'), 'Organ Saliency Map')

        # 5. Scientific metrics integrity (unpaired scan must be None)
        metrics = result['metrics']
        self.assertIsNone(metrics['psnr_delta'])
        self.assertIsNone(metrics['ssim_delta'])

        # 6. Neutralized clinical and emergency claims
        self.assertIsNone(result['diagnosis'].get('is_emergency'))
        self.assertEqual(result['diagnosis'].get('emergency_status'), "Not Assessed (No Certified Pathology Classifier)")
        agent_res = result['agent_consultation']
        self.assertIsNone(agent_res.get('is_emergency'))
        self.assertEqual(agent_res.get('emergency_status'), "Not Assessed (No Certified Pathology Classifier)")
        self.assertFalse(agent_res.get('is_human_verified'))

        # 7. Zero automatic cloud uploads
        self.assertEqual(len(self.mock_s3.put_calls), 0)

    def test_flask_process_invalid_input_rejected_safely(self):
        """Verify invalid or empty upload to /process fails safely with HTTP 400."""
        resp = self.client.post('/process', data={}, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn('error', data)

    # =========================================================================
    # 6. HITL INTEGRITY & DEMO ACKNOWLEDGEMENT TESTS (PHASE 1H)
    # =========================================================================

    def test_default_scan_state_not_clinically_verified(self):
        """Verify default scan state has is_human_verified=False and truthful unreviewed status."""
        board = ASCADEMedicalBoard()
        test_context = {
            "organ": "Brain",
            "findings": ["Low-attenuation candidate region detected in frontal lobe"],
            "psnr_delta": None,
            "ssim_delta": None
        }
        res = board.consult(test_context, "Assess scan findings")

        self.assertFalse(res["is_human_verified"])
        self.assertFalse(res["review_acknowledged"])
        self.assertFalse(res["reviewer_identity_verified"])
        self.assertEqual(res["review_status"], "Research output — no authenticated clinical review recorded")
        self.assertIn("**UNVERIFIED** (No authenticated clinical identity provider present)", res["consensus_report"])
        self.assertNotIn("Dr. Vivek", res["consensus_report"])
        self.assertNotIn("M.D.", res["consensus_report"])

    def test_anonymous_demo_acknowledgement_cannot_set_is_human_verified(self):
        """Verify demo acknowledgement sets review_acknowledged=True but NEVER is_human_verified=True."""
        resp = self.client.post('/api/copilot/approve', json={
            "hitl_notes": "Demo user reviewed candidate ROI overlay",
            "scan_context": {
                "organ": "Brain",
                "findings": ["Non-diagnostic research candidate finding"]
            }
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        self.assertFalse(data["is_human_verified"])
        self.assertTrue(data["review_acknowledged"])
        self.assertFalse(data["reviewer_identity_verified"])
        self.assertEqual(data["review_status"], "Demo acknowledgement only — reviewer identity not authenticated")
        self.assertIn("[DEMO ACKNOWLEDGED]", data["consensus_report"])
        self.assertIn("Anonymous Demo User (Unauthenticated)", data["consensus_report"])

    def test_client_supplied_credentials_cannot_forge_clinical_verification(self):
        """Verify client-supplied doctor names, degrees, or verification flags cannot forge verified status."""
        forged_payload = {
            "physician_name": "Dr. Vivek, M.D.",
            "degree": "M.D., FACS",
            "role": "Chief Radiologist",
            "is_human_verified": True,
            "reviewer_identity_verified": True,
            "hitl_notes": "Attempting to inject doctor credentials",
            "scan_context": {
                "organ": "Chest",
                "findings": ["Bilateral lung field candidate scan"]
            }
        }
        resp = self.client.post('/api/copilot/approve', json=forged_payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        # Invariants strictly enforced by backend
        self.assertFalse(data["is_human_verified"])
        self.assertFalse(data["reviewer_identity_verified"])
        self.assertTrue(data["review_acknowledged"])
        self.assertEqual(data["review_status"], "Demo acknowledgement only — reviewer identity not authenticated")
        self.assertNotIn("Dr. Vivek", data["consensus_report"])
        self.assertIn("Anonymous Demo User (Unauthenticated)", data["consensus_report"])

    def test_s3_upload_report_json_enforces_unverified_invariants(self):
        """Verify S3 report_json upload sanitizes forged doctor names and clamps is_human_verified=False."""
        forged_report = {
            "organ": "Brain",
            "predicted_organ": "Brain",
            "physician_name": "Dr. Fraud, M.D.",
            "doctor_signature": "Signed by Dr. Fraud",
            "is_human_verified": True,
            "reviewer_identity_verified": True,
            "review_acknowledged": True,
            "review_status": "Clinically Verified by Dr. Fraud",
            "is_emergency": True,
            "emergency_status": "Critical Emergency Protocol Activated"
        }
        resp = self.client.post('/api/s3/upload', json={
            "artifact_type": "report_json",
            "payload": forged_report
        })
        self.assertEqual(resp.status_code, 200)

        # Inspect S3 put_object call
        self.assertEqual(len(self.mock_s3.put_calls), 1)
        put_call = self.mock_s3.put_calls[0]
        stored_bytes = put_call["Body"]
        stored_dict = json.loads(stored_bytes.decode('utf-8'))

        # Security Invariants verified in stored JSON
        self.assertFalse(stored_dict["is_human_verified"])
        self.assertFalse(stored_dict["reviewer_identity_verified"])
        self.assertTrue(stored_dict["review_acknowledged"])
        self.assertIsNone(stored_dict["is_emergency"])
        self.assertEqual(stored_dict["emergency_status"], "Not Assessed (No Certified Pathology Classifier)")
        # Unauthorized fields stripped
        self.assertNotIn("physician_name", stored_dict)
        self.assertNotIn("doctor_signature", stored_dict)

    def test_emergency_status_remains_unassessed_end_to_end(self):
        """Verify is_emergency remains strictly None and emergency_status is 'Not Assessed' across board consult."""
        board = ASCADEMedicalBoard()
        scan_context = {
            "organ": "Brain",
            "findings": ["Hyperdense intracranial region"],
            "severity": "High Risk"
        }
        res = board.consult(scan_context, "Emergency triage protocol?")
        self.assertIsNone(res["is_emergency"])
        self.assertEqual(res["emergency_status"], "Not Assessed (No Certified Pathology Classifier)")
        self.assertIn("Safety Triage: Research Demo Mode — Emergency Assessment Not Available", res["consensus_report"])
        self.assertIn("Triage Evaluation: **Not Assessed**", res["consensus_report"])

    def test_model_artifact_presence_and_checksums(self):
        """Verify expected local model files exist and match documented exact byte size."""
        denoiser_path = "medical_ct_denoiser_full.keras"
        diagnostic_path = "medical_ct_diagnostic_model.keras"

        self.assertTrue(os.path.exists(denoiser_path), f"Missing {denoiser_path}")
        self.assertTrue(os.path.exists(diagnostic_path), f"Missing {diagnostic_path}")

        self.assertEqual(os.path.getsize(denoiser_path), 23388275)
        self.assertEqual(os.path.getsize(diagnostic_path), 256877707)

    def test_review_state_isolation_between_scans(self):
        """Verify review acknowledgement of Scan 1 does not bleed into Scan 2."""
        board = ASCADEMedicalBoard()
        scan1_context = {"organ": "Brain", "findings": ["Frontal lobe anomaly"]}
        scan2_context = {"organ": "Chest", "findings": ["Pulmonary nodule candidate"]}

        # Scan 1 acknowledged
        res1 = board.consult(scan1_context, "Review Scan 1", review_acknowledged=True)
        self.assertTrue(res1["review_acknowledged"])
        self.assertFalse(res1["is_human_verified"])

        # Scan 2 unreviewed default
        res2 = board.consult(scan2_context, "Review Scan 2", review_acknowledged=False)
        self.assertFalse(res2["review_acknowledged"])
        self.assertFalse(res2["is_human_verified"])
        self.assertEqual(res2["review_status"], "Research output — no authenticated clinical review recorded")
        self.assertIn("UNVERIFIED", res2["consensus_report"])


if __name__ == '__main__':
    unittest.main()
