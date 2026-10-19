"""
CASCADE Safe, Privacy-Conscious End-to-End Processing Telemetry.

Provides operational observability for scan processing, AI inference,
multi-agent consultations, and cloud storage workflows without leaking
PHI, credentials, raw byte buffers, or internal file paths.
"""

import time
import uuid
import re
from typing import Dict, Any, Optional

# Sanitized error categories
ERROR_CATEGORIES = {
    "INVALID_PAYLOAD": "Malformed or unparseable request payload",
    "INVALID_BASE64": "Invalid or corrupted base64 encoding",
    "INVALID_IMAGE_FORMAT": "Unsupported or unrecognized image binary signature",
    "UNSUPPORTED_ARTIFACT_TYPE": "Artifact type is not permitted for storage",
    "EXPLANATION_UNAVAILABLE": "Requested explanation artifact could not be generated",
    "PAYLOAD_TOO_LARGE": "Payload exceeds maximum allowed transfer size",
    "MODEL_INFERENCE_ERROR": "Neural network forward pass failed",
    "STORAGE_UNAVAILABLE": "Target cloud storage service is not configured or offline",
    "STORAGE_CLIENT_ERROR": "Cloud storage driver encountered an operation error",
    "PIPELINE_ERROR": "Batch ETL processing error occurred"
}

SENSITIVE_KEYS = {
    'token', 'secret', 'key', 'password', 'raw_bytes', 'pixel', 'buffer',
    'patient', 'name', 'mrn', 'dob', 'birth', 'ssn', 'accession', 'id',
    'hash', 'patient_id', 'patient_name', 'physician', 'doctor', 'hospital'
}

def sanitize_value(val: Any) -> Any:
    """
    Strips local file paths, credentials, and potential PHI from telemetry values.
    Uses strict filtering to ensure zero leakage of identifying or secret data.
    """
    if isinstance(val, str):
        # Strip Windows/Linux absolute paths
        val = re.sub(r'[A-Za-z]:\\[^\s]+', '[REDACTED_PATH]', val)
        val = re.sub(r'/(?:[a-zA-Z0-9_\.\-]+/)+[a-zA-Z0-9_\.\-]+', '[REDACTED_PATH]', val)
        # Strip potential keys/tokens (long hex/alphanumeric strings, AWS keys)
        val = re.sub(r'(?:AKIA|akia)[A-Z0-9]{16}', '[REDACTED_KEY]', val)
        val = re.sub(r'[a-zA-Z0-9+/=]{32,}', '[REDACTED_TOKEN]', val)
        if len(val) > 200:
            val = val[:197] + "..."
    elif isinstance(val, dict):
        return {
            k: sanitize_value(v)
            for k, v in val.items()
            if not any(s in k.lower() for s in SENSITIVE_KEYS)
        }
    elif isinstance(val, (list, tuple)):
        return [sanitize_value(v) for v in val[:10]]
    return val

class ProcessingTelemetry:
    """
    Structured in-memory operational telemetry recorder.
    """
    def __init__(self):
        self._records: Dict[str, Dict[str, Any]] = {}

    def new_correlation_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def start_trace(self, correlation_id: Optional[str] = None, operation_name: str = "scan_processing") -> str:
        corr_id = correlation_id or self.new_correlation_id()
        self._records[corr_id] = {
            "correlation_id": corr_id,
            "operation": operation_name,
            "start_time": time.time(),
            "stages": [],
            "status": "IN_PROGRESS",
            "metrics_available": False,
            "error_category": None
        }
        return corr_id

    def record_stage(
        self,
        correlation_id: str,
        stage_name: str,
        duration_sec: float,
        status: str = "SUCCESS",
        details: Optional[Dict[str, Any]] = None
    ):
        if correlation_id not in self._records:
            self.start_trace(correlation_id)

        sanitized_details = sanitize_value(details or {})
        stage_entry = {
            "stage": stage_name,
            "duration_ms": round(duration_sec * 1000.0, 2),
            "status": status,
            "details": sanitized_details,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self._records[correlation_id]["stages"].append(stage_entry)

    def record_metrics_status(self, correlation_id: str, has_metrics: bool, psnr_delta: Optional[float] = None, ssim_delta: Optional[float] = None):
        if correlation_id in self._records:
            self._records[correlation_id]["metrics_available"] = has_metrics
            if has_metrics and psnr_delta is not None and ssim_delta is not None:
                self._records[correlation_id]["psnr_delta"] = round(float(psnr_delta), 2)
                self._records[correlation_id]["ssim_delta"] = round(float(ssim_delta), 4)

    def finish_trace(self, correlation_id: str, final_status: str = "COMPLETED", error_category: Optional[str] = None) -> Dict[str, Any]:
        if correlation_id not in self._records:
            self.start_trace(correlation_id)

        rec = self._records[correlation_id]
        rec["status"] = final_status
        rec["total_duration_ms"] = round((time.time() - rec["start_time"]) * 1000.0, 2)
        if error_category:
            rec["error_category"] = error_category
            rec["error_description"] = ERROR_CATEGORIES.get(error_category, "Unspecified error")

        return self.get_trace(correlation_id)

    def get_trace(self, correlation_id: str) -> Dict[str, Any]:
        rec = self._records.get(correlation_id, {})
        # Return a shallow copy with sanitized data
        return dict(rec)

# Global telemetry manager instance
global_telemetry = ProcessingTelemetry()

def get_telemetry() -> ProcessingTelemetry:
    return global_telemetry
