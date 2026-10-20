"""
CASCADE Upload Safety & Cloud Run Hardening Unit Tests.

Verifies:
1. Temporary uploads in /process are strictly stored in a private directory OUTSIDE static/.
2. Temporary files are completely unlinked after successful processing.
3. Temporary files are unlinked even if an unhandled exception occurs during processing.
4. Primary temporary file is unlinked if saving the secondary clean file fails.
5. Unsupported file extensions (e.g., .sh, .py, .exe, .bin) are rejected with HTTP 400.
6. Path traversal filenames cannot escape the private upload folder and are safely cleaned up.
7. Declared Content-Length exceeding 32 MB returns HTTP 413.
8. Unbounded/oversized payload parsing triggers Flask's MAX_CONTENT_LENGTH returning sanitized HTTP 413.
9. Zero patient data or temporary scans are stored in static/uploads/.
"""

import io
import os
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

from app import app, ALLOWED_IMAGE_EXTENSIONS, MAX_IMAGE_UPLOAD_SIZE


def create_test_png_bytes(width=64, height=64):
    buf = io.BytesIO()
    img = Image.new('L', (width, height), color=128)
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.getvalue()


class TestUploadSafetyHardening(unittest.TestCase):

    def setUp(self):
        self.app = app
        self.app.config['TESTING'] = True
        self.upload_folder = self.app.config.get('UPLOAD_FOLDER')
        os.makedirs(self.upload_folder, exist_ok=True)
        self.static_uploads = os.path.join(self.app.root_path, 'static', 'uploads')
        self.client = self.app.test_client()

    def _get_upload_files(self):
        """Returns set of all filenames in the private upload directory."""
        if not os.path.exists(self.upload_folder):
            return set()
        return set(os.listdir(self.upload_folder))

    def _get_static_uploads_files(self):
        """Returns set of all filenames in static/uploads/."""
        if not os.path.exists(self.static_uploads):
            return set()
        return set(os.listdir(self.static_uploads))

    def test_upload_directory_isolated_from_static(self):
        """Verify upload folder is strictly located outside the publicly accessible static/ directory."""
        upload_abs = os.path.abspath(self.upload_folder)
        static_abs = os.path.abspath(os.path.join(self.app.root_path, 'static'))

        # Must not be equal to static or any subdirectory of static
        self.assertFalse(
            upload_abs.startswith(static_abs),
            f"Security violation! UPLOAD_FOLDER ({upload_abs}) is located inside public static path ({static_abs})"
        )

    def test_process_cleans_up_on_success(self):
        """Verify temporary files are completely unlinked after successful processing."""
        initial_upload_files = self._get_upload_files()
        initial_static_files = self._get_static_uploads_files()
        png_data = create_test_png_bytes()

        data = {
            'image': (io.BytesIO(png_data), 'test_sample.png'),
            'organ': 'brain'
        }
        resp = self.client.post('/process', data=data, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 200)

        # Confirm zero leakage in private upload folder
        remaining_upload_files = self._get_upload_files()
        self.assertEqual(
            initial_upload_files,
            remaining_upload_files,
            f"Temporary files leaked in private upload folder: {remaining_upload_files - initial_upload_files}"
        )

        # Confirm static/uploads was never touched
        self.assertEqual(
            initial_static_files,
            self._get_static_uploads_files(),
            "Files were inappropriately written to public static/uploads/ directory!"
        )

    def test_process_cleans_up_on_exception(self):
        """Verify temporary files are unlinked even if an unhandled exception occurs."""
        initial_files = self._get_upload_files()
        png_data = create_test_png_bytes()

        data = {
            'image': (io.BytesIO(png_data), 'crash_test.png'),
            'clean_image': (io.BytesIO(png_data), 'clean_crash_test.png'),
            'organ': 'chest'
        }

        with patch('app.process_image', side_effect=RuntimeError("Simulated pipeline crash")):
            with self.assertRaises(RuntimeError):
                self.client.post('/process', data=data, content_type='multipart/form-data')

        remaining_files = self._get_upload_files()
        self.assertEqual(
            initial_files,
            remaining_files,
            f"Temporary files leaked on exception: {remaining_files - initial_files}"
        )

    def test_process_cleans_up_when_clean_file_save_fails(self):
        """Verify primary file is cleaned up if saving the reference/clean file raises an error."""
        initial_files = self._get_upload_files()
        png_data = create_test_png_bytes()

        with patch('werkzeug.datastructures.FileStorage.save', side_effect=[None, IOError("Disk write error")]):
            data = {
                'image': (io.BytesIO(png_data), 'primary.png'),
                'clean_image': (io.BytesIO(png_data), 'clean.png'),
                'organ': 'chest'
            }
            try:
                self.client.post('/process', data=data, content_type='multipart/form-data')
            except IOError:
                pass

        remaining_files = self._get_upload_files()
        self.assertEqual(
            initial_files,
            remaining_files,
            f"Primary temp file was not cleaned up when clean file save failed: {remaining_files - initial_files}"
        )

    def test_process_rejects_unsupported_extensions(self):
        """Verify unsupported file extensions are rejected with HTTP 400 and never processed."""
        bad_extensions = ['malicious.sh', 'exploit.py', 'payload.exe', 'data.bin', 'archive.zip']
        for bad_name in bad_extensions:
            initial_files = self._get_upload_files()
            data = {
                'image': (io.BytesIO(b"dummy payload"), bad_name),
                'organ': 'brain'
            }
            resp = self.client.post('/process', data=data, content_type='multipart/form-data')
            self.assertEqual(resp.status_code, 400)
            data_json = resp.get_json()
            self.assertIn('error', data_json)
            self.assertIn('Unsupported file extension', data_json['error'])

            # Confirm zero temporary files leaked
            self.assertEqual(self._get_upload_files(), initial_files)

    def test_process_path_traversal_neutralized(self):
        """Verify path traversal characters in filename cannot escape the upload directory."""
        initial_files = self._get_upload_files()
        png_data = create_test_png_bytes()
        traversal_name = "../../traversal_attack.png"
        data = {
            'image': (io.BytesIO(png_data), traversal_name),
            'organ': 'chest'
        }

        parent_dir = os.path.dirname(os.path.abspath(self.upload_folder))
        escaped_file = os.path.join(parent_dir, "traversal_attack.png")
        self.assertFalse(os.path.exists(escaped_file))

        resp = self.client.post('/process', data=data, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 200)

        # Confirm file was never written to parent dir and no temp files leak in upload folder
        self.assertFalse(os.path.exists(escaped_file))
        self.assertEqual(self._get_upload_files(), initial_files)

    def test_process_payload_size_limit_with_declared_content_length(self):
        """Verify oversized request with declared Content-Length returns HTTP 413."""
        png_data = create_test_png_bytes()
        data = {
            'image': (io.BytesIO(png_data), 'test.png'),
            'organ': 'chest'
        }
        oversized_environ = {
            'CONTENT_LENGTH': str(MAX_IMAGE_UPLOAD_SIZE + 1024)
        }
        resp = self.client.post(
            '/process',
            data=data,
            content_type='multipart/form-data',
            environ_overrides=oversized_environ
        )
        self.assertEqual(resp.status_code, 413)
        data_json = resp.get_json()
        self.assertEqual(data_json.get('status'), 'REJECTED')
        self.assertIn('Upload payload exceeds', data_json.get('error', ''))

    def test_process_payload_size_limit_via_werkzeug_max_content_length(self):
        """Verify Werkzeug RequestEntityTooLarge error is caught and sanitized as HTTP 413."""
        # Test errorhandler directly with RequestEntityTooLarge
        from werkzeug.exceptions import RequestEntityTooLarge
        with self.app.test_request_context():
            response, code = self.app.error_handler_spec[None][413][RequestEntityTooLarge](RequestEntityTooLarge())
            self.assertEqual(code, 413)
            data = response.get_json()
            self.assertEqual(data.get('status'), 'REJECTED')
            self.assertIn('Upload payload exceeds', data.get('error', ''))

    def test_framing_policy_allows_huggingface_origins(self):
        """Verify Content-Security-Policy allows framing strictly from self and Hugging Face origins."""
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        csp = resp.headers.get('Content-Security-Policy', '')
        self.assertIn("frame-ancestors 'self' https://huggingface.co https://*.hf.space", csp)
        self.assertNotIn('X-Frame-Options', resp.headers)

    def test_research_endpoint_returns_503_when_checkpoint_missing(self):
        """Verify /api/research/denoise_dicom gracefully returns HTTP 503 if model is uninstalled/offline."""
        with patch('research_inference.denoise_dicom_research', side_effect=FileNotFoundError("Model file missing")):
            fake_dcm = io.BytesIO(b"FAKE_DICOM_DATA")
            resp = self.client.post(
                '/api/research/denoise_dicom',
                data={'dicom_scan': (fake_dcm, 'test.dcm')},
                content_type='multipart/form-data'
            )
            self.assertEqual(resp.status_code, 503)
            data = resp.get_json()
            self.assertEqual(data.get('status'), 'UNAVAILABLE')
            self.assertIn('offline or uninstalled', data.get('error', ''))


if __name__ == '__main__':
    unittest.main()
