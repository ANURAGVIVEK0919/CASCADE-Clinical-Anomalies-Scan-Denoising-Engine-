import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import json
import uuid
import time
from gradcam_utils import generate_gradcam, overlay_gradcam
import tempfile
from werkzeug.exceptions import RequestEntityTooLarge
from medical_agent import ASCADEMedicalBoard
from data_pipeline import CASCADEDICOMPipeline
from aws_s3_storage import S3StorageVault, AWSCloudServicesManager
from telemetry import get_telemetry, sanitize_value


# Load environment variables from .env file
load_dotenv()

# Upload limits and allowed formats
MAX_IMAGE_UPLOAD_SIZE = 32 * 1024 * 1024  # 32 MB limit per request
ALLOWED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.dcm', '.dicom'}

app = Flask(__name__)

# Secure private temporary directory completely isolated from publicly served static/ directory
DEFAULT_UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'cascade_uploads')
app.config['UPLOAD_FOLDER'] = os.environ.get('CASCADE_UPLOAD_FOLDER', DEFAULT_UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_IMAGE_UPLOAD_SIZE
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def handle_payload_too_large(error):
    """
    Standardized HTTP 413 handler preventing raw server tracebacks or internal path leakage.
    """
    return jsonify({
        'status': 'REJECTED',
        'error': 'Upload payload exceeds maximum allowable limit of 32 MB.'
    }), 413


@app.after_request
def set_secure_framing_policy(response):
    """
    Allow framing strictly from self and official Hugging Face Space origins.
    Protects against malicious clickjacking while permitting authorized embedding
    inside Hugging Face Space iframes.
    """
    response.headers['Content-Security-Policy'] = (
        "frame-ancestors 'self' https://huggingface.co https://*.hf.space"
    )
    # Ensure legacy X-Frame-Options does not unconditionally deny Hugging Face
    response.headers.pop('X-Frame-Options', None)
    return response

# Initialize Enterprise Managers
medical_board = ASCADEMedicalBoard()
dicom_pipeline = CASCADEDICOMPipeline()
s3_vault = S3StorageVault()
aws_manager = AWSCloudServicesManager()

def get_s3_vault():
    """
    Returns configured S3 vault or mock injected via app.config['S3_VAULT'].
    Enables isolated unit testing without touching AWS network services.
    """
    return app.config.get('S3_VAULT') or s3_vault



# Load the trained models
MODEL_PATH = 'medical_ct_denoiser_full.keras'
DIAGNOSTIC_MODEL_PATH = 'medical_ct_diagnostic_model.keras'

model = None
diagnostic_model = None

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

def get_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            try:
                model = load_model(MODEL_PATH, compile=False)
            except Exception as e:
                print(f"[!] Error loading denoising model: {e}")
                model = None
        else:
            model = None
    return model

def get_diagnostic_model():
    global diagnostic_model
    if diagnostic_model is None:
        if os.path.exists(DIAGNOSTIC_MODEL_PATH):
            try:
                from tensorflow.keras.saving import register_keras_serializable

                @register_keras_serializable()
                def random_rot90(x):
                    # Return input unchanged during inference to prevent random fluctuations
                    return x

                diagnostic_model = load_model(
                    DIAGNOSTIC_MODEL_PATH,
                    custom_objects={'random_rot90': random_rot90},
                    compile=False
                )
            except Exception as e:
                print(f"[!] Error loading diagnostic model: {e}")
                diagnostic_model = None
        else:
            diagnostic_model = None
    return diagnostic_model

def process_image(image_path, clean_path=None):
    patient_metadata = None
    is_dicom = image_path.lower().endswith('.dcm') or image_path.lower().endswith('.dicom')

    if is_dicom and PYDICOM_AVAILABLE:
        try:
            ds = pydicom.dcmread(image_path)
            # Extract demographics
            patient_metadata = {
                'name': str(ds.get('PatientName', 'Jane Doe')),
                'id': str(ds.get('PatientID', 'ACC-' + str(np.random.randint(100000, 999999)))),
                'age': str(ds.get('PatientAge', 'N/A')),
                'sex': str(ds.get('PatientSex', 'N/A')),
                'date': str(ds.get('StudyDate', 'N/A')),
                'manufacturer': str(ds.get('Manufacturer', 'N/A')),
                'modality': str(ds.get('Modality', 'CT'))
            }

            # Format date nicely
            if patient_metadata['date'] and len(patient_metadata['date']) == 8:
                d = patient_metadata['date']
                patient_metadata['date'] = f"{d[:4]}-{d[4:6]}-{d[6:]}"

            # Extract pixel array
            pixel_array = ds.pixel_array

            # Apply min-max HU scaling to 0.0 - 1.0
            p_min = np.min(pixel_array)
            p_max = np.max(pixel_array)
            if p_max - p_min > 0:
                normalized_pixel_array = (pixel_array - p_min) / (p_max - p_min)
            else:
                normalized_pixel_array = np.zeros_like(pixel_array, dtype=np.float32)

            # Convert to 256x256 image
            img = Image.fromarray((normalized_pixel_array * 255).astype(np.uint8))
            img = img.resize((256, 256), Image.Resampling.BILINEAR)
            img_array = np.expand_dims(np.array(img) / 255.0, axis=-1)
        except Exception as e:
            print(f"[!] Error processing DICOM: {e}")
            # Fallback to standard loading if pydicom fails
            img = load_img(image_path, target_size=(256, 256), color_mode='grayscale')
            img_array = img_to_array(img) / 255.0
    else:
        # Standard image loading (PNG/JPG)
        img = load_img(image_path, target_size=(256, 256), color_mode='grayscale')
        img_array = img_to_array(img) / 255.0

    # Denoise
    m = get_model()
    denoised_img = None
    denoiser_engine = "CASCADE U-Net Neural Denoiser"

    if m is not None:
        try:
            # Predict (input: 1, 256, 256, 1)
            input_tensor = np.expand_dims(img_array, axis=0)
            denoised_tensor = m.predict(input_tensor)
            denoised_img = denoised_tensor[0]
        except Exception:
            m = None

    if m is None or denoised_img is None:
        # Fallback NumPy local smoothing box-filter - explicitly labeled as classical baseline
        denoiser_engine = "Classical 3x3 Smoothing Baseline (Classical Fallback)"
        smoothed = np.copy(img_array)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                smoothed += np.roll(np.roll(img_array, dx, axis=0), dy, axis=1)
        smoothed = smoothed / 10.0  # 3x3 local average
        # Apply a mild contrast enhancement and clamp output
        denoised_img = np.clip((smoothed - 0.05) * 1.05 + 0.05, 0.0, 1.0)

    clean_found = False
    clean_array = None

    # Check if a user-uploaded custom clean reference was provided
    if clean_path and os.path.exists(clean_path):
        try:
            clean_img = load_img(clean_path, target_size=(256, 256), color_mode='grayscale')
            clean_array = img_to_array(clean_img) / 255.0
            clean_found = True
        except Exception as e:
            print(f"[!] Error loading user-uploaded clean scan: {e}")
            clean_found = False

    # Fallback to local dataset directory if not uploaded directly
    if not clean_found:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        clean_dirs = [
            os.path.join(base_dir, "data", "processed", "deeplesion_localization", "clean"),
            os.path.join(base_dir, "data", "unified_dataset", "clean"),
            os.path.join(base_dir, "data", "separated_scans")
        ]
        filename = os.path.basename(image_path)
        if filename.startswith("temp_"):
            parts = filename.split("_")
            original_filename = "_".join(parts[2:])
        else:
            original_filename = filename

        clean_candidates = [
            original_filename.replace('_noisy', '_clean'),
            original_filename.replace('_noisy', '_clear'),
            original_filename.replace('noisy', 'clean'),
            original_filename.replace('noisy', 'clear')
        ]

        # 1. Search in unified_dataset/clean
        unified_clean_dir = clean_dirs[0]
        if os.path.exists(unified_clean_dir):
            for candidate in clean_candidates:
                cand_path = os.path.join(unified_clean_dir, candidate)
                if os.path.exists(cand_path):
                    try:
                        clean_img = load_img(cand_path, target_size=(256, 256), color_mode='grayscale')
                        clean_array = img_to_array(clean_img) / 255.0
                        clean_found = True
                        break
                    except Exception as e:
                        print(f"[!] Error loading clean scan reference from dataset: {e}")
                        clean_found = False

        # 2. If not found in unified, search separated_scans
        if not clean_found and os.path.exists(clean_dirs[1]):
            import glob
            sep_dir = clean_dirs[1]
            for candidate in clean_candidates:
                matches = glob.glob(os.path.join(sep_dir, "**", candidate), recursive=True)
                if matches and os.path.exists(matches[0]):
                    try:
                        clean_img = load_img(matches[0], target_size=(256, 256), color_mode='grayscale')
                        clean_array = img_to_array(clean_img) / 255.0
                        clean_found = True
                        break
                    except Exception as e:
                        print(f"[!] Error loading clean scan reference from separated scans: {e}")
                        clean_found = False

    if clean_found:
        # True scientific relative comparison against clinical ground truth!
        psnr_noisy = float(tf.image.psnr(clean_array, img_array, max_val=1.0).numpy())
        ssim_noisy = float(tf.image.ssim(tf.convert_to_tensor(clean_array), tf.convert_to_tensor(img_array), max_val=1.0).numpy())

        psnr_restored = float(tf.image.psnr(clean_array, denoised_img, max_val=1.0).numpy())
        ssim_restored = float(tf.image.ssim(tf.convert_to_tensor(clean_array), tf.convert_to_tensor(denoised_img), max_val=1.0).numpy())

        psnr_delta = psnr_restored - psnr_noisy
        ssim_delta = ssim_restored - ssim_noisy

        # Improvement % based on PSNR delta relative to original noisy baseline
        improvement_pct = min(99.9, (psnr_delta / max(1e-5, psnr_noisy)) * 100)
    else:
        # FIX: values must come from computation only. Removed hardcoded fallbacks.
        psnr_noisy = None
        ssim_noisy = None
        psnr_restored = None
        ssim_restored = None
        psnr_delta = None
        ssim_delta = None
        improvement_pct = None

    # Difference Map (What was removed)
    diff_map = np.abs(img_array - denoised_img)
    if np.max(diff_map) > 0:
        diff_map = diff_map / np.max(diff_map)

    return img_array, denoised_img, diff_map, {
        'psnr_noisy': psnr_noisy,
        'psnr_restored': psnr_restored,
        'psnr_delta': psnr_delta,
        'ssim_noisy': ssim_noisy,
        'ssim_restored': ssim_restored,
        'ssim_delta': ssim_delta,
        'improvement': improvement_pct,
        'denoiser_engine': denoiser_engine
    }, patient_metadata

def run_diagnostic_analyzer(img_array, organ_type, bbox=None, predicted_organ=None, filename=None):
    # Calculate actual statistics of the denoised scan
    mean_val = float(np.mean(img_array))
    std_val = float(np.std(img_array))
    max_val = float(np.max(img_array))

    # Target organ is determined primarily by AI model prediction, falling back to user selection
    effective_organ = (predicted_organ or organ_type).lower()
    organ_agreement = (predicted_organ.lower() == organ_type.lower()) if predicted_organ else True

    # Extract and sanitize candidate Region of Interest (ROI) from ResNet Diagnostic Model
    sanitized_bbox = None
    has_candidate_roi = False
    if bbox is not None and len(bbox) == 4 and all(np.isfinite(bbox)):
        r_ymin, r_xmin, r_ymax, r_xmax = bbox
        ymin = max(0.0, min(1.0, min(r_ymin, r_ymax)))
        xmin = max(0.0, min(1.0, min(r_xmin, r_xmax)))
        ymax = max(0.0, min(1.0, max(r_ymin, r_ymax)))
        xmax = max(0.0, min(1.0, max(r_xmin, r_xmax)))
        if (xmax - xmin) >= 0.01 and (ymax - ymin) >= 0.01:
            sanitized_bbox = [round(ymin, 4), round(xmin, 4), round(ymax, 4), round(xmax, 4)]
            has_candidate_roi = True
    bbox = sanitized_bbox

    findings = []
    recommendations = []

    if not organ_agreement and predicted_organ:
        findings.append(f"Organ discrepancy noted: Model classified as {predicted_organ.capitalize()}, user selected {organ_type.capitalize()}")

    # Neutralized Screening Status: Candidate ROI presence rather than unsupported clinical disease diagnoses
    if has_candidate_roi:
        severity = "Candidate ROI Detected"
        diagnostic_text = (
            f"Automated candidate region of interest localized in {effective_organ} scan. "
            f"Non-diagnostic research screening observation; requires qualified radiologist evaluation."
        )
        ymin, xmin, ymax, xmax = bbox
        findings.append(f"Stage-2 ResNet candidate ROI localized at [{ymin:.2f}, {xmin:.2f}, {ymax:.2f}, {xmax:.2f}] (experimental research localization)")
        recommendations.append("Investigational candidate ROI; correlate with primary PACS study and patient history")
        recommendations.append("Qualified radiologist interpretation required (non-diagnostic AI tool)")
    else:
        severity = "No Candidate ROI Flagged"
        diagnostic_text = (
            f"No candidate region of interest localized above threshold in {effective_organ} scan. "
            f"Non-diagnostic research observation; does not rule out underlying pathology."
        )
        findings.append("No candidate ROI localized above spatial threshold")
        recommendations.append("Standard clinical image review protocol")
        recommendations.append("Non-diagnostic AI research tool")

    return {
        'organ': organ_type.capitalize(),
        'predicted_organ': predicted_organ.capitalize() if predicted_organ else organ_type.capitalize(),
        'effective_organ': effective_organ.capitalize(),
        'organ_agreement': organ_agreement,
        'severity': severity,
        'diagnostic_text': diagnostic_text,
        'findings': findings,
        'recommendations': recommendations,
        'metrics_info': {
            'mean': round(mean_val, 4),
            'std': round(std_val, 4)
        },
        'bbox': bbox,
        'is_emergency': None,
        'emergency_status': "Not Assessed (No Certified Pathology Classifier)",
        'human_review_status': "Pending Human Review",
        'is_human_verified': False,
        'is_heuristic_pathology': True,
        'pathology_note': "Research demonstration only. CASCADE Stage-2 model performs organ classification and candidate ROI localization. It is not an FDA-cleared diagnostic device and does not diagnose disease or emergency conditions."
    }

def array_to_base64(arr):
    # Squeeze to handle grayscale (H, W, 1) -> (H, W)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.after_request
def add_header(response):
    """
    Force browser to never cache templates and assets during development.
    This prevents old templates from sticking around when files are edited.
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if request.content_length and request.content_length > MAX_IMAGE_UPLOAD_SIZE:
        return jsonify({
            'status': 'REJECTED',
            'error': 'Upload payload exceeds maximum allowable limit of 32 MB.'
        }), 413

    # Supports both singular 'image' or multiple files under 'images'
    uploaded_files = request.files.getlist('images')
    clean_files = request.files.getlist('clean_images')
    has_clean_tags = request.form.getlist('has_clean')
    organ_tags = request.form.getlist('organs')

    # Fallback to single 'image' upload if 'images' is not provided
    if not uploaded_files or uploaded_files[0].filename == '':
        single_file = request.files.get('image')
        if single_file and single_file.filename != '':
            uploaded_files = [single_file]
            organ_tags = [request.form.get('organ', 'chest')]
            clean_files = [request.files.get('clean_image')]
            has_clean_tags = ['true' if request.files.get('clean_image') else 'false']

    if not uploaded_files or uploaded_files[0].filename == '':
        return jsonify({'error': 'No images uploaded'}), 400

    results = []

    for i, file in enumerate(uploaded_files):
        # Determine organ type
        organ = organ_tags[i] if i < len(organ_tags) else 'chest'
        organ = organ.lower()

        ext = os.path.splitext(file.filename)[1].lower() if file.filename else ''
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            return jsonify({'error': f"Unsupported file extension '{ext}'. Allowed formats: PNG, JPG, JPEG, BMP, DCM, DICOM."}), 400

        filepath = None
        clean_filepath = None
        try:
            temp_id = uuid.uuid4().hex
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{temp_id}_{i}{ext}")
            file.save(filepath)

            has_clean = has_clean_tags[i] == 'true' if i < len(has_clean_tags) else False
            if has_clean and i < len(clean_files):
                clean_file = clean_files[i]
                if clean_file and clean_file.filename != '':
                    c_ext = os.path.splitext(clean_file.filename)[1].lower() if clean_file.filename else ''
                    if c_ext not in ALLOWED_IMAGE_EXTENSIONS:
                        return jsonify({'error': f"Unsupported reference file extension '{c_ext}'."}), 400
                    clean_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_clean_{temp_id}_{i}{c_ext}")
                    clean_file.save(clean_filepath)

            noisy, denoised, diff, metrics, patient_metadata = process_image(filepath, clean_filepath)

            if noisy is None:
                continue

            # Run diagnostic model inference (Option 1)
            m_diag = get_diagnostic_model()
            predicted_organ = None
            bbox = None
            model_confidence = None
            top2 = []
            gradcam_overlay = None
            organ_agreement = True
            organ_warning = None

            if m_diag is not None:
                try:
                    # Input tensor is (1, 256, 256, 1)
                    input_tensor = np.expand_dims(denoised, axis=0)
                    class_pred, bbox_pred = m_diag.predict(input_tensor, verbose=0)

                    class_idx = int(np.argmax(class_pred[0]))
                    class_map = {0: 'chest', 1: 'brain', 2: 'abdomen'}
                    predicted_organ = class_map.get(class_idx, 'chest')

                    # Independent confidence for predicted organ
                    model_confidence = float(class_pred[0][class_idx]) * 100.0

                    # Top-2 prediction distribution
                    organ_labels = ["Chest", "Brain", "Abdomen"]
                    top2_idx = np.argsort(class_pred[0])[::-1][:2]
                    top2 = [
                        {"label": organ_labels[idx], "prob": round(float(class_pred[0][idx]) * 100.0, 1)}
                        for idx in top2_idx
                    ]

                    # Organ agreement verification
                    organ_agreement = (predicted_organ.lower() == organ.lower())
                    if not organ_agreement:
                        organ_warning = f"Selected organ is '{organ.capitalize()}', but AI classified this scan as '{predicted_organ.capitalize()}' (uncalibrated model score: {model_confidence:.1f}%)."
                    elif model_confidence < 60.0:
                        organ_warning = f"Low model classification score ({model_confidence:.1f}% uncalibrated). Findings require clinical correlation."

                    # Extract and sanitize candidate ROI bounding box
                    raw_coords = bbox_pred[0].tolist()
                    if len(raw_coords) == 4 and all(np.isfinite(raw_coords)):
                        r_ymin, r_xmin, r_ymax, r_xmax = raw_coords
                        ymin = max(0.0, min(1.0, min(r_ymin, r_ymax)))
                        xmin = max(0.0, min(1.0, min(r_xmin, r_xmax)))
                        ymax = max(0.0, min(1.0, max(r_ymin, r_ymax)))
                        xmax = max(0.0, min(1.0, max(r_xmin, r_xmax)))
                        if (xmax - xmin) >= 0.01 and (ymax - ymin) >= 0.01:
                            bbox = [round(ymin, 4), round(xmin, 4), round(ymax, 4), round(xmax, 4)]
                        else:
                            bbox = None
                    else:
                        bbox = None

                    # Generate Grad-CAM Saliency Map explaining the model's predicted organ classification
                    denoised_uint8 = np.uint8(np.squeeze(denoised, axis=-1) * 255)
                    heatmap = generate_gradcam(m_diag, denoised, class_idx)
                    gradcam_overlay = overlay_gradcam(denoised_uint8, heatmap, alpha=0.45)

                except Exception as e:
                    print(f"[!] Error running diagnostic inference: {e}")
                    m_diag = None
                    predicted_organ = organ
                    organ_warning = "Diagnostic AI model inference encountered an error."
            else:
                predicted_organ = organ
                organ_warning = "Diagnostic AI model offline; displaying standard imaging mode."

            # Run diagnostic analyzer
            diagnosis = run_diagnostic_analyzer(denoised, organ, bbox, predicted_organ, filename=file.filename)

            # 1. Execute LangGraph Multi-Agent Board Consultation
            scan_context = {
                'organ': (predicted_organ or organ).capitalize(),
                'severity': diagnosis['severity'],
                'psnr_delta': metrics['psnr_delta'],
                'ssim_delta': metrics['ssim_delta'],
                'findings': diagnosis['findings'],
                'bbox': diagnosis.get('bbox', None)
            }
            agent_consultation = medical_board.consult(
                scan_context=scan_context,
                user_query="Multi-Agent Automated Scan Diagnostic & Denoising Assessment"
            )

            # 2. Automated Background DICOM ETL Data Pipeline Execution
            try:
                etl_summary = dicom_pipeline.run_pipeline()
            except Exception as e:
                print(f"[!] Background DICOM ETL error: {e}")
                etl_summary = {"status": "Executed (Local Fallback)", "data_lake_directory": "data/pipeline_output"}

            # 3. Automated Background AWS Cloud Status (Safe, truthful local mode representation)
            # Raw scan files containing potential PHI are never auto-uploaded to cloud storage
            aws_cloud_res = {
                "status": "Local PACS Mode (Cloud Storage Offline)",
                "aws_services": {
                    "1_s3_storage": {
                        "status": "Unavailable",
                        "message": "Automatic cloud upload of raw input scans is disabled. Artifacts must be synced explicitly via S3 vault.",
                        "storage_mode": "Offline / Local Storage Only"
                    },
                    "2_ec2_server": {
                        "status": "Healthy",
                        "hosting_environment": "Local PACS Engine"
                    },
                    "3_cloudwatch_logging": {
                        "status": "Local Logging Active",
                        "log_stream": "/local/cascade/pacs-engine"
                    }
                }
            }

            results.append({
                'name': file.filename,
                'pipeline': 'legacy_image_pipeline',
                'model_version': 'Legacy U-Net (medical_ct_denoiser_full.keras)' if get_model() is not None else 'Classical Fallback Baseline',
                'is_research_model': False,
                'organ': organ.capitalize(),
                'predicted_organ': predicted_organ.capitalize() if predicted_organ else organ.capitalize(),
                'organ_agreement': organ_agreement,
                'organ_warning': organ_warning,
                'confidence': round(model_confidence, 1) if model_confidence is not None else None,
                'confidence_label': 'Uncalibrated Model Score',
                'top2_predictions': top2,
                'noisy': array_to_base64(noisy),
                'denoised': array_to_base64(denoised),
                'diff': array_to_base64(diff),
                'metrics': metrics,
                'diagnosis': diagnosis,
                'dicom_metadata': patient_metadata,
                'gradcam_overlay': gradcam_overlay,
                'gradcam_type': 'Organ Saliency Map',
                'gradcam_note': 'Explains convolutional features supporting organ classification; not a disease or lesion localization map.',
                'agent_consultation': agent_consultation,
                'etl_summary': etl_summary,
                'aws_cloud_res': aws_cloud_res
            })

        finally:
            # Clean up
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass
            if clean_filepath and os.path.exists(clean_filepath):
                try:
                    os.remove(clean_filepath)
                except OSError:
                    pass

    if not results:
        return jsonify({'error': 'Failed to process any CT scans. Ensure you upload valid image files.'}), 500

    return jsonify({
        'results': results
    })

@app.route('/api/copilot/chat', methods=['POST'])
def copilot_chat():
    """
    Multi-Agent LangGraph Medical Board Consultation Endpoint.
    """
    data = request.json or {}
    scan_context = data.get('scan_context', {
        'organ': 'Brain',
        'severity': 'High Risk',
        # FIX: values must come from computation only.
        'psnr_delta': None,
        'ssim_delta': None,
        'findings': ['Hyperdense region identified']
    })
    user_query = data.get('query', '')

    response = medical_board.consult(scan_context, user_query)
    return jsonify(response)

@app.route('/api/copilot/approve', methods=['POST'])
def copilot_approve():
    """
    Demo Review Acknowledgement Endpoint.
    Clinical Safety Note: Genuine clinical sign-off is unsupported because the platform does
    not incorporate an authenticated clinician identity provider or authorized credentialing
    mechanism. This endpoint records only an unauthenticated demo acknowledgement.
    is_human_verified and reviewer_identity_verified are strictly False on the server.
    """
    data = request.json or {}
    scan_context = data.get('scan_context', {})
    if not scan_context:
        scan_context = {
            'organ': 'Brain',
            'severity': 'No Candidate ROI Flagged',
            'psnr_delta': None,
            'ssim_delta': None,
            'findings': ['Demonstration scan observation']
        }

    hitl_notes = data.get('hitl_notes', 'Demo user acknowledged review of reconstructed scan.')

    response = medical_board.consult(
        scan_context=scan_context,
        user_query="Demo Review Acknowledgement",
        review_acknowledged=True,
        hitl_notes=hitl_notes
    )
    return jsonify(response)

@app.route('/api/pipeline/run', methods=['POST', 'GET'])
def run_pipeline_api():
    """
    Triggers automated DICOM Ingestion, HIPAA Anonymization & Transformation Data Pipeline.
    """
    summary = dicom_pipeline.run_pipeline()
    return jsonify(summary)

@app.route('/api/s3/upload', methods=['POST'])
def s3_upload_api():
    """
    CASCADE S3 Artifact Storage Vault Endpoint.
    Accepts real generated CASCADE artifacts (restored_image, gradcam_overlay, diff_map, report_json).
    Strictly forbids raw server filesystem paths, path traversal, and unapproved artifact types.
    Enforces maximum upload bounds (10MB) and validates image signatures / JSON schema.
    Applies server-generated safe artifact keys with AES256 server-side encryption.
    """
    data = request.json or {}
    telemetry = get_telemetry()
    corr_id = telemetry.new_correlation_id()
    telemetry.start_trace(correlation_id=corr_id, operation_name="s3_artifact_upload")

    artifact_type = data.get('artifact_type', 'restored_image')
    payload = data.get('payload')

    from aws_s3_storage import ALLOWED_ARTIFACT_TYPES, MAX_ARTIFACT_SIZE_BYTES

    # 1. Validate artifact type against approved workflow types
    if artifact_type not in ALLOWED_ARTIFACT_TYPES:
        telemetry.finish_trace(corr_id, final_status="FAILED", error_category="UNSUPPORTED_ARTIFACT_TYPE")
        return jsonify({
            "status": "Failed",
            "error": f"Unsupported artifact type '{artifact_type}'. Allowed types: {list(ALLOWED_ARTIFACT_TYPES.keys())}"
        }), 400

    # 2. Grad-CAM explanation integrity check (no fabricated overlays)
    if artifact_type == 'gradcam_overlay' and (payload is None or payload == '' or payload == 'null'):
        telemetry.finish_trace(corr_id, final_status="FAILED", error_category="EXPLANATION_UNAVAILABLE")
        return jsonify({
            "status": "Failed",
            "error": "Grad-CAM explanation is unavailable for this scan. Refusing to upload placeholder as real explanation."
        }), 400

    # 3. Prevent empty or missing payloads
    if not payload:
        telemetry.finish_trace(corr_id, final_status="FAILED", error_category="INVALID_PAYLOAD")
        return jsonify({
            "status": "Failed",
            "error": "Artifact payload is required. Must provide valid base64 image bytes or structured report JSON."
        }), 400

    # 4. Parse and validate payload bytes
    payload_bytes = None
    MAX_BASE64_CHAR_LEN = int(MAX_ARTIFACT_SIZE_BYTES * 1.37) + 2048

    if artifact_type in ['restored_image', 'gradcam_overlay', 'diff_map']:
        if not isinstance(payload, str):
            telemetry.finish_trace(corr_id, final_status="FAILED", error_category="INVALID_BASE64")
            return jsonify({"status": "Failed", "error": "Image artifact payload must be a base64-encoded string."}), 400

        # Enforce base64 string length limit before decoding to account for expansion
        if len(payload) > MAX_BASE64_CHAR_LEN:
            telemetry.finish_trace(corr_id, final_status="FAILED", error_category="PAYLOAD_TOO_LARGE")
            return jsonify({
                "status": "Failed",
                "error": f"Payload text length ({len(payload)} chars) exceeds maximum allowable limit."
            }), 413

        # Strip data URL prefix if present (e.g., data:image/png;base64,...)
        b64_str = payload
        if ',' in b64_str and ('base64' in b64_str[:50]):
            b64_str = b64_str.split(',', 1)[1]

        try:
            payload_bytes = base64.b64decode(b64_str)
        except Exception as e:
            telemetry.finish_trace(corr_id, final_status="FAILED", error_category="INVALID_BASE64")
            return jsonify({"status": "Failed", "error": "Malformed or corrupted base64 encoding."}), 400

        # Enforce maximum binary size immediately before image parsing
        if len(payload_bytes) > MAX_ARTIFACT_SIZE_BYTES:
            telemetry.finish_trace(corr_id, final_status="FAILED", error_category="PAYLOAD_TOO_LARGE")
            return jsonify({
                "status": "Failed",
                "error": f"Payload size ({len(payload_bytes)} bytes) exceeds maximum allowable limit of {MAX_ARTIFACT_SIZE_BYTES} bytes."
            }), 413

        # Safe binary verification: check magic header and verify structure with safe image parser
        if len(payload_bytes) < 8 or not (payload_bytes.startswith(b'\x89PNG\r\n\x1a\n') or payload_bytes.startswith(b'\xff\xd8\xff')):
            telemetry.finish_trace(corr_id, final_status="FAILED", error_category="INVALID_IMAGE_FORMAT")
            return jsonify({"status": "Failed", "error": "Decoded payload does not match valid image signature (PNG or JPEG required)."}), 400

        try:
            with Image.open(BytesIO(payload_bytes)) as img_verifier:
                img_verifier.verify()
                if img_verifier.format not in ['PNG', 'JPEG']:
                    telemetry.finish_trace(corr_id, final_status="FAILED", error_category="INVALID_IMAGE_FORMAT")
                    return jsonify({"status": "Failed", "error": f"Image format '{img_verifier.format}' is not supported. PNG and JPEG only."}), 400
        except Exception as e:
            telemetry.finish_trace(corr_id, final_status="FAILED", error_category="INVALID_IMAGE_FORMAT")
            return jsonify({"status": "Failed", "error": "Corrupted or malformed image binary payload."}), 400

    elif artifact_type == 'report_json':
        ALLOWED_REPORT_FIELDS = {
            'consensus_report', 'differential_diagnosis', 'is_emergency', 'emergency_status',
            'hitl_status', 'is_human_verified', 'engine', 'organ', 'severity',
            'findings', 'recommendations', 'metrics', 'timestamp',
            'review_acknowledged', 'review_status', 'reviewer_identity_verified'
        }
        ALLOWED_METRICS_KEYS = {
            'psnr_baseline_db', 'psnr_restored_db', 'psnr_gain_db',
            'ssim_baseline', 'ssim_restored', 'ssim_gain',
            'psnr_noisy', 'psnr_restored', 'psnr_delta',
            'ssim_noisy', 'ssim_delta', 'noise_reduction_pct'
        }

        parsed_data = None
        if isinstance(payload, dict):
            parsed_data = payload
        elif isinstance(payload, str):
            if len(payload) > MAX_BASE64_CHAR_LEN:
                telemetry.finish_trace(corr_id, final_status="FAILED", error_category="PAYLOAD_TOO_LARGE")
                return jsonify({"status": "Failed", "error": "Report payload text length exceeds maximum limit."}), 413
            try:
                parsed_data = json.loads(payload)
            except Exception as e:
                telemetry.finish_trace(corr_id, final_status="FAILED", error_category="INVALID_PAYLOAD")
                return jsonify({"status": "Failed", "error": "Report payload is not valid JSON."}), 400
        else:
            telemetry.finish_trace(corr_id, final_status="FAILED", error_category="INVALID_PAYLOAD")
            return jsonify({"status": "Failed", "error": "Report payload must be a JSON object or string."}), 400

        if not isinstance(parsed_data, dict):
            telemetry.finish_trace(corr_id, final_status="FAILED", error_category="INVALID_PAYLOAD")
            return jsonify({"status": "Failed", "error": "Report payload must evaluate to a JSON object dictionary."}), 400

        # Strict allowlist sanitization: strip any field not explicitly permitted
        sanitized_report = {}
        for k, v in parsed_data.items():
            if k in ALLOWED_REPORT_FIELDS:
                if k == 'metrics' and isinstance(v, dict):
                    sanitized_report['metrics'] = {
                        mk: mv for mk, mv in v.items()
                        if mk in ALLOWED_METRICS_KEYS and (isinstance(mv, (int, float, type(None))))
                    }
                elif isinstance(v, (str, int, float, bool, type(None))):
                    sanitized_report[k] = v
                elif isinstance(v, list):
                    sanitized_report[k] = [str(item) for item in v if isinstance(item, (str, int, float))]

        # Safety Gate: Client cannot assert genuine clinical verification or emergency triage
        sanitized_report['is_human_verified'] = False
        sanitized_report['reviewer_identity_verified'] = False
        sanitized_report['is_emergency'] = None
        sanitized_report['emergency_status'] = "Not Assessed (No Certified Pathology Classifier)"

        if not sanitized_report:
            telemetry.finish_trace(corr_id, final_status="FAILED", error_category="INVALID_PAYLOAD")
            return jsonify({"status": "Failed", "error": "Report payload contains no valid, approved clinical report fields."}), 400

        payload_bytes = json.dumps(sanitized_report, indent=2).encode('utf-8')

    # 5. Bound maximum binary payload size
    if len(payload_bytes) > MAX_ARTIFACT_SIZE_BYTES:
        telemetry.finish_trace(corr_id, final_status="FAILED", error_category="PAYLOAD_TOO_LARGE")
        return jsonify({
            "status": "Failed",
            "error": f"Payload size ({len(payload_bytes)} bytes) exceeds maximum allowable limit of {MAX_ARTIFACT_SIZE_BYTES} bytes."
        }), 413

    # 6. Upload through dependency-injected S3 Vault
    vault = get_s3_vault()
    upload_start = time.time()
    result = vault.upload_artifact(
        payload_bytes=payload_bytes,
        artifact_type=artifact_type,
        request_id=corr_id
    )
    upload_duration = time.time() - upload_start

    stage_status = "SUCCESS" if result.get("status") == "Success" else ("SKIPPED" if result.get("status") == "Unavailable" else "FAILED")
    telemetry.record_stage(
        corr_id,
        "s3_artifact_upload",
        upload_duration,
        status=stage_status,
        details={
            "artifact_type": artifact_type,
            "artifact_id": result.get("artifact_id"),
            "size_bytes": len(payload_bytes),
            "storage_mode": result.get("storage_mode")
        }
    )

    final_status = "COMPLETED" if result.get("status") in ["Success", "Unavailable"] else "FAILED"
    telemetry.finish_trace(corr_id, final_status=final_status)

    return jsonify(result)

MAX_DICOM_UPLOAD_SIZE = 32 * 1024 * 1024  # 32 MB limit per file

@app.route('/api/research/denoise_dicom', methods=['POST'])
def research_denoise_dicom_api():
    """
    Dedicated Research Inference Endpoint for Calibrated CT DICOM Slices (Residual U-Net v3).

    Accepts:
      - 'dicom_scan': Multipart file upload of a calibrated 2D CT DICOM slice (max 32 MB).
      - 'reference_scan': (Optional) Multipart file upload of matched routine-dose CT DICOM.
      - 'window': (Optional) Form field specifying clinical display window:
                  'soft_tissue' (default), 'lung', 'bone', 'full_body'.

    Strict Safety & Hardening Policy:
      - Enforces 32 MB upload limit (HTTP 413 on payload too large).
      - Rejects empty, corrupt, multiframe, or non-CT inputs (HTTP 400).
      - Ordinary 8-bit images (PNG, JPEG, BMP) strictly rejected (HTTP 400).
      - Zero persistent scan storage; strictly in-memory processing.
      - Controlled error messages without server path or stack trace leakage.
      - Output is explicitly marked as PRELIMINARY_RESEARCH_CANDIDATE (not for clinical use).
    """
    from research_inference import (
        denoise_dicom_research,
        record_research_request,
        verify_checkpoint_integrity
    )

    t_start = time.time()

    # Bounded content length check
    if request.content_length and request.content_length > MAX_DICOM_UPLOAD_SIZE * 2:
        record_research_request(success=False, duration_ms=0.0, status="PAYLOAD_TOO_LARGE")
        return jsonify({
            'status': 'REJECTED',
            'error': 'Upload payload exceeds maximum allowable limit of 32 MB.'
        }), 413

    if 'dicom_scan' not in request.files:
        record_research_request(success=False, duration_ms=0.0, status="MISSING_FILE")
        return jsonify({
            'status': 'REJECTED',
            'error': 'Missing required file field: dicom_scan. A calibrated CT DICOM slice is required.'
        }), 400

    file = request.files['dicom_scan']
    if not file or file.filename == '':
        record_research_request(success=False, duration_ms=0.0, status="EMPTY_FILE")
        return jsonify({
            'status': 'REJECTED',
            'error': 'Empty file provided for dicom_scan.'
        }), 400

    fname_lower = file.filename.lower()
    if fname_lower.endswith('.png') or fname_lower.endswith('.jpg') or fname_lower.endswith('.jpeg') or fname_lower.endswith('.bmp'):
        record_research_request(success=False, duration_ms=0.0, status="UNSUPPORTED_FORMAT")
        return jsonify({
            'status': 'REJECTED',
            'error': (
                f"Unsupported file format. Ordinary 8-bit images lack physical "
                "Hounsfield calibration metadata and are rejected. Residual U-Net v3 requires genuine CT DICOM."
            )
        }), 400

    # Bounded stream read to prevent memory exhaustion
    scan_bytes = file.read(MAX_DICOM_UPLOAD_SIZE + 1024)
    if len(scan_bytes) > MAX_DICOM_UPLOAD_SIZE:
        record_research_request(success=False, duration_ms=0.0, status="FILE_TOO_LARGE")
        return jsonify({
            'status': 'REJECTED',
            'error': 'Uploaded DICOM file exceeds maximum allowable limit of 32 MB.'
        }), 413

    if len(scan_bytes) == 0:
        record_research_request(success=False, duration_ms=0.0, status="EMPTY_FILE")
        return jsonify({
            'status': 'REJECTED',
            'error': 'Empty file provided for dicom_scan.'
        }), 400

    window_mode = request.form.get('window', 'soft_tissue').lower()

    ref_file = request.files.get('reference_scan')
    ref_bytes = None
    if ref_file and ref_file.filename != '':
        ref_bytes = ref_file.read(MAX_DICOM_UPLOAD_SIZE + 1024)
        if len(ref_bytes) > MAX_DICOM_UPLOAD_SIZE:
            record_research_request(success=False, duration_ms=0.0, status="REFERENCE_TOO_LARGE")
            return jsonify({
                'status': 'REJECTED',
                'error': 'Uploaded reference DICOM file exceeds maximum allowable limit of 32 MB.'
            }), 413

    try:
        result = denoise_dicom_research(
            input_source=scan_bytes,
            reference_source=ref_bytes,
            window_mode=window_mode
        )
        duration_ms = (time.time() - t_start) * 1000.0
        record_research_request(success=True, duration_ms=duration_ms, status="SUCCESS")
        return jsonify(result), 200
    except ValueError as ve:
        duration_ms = (time.time() - t_start) * 1000.0
        record_research_request(success=False, duration_ms=duration_ms, status="VALIDATION_REJECTED")
        sanitized_msg = sanitize_value(str(ve))
        return jsonify({
            'status': 'REJECTED',
            'error': sanitized_msg
        }), 400
    except FileNotFoundError as fnfe:
        duration_ms = (time.time() - t_start) * 1000.0
        record_research_request(success=False, duration_ms=duration_ms, status="MODEL_UNAVAILABLE")
        return jsonify({
            'status': 'UNAVAILABLE',
            'error': 'Residual U-Net v3 research candidate model is offline or uninstalled on this server.'
        }), 503
    except Exception as e:
        duration_ms = (time.time() - t_start) * 1000.0
        record_research_request(success=False, duration_ms=duration_ms, status="ERROR")
        corr_id = telemetry.new_correlation_id()
        telemetry.finish_trace(corr_id, final_status="FAILED", error_category="MODEL_INFERENCE_ERROR")
        return jsonify({
            'status': 'ERROR',
            'error': 'Internal processing error during research inference.',
            'trace_id': corr_id
        }), 500

@app.route('/metrics', methods=['GET'])
def metrics_api():
    """
    Prometheus & System Health Monitoring Observability Endpoint.
    Exposes distinct telemetry for the legacy image pipeline and research DICOM pipeline.
    """
    s3_status = "Connected" if getattr(s3_vault, 'is_connected', False) else "Local Offline Vault"

    from research_inference import (
        V3_CHECKPOINT_PATH,
        EXPECTED_V3_SHA256,
        verify_checkpoint_integrity,
        get_research_telemetry
    )

    v3_available = verify_checkpoint_integrity()
    research_telemetry = get_research_telemetry()

    metrics = {
        "status": "Healthy",
        "service": "CASCADE Clinical Anomalies & Scan Denoising Engine",
        "deployment_environment": "LOCAL_DEVELOPMENT (Local Workstation Only / Not Externally Deployed)",
        "uptime_sec": time.time(),
        "pipelines": {
            "legacy_image_pipeline": {
                "model_id": "medical_ct_denoiser_full.keras",
                "task": "256x256 8-bit Image / PNG Denoising (Legacy Web)",
                "status": "Loaded" if model is not None else "Fallback Active",
                "weights_available": os.path.exists(MODEL_PATH)
            },
            "diagnostic_localization_pipeline": {
                "model_id": "medical_ct_diagnostic_model.keras",
                "task": "ResNet50V2 Region of Interest Candidate Classification",
                "status": "Loaded" if diagnostic_model is not None else "Fallback Active",
                "weights_available": os.path.exists(DIAGNOSTIC_MODEL_PATH)
            },
            "research_dicom_pipeline": {
                "model_id": "candidate_unet_residual_edge_v3.keras",
                "checkpoint_path": "artifacts/experiments/unet_residual_edge_v3/run_001/candidate_unet_residual_edge_v3.keras",
                "checkpoint_sha256": EXPECTED_V3_SHA256,
                "task": "Native 512x512 Calibrated CT DICOM Denoising",
                "status": "Available" if v3_available else "Offline",
                "weights_available": os.path.exists(V3_CHECKPOINT_PATH),
                "research_classification": "PRELIMINARY_RESEARCH_CANDIDATE",
                "endpoint": "/api/research/denoise_dicom"
            }
        },
        "models": {
            "u_net_denoiser": "Loaded" if model is not None else "Fallback Active",
            "resnet50v2_diagnostic": "Loaded" if diagnostic_model is not None else "Fallback Active",
            "residual_unet_v3_candidate": "Available" if v3_available else "Offline"
        },
        "research_telemetry": research_telemetry,
        "performance_benchmarks": {
            "avg_psnr_gain_db": None,
            "avg_ssim_gain": None,
            "triage_accuracy_pct": None,
            "normal_specificity_pct": None
        },
        "cloud_vault": s3_status,
        "langgraph_multi_agent": "Active"
    }
    return jsonify(metrics)



if __name__ == '__main__':

    # Preload deep learning models at startup to avoid runtime thread contention and lag on first request
    print("[*] Pre-loading Cascaded Neural Network Models...")
    try:
        get_model()
        get_diagnostic_model()
        print("[+] Models pre-loaded successfully and cached in RAM.")
    except Exception as e:
        print(f"[!] Warning: Pre-loading models encountered an error (fallbacks will be used): {e}")

    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
