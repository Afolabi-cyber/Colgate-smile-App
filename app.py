from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageEnhance
import io, base64, uuid, os, secrets
from datetime import datetime, timedelta

app = Flask(__name__, template_folder='templates')
app.config.update(
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=False,
)
app.secret_key = secrets.token_hex(32)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": ["http://localhost:5000"]}})

UPLOAD_FOLDER = 'uploads'
SMILE_THRESHOLD = 60
MAX_PHOTO_AGE_HOURS = 6

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh

# Key facial landmarks for smile detection
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
UPPER_LIP = 13
LOWER_LIP = 14

def decode_base64_image(data_url):
    if not data_url:
        return None
    if ',' in data_url:
        data = data_url.split(',', 1)[1]
    else:
        data = data_url
    try:
        image_bytes = base64.b64decode(data)
    except Exception as e:
        raise ValueError("Invalid base64 image data: " + str(e))
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def compute_smile_score(landmarks, image_width, image_height):
    """
    Compute smile score based on mouth width and lip separation.
    Returns a score from 0-100.
    """
    try:
        # Get landmark coordinates
        lx, ly = landmarks[LEFT_MOUTH]
        rx, ry = landmarks[RIGHT_MOUTH]
        ux, uy = landmarks[UPPER_LIP]
        lower_x, lower_y = landmarks[LOWER_LIP]
        
        # Convert to pixel coordinates
        lx_px = int(lx * image_width)
        rx_px = int(rx * image_width)
        upper_y_px = int(uy * image_height)
        lower_y_px = int(lower_y * image_height)
        
        # Calculate mouth width and lip separation
        mouth_width = max(1, abs(rx_px - lx_px))
        lip_sep = max(0, lower_y_px - upper_y_px)
        
        # Calculate ratio and score
        ratio = lip_sep / mouth_width
        score = int(min(100, max(0, (ratio / 0.25) * 100)))
        
        return score
    except Exception as e:
        print(f"Error computing smile score: {e}")
        return 0

def apply_filter_to_cv2_image(img_bgr, filter_type):
    """
    Apply Colgate brand-friendly filters to OpenCV BGR image
    Returns filtered BGR image
    """
    if filter_type == 'none' or not filter_type:
        return img_bgr
    
    # Convert to PIL for easier manipulation
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    if filter_type == 'warm_glow':
        img_array = np.array(pil_img, dtype=np.float32)
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(1.5)
        img_array = np.array(pil_img, dtype=np.float32)
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.15 + 15, 0, 255)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.05 + 10, 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 0.95, 0, 255)
        pil_img = Image.fromarray(img_array.astype(np.uint8))
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.25)
    
    elif filter_type == 'vibrant':
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(1.7)
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(1.15)
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.2)
    
    elif filter_type == 'clarendon':
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.4)
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(1.3)
    
    elif filter_type == 'natural_bright':
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(1.25)
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.15)
    
    elif filter_type == 'fresh':
        img_array = np.array(pil_img, dtype=np.float32)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.1 + 10, 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.05 + 5, 0, 255)
        pil_img = Image.fromarray(img_array.astype(np.uint8))
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(1.18)
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(1.3)
    
    elif filter_type == 'confident':
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.5)
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(1.5)
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
    
    # Convert back to BGR for OpenCV
    filtered_rgb = np.array(pil_img)
    filtered_bgr = cv2.cvtColor(filtered_rgb, cv2.COLOR_RGB2BGR)
    return filtered_bgr

def whiten_teeth_mediapipe(img):
    original = img.copy()
    h, w, _ = img.shape

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            print("No face detected")
            return original

        INNER_MOUTH = [
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308
        ]

        # ---------------- Mouth Mask ----------------
        mouth_mask = np.zeros((h, w), dtype=np.uint8)

        for face in results.multi_face_landmarks:
            pts = []
            for idx in INNER_MOUTH:
                lm = face.landmark[idx]
                pts.append([int(lm.x * w), int(lm.y * h)])
            pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(mouth_mask, [pts], 255)

        mouth_mask = cv2.GaussianBlur(mouth_mask, (31, 31), 0)
        mouth_mask_f = mouth_mask.astype(np.float32) / 255.0

        # ---------------- TONGUE SEGMENTATION ----------------
        # Tongue is pink/reddish and not bright like teeth.
        hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        
        # Pink / reddish color range
        lower_tongue1 = np.array([0, 50, 30])
        upper_tongue1 = np.array([15, 255, 255])

        lower_tongue2 = np.array([160, 50, 30])
        upper_tongue2 = np.array([180, 255, 255])

        tongue_mask1 = cv2.inRange(hsv, lower_tongue1, upper_tongue1)
        tongue_mask2 = cv2.inRange(hsv, lower_tongue2, upper_tongue2)

        tongue_mask = cv2.bitwise_or(tongue_mask1, tongue_mask2)

        # Only consider tongue inside the mouth area
        tongue_mask = cv2.bitwise_and(tongue_mask, mouth_mask)
        tongue_mask = cv2.GaussianBlur(tongue_mask, (25, 25), 0)
        tongue_f = tongue_mask.astype(np.float32) / 255.0

        # Teeth mask = mouth - tongue
        teeth_mask_f = np.clip(mouth_mask_f - tongue_f, 0, 1)

        # ---------------- Brightness Equalization ----------------
        lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        L_eq = clahe.apply(L)

        lab_eq = cv2.merge([L_eq, A, B])
        eq_img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # ---------------- Whitening ----------------
        lab2 = cv2.cvtColor(eq_img, cv2.COLOR_BGR2LAB)
        L2, A2, B2 = cv2.split(lab2)

        L2 = np.clip(L2 + 75 * teeth_mask_f, 0, 255)
        B2 = np.clip(B2 - 12 * teeth_mask_f, 0, 255)  # reduce yellow ONLY

        whitened_lab = cv2.merge([
            L2.astype(np.uint8),
            A2.astype(np.uint8),
            B2.astype(np.uint8)
        ])

        whitened = cv2.cvtColor(whitened_lab, cv2.COLOR_LAB2BGR)

        # ---------------- Sharpening ----------------
        blurred = cv2.GaussianBlur(whitened, (0, 0), 2.5)
        sharp = cv2.addWeighted(whitened, 1.6, blurred, -0.6, 0)

        mask3 = np.stack([teeth_mask_f]*3, axis=2)
        final = (original * (1 - mask3) + sharp * mask3).astype(np.uint8)

        return final

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def enhance_image_snapchat_style(image_path, output_path=None):
    """
    Enhance image similar to Snapchat's default camera enhancement.
    
    Args:
        image_path: Path to input image (string or numpy array)
        output_path: Path to save enhanced image (optional)
    
    Returns:
        Enhanced image as numpy array
    """
    # Load image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image_path
    
    # Convert to PIL for easier manipulation
    pil_img = Image.fromarray(img_rgb)
    
    # 1. Brightness enhancement (subtle boost)
    brightness_enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = brightness_enhancer.enhance(1.1)  # 10% brighter
    
    # 2. Contrast enhancement
    contrast_enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = contrast_enhancer.enhance(1.15)  # 15% more contrast
    
    # 3. Color/Saturation boost
    color_enhancer = ImageEnhance.Color(pil_img)
    pil_img = color_enhancer.enhance(1.2)  # 20% more saturation
    
    # 4. Slight sharpening
    pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=3))
    
    # Convert back to numpy array
    enhanced = np.array(pil_img)
    
    # 5. Apply subtle skin smoothing (optional - reduces noise)
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=20, sigmaSpace=20)
    
    # 6. Warm tone adjustment (slight yellow tint for better skin tones)
    enhanced = enhanced.astype(np.float32)
    enhanced[:, :, 0] = np.clip(enhanced[:, :, 0] * 1.02, 0, 255)  # Red
    enhanced[:, :, 1] = np.clip(enhanced[:, :, 1] * 1.01, 0, 255)  # Green
    enhanced = enhanced.astype(np.uint8)
    
    # Save if output path provided
    if output_path:
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, enhanced_bgr)
        print(f"Enhanced image saved to: {output_path}")
    
    return enhanced


def enhance_image_advanced(image_path, output_path=None, 
                          brightness=1.1, contrast=1.15, 
                          saturation=1.2, sharpness=1.0):
    """
    Advanced version with customizable parameters.
    
    Args:
        image_path: Path to input image
        output_path: Path to save enhanced image (optional)
        brightness: Brightness multiplier (1.0 = no change, >1.0 = brighter)
        contrast: Contrast multiplier (1.0 = no change, >1.0 = more contrast)
        saturation: Saturation multiplier (1.0 = no change, >1.0 = more vivid)
        sharpness: Sharpness level (0.0 to 2.0, 1.0 = original)
    
    Returns:
        Enhanced image as numpy array
    """
    # Load image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image_path
    
    pil_img = Image.fromarray(img_rgb)
    
    # Apply enhancements
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(brightness)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(contrast)
    
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(saturation)
    
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(sharpness)
    
    enhanced = np.array(pil_img)
    
    # Save if output path provided
    if output_path:
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, enhanced_bgr)
        print(f"Enhanced image saved to: {output_path}")
    
    return enhanced

                
def create_person_mask(img_bgr):
    """
    Create a mask that isolates the person from the background.
    Uses MediaPipe Selfie Segmentation for accurate person detection.
    Returns a binary mask where 255 = person, 0 = background
    """
    try:
        import mediapipe as mp
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        
        with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_seg:
            # Process the image
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = selfie_seg.process(rgb)
            
            if results.segmentation_mask is not None:
                # Convert to binary mask (threshold at 0.5)
                mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
                
                # Apply morphological operations to smooth the mask
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                
                # Blur the edges for smoother transition
                mask = cv2.GaussianBlur(mask, (21, 21), 0)
                
                return mask
            else:
                # Fallback: return full white mask (keep entire image)
                return np.ones((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8) * 255
                
    except Exception as e:
        print(f"Error creating person mask: {e}")
        # Fallback: return full white mask
        return np.ones((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.uint8) * 255

def generate_colgate_output(img_bgr, smile_score, filter_type='none', background=None):
    """
    Creates the final Colgate-branded downloaded photo with optional filter and background.
    Background appears everywhere EXCEPT where the person is.
    The person (with filter and teeth whitening) is composited on top of the background.
    """
    try:
        # STEP 1: Apply filter first
        if filter_type and filter_type != 'none':
            img_bgr = apply_filter_to_cv2_image(img_bgr, filter_type)
        
        # STEP 2: Apply teeth whitening
        result = whiten_teeth_mediapipe(img_bgr)
        result = enhance_image_snapchat_style(result)
    except Exception as e:
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()
        result = img_bgr

    # Convert to PIL RGB
    img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    W, H = img.size

    # STEP 3: Composite person onto background pattern if selected
    if background and background != 'none':
        try:
            bg_path = f"static/assets/bg_{background}.png"
            print(f"Attempting to load background from: {bg_path}")
            
            if os.path.exists(bg_path):
                # Load and resize background pattern
                bg_img = Image.open(bg_path).convert("RGB")
                bg_resized = bg_img.resize((W, H), Image.LANCZOS)
                
                # Create person mask using MediaPipe segmentation
                print("Creating person segmentation mask...")
                person_mask = create_person_mask(result)
                
                # Convert mask to PIL Image
                mask_pil = Image.fromarray(person_mask)
                
                # Composite: background everywhere, person on top
                # This puts the full vibrant background in all non-person areas
                final_composite = Image.composite(img, bg_resized, mask_pil)
                img = final_composite
                
                print(f"✓ Background '{background}' applied with person segmentation")
            else:
                print(f"✗ Background file not found: {bg_path}")
                print(f"  Current working directory: {os.getcwd()}")
                print(f"  Looking in: {os.path.abspath('static/assets/')}")
        except Exception as e:
            print(f"✗ Error applying background: {e}")
            import traceback
            traceback.print_exc()

    # STEP 4: Create the final output with Colgate branding

    footer_h = int(H * 0.55)
    final_h = H + footer_h

    final = Image.new("RGB", (W, final_h), (255, 255, 255))
    draw = ImageDraw.Draw(final)

    final.paste(img, (0, 0))

    frame_margin = int(W * 0.08)
    x0, y0 = frame_margin, frame_margin
    x1, y1 = W - frame_margin, H - frame_margin
    draw.rounded_rectangle([x0, y0, x1, y1], outline="white", width=int(W*0.012), radius=30)

    badge_r = int(W * 0.25)
    bx = W - badge_r - int(W * 0.06)
    by = int(W * 0.06)
    draw.ellipse([bx, by, bx + badge_r, by + badge_r], fill="white", outline="#E30000", width=8)

    try:
        score_font = ImageFont.truetype("static/fonts/Inter-Bold.ttf", int(badge_r * 0.35))
        score_label_font = ImageFont.truetype("static/fonts/Inter-Regular.ttf", int(badge_r * 0.18))
    except Exception:
        score_font = ImageFont.load_default()
        score_label_font = ImageFont.load_default()

    score_bbox = draw.textbbox((0, 0), "SCORE", font=score_label_font)
    score_text_w = score_bbox[2] - score_bbox[0]
    draw.text((bx + (badge_r - score_text_w) / 2, by + badge_r * 0.22), 
              "SCORE", fill="#E30000", font=score_label_font)
    
    pct_text = f"{smile_score}%"
    pct_bbox = draw.textbbox((0, 0), pct_text, font=score_font)
    pct_text_w = pct_bbox[2] - pct_bbox[0]
    draw.text((bx + (badge_r - pct_text_w) / 2, by + badge_r * 0.45),
              pct_text, fill="#E30000", font=score_font)

    footer = Image.new("RGB", (W, footer_h), (227, 0, 0))
    final.paste(footer, (0, H))

    try:
        logo = Image.open("static/assets/colgate_logo.png").convert("RGBA")
        logo_w = int(W * 0.30)
        logo = logo.resize((logo_w, int(logo_w * logo.height / logo.width)), Image.LANCZOS)
        final.paste(logo, (int(W * 0.05), H + int(footer_h * 0.08)), logo)
    except FileNotFoundError:
        print("Colgate logo not found. Skipping.")

    try:
        main_font = ImageFont.truetype("static/fonts/Inter-Bold.ttf", int(W * 0.08))
        hash_font = ImageFont.truetype("static/fonts/Inter-Regular.ttf", int(W * 0.055))
        sub_font = ImageFont.truetype("static/fonts/Inter-Medium.ttf", int(W * 0.04))
    except Exception:
        main_font = ImageFont.load_default()
        hash_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()

    draw.text((int(W * 0.05), H + int(footer_h * 0.33)),
              "You've Got the\nColgate Smile!", fill="white", font=main_font)

    draw.text((int(W * 0.05), H + int(footer_h * 0.63)),
              "#ColgateSmileMeter", fill="white", font=hash_font)

    sub_h = int(footer_h * 0.22)
    sub = Image.new("RGB", (W, sub_h), (200, 0, 0))
    final.paste(sub, (0, H + footer_h - sub_h))

    draw.text((int(W * 0.05), H + footer_h - sub_h + int(sub_h * 0.27)),
              "Captured by the Colgate AI Smile Meter", fill="white", font=sub_font)

    return final

@app.route('/')
def index():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        session['attempts'] = 0
        session['highest_score'] = 0

    logo_path = "static/assets/colgate_logo.png"
    tooth_icon_path = "static/assets/tooth_graphic.png"

    return render_template('index.html', logo_path=logo_path, tooth_path=tooth_icon_path)

@app.route('/analyze-frame', methods=['POST'])
def analyze_frame():
    try:
        data = request.json or {}
        image_data = data.get('image') or ''
        image = decode_base64_image(image_data)
        if image is None:
            return jsonify({'error': 'No image'}), 400
        h, w = image.shape[:2]

        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

        quality_checks = {'face_detected': False, 'all_passed': False, 'centered': False, 'head_straight': False}
        smile_score = 0

        if results and results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            landmarks = {idx: (p.x, p.y) for idx, p in enumerate(lm)}
            quality_checks['face_detected'] = True

            if 1 in landmarks:
                nose_x = landmarks[1][0] * w
                quality_checks['centered'] = abs(nose_x - w / 2) < w * 0.2

            if 33 in landmarks and 263 in landmarks:
                left_eye_y = landmarks[33][1] * h
                right_eye_y = landmarks[263][1] * h
                quality_checks['head_straight'] = abs(left_eye_y - right_eye_y) < h * 0.03

            if LEFT_MOUTH in landmarks and RIGHT_MOUTH in landmarks and UPPER_LIP in landmarks and LOWER_LIP in landmarks:
                smile_score = compute_smile_score(landmarks, w, h)

            quality_checks['all_passed'] = all([
                quality_checks['face_detected'],
                quality_checks['centered'],
                quality_checks['head_straight'],
                smile_score >= SMILE_THRESHOLD
            ])

        session['attempts'] = session.get('attempts', 0) + 1
        session['highest_score'] = max(session.get('highest_score', 0), int(smile_score))

        message = 'Keep trying! Smile wider!' if smile_score < SMILE_THRESHOLD else 'Smile Perfect! Ready to capture.'
        
        if smile_score < SMILE_THRESHOLD:
            if not quality_checks['face_detected']:
                 message = 'Position your face in the frame 🙂'
            elif not quality_checks['centered'] or not quality_checks['head_straight']:
                 message = 'Center your face and keep your head straight.'

        return jsonify({
            'smile_score': int(smile_score),
            'quality_checks': quality_checks,
            'message': message,
            'can_capture': quality_checks.get('all_passed', False)
        })
    except Exception as e:
        print(f"Error in analyze_frame: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/capture-photo', methods=['POST'])
def capture_photo():
    try:
        data = request.json or {}

        image_data = data.get('image') or ''
        filter_type = data.get('filter', 'none')
        selected_bg = data.get('background', 'smiles')

        print(f"\n{'='*60}")
        print(f"CAPTURE REQUEST RECEIVED")
        print(f"{'='*60}")
        print(f"Filter: {filter_type}")
        print(f"Background: {selected_bg}")
        print(f"Image data length: {len(image_data) if image_data else 0}")
        print(f"{'='*60}\n")

        image = decode_base64_image(image_data)
        if image is None:
            return jsonify({'error': 'No image'}), 400

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

        smile_score = 0

        if results and results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            landmarks = {idx: (p.x, p.y) for idx, p in enumerate(lm)}
            if LEFT_MOUTH in landmarks:
                smile_score = compute_smile_score(
                    landmarks,
                    image.shape[1],
                    image.shape[0]
                )

        print(f"Smile score computed: {smile_score}")
        print(f"Generating output with background: {selected_bg}")

        final_img = generate_colgate_output(
            image,
            int(smile_score),
            filter_type,
            selected_bg
        )

        photo_id = uuid.uuid4().hex
        file_path = os.path.join(UPLOAD_FOLDER, f"{photo_id}.jpg")
        final_img.save(file_path, "JPEG", quality=95)

        print(f"✓ Photo saved: {file_path}")
        print(f"{'='*60}\n")

        coupon = f"SMILE-{secrets.token_hex(3).upper()}"

        return jsonify({
            'success': True,
            'photo_id': photo_id,
            'smile_score': int(smile_score),
            'coupon_code': coupon
        })

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ ERROR IN CAPTURE_PHOTO")
        print(f"{'='*60}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<photo_id>')
def download_photo(photo_id):
    try:
        if not photo_id or len(photo_id) < 8:
            return jsonify({'error': 'Invalid photo id'}), 400

        filepath = os.path.join(UPLOAD_FOLDER, f"{photo_id}.jpg")

        if not os.path.exists(filepath):
            return jsonify({'error': 'Photo not found'}), 404

        return send_file(
            filepath,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f"colgate_{photo_id}.jpg"
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def get_stats():
    return jsonify({
        'attempts': session.get('attempts', 0),
        'highest_score': session.get('highest_score', 0),
        'user_id': session.get('user_id', '')
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)