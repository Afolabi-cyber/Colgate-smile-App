from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageEnhance
import io, base64, uuid, os, secrets
from datetime import datetime, timedelta

app = Flask(__name__, template_folder='templates')
# Session & cookie settings
app.config.update(
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=False,  # set True in production with HTTPS
)
app.secret_key = secrets.token_hex(32)
# Allow local origin and send credentials (adjust origin in production)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": ["http://localhost:5000"]}})

# Configuration
UPLOAD_FOLDER = 'uploads'
SMILE_THRESHOLD = 60  # scale 0-100; adjust as required
MAX_PHOTO_AGE_HOURS = 6

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh

# Useful landmark indices for smile detection (MediaPipe Face Mesh)
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
UPPER_LIP = 13
LOWER_LIP = 14

def decode_base64_image(data_url):
    """Decode a data URL (data:image/jpeg;base64,...) into an OpenCV BGR image"""
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
    Basic smile scoring using mouth opening relative to mouth width.
    landmarks: dict idx -> (x,y) normalized coords in [0..1]
    returns int in 0..100
    """
    try:
        lx, ly = landmarks[LEFT_MOUTH]
        rx, ry = landmarks[RIGHT_MOUTH]
        ux, uy = landmarks[UPPER_LIP]
        lower_x, lower_y = landmarks[LOWER_LIP]
        lx_px = int(lx * image_width)
        rx_px = int(rx * image_width)
        upper_y_px = int(uy * image_height)
        lower_y_px = int(lower_y * image_height)
        mouth_width = max(1, abs(rx_px - lx_px))
        lip_sep = max(0, lower_y_px - upper_y_px)
        # ratio tuned empirically; 0.25 roughly corresponds to open smile in this simple heuristic
        ratio = lip_sep / mouth_width
        score = int(min(100, max(0, (ratio / 0.25) * 100)))
        return score
    except Exception:
        return 0

# ============================================================================
# PHOTO FILTERS - Applied to captured images
# ============================================================================

def apply_filter_to_cv2_image(img_bgr, filter_type):
    """
    Apply Instagram-style filters to OpenCV BGR image
    Returns filtered BGR image
    """
    if filter_type == 'none':
        return img_bgr
    
    # Convert to PIL for easier manipulation
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    if filter_type == 'grayscale':
        pil_img = pil_img.convert('L').convert('RGB')
    
    elif filter_type == 'sepia':
        # Sepia tone
        img_array = np.array(pil_img, dtype=np.float32)
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                  [0.349, 0.686, 0.168],
                                  [0.272, 0.534, 0.131]])
        img_array = img_array @ sepia_filter.T
        img_array = np.clip(img_array, 0, 255)
        pil_img = Image.fromarray(img_array.astype(np.uint8))
    
    elif filter_type == 'vintage':
        # Vintage with slight color shift
        img_array = np.array(pil_img, dtype=np.float32)
        vintage_filter = np.array([[0.393, 0.769, 0.189],
                                    [0.349, 0.686, 0.168],
                                    [0.272, 0.534, 0.131]])
        img_array = img_array @ vintage_filter.T
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] + 30, 0, 255)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] + 20, 0, 255)
        pil_img = Image.fromarray(img_array.astype(np.uint8))
    
    elif filter_type == 'brighten':
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(1.3)
    
    elif filter_type == 'contrast':
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.5)
    
    elif filter_type == 'cool':
        # Blue tones
        img_array = np.array(pil_img, dtype=np.float32)
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] - 20, 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] + 30, 0, 255)
        pil_img = Image.fromarray(img_array.astype(np.uint8))
    
    elif filter_type == 'warm':
        # Orange/warm tones
        img_array = np.array(pil_img, dtype=np.float32)
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] + 40, 0, 255)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] + 15, 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] - 20, 0, 255)
        pil_img = Image.fromarray(img_array.astype(np.uint8))
    
    elif filter_type == 'saturate':
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(1.8)
    
    elif filter_type == 'dreamy':
        # Soft, hazy effect
        img_array = np.array(pil_img, dtype=np.float32)
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.1 + 20, 0, 255)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.05 + 15, 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.05 + 25, 0, 255)
        pil_img = Image.fromarray(img_array.astype(np.uint8))
    
    elif filter_type == 'dramatic':
        # High contrast dramatic
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.6)
        img_array = np.array(pil_img, dtype=np.float32)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] + 10, 0, 255)
        pil_img = Image.fromarray(img_array.astype(np.uint8))
    
    elif filter_type == 'sunset':
        # Golden hour effect
        img_array = np.array(pil_img, dtype=np.float32)
        img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.15 + 30, 0, 255)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 0.95 + 10, 0, 255)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 0.85 - 20, 0, 255)
        pil_img = Image.fromarray(img_array.astype(np.uint8))
    
    # Convert back to BGR for OpenCV
    filtered_rgb = np.array(pil_img)
    filtered_bgr = cv2.cvtColor(filtered_rgb, cv2.COLOR_RGB2BGR)
    return filtered_bgr

def whiten_teeth_mediapipe(img, whitening_intensity=70):
    """
    Whiten teeth using MediaPipe face mesh for accurate mouth detection.
    """
    # Initialize MediaPipe Face Mesh
    # Note: In a real environment, you'd initialize this once outside the function for efficiency
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
    
        original = img.copy()
        h, w, _ = img.shape

        if img is None:
            raise ValueError("Could not load image")
        
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process image to get face landmarks
        results = face_mesh.process(img_rgb)
        
        if not results.multi_face_landmarks:
            print("No faces detected in the image!")
            return original
        
        # Teeth landmarks
        UPPER_INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415]
        LOWER_INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]
        INNER_MOUTH = list(set(UPPER_INNER_LIP + LOWER_INNER_LIP))
        
        # Create a combined mask for all faces
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Process each detected face
        for face_landmarks in results.multi_face_landmarks:
            # Get mouth region coordinates
            mouth_points = []
            for idx in INNER_MOUTH:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                mouth_points.append([x, y])
            
            mouth_points = np.array(mouth_points, dtype=np.int32)
            
            # Create mask for this mouth
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [mouth_points], 255)
            
            # Expand the mask slightly
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # Add to combined mask
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Detect teeth within the mouth region using color
        hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        
        # Detect bright pixels (teeth)
        lower_teeth = np.array([0, 0, 120])
        upper_teeth = np.array([30, 100, 255])
        teeth_mask = cv2.inRange(hsv, lower_teeth, upper_teeth)
        
        # Combine: only teeth colors inside mouth
        teeth_mask = cv2.bitwise_and(teeth_mask, combined_mask)
        
        # Clean up the teeth mask
        kernel_small = np.ones((3, 3), np.uint8)
        teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        
        # Blur for smooth blending
        teeth_mask_blurred = cv2.GaussianBlur(teeth_mask, (15, 15), 0)
        
        # Normalize mask
        mask_normalized = teeth_mask_blurred.astype(np.float32) / 255.0
        
        # Convert to LAB for better color control
        lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        lab_whitened = lab.copy()
        
        # Increase lightness
        lab_whitened[:, :, 0] = np.clip(
            lab_whitened[:, :, 0] + (whitening_intensity * mask_normalized),
            0, 255
        ).astype(np.uint8)
        
        # Reduce yellow
        lab_whitened[:, :, 1] = np.clip(
            lab_whitened[:, :, 1] - (8 * mask_normalized),
            0, 255
        ).astype(np.uint8)
        
        lab_whitened[:, :, 2] = np.clip(
            lab_whitened[:, :, 2] - (8 * mask_normalized),
            0, 255
        ).astype(np.uint8)
        
        # Convert back to BGR
        whitened = cv2.cvtColor(lab_whitened, cv2.COLOR_LAB2BGR)
        
        # Blend with original
        mask_3channel = np.stack([mask_normalized] * 3, axis=2)
        img_float = original.astype(np.float32)
        whitened_float = whitened.astype(np.float32)
        result = (img_float * (1 - mask_3channel) + whitened_float * mask_3channel).astype(np.uint8)
        
        return result

def generate_colgate_output(img_bgr, smile_score, filter_type='none'):
    """
    Creates the final Colgate-branded downloaded photo with optional filter.
    """
    try:
        # Apply filter first (before teeth whitening for best results)
        if filter_type and filter_type != 'none':
            img_bgr = apply_filter_to_cv2_image(img_bgr, filter_type)
        
        # Then apply teeth whitening
        result = whiten_teeth_mediapipe(img_bgr, whitening_intensity=40)
    except Exception as e:
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()
        result = img_bgr

    # Convert BGR → PIL
    img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    W, H = img.size

    # How tall is footer?
    footer_h = int(H * 0.55)
    final_h = H + footer_h

    # Final canvas
    final = Image.new("RGB", (W, final_h), (255, 255, 255))
    draw = ImageDraw.Draw(final)

    # Paste original
    final.paste(img, (0, 0))

    # FRAME AROUND FACE
    frame_margin = int(W * 0.08)
    x0, y0 = frame_margin, frame_margin
    x1, y1 = W - frame_margin, H - frame_margin
    draw.rounded_rectangle([x0, y0, x1, y1], outline="white", width=int(W*0.012), radius=30)

    # SCORE BADGE
    badge_r = int(W * 0.25)
    bx = W - badge_r - int(W * 0.06)
    by = int(W * 0.06)
    draw.ellipse([bx, by, bx + badge_r, by + badge_r], fill="white", outline="#E30000", width=8)

    # Score text
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

    # BIG RED FOOTER
    footer = Image.new("RGB", (W, footer_h), (227, 0, 0))
    final.paste(footer, (0, H))

    # Colgate logo (Assuming static assets are available)
    try:
        logo = Image.open("static/assets/colgate_logo.png").convert("RGBA")
        logo_w = int(W * 0.30)
        logo = logo.resize((logo_w, int(logo_w * logo.height / logo.width)), Image.LANCZOS)
        final.paste(logo, (int(W * 0.05), H + int(footer_h * 0.08)), logo)
    except FileNotFoundError:
        # Placeholder if logo is missing
        print("Colgate logo not found. Skipping.")

    # Main footer text
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

    # SUBFOOTER
    sub_h = int(footer_h * 0.22)
    sub = Image.new("RGB", (W, sub_h), (200, 0, 0))
    final.paste(sub, (0, H + footer_h - sub_h))

    draw.text((int(W * 0.05), H + footer_h - sub_h + int(sub_h * 0.27)),
              "Captured by the Colgate AI Smile Meter", fill="white", font=sub_font)

    return final

@app.route('/')
def index():
    # initialize session fields
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

        # Run MediaPipe Face Mesh on a single frame
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
        
        # Add a specific message for positioning if not centered/straight
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
        return jsonify({'error': str(e)}), 500

@app.route('/capture-photo', methods=['POST'])
def capture_photo():
    try:
        data = request.json or {}
        image_data = data.get('image') or ''
        filter_type = data.get('filter', 'none')  # Get selected filter
        
        image = decode_base64_image(image_data)
        if image is None:
            return jsonify({'error': 'No image'}), 400

        # compute smile
        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

        smile_score = 0
        if results and results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            landmarks = {idx: (p.x, p.y) for idx, p in enumerate(lm)}
            if LEFT_MOUTH in landmarks:
                smile_score = compute_smile_score(landmarks, image.shape[1], image.shape[0])

        # Create final Colgate-branded output with filter
        final_img = generate_colgate_output(image, int(smile_score), filter_type)

        photo_id = uuid.uuid4().hex
        path = os.path.join(UPLOAD_FOLDER, f"{photo_id}.jpg")
        final_img.save(path, "JPEG", quality=95)

        coupon = f"SMILE-{secrets.token_hex(3).upper()}"
        return jsonify({'success': True, 'photo_id': photo_id, 'smile_score': int(smile_score), 'coupon_code': coupon})
    except Exception as e:
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
    # You must have 'static/assets/colgate_logo.png', 'static/assets/tooth_graphic.png', 
    # and 'static/fonts/Inter-Bold.ttf', 'static/fonts/Inter-Regular.ttf', 'static/fonts/Inter-Medium.ttf' 
    # available in your file system for the full application to run correctly.
    app.run(debug=True, host='0.0.0.0', port=5000)