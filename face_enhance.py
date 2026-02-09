# import io
# import os
# from flask import Flask, request, render_template_string, send_file, redirect, url_for
# from PIL import Image
# import cv2
# import numpy as np
# import mediapipe as mp

# app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB uploads

# mp_face_mesh = mp.solutions.face_mesh

# # --- Helpers: build skin mask from Mediapipe landmarks ---
# # Indices to exclude (eyes, lips, eyebrows) - Mediapipe face mesh has 468 landmarks.
# # We'll use landmark groups common in mediapipe face mesh mapping.
# LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# LIPS = [61,146,91,181,84,17,314,405,321,375,291,308]
# INNER_LIPS = [78,95,88,178,87,14,317,402,318,324,308]
# LEFT_BROW = [70,63,105,66,107,55,65,52,53,46]
# RIGHT_BROW = [336,296,334,293,300,383,282,295,285,276]

# def landmarks_to_points(landmarks, h, w):
#     pts = []
#     for lm in landmarks:
#         pts.append((int(lm.x * w), int(lm.y * h)))
#     return pts

# def build_face_mask(image_bgr, face_landmarks):
#     h, w = image_bgr.shape[:2]
#     # Collect face contour / overall face indices for mask
#     # Use a larger area by picking forehead & jawline landmarks.
#     # We'll use the face oval indices from Mediapipe mesh:
#     FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
#                  361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
#                  176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
#                  162, 21, 54, 103, 67, 109]
#     pts = []
#     for idx in FACE_OVAL:
#         lm = face_landmarks[idx]
#         pts.append((int(lm.x * w), int(lm.y * h)))
#     mask = np.zeros((h, w), dtype=np.uint8)
#     # fill face oval
#     cv2.fillConvexPoly(mask, np.array(pts, dtype=np.int32), 255)

#     # subtract eyes, eyebrows, lips
#     def subtract_region(indices):
#         region = []
#         for i in indices:
#             lm = face_landmarks[i]
#             region.append((int(lm.x * w), int(lm.y * h)))
#         if len(region) >= 3:
#             cv2.fillConvexPoly(mask, np.array(region, dtype=np.int32), 0)

#     subtract_region(LEFT_EYE)
#     subtract_region(RIGHT_EYE)
#     subtract_region(LIPS)
#     subtract_region(INNER_LIPS)
#     subtract_region(LEFT_BROW)
#     subtract_region(RIGHT_BROW)

#     # Smooth mask edges
#     mask = cv2.GaussianBlur(mask, (15, 15), 0)
#     # Normalize to 0..1 float
#     mask_f = (mask.astype(np.float32) / 255.0)[:, :, None]
#     return mask_f

# # --- Skin enhancement function (works on BGR numpy arrays) ---
# def enhance_skin_only(bgr_img, strength=0.9):
#     """
#     bgr_img: uint8 BGR image (cv2)
#     strength: 0..1 how strong the skin enhancement should be (1 is stronger)
#     """
#     h, w = bgr_img.shape[:2]
#     img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
#     with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
#                                refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
#         results = face_mesh.process(img_rgb)
#         if not results.multi_face_landmarks:
#             # No face — return original
#             return bgr_img
#         face_landmarks = results.multi_face_landmarks[0].landmark

#     mask_f = build_face_mask(bgr_img, face_landmarks)  # float HxWx1

#     # 1) Light smoothing on skin: bilateral filter on the whole image copy
#     smooth = cv2.bilateralFilter(bgr_img, d=9, sigmaColor=75, sigmaSpace=75)

#     # 2) Color/brightness correction in YCrCb
#     ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
#     ycrcb_smooth = cv2.cvtColor(smooth, cv2.COLOR_BGR2YCrCb).astype(np.float32)

#     # Apply CLAHE to Y channel of the smooth image for even brightness on skin
#     y = ycrcb_smooth[:, :, 0].astype(np.uint8)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     y_clahe = clahe.apply(y).astype(np.float32)

#     # Color correction: gently move Cr,Cb channels of the original toward the smoothed ones
#     cr = ycrcb[:, :, 1]
#     cb = ycrcb[:, :, 2]
#     cr_s = ycrcb_smooth[:, :, 1]
#     cb_s = ycrcb_smooth[:, :, 2]

#     # blend channels but only where mask applies
#     # compute blended channels
#     Y_new = (1.0 - strength) * ycrcb[:, :, 0] + strength * y_clahe
#     CR_new = (1.0 - strength) * cr + strength * cr_s
#     CB_new = (1.0 - strength) * cb + strength * cb_s

#     ycrcb_new = np.stack([Y_new, CR_new, CB_new], axis=2).astype(np.uint8)
#     enhanced_bgr = cv2.cvtColor(ycrcb_new, cv2.COLOR_YCrCb2BGR)

#     # Additional subtle warm/cool balance toward natural skin tone:
#     # convert to LAB and nudge A/B channels slightly toward a mild target
#     lab_orig = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)
#     lab_enh = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

#     # target biases (small)
#     a_bias = -2.0 * strength  # pushes slightly away from magenta
#     b_bias = 2.0 * strength   # slight warm push
#     lab_enh[:, :, 1] = lab_enh[:, :, 1] + a_bias
#     lab_enh[:, :, 2] = lab_enh[:, :, 2] + b_bias
#     lab_enh = np.clip(lab_enh, 0, 255).astype(np.uint8)
#     warm_bgr = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)

#     # Combine original and warm_bgr using mask
#     mask_3 = np.concatenate([mask_f, mask_f, mask_f], axis=2)
#     combined = (warm_bgr.astype(np.float32) * mask_3 +
#                 bgr_img.astype(np.float32) * (1.0 - mask_3)).astype(np.uint8)

#     # final gentle dodge: slightly increase contrast in skin areas
#     lab_comb = cv2.cvtColor(combined, cv2.COLOR_BGR2LAB).astype(np.float32)
#     L = lab_comb[:, :, 0]
#     # apply small gamma/contrast change only on skin region
#     L_skin = 255.0 * ((L / 255.0) ** (1.0 - 0.05 * strength))
#     lab_comb[:, :, 0] = (1.0 - (0.6*strength)) * L + (0.6*strength) * L_skin
#     lab_comb = np.clip(lab_comb, 0, 255).astype(np.uint8)
#     final_bgr = cv2.cvtColor(lab_comb, cv2.COLOR_LAB2BGR)

#     return final_bgr

# # --- Flask routes ---
# UPLOAD_FORM = """
# <!doctype html>
# <title>Skin Tone Enhancer</title>
# <h2>Upload image to enhance skin tone (only)</h2>
# <form method=post enctype=multipart/form-data action="/upload">
#   <input type=file name=file accept="image/*">
#   <input type=submit value="Upload & Enhance">
# </form>
# <p>Output will preserve eyes, lips, hair and background. Enhancement is focused on skin only.</p>
# """

# @app.route('/')
# def index():
#     return render_template_string(UPLOAD_FORM)

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'file' not in request.files:
#         return redirect(url_for('index'))
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(url_for('index'))

#     # read image into OpenCV
#     in_memory = file.read()
#     nparr = np.frombuffer(in_memory, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if img is None:
#         return "Could not read image", 400

#     # enhance skin
#     try:
#         out = enhance_skin_only(img, strength=0.92)
#     except Exception as e:
#         # if face not found or error, return original with message
#         print("Enhancement error:", e)
#         out = img

#     # encode to JPEG and send
#     is_success, buffer = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
#     io_buf = io.BytesIO(buffer)
#     io_buf.seek(0)
#     return send_file(io_buf, mimetype='image/jpeg', download_name='enhanced.jpg')

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
#     import cv2
#     import sys
#     from pathlib import Path

#     # from app import enhance_skin_only  # if placed next to app.py

#     def main(in_path, out_path):
#         img = cv2.imread(in_path)
#         if img is None:
#             print("Couldn't read", in_path); return
#         out = enhance_skin_only(img, strength=0.9)
#         cv2.imwrite(out_path, out)
#         print("Saved", out_path)

#     if __name__ == '__main__':
#         if len(sys.argv) < 3:
#             print("Usage: python enhance_skin.py input.jpg output.jpg")
#         else:
#             main(sys.argv[1], sys.argv[2])

# import cv2
# import numpy as np
# from PIL import Image
# import streamlit as st
# import io
# from typing import Tuple, Optional
# import warnings
# warnings.filterwarnings('ignore')

# class SkinToneEnhancer:
#     def __init__(self):
#         """Initialize the skin tone enhancer"""
#         pass
    
#     def detect_skin_mask(self, image: np.ndarray) -> np.ndarray:
#         """
#         Detect skin regions using YCrCb color space
#         Based on proven skin color ranges in YCrCb space
#         """
#         # Convert to YCrCb color space (better for skin detection)
#         ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
#         # Define skin color range in YCrCb
#         # These values are based on research papers for skin detection
#         min_YCrCb = np.array([0, 133, 77], np.uint8)
#         max_YCrCb = np.array([255, 173, 127], np.uint8)
        
#         # Create skin mask
#         skin_mask = cv2.inRange(ycrcb, min_YCrCb, max_YCrCb)
        
#         # Apply morphological operations to clean the mask
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#         skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
#         skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
#         skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
        
#         return skin_mask
    
#     def adjust_skin_tone(self, image: np.ndarray, 
#                          brightness: float = 1.05,
#                          saturation: float = 1.1,
#                          warmth: float = 1.02) -> np.ndarray:
#         """
#         Adjust skin tone parameters carefully
#         """
#         # Convert to HSV for better color manipulation
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
#         # Adjust saturation (carefully)
#         hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        
#         # Adjust value (brightness) - very subtle
#         hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
        
#         # Convert back to BGR
#         enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
#         # Apply subtle warmth adjustment
#         # Increase red channel slightly, decrease blue channel
#         b, g, r = cv2.split(enhanced)
#         r = np.clip(r.astype(np.float32) * warmth, 0, 255).astype(np.uint8)
#         b = np.clip(b.astype(np.float32) / warmth, 0, 255).astype(np.uint8)
#         enhanced = cv2.merge([b, g, r])
        
#         return enhanced
    
#     def selective_color_balance(self, image: np.ndarray, 
#                                skin_mask: np.ndarray) -> np.ndarray:
#         """
#         Apply color balance only to skin regions
#         """
#         # Convert to LAB color space for better color balance
#         lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#         l, a, b = cv2.split(lab)
        
#         # Apply CLAHE to L channel (improves contrast while preserving details)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         l = clahe.apply(l)
        
#         # Merge back
#         lab = cv2.merge([l, a, b])
#         balanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
#         # Apply only to skin regions
#         skin_mask_3ch = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR) / 255.0
        
#         # Blend original and balanced based on skin mask
#         result = image * (1 - skin_mask_3ch) + balanced * skin_mask_3ch
#         result = result.astype(np.uint8)
        
#         return result
    
#     def enhance_skin_texture(self, image: np.ndarray, 
#                             skin_mask: np.ndarray,
#                             blur_strength: float = 0.5) -> np.ndarray:
#         """
#         Gently smooth skin texture while preserving details
#         """
#         # Create a very subtle blur for skin areas
#         blurred = cv2.bilateralFilter(image, 9, 75, 75)
        
#         # Create mask for blending (soft edges)
#         soft_mask = cv2.GaussianBlur(skin_mask, (15, 15), 5)
#         soft_mask = soft_mask.astype(np.float32) / 255.0 * blur_strength
        
#         # Expand mask to 3 channels
#         soft_mask_3ch = cv2.cvtColor(soft_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
#         soft_mask_3ch = soft_mask_3ch.astype(np.float32) / 255.0
        
#         # Blend original with blurred based on mask
#         enhanced = image * (1 - soft_mask_3ch) + blurred * soft_mask_3ch
#         enhanced = enhanced.astype(np.uint8)
        
#         return enhanced
    
#     def enhance_image(self, image: np.ndarray) -> np.ndarray:
#         """
#         Main enhancement pipeline focusing on skin tone
#         """
#         # Make a copy of the original
#         original = image.copy()
        
#         # Step 1: Detect skin regions
#         skin_mask = self.detect_skin_mask(image)
        
#         # Step 2: Apply selective color balance
#         image = self.selective_color_balance(image, skin_mask)
        
#         # Step 3: Adjust skin tone parameters
#         image = self.adjust_skin_tone(
#             image,
#             brightness=1.05,  # Very subtle brightness increase
#             saturation=1.08,  # Slight saturation boost
#             warmth=1.01       # Tiny warmth adjustment
#         )
        
#         # Step 4: Gentle skin texture enhancement
#         image = self.enhance_skin_texture(image, skin_mask, blur_strength=0.3)
        
#         # Step 5: Final sharpening to preserve details
#         kernel = np.array([[-1, -1, -1],
#                           [-1,  9, -1],
#                           [-1, -1, -1]]) / 9.0
#         image = cv2.filter2D(image, -1, kernel)
        
#         # Blend with original to maintain natural look (70% enhanced, 30% original)
#         final = cv2.addWeighted(image, 0.7, original, 0.3, 0)
        
#         return final, skin_mask

# def main():
#     """Streamlit UI for the skin tone enhancer"""
#     st.set_page_config(
#         page_title="Professional Skin Tone Enhancer",
#         page_icon="✨",
#         layout="wide"
#     )
    
#     st.title("✨ Professional Skin Tone Enhancer")
#     st.markdown("""
#     Upload an image to enhance skin tones naturally and beautifully.
#     Focuses on:
#     - Natural skin tone improvement
#     - Gentle texture smoothing
#     - Balanced color correction
#     - Preservation of facial features
#     """)
    
#     # Initialize enhancer
#     enhancer = SkinToneEnhancer()
    
#     # File uploader
#     uploaded_file = st.file_uploader(
#         "Choose an image...", 
#         type=['jpg', 'jpeg', 'png', 'bmp'],
#         help="Upload portrait or facial images for best results"
#     )
    
#     if uploaded_file is not None:
#         # Read image
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
#         # Convert BGR to RGB for display
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         # Create columns for comparison
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("Original Image")
#             st.image(image_rgb, use_container_width=True)
        
#         with st.spinner('Enhancing skin tone...'):
#             # Enhance image
#             enhanced, skin_mask = enhancer.enhance_image(image)
            
#             # Convert enhanced to RGB for display
#             enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            
#             with col2:
#                 st.subheader("Enhanced Image")
#                 st.image(enhanced_rgb, use_container_width=True)
        
#         # Create comparison slider
#         st.subheader("Comparison View")
#         comparison = st.slider(
#             "Slide to compare before/after",
#             min_value=0.0,
#             max_value=1.0,
#             value=0.5,
#             help="Drag left to see original, right to see enhanced"
#         )
        
#         # Create comparison image
#         h, w = image_rgb.shape[:2]
#         comparison_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
        
#         # Calculate split point
#         split_point = int(w * comparison)
        
#         # Combine images
#         comparison_img[:, :split_point] = image_rgb[:, :split_point]
#         comparison_img[:, split_point:] = enhanced_rgb[:, split_point:]
        
#         # Add divider line
#         cv2.line(comparison_img, (split_point, 0), (split_point, h), (255, 255, 255), 2)
        
#         st.image(comparison_img, use_container_width=True)
        
#         # Download buttons
#         st.subheader("Download Enhanced Image")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Convert enhanced image to bytes for download
#             is_success, buffer = cv2.imencode(".jpg", enhanced)
#             if is_success:
#                 st.download_button(
#                     label="Download as JPG",
#                     data=buffer.tobytes(),
#                     file_name="enhanced_skin_tone.jpg",
#                     mime="image/jpeg"
#                 )
        
#         with col2:
#             is_success_png, buffer_png = cv2.imencode(".png", enhanced)
#             if is_success_png:
#                 st.download_button(
#                     label="Download as PNG",
#                     data=buffer_png.tobytes(),
#                     file_name="enhanced_skin_tone.png",
#                     mime="image/png"
#                 )
        
#         # Additional options in expander
#         with st.expander("Advanced Options"):
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 brightness = st.slider("Brightness", 0.9, 1.2, 1.05, 0.01)
#             with col2:
#                 saturation = st.slider("Saturation", 0.9, 1.3, 1.08, 0.01)
#             with col3:
#                 warmth = st.slider("Warmth", 0.95, 1.05, 1.01, 0.01)
            
#             if st.button("Apply Custom Settings"):
#                 custom_enhanced = enhancer.adjust_skin_tone(
#                     image, brightness, saturation, warmth
#                 )
#                 custom_enhanced_rgb = cv2.cvtColor(custom_enhanced, cv2.COLOR_BGR2RGB)
#                 st.image(custom_enhanced_rgb, caption="Custom Enhancement", use_container_width=True)
        
#         st.info("💡 **Tip:** For best results, use well-lit portrait photos with clear facial features.")

# if __name__ == "__main__":
#     main()


import cv2
import numpy as np
from tkinter import Tk, Button, Label, Scale, HORIZONTAL, filedialog, Frame
from PIL import Image, ImageTk
import tkinter as tk

class PerfectSkinEnhancer:
    def __init__(self):
        self.root = Tk()
        self.root.title("Perfect Skin Enhancement Studio")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a1a')
        
        self.original_image = None
        self.current_image = None
        self.skin_mask = None
        
        self.setup_ui()
    def setup_ui(self):
        # Header
        header = Frame(self.root, bg='#2d2d2d', height=80)
        header.pack(fill='x')
        Label(header, text="🎨 Perfect Skin Enhancement Studio", 
            font=("Arial", 20, "bold"), bg='#2d2d2d', fg='white').pack(pady=20)

        # Initialize controls dictionary BEFORE creating sliders
        self.controls = {}

        # Main container
        main_container = Frame(self.root, bg='#1a1a1a')
        main_container.pack(fill='both', expand=True, padx=20, pady=10)

        
    # def setup_ui(self):
    #     # Header
    #     header = Frame(self.root, bg='#2d2d2d', height=80)
    #     header.pack(fill='x')
    #     Label(header, text="🎨 Perfect Skin Enhancement Studio", 
    #           font=("Arial", 20, "bold"), bg='#2d2d2d', fg='white').pack(pady=20)
        
        # Main container
        # main_container = Frame(self.root, bg='#1a1a1a')
        # main_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel - Controls
        control_panel = Frame(main_container, bg='#2d2d2d', width=350)
        control_panel.pack(side='left', fill='y', padx=(0, 10))
        control_panel.pack_propagate(False)
        
        # Upload button
        Button(control_panel, text="📁 Upload Image", command=self.upload_image,
               bg='#4CAF50', fg='white', font=("Arial", 12, "bold"),
               padx=30, pady=15, relief='flat', cursor='hand2').pack(pady=20, padx=20)
        
        # Scrollable controls frame
        canvas = tk.Canvas(control_panel, bg='#2d2d2d', highlightthickness=0)
        scrollbar = tk.Scrollbar(control_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas, bg='#2d2d2d')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10)
        scrollbar.pack(side="right", fill="y")
        
        # Enhancement controls
        self.create_control(scrollable_frame, "✨ Overall Enhancement", 0, 100, 50, 'enhancement')
        self.create_control(scrollable_frame, "💡 Brightness", -30, 30, 8, 'brightness')
        self.create_control(scrollable_frame, "🌟 Radiance", 0, 100, 45, 'radiance')
        self.create_control(scrollable_frame, "🎭 Smoothness Level", 0, 100, 60, 'smoothness')
        self.create_control(scrollable_frame, "🔥 Warmth", -30, 30, 12, 'warmth')
        self.create_control(scrollable_frame, "🎨 Tone Evenness", 0, 100, 55, 'evenness')
        self.create_control(scrollable_frame, "💫 Clarity", 0, 100, 40, 'clarity')
        self.create_control(scrollable_frame, "🌈 Saturation", -30, 30, 8, 'saturation')
        self.create_control(scrollable_frame, "⚡ Sharpness", 0, 100, 25, 'sharpness')
        self.create_control(scrollable_frame, "👁️ Eye Enhancement", 0, 100, 30, 'eyes')
        
        # Preset buttons
        preset_frame = Frame(scrollable_frame, bg='#2d2d2d')
        preset_frame.pack(pady=20, padx=10)
        
        Label(preset_frame, text="Quick Presets", font=("Arial", 11, "bold"),
              bg='#2d2d2d', fg='white').pack(pady=5)
        
        Button(preset_frame, text="Natural", command=lambda: self.apply_preset('natural'),
               bg='#3d3d3d', fg='white', width=15).pack(pady=3)
        Button(preset_frame, text="Flawless", command=lambda: self.apply_preset('flawless'),
               bg='#3d3d3d', fg='white', width=15).pack(pady=3)
        Button(preset_frame, text="Instagram", command=lambda: self.apply_preset('instagram'),
               bg='#3d3d3d', fg='white', width=15).pack(pady=3)
        Button(preset_frame, text="Reset", command=self.reset_controls,
               bg='#d32f2f', fg='white', width=15).pack(pady=3)
        
        # Save button
        Button(control_panel, text="💾 Save Enhanced Image", command=self.save_image,
               bg='#2196F3', fg='white', font=("Arial", 12, "bold"),
               padx=30, pady=15, relief='flat', cursor='hand2').pack(side='bottom', pady=20, padx=20)
        
        # Right panel - Image display
        image_panel = Frame(main_container, bg='#1a1a1a')
        image_panel.pack(side='right', fill='both', expand=True)
        
        self.image_label = Label(image_panel, bg='#1a1a1a')
        self.image_label.pack(expand=True)
        
        self.controls = {}
        
    def create_control(self, parent, label, min_val, max_val, default, key):
        frame = Frame(parent, bg='#2d2d2d')
        frame.pack(fill='x', pady=8, padx=10)
        
        Label(frame, text=label, font=("Arial", 10), bg='#2d2d2d', 
              fg='white', anchor='w').pack(fill='x')
        
        scale = Scale(frame, from_=min_val, to=max_val, orient=HORIZONTAL,
                     bg='#2d2d2d', fg='white', highlightthickness=0,
                     troughcolor='#1a1a1a', activebackground='#4CAF50',
                     command=lambda v: self.update_enhancement())
        scale.set(default)
        scale.pack(fill='x')
        
        self.controls[key] = scale
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.current_image = self.original_image.copy()
            self.update_enhancement()
    
    def advanced_skin_detection(self, img):
        """Multi-method skin detection for accuracy"""
        # Method 1: YCrCb
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
        upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        mask1 = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Method 2: HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
        upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # Combine masks
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small noise
        skin_mask = cv2.medianBlur(skin_mask, 5)
        
        # Smooth edges
        skin_mask = cv2.GaussianBlur(skin_mask, (9, 9), 0)
        
        return skin_mask
    
    def frequency_separation(self, img, mask, d=9, sigma_color=75, sigma_space=75):
        """Advanced skin smoothing using frequency separation"""
        # Convert to float
        img_float = img.astype(float)
        
        # Create low frequency (color/tone)
        low_freq = cv2.bilateralFilter(img.astype(np.uint8), d, sigma_color, sigma_space)
        low_freq = low_freq.astype(float)
        
        # Create high frequency (texture/detail)
        high_freq = img_float - low_freq
        
        # Blur the low frequency more for smoothing
        smooth_low = cv2.GaussianBlur(low_freq.astype(np.uint8), (0, 0), 3)
        
        # Recombine with reduced high frequency for smoothness
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) / 255.0
        
        result = smooth_low.astype(float) + high_freq * 0.3  # Reduce texture by 70%
        result = img_float * (1 - mask_3ch) + result * mask_3ch
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def enhance_eyes(self, img, mask, intensity):
        """Brighten and enhance eye regions"""
        if intensity == 0:
            return img
        
        # Detect face region (simplified - assumes center focus)
        h, w = img.shape[:2]
        eye_region = img.copy()
        
        # Create eye enhancement mask (upper 40% of image)
        eye_mask = np.zeros_like(mask)
        eye_mask[int(h*0.25):int(h*0.5), :] = 255
        eye_mask = cv2.GaussianBlur(eye_mask, (31, 31), 0)
        
        # Combine with skin mask
        combined_mask = cv2.bitwise_and(mask, eye_mask)
        combined_mask_3ch = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2RGB) / 255.0
        
        # Brighten
        brightened = np.clip(img.astype(float) + intensity * 0.5, 0, 255)
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * (intensity / 200)
        sharpened = cv2.filter2D(brightened.astype(np.uint8), -1, kernel)
        
        result = img * (1 - combined_mask_3ch) + sharpened * combined_mask_3ch
        return result.astype(np.uint8)
    
    def perfect_skin_enhancement(self, img, params):
        """Apply all enhancements with professional algorithms"""
        enhanced = img.copy().astype(float)
        
        # Detect skin
        skin_mask = self.advanced_skin_detection(img)
        skin_mask_3ch = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2RGB) / 255.0
        
        # Store for other functions
        self.skin_mask = skin_mask
        
        # Overall enhancement multiplier
        enhancement_factor = params['enhancement'] / 50.0
        
        # 1. Frequency separation smoothing
        if params['smoothness'] > 0:
            smooth_strength = int(9 + params['smoothness'] / 10)
            smoothed = self.frequency_separation(img, skin_mask, 
                                                 smooth_strength, 
                                                 75 + params['smoothness'], 
                                                 75 + params['smoothness'])
            blend = params['smoothness'] / 100.0 * enhancement_factor
            enhanced = enhanced * (1 - blend * skin_mask_3ch) + smoothed * (blend * skin_mask_3ch)
        
        # 2. Tone evenness using adaptive histogram equalization
        if params['evenness'] > 0:
            lab = cv2.cvtColor(enhanced.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_eq = clahe.apply(l)
            
            # Blend equalized with original
            blend_factor = params['evenness'] / 200.0 * enhancement_factor
            l = (l * (1 - blend_factor) + l_eq * blend_factor).astype(np.uint8)
            
            lab = cv2.merge([l, a, b])
            even_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            enhanced = enhanced * (1 - skin_mask_3ch) + even_img * skin_mask_3ch
        
        # 3. Brightness and radiance
        if params['brightness'] != 0 or params['radiance'] > 0:
            # Convert to LAB for better brightness control
            lab = cv2.cvtColor(enhanced.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply brightness
            brightness_adj = params['brightness'] * enhancement_factor
            l = np.clip(l.astype(float) + brightness_adj * skin_mask[:, :], 0, 255)
            
            # Apply radiance (glow effect)
            if params['radiance'] > 0:
                l_blur = cv2.GaussianBlur(l.astype(np.uint8), (21, 21), 0)
                radiance_strength = params['radiance'] / 100.0 * enhancement_factor * 0.3
                l = l * (1 - radiance_strength) + l_blur * radiance_strength
            
            lab = cv2.merge([l.astype(np.uint8), a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(float)
        
        # 4. Warmth adjustment
        if params['warmth'] != 0:
            warmth = params['warmth'] * enhancement_factor
            warmth_adj = np.zeros_like(enhanced)
            warmth_adj[:, :, 0] = warmth * 1.5  # Red
            warmth_adj[:, :, 1] = warmth * 0.7  # Green
            warmth_adj[:, :, 2] = -warmth * 0.5  # Blue
            enhanced = enhanced + (warmth_adj * skin_mask_3ch)
        
        # 5. Clarity (local contrast enhancement)
        if params['clarity'] > 0:
            gray = cv2.cvtColor(enhanced.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (0, 0), 10)
            clarity_mask = cv2.subtract(gray, blur)
            
            clarity_strength = params['clarity'] / 100.0 * enhancement_factor * 2
            for i in range(3):
                enhanced[:, :, i] = enhanced[:, :, i] + clarity_mask * clarity_strength * skin_mask / 255.0
        
        # 6. Saturation
        if params['saturation'] != 0:
            hsv = cv2.cvtColor(enhanced.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(float)
            sat_factor = 1.0 + (params['saturation'] / 100.0) * enhancement_factor
            hsv[:, :, 1] = hsv[:, :, 1] * (1 - skin_mask_3ch[:, :, 0]) + \
                          np.clip(hsv[:, :, 1] * sat_factor, 0, 255) * skin_mask_3ch[:, :, 0]
            enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(float)
        
        # 7. Sharpness
        if params['sharpness'] > 0:
            sharp_strength = params['sharpness'] / 100.0 * enhancement_factor
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]]) * sharp_strength * 0.5
            sharpened = cv2.filter2D(enhanced.astype(np.uint8), -1, kernel)
            enhanced = enhanced * (1 - skin_mask_3ch) + sharpened * skin_mask_3ch
        
        # Clip and convert
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # 8. Eye enhancement (done separately)
        if params['eyes'] > 0:
            enhanced = self.enhance_eyes(enhanced, skin_mask, 
                                       params['eyes'] * enhancement_factor)
        
        return enhanced
    
    def update_enhancement(self, *args):
        if self.original_image is None:
            return
        
        params = {
            'enhancement': self.controls['enhancement'].get(),
            'brightness': self.controls['brightness'].get(),
            'radiance': self.controls['radiance'].get(),
            'smoothness': self.controls['smoothness'].get(),
            'warmth': self.controls['warmth'].get(),
            'evenness': self.controls['evenness'].get(),
            'clarity': self.controls['clarity'].get(),
            'saturation': self.controls['saturation'].get(),
            'sharpness': self.controls['sharpness'].get(),
            'eyes': self.controls['eyes'].get()
        }
        
        self.current_image = self.perfect_skin_enhancement(self.original_image, params)
        self.display_current_image()
    
    def display_current_image(self):
        if self.current_image is None:
            return
        
        h, w = self.current_image.shape[:2]
        max_size = 700
        
        if h > max_size or w > max_size:
            scale = min(max_size / h, max_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            display = cv2.resize(self.current_image, (new_w, new_h), 
                               interpolation=cv2.INTER_LANCZOS4)
        else:
            display = self.current_image
        
        img_pil = Image.fromarray(display)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk
    
    def apply_preset(self, preset):
        if self.original_image is None:
            return
        
        presets = {
            'natural': {
                'enhancement': 40, 'brightness': 5, 'radiance': 30,
                'smoothness': 40, 'warmth': 8, 'evenness': 35,
                'clarity': 25, 'saturation': 5, 'sharpness': 15, 'eyes': 20
            },
            'flawless': {
                'enhancement': 70, 'brightness': 12, 'radiance': 60,
                'smoothness': 75, 'warmth': 15, 'evenness': 70,
                'clarity': 45, 'saturation': 12, 'sharpness': 30, 'eyes': 40
            },
            'instagram': {
                'enhancement': 60, 'brightness': 10, 'radiance': 50,
                'smoothness': 65, 'warmth': 18, 'evenness': 60,
                'clarity': 50, 'saturation': 15, 'sharpness': 35, 'eyes': 35
            }
        }
        
        if preset in presets:
            for key, value in presets[preset].items():
                self.controls[key].set(value)
            self.update_enhancement()
    
    def reset_controls(self):
        defaults = {
            'enhancement': 50, 'brightness': 8, 'radiance': 45,
            'smoothness': 60, 'warmth': 12, 'evenness': 55,
            'clarity': 40, 'saturation': 8, 'sharpness': 25, 'eyes': 30
        }
        
        for key, value in defaults.items():
            self.controls[key].set(value)
        self.update_enhancement()
    
    def save_image(self):
        if self.current_image is None:
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        
        if file_path:
            img_bgr = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            print(f"✅ Image saved successfully: {file_path}")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = PerfectSkinEnhancer()
    app.run()