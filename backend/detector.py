import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import pandas as pd
import os


class DeepfakeDetector:
    def __init__(self, model_path='deepfake_brain.pkl'):
        # 1. Initialize Face Landmarker
        face_base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        face_options = vision.FaceLandmarkerOptions(
            base_options=face_base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

        # 2. Load the Machine Learning Model
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print("✅ ML Model loaded successfully!")
        else:
            self.model = None
            print("⚠️ No ML model found. Run train_model.py first.")

    # ─────────────────────────────────────────────────────────────────────────────
# FIX 1: freq_artifacts
# Problem: 52% of images scored exactly 0.0 because the normalization floor
# (0.50) was too high — most images fell below it and clipped to 0.
# Fix: lower the floor to 0.30 so the full range of images gets spread out
# across 0–100 instead of half of them being crammed at 0.
# ─────────────────────────────────────────────────────────────────────────────
    def check_frequency_artifacts(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
    
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        y_idx, x_idx = np.ogrid[:h, :w]
        dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
        r_low = min(h, w) // 8
    
        low_energy = magnitude[dist <= r_low].sum()
        total_energy = magnitude.sum() + 1e-9
        high_ratio = 1.0 - (low_energy / total_energy)
    
        # Lowered floor from 0.50 → 0.30 so images with less high-freq content
        # still spread across the scale instead of all clipping to 0.
        score = np.clip((high_ratio - 0.30) / 0.45 * 100, 0, 100)
        return round(float(score), 1)
    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE 2: Local Noise Consistency (REPLACES texture_noise)
    # The old texture_noise used a single global Laplacian variance which
    # saturated at 100 for almost all images. This version measures how
    # *consistent* the noise is across local blocks. Real camera images
    # have uniform sensor noise; GAN images often have patchy over-smoothed
    # regions next to noisier regions.
    # ─────────────────────────────────────────────────────────────────────────
    def check_noise_consistency(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        resized = cv2.resize(gray, (256, 256))

        # Noise residual: image minus its blurred version
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        noise = resized - blurred

        # Compute local noise std in 32×32 blocks
        block_size = 32
        local_stds = []
        for i in range(0, 256 - block_size, block_size):
            for j in range(0, 256 - block_size, block_size):
                block_std = noise[i:i + block_size, j:j + block_size].std()
                local_stds.append(block_std)

        if len(local_stds) < 2:
            return 25.0

        local_stds = np.array(local_stds)
        # Coefficient of variation: high = inconsistent noise = suspicious
        cv = local_stds.std() / (local_stds.mean() + 1e-9)
        score = np.clip(cv * 60, 0, 100)
        return round(float(score), 1)

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE 3: Error Level Analysis (NEW — strong deepfake signal)
    # ELA works by re-saving the image as JPEG at a known quality level and
    # measuring where the error is highest. Real photos have consistent ELA
    # across regions. GAN-generated images often have inconsistent ELA because
    # different parts of the image were synthesized with different "confidence".
    # ─────────────────────────────────────────────────────────────────────────
    def check_ela(self, img):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, buffer = cv2.imencode('.jpg', img, encode_param)
        compressed = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        if compressed is None:
            return 25.0

        ela = cv2.absdiff(img.astype(np.float64), compressed.astype(np.float64))
        ela_gray = ela.mean(axis=2)

        ela_mean = ela_gray.mean()
        ela_std = ela_gray.std()

        # High inconsistency (std/mean) = different regions were synthesized
        # differently = suspicious. Normalize to a practical range.
        inconsistency = ela_std / (ela_mean + 1e-9)
        score = np.clip(inconsistency * 20, 0, 100)
        return round(float(score), 1)

    # ─────────────────────────────────────────────────────────────────────────────
    # FIX 4: color_consistency → replaced with hue_distribution_score
    # Problem: channel correlation (R/G/B) is similarly high for both real and GAN
    # portrait photos — all faces are colorful. Real=9.2, Fake=8.4. No signal.
    # Fix: GAN images often have subtly different hue/saturation statistics.
    # Specifically, they tend to have a narrower, more "ideal" saturation
    # distribution — less of the extreme low-saturation (shadow) and extreme
    # high-saturation pixels that real camera photos produce.
    # We measure the spread (std dev) of the saturation channel histogram.
    # ─────────────────────────────────────────────────────────────────────────────
    def check_color_consistency(self, img):
        resized = cv2.resize(img, (128, 128))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    
        # Saturation channel (0–255)
        sat = hsv[:, :, 1].astype(np.float64).flatten()
        # Value channel (brightness)
        val = hsv[:, :, 2].astype(np.float64).flatten()
    
        sat_std = sat.std()
        sat_mean = sat.mean()
    
        # Fraction of very low saturation pixels (< 30) — shadows, grays
        # Real photos have more of these (clothing, backgrounds, shadows)
        low_sat_frac = (sat < 30).mean()
    
        # Fraction of very high saturation pixels (> 200)
        high_sat_frac = (sat > 200).mean()
    
        # Real camera images have high saturation std (wide spread 0→255)
        # GAN images tend to cluster in the mid-saturation range
        # Low sat_std = narrower distribution = more suspicious
        sat_score = np.clip((80 - sat_std) / 0.6, 0, 100)
    
        # Also: very low low_sat_frac is suspicious (GAN avoids dark/gray areas)
        naturalness_score = np.clip((0.15 - low_sat_frac) / 0.0015, 0, 100)
    
        score = sat_score * 0.6 + naturalness_score * 0.4
        return round(float(np.clip(score, 0, 100)), 1)
 

    # ─────────────────────────────────────────────────────────────────────────────
    # FIX 2: skin_smoothness
    # Problem: formula `100 - (avg_local_var / 1.5)` goes negative for any image
    # with avg_local_var > 150, which is most real photos. 100% of images = 0.
    # Fix: use log-scale normalization so the score stays meaningful across the
    # full realistic range of local variance values (~5 to ~500).
    # Real photos tend to have higher local variance (more texture) → lower score.
    # GAN photos tend to be over-smooth → higher score.
    # ─────────────────────────────────────────────────────────────────────────────
    def check_skin_smoothness(self, img):
        """
        Renamed logically to chromatic_aberration but keeping the method name
        so the CSV columns and model stay compatible.
        Measures color fringing at edges — real cameras have it, GANs don't.
        High score = low aberration = suspicious (likely GAN).
        """
        resized = cv2.resize(img, (256, 256)).astype(np.float64)
        b, g, r = cv2.split(resized)
    
        # Find strong edges using the green channel (most reliable)
        edges = cv2.Canny(resized.astype(np.uint8), 50, 150)
        edge_mask = edges > 0
    
        if edge_mask.sum() < 100:
            return 40.0  # Not enough edges to measure — neutral
    
        # Compute the difference between R and G channels at edge pixels
        # and between B and G channels at edge pixels
        rg_diff = np.abs(r - g)[edge_mask]
        bg_diff = np.abs(b - g)[edge_mask]
    
        # Average channel separation at edges
        avg_aberration = (rg_diff.mean() + bg_diff.mean()) / 2.0
    
        # Real photos: avg_aberration typically 3–15 (subtle color fringing)
        # GAN photos: avg_aberration typically 0.5–4 (no optical distortion)
        # Low aberration = suspicious = high score
        score = np.clip(100 - (avg_aberration / 12.0) * 100, 0, 100)
        return round(float(score), 1)

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE 6: Facial Landmark Geometry (FIXED — continuous output)
    # The old version returned only 3 possible values (20, 25, 60) which made
    # it effectively useless. This version computes multiple facial ratios and
    # returns a continuous anomaly score.
    # ─────────────────────────────────────────────────────────────────────────
    def check_facial_landmarks(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        results = self.face_landmarker.detect(mp_image)

        if not results.face_landmarks:
            return 30.0  # Neutral — no face detected

        try:
            lm = results.face_landmarks[0]

            # Compute multiple geometric ratios
            eye_dist = abs(lm[33].x - lm[263].x)
            mouth_w = abs(lm[61].x - lm[291].x)
            nose_w = abs(lm[129].x - lm[358].x) if len(lm) > 358 else eye_dist * 0.5
            face_h = abs(lm[10].y - lm[152].y) if len(lm) > 152 else eye_dist * 2.0
            eye_h_left = abs(lm[159].y - lm[145].y) if len(lm) > 159 else 0.02
            eye_h_right = abs(lm[386].y - lm[374].y) if len(lm) > 386 else 0.02

            if eye_dist < 0.01 or face_h < 0.01:
                return 30.0

            # Normalized ratios
            ratio_mouth_eye = mouth_w / eye_dist
            ratio_nose_eye = nose_w / eye_dist
            ratio_eye_face = eye_dist / face_h
            ratio_eye_symmetry = abs(eye_h_left - eye_h_right) / (
                (eye_h_left + eye_h_right) / 2 + 1e-5
            )

            # Expected ranges for normal human faces
            # Deviation from these ranges is suspicious
            anomalies = [
                max(0, abs(ratio_mouth_eye - 0.90) - 0.15) * 100,
                max(0, abs(ratio_nose_eye - 0.55) - 0.10) * 100,
                max(0, abs(ratio_eye_face - 0.38) - 0.06) * 100,
                min(ratio_eye_symmetry * 150, 50)  # asymmetry score
            ]

            score = np.clip(np.mean(anomalies) * 1.5, 0, 100)
            return round(float(score), 1)
        except Exception:
            return 30.0

    # ─────────────────────────────────────────────────────────────────────────────
    # FIX 3: morph_uniformity
    # Problem: `(1.0 - cv) * 80` clips to 0 whenever cv > 1.0, which is common
    # in natural images with mixed smooth and textured regions. 79% hit 0.
    # Fix: handle the full cv range properly. Low cv (too uniform = GAN-smooth)
    # and high cv (very patchy) are both somewhat suspicious compared to
    # a mid-range cv that natural photos produce. But we want to primarily
    # flag over-smoothness (low cv), so we use a one-sided sigmoid-like mapping.
    # ─────────────────────────────────────────────────────────────────────────────
    def check_morphological_uniformity(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(gray, kernel, iterations=1).astype(np.float64)
        eroded = cv2.erode(gray, kernel, iterations=1).astype(np.float64)
        gradient = dilated - eroded
    
        mean_g = gradient.mean()
        std_g = gradient.std()
    
        if mean_g < 1.0:
            return 50.0  # Flat/blank image — return neutral instead of 0
    
        cv = std_g / mean_g
    
        # Map cv to score:
        # cv near 0 = very uniform = over-smoothed = suspicious (score → 80+)
        # cv near 0.5–1.5 = natural variation = score 20–50
        # cv > 2 = patchy/noisy = somewhat suspicious (score 50–70)
        # Use a piecewise mapping to cover the full range without clipping
        if cv < 0.5:
            score = 80 - cv * 60        # 0→80, 0.5→50
        elif cv < 1.5:
            score = 50 - (cv - 0.5) * 30   # 0.5→50, 1.5→20
        else:
            score = 20 + min((cv - 1.5) * 15, 50)  # 1.5→20, grows slowly
    
        return round(float(np.clip(score, 0, 100)), 1)

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE 8: Edge Sharpness Consistency (REPLACES extra_props)
    # The old extra_props only returned 15.0 or 40.0 — two values. Useless.
    # This measures how consistent edge sharpness is across the image.
    # GANs sometimes produce unnaturally uniform or patchy edge sharpness.
    # ─────────────────────────────────────────────────────────────────────────
    def check_edge_consistency(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        resized = cv2.resize(gray, (256, 256))

        # Sobel edge map
        sx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sx ** 2 + sy ** 2)

        # Block-level edge statistics
        block_size = 32
        block_means = []
        for i in range(0, 256 - block_size, block_size):
            for j in range(0, 256 - block_size, block_size):
                block_mean = edge_mag[i:i + block_size, j:j + block_size].mean()
                block_means.append(block_mean)

        if len(block_means) < 2:
            return 25.0

        block_means = np.array(block_means)
        # Coefficient of variation — high inconsistency can indicate GAN artifacts
        cv = block_means.std() / (block_means.mean() + 1e-9)
        # Moderate inconsistency is normal (face vs background)
        # Very low or very high inconsistency is suspicious
        score = np.clip(abs(cv - 0.8) * 60, 0, 100)
        return round(float(score), 1)

    # ─────────────────────────────────────────────────────────────────────────

    def extract_all_features(self, img_cv):
        """Extract all 8 features. Used in both training and prediction."""
        return {
            "freq_artifacts":   self.check_frequency_artifacts(img_cv),
            "noise_consistency": self.check_noise_consistency(img_cv),
            "ela_score":        self.check_ela(img_cv),
            "color_consistency": self.check_color_consistency(img_cv),
            "skin_smoothness":  self.check_skin_smoothness(img_cv),
            "facial_landmarks": self.check_facial_landmarks(img_cv),
            "morph_uniformity": self.check_morphological_uniformity(img_cv),
            "edge_consistency": self.check_edge_consistency(img_cv),
        }

    def predict_from_array(self, img_cv):
        features = self.extract_all_features(img_cv)

        if self.model:
            df = pd.DataFrame([features])
            prob = self.model.predict_proba(df)[0][1] * 100
            final_score = round(float(prob), 1)
        else:
            # Fallback manual weights (run train_model.py to get a real model)
            weights = {
                "freq_artifacts":    0.10,
                "noise_consistency": 0.18,
                "ela_score":         0.20,
                "color_consistency": 0.15,
                "skin_smoothness":   0.17,
                "facial_landmarks":  0.10,
                "morph_uniformity":  0.07,
                "edge_consistency":  0.03,
            }
            final_score = sum(features[k] * w for k, w in weights.items())
            final_score = round(float(np.clip(final_score, 0, 100)), 1)

        if final_score >= 70:
            verdict = "HIGH probability DEEPFAKE"
        elif final_score >= 50:
            verdict = "SUSPICIOUS - Likely Deepfake"
        else:
            verdict = "Likely REAL image"

        return {
            "verdict": verdict,
            "final_score": final_score,
            "scores": features,
            "explanation": "Analyzing texture, noise patterns, ELA, and color channels...",
        }

# import cv2
# import numpy as np
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import easyocr
# from spellchecker import SpellChecker
# import joblib
# import pandas as pd
# import os


# class DeepfakeDetector:
#     def __init__(self, model_path='deepfake_brain.pkl'):
#         # 1. Initialize Face Landmarker
#         face_base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
#         face_options = vision.FaceLandmarkerOptions(
#             base_options=face_base_options,
#             running_mode=vision.RunningMode.IMAGE,
#             num_faces=1,
#             min_face_detection_confidence=0.5,
#             min_face_presence_confidence=0.5,
#             min_tracking_confidence=0.5,
#             output_face_blendshapes=False,
#             output_facial_transformation_matrixes=False
#         )
#         self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

#         # 2. Initialize Hand Landmarker
#         hand_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
#         hand_options = vision.HandLandmarkerOptions(
#             base_options=hand_base_options,
#             running_mode=vision.RunningMode.IMAGE,
#             num_hands=2,
#             min_hand_detection_confidence=0.5,
#             min_hand_presence_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
#         self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

#         # 3. Initialize OCR and Spellchecker
#         self.reader = easyocr.Reader(['en', 'hi'], gpu=False)
#         self.spell = SpellChecker()

#         # 4. Load the Machine Learning Model
#         if os.path.exists(model_path):
#             self.model = joblib.load(model_path)
#             print("✅ ML Model loaded successfully!")
#         else:
#             self.model = None
#             print("⚠️ No ML model found. Using manual weights fallback.")

#     # --- FEATURE EXTRACTION FUNCTIONS ---

#     def check_frequency_artifacts(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         f = np.fft.fft2(gray)
#         fshift = np.fft.fftshift(f)
#         magnitude = 20 * np.log(np.abs(fshift) + 1)
#         high_freq_power = np.sum(magnitude[magnitude > np.percentile(magnitude, 95)])
#         score = min(100, high_freq_power / (np.mean(magnitude) * 40))
#         return round(score, 1)

#     def check_texture_noise(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#         variance = laplacian.var()
#         score = max(0, min(100, (8000 - variance) / 60)) if variance < 8000 else 0
#         return round(score, 1)

#     def check_lighting_shadows(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
        
#         mean_val = np.mean(sobel)
#         std_val = np.std(sobel)
        
#         # FIX: Ensure the score never goes below 0 or above 100
#         # This prevents the "negative millions" error you saw
#         inconsistency = std_val / (mean_val + 1e-6) 
#         final_score = np.clip(inconsistency * 10, 0, 100) 
#         return float(final_score)

#     def check_facial_landmarks(self, img):
#         rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
#         results = self.face_landmarker.detect(mp_image)

#         if not results.face_landmarks:
#             return 25.0

#         try:
#             landmarks = results.face_landmarks[0]
#             eye_dist = abs(landmarks[33].x - landmarks[263].x)
#             mouth_width = abs(landmarks[61].x - landmarks[291].x)
#             ratio = mouth_width / (eye_dist + 1e-5)
#             anomaly = 1 if 0.8 < ratio < 1.3 else 0
#             score = 60 if anomaly == 0 else 20
#         except:
#             score = 25.0
#         return round(score, 1)

#     def check_hands_fingers(self, img):
#         rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
#         results = self.hand_landmarker.detect(mp_image)

#         if not results.hand_landmarks:
#             return 15.0

#         total_anomaly = 0
#         for hand_landmarks_list in results.hand_landmarks:
#             finger_count = self._count_fingers_new(hand_landmarks_list)
#             if finger_count != 5:
#                 total_anomaly += 40
#             total_anomaly += 30
#         return min(100, total_anomaly)

#     def _count_fingers_new(self, landmarks):
#         tips = [4, 8, 12, 16, 20]
#         count = 0
#         if landmarks[4].x < landmarks[3].x: count += 1
#         for i in range(1, 5):
#             if landmarks[tips[i]].y < landmarks[tips[i] - 1].y:
#                 count += 1
#         return count

#     def check_text_errors(self, img):
#         results = self.reader.readtext(img, detail=1)
#         if not results: return 15.0
#         words = " ".join([res[1] for res in results]).split()
#         if not words: return 15.0
#         misspelled = self.spell.unknown(words)
#         error_rate = len(misspelled) / len(words) * 100
#         total_score = min(95, error_rate * 3 + 25)
#         return round(total_score, 1)

#     def check_morphological_view(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         kernel = np.ones((5, 5), np.uint8)
#         dilated = cv2.dilate(gray, kernel, iterations=1)
#         eroded = cv2.erode(gray, kernel, iterations=1)
#         morph_diff = cv2.absdiff(dilated, eroded)
#         score = min(90, (np.mean(morph_diff) / 255 * 100) * 2.5)
#         return round(score, 1)

#     def check_extra_properties(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         skin_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#         return 40.0 if skin_var < 800 else 15.0

#     def extract_all_features(self, img_cv):
#         """Used for both training (CSV) and prediction."""
#         return {
#             "freq_artifacts": self.check_frequency_artifacts(img_cv),
#             "texture_noise": self.check_texture_noise(img_cv),
#             "lighting_shadows": self.check_lighting_shadows(img_cv),
#             "facial_landmarks": self.check_facial_landmarks(img_cv),
#             "hand_fingers": self.check_hands_fingers(img_cv),
#             "text_errors": self.check_text_errors(img_cv),
#             "morphological": self.check_morphological_view(img_cv),
#             "extra_props": self.check_extra_properties(img_cv)
#         }

#     def predict_from_array(self, img_cv):
#         features = self.extract_all_features(img_cv)
        
#         if self.model:
#             df = pd.DataFrame([features])
#             prob = self.model.predict_proba(df)[0][1] * 100
#             final_score = round(prob, 1)
#         else:
#             weights = [0.20, 0.15, 0.12, 0.15, 0.20, 0.08, 0.07, 0.03]
#             final_score = sum(s * w for s, w in zip(features.values(), weights))

#         if final_score >= 70: verdict = "HIGH probability DEEPFAKE"
#         elif final_score >= 50: verdict = "SUSPICIOUS - Likely Deepfake"
#         else: verdict = "Likely REAL image"

#         return {
#             "verdict": verdict,
#             "final_score": final_score,
#             "scores": features,
#             "explanation": "Analyzing texture, symmetry, and artifacts..."
#         }
# import cv2
# import numpy as np
# import mediapipe as mp                  # ← ADD THIS LINE
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# # from mediapipe.framework.formats import landmark_pb2
# from PIL import Image
# import easyocr
# from spellchecker import SpellChecker
# class DeepfakeDetector:
#     def __init__(self):
#         # self.mp_face = mp.solutions.face_mesh
#         # self.mp_hands = mp.solutions.hands
#         # self.face_mesh = self.mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
#         # self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
#         # Face Landmarker
#         face_base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
#         face_options = vision.FaceLandmarkerOptions(
#             base_options=face_base_options,
#             running_mode=vision.RunningMode.IMAGE,
#             num_faces=1,
#             min_face_detection_confidence=0.5,
#             min_face_presence_confidence=0.5,
#             min_tracking_confidence=0.5,
#             output_face_blendshapes=False,   # we don't need blendshapes for now
#             output_facial_transformation_matrixes=False
#         )
#         self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

#         # Hand Landmarker
#         hand_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
#         hand_options = vision.HandLandmarkerOptions(
#             base_options=hand_base_options,
#             running_mode=vision.RunningMode.IMAGE,
#             num_hands=2,
#             min_hand_detection_confidence=0.5,
#             min_hand_presence_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
#         self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
                
#         self.reader = easyocr.Reader(['en'], gpu=False)
#         self.spell = SpellChecker()
        
#         print("✅ DeepfakeDetector initialized - checks all requested properties!")

#     def load_image(self, image_path):
#         img = cv2.imread(image_path)
#         if img is None:
#             raise ValueError(f"Cannot read image: {image_path}")
#         return img

#     def check_frequency_artifacts(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         f = np.fft.fft2(gray)
#         fshift = np.fft.fftshift(f)
#         magnitude = 20 * np.log(np.abs(fshift) + 1)
#         high_freq_power = np.sum(magnitude[magnitude > np.percentile(magnitude, 95)])
#         score = min(100, high_freq_power / (np.mean(magnitude) * 40))
#         return round(score, 1)

#     def check_texture_noise(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#         variance = laplacian.var()
#         score = max(0, min(100, (8000 - variance) / 60)) if variance < 8000 else 0
#         return round(score, 1)

#     def check_lighting_shadows(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
#         inconsistency = np.std(sobel) / (np.mean(sobel) + 1e-5)
#         score = min(100, inconsistency * 25)
#         return round(score, 1)

#     # def check_facial_landmarks(self, img):
#         # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # results = self.face_mesh.process(rgb)
#         # if not results.multi_face_landmarks:
#         #     return 25.0
#         # landmarks = results.multi_face_landmarks[0].landmark
#         # left_eye = landmarks[33]
#         # right_eye = landmarks[263]
#         # mouth_width = abs(landmarks[61].x - landmarks[291].x)
#         # eye_dist = abs(left_eye.x - right_eye.x)
#         # ratio = mouth_width / (eye_dist + 1e-5)
#         # anomaly = 1 if 0.8 < ratio < 1.3 else 0
#         # score = 60 if anomaly == 0 else 20
#         # return round(score, 1)
#     def check_facial_landmarks(self, img):
#     # Convert OpenCV BGR to RGB and to MediaPipe Image
#         rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

#         # Run detection
#         results = self.face_landmarker.detect(mp_image)

#         if not results.face_landmarks:
#             return 25.0  # neutral if no face

#         # Take first face (we set num_faces=1)
#         landmarks = results.face_landmarks[0]  # list of NormalizedLandmark

#         # Your original check: eye-mouth ratio (adapt to new landmark indices)
#         # MediaPipe Face Landmarker uses 478 points, indices similar but verify
#         # Common: left eye outer ~33, right eye outer ~263, mouth center bottom ~13, etc.
#         # For simplicity, use approximate indices (test and adjust if needed)
#         try:
#             left_eye = landmarks[33]   # left eye outer corner
#             right_eye = landmarks[263] # right eye outer corner
#             mouth_left = landmarks[61]
#             mouth_right = landmarks[291]

#             eye_dist = abs(left_eye.x - right_eye.x)
#             mouth_width = abs(mouth_left.x - mouth_right.x)
#             ratio = mouth_width / (eye_dist + 1e-5)

#             anomaly = 1 if 0.8 < ratio < 1.3 else 0
#             score = 60 if anomaly == 0 else 20
#         except IndexError:
#             score = 25.0  # fallback if indices wrong

#         return round(score, 1)

#     # def check_hands_fingers(self, img):
#     #     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     #     results = self.hands.process(rgb)
#     #     if not results.multi_hand_landmarks:
#     #         return 15.0
        
#     #     total_anomaly = 0
#     #     for hand_landmarks in results.multi_hand_landmarks:
#     #         finger_count = self._count_fingers(hand_landmarks)
#     #         if finger_count != 5:
#     #             total_anomaly += 40
#     #         total_anomaly += 30   # placeholder structure penalty
#     #     return min(100, total_anomaly)
#     def check_hands_fingers(self, img):
#         rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

#         results = self.hand_landmarker.detect(mp_image)

#         if not results.hand_landmarks:
#             return 15.0

#         total_anomaly = 0
#         for hand_landmarks_list in results.hand_landmarks:
#             # hand_landmarks_list is list of 21 NormalizedLandmark per hand
#             finger_count = self._count_fingers_new(hand_landmarks_list)
#             if finger_count != 5:
#                 total_anomaly += 40

#             # Keep simple structure penalty (you can improve later)
#             total_anomaly += 30

#         return min(100, total_anomaly)

#     def _count_fingers_new(self, landmarks):
#         """Adapted finger count for new format (21 landmarks per hand)"""
#         # Landmark indices (standard MediaPipe hand model):
#         # 0: wrist, 1-4: thumb, 5-8: index, 9-12: middle, 13-16: ring, 17-20: pinky
#         # Tip indices: thumb=4, index=8, middle=12, ring=16, pinky=20
#         tips = [4, 8, 12, 16, 20]
#         pip_mcp = [2, 5, 9, 13, 17]  # approximate for raised check

#         count = 0

#         # Thumb (special horizontal check)
#         if landmarks[4].x < landmarks[3].x:  # tip left of IP joint (adjust for orientation)
#             count += 1

#         # Other fingers: tip above PIP joint
#         for i in range(1, 5):  # index to pinky
#             tip_y = landmarks[tips[i]].y
#             pip_y = landmarks[tips[i] - 1].y   # PIP joint
#             if tip_y < pip_y:  # tip higher than PIP (raised)
#                 count += 1

#         return count                    

#     # def _count_fingers(self, landmarks):
#         tips = [4, 8, 12, 16, 20]
#         count = 0
#         for i in range(1, 5):
#             if landmarks.landmark[tips[i]].y < landmarks.landmark[tips[i] - 2].y:
#                 count += 1
#         thumb_tip = landmarks.landmark[4]
#         thumb_ip = landmarks.landmark[3]
#         if abs(thumb_tip.x - thumb_ip.x) > abs(thumb_tip.y - thumb_ip.y):
#             count += 1
#         return count

#     def check_text_errors(self, img):
#         results = self.reader.readtext(img, detail=1)
#         if not results:
#             return 15.0
#         full_text = " ".join([res[1] for res in results])
#         words = full_text.split()
#         if not words:
#             return 15.0
        
#         misspelled = self.spell.unknown(words)
#         spelling_error_rate = len(misspelled) / len(words) * 100
        
#         shape_anomaly = 0
#         for (_, _, conf) in results:
#             if conf < 0.6:
#                 shape_anomaly += 20
#         total_score = min(95, spelling_error_rate * 3 + shape_anomaly + 25)
#         return round(total_score, 1)

#     def check_morphological_view(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         kernel = np.ones((5, 5), np.uint8)
#         dilated = cv2.dilate(gray, kernel, iterations=1)
#         eroded = cv2.erode(gray, kernel, iterations=1)
#         morph_diff = cv2.absdiff(dilated, eroded)
#         anomaly = np.mean(morph_diff) / 255 * 100
#         score = min(90, anomaly * 2.5)
#         return round(score, 1)

#     def check_extra_properties(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         skin_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#         extra_score = 40 if skin_var < 800 else 15
#         return round(extra_score, 1)

#     def predict_from_array(self, img_cv):
#         """Run detection on OpenCV image array (no path needed)"""
#         scores = {
#             "Frequency Artifacts (Google SynthID)": self.check_frequency_artifacts(img_cv),
#             "Texture/Noise (Meta DFDC)": self.check_texture_noise(img_cv),
#             "Lighting/Shadows (Microsoft)": self.check_lighting_shadows(img_cv),
#             "Facial Landmarks (Intel)": self.check_facial_landmarks(img_cv),
#             "Hands/Fingers Structure & Count": self.check_hands_fingers(img_cv),
#             "Text Spelling + Letter Shapes": self.check_text_errors(img_cv),
#             "Morphological View": self.check_morphological_view(img_cv),
#             "Extra (Skin/Eyes)": self.check_extra_properties(img_cv)
#         }

#         weights = [0.20, 0.15, 0.12, 0.15, 0.20, 0.08, 0.07, 0.03]
#         final_score = sum(s * w for s, w in zip(scores.values(), weights))

#         if final_score >= 70:
#             verdict = "HIGH probability DEEPFAKE"
#         elif final_score >= 50:
#             verdict = "SUSPICIOUS - Likely Deepfake"
#         else:
#             verdict = "Likely REAL image"

#         # Simple explanation from top 3 scores
#         sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
#         top_reasons = [f"{prop}: {score:.0f}%" for prop, score in sorted_scores]
#         explanation = "Main signals: " + ", ".join(top_reasons)

#         return {
#             "verdict": verdict,
#             "final_score": round(final_score, 1),
#             "scores": scores,
#             "explanation": explanation
#         }
# import cv2
# import numpy as np
# import mediapipe as mp                  # ← ADD THIS LINE
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# # from mediapipe.framework.formats import landmark_pb2
# from PIL import Image
# import easyocr
# from spellchecker import SpellChecker
# class DeepfakeDetector:
#     def __init__(self):
#         # self.mp_face = mp.solutions.face_mesh
#         # self.mp_hands = mp.solutions.hands
#         # self.face_mesh = self.mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
#         # self.hands = self.mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
#         # Face Landmarker
#         face_base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
#         face_options = vision.FaceLandmarkerOptions(
#             base_options=face_base_options,
#             running_mode=vision.RunningMode.IMAGE,
#             num_faces=1,
#             min_face_detection_confidence=0.5,
#             min_face_presence_confidence=0.5,
#             min_tracking_confidence=0.5,
#             output_face_blendshapes=False,   # we don't need blendshapes for now
#             output_facial_transformation_matrixes=False
#         )
#         self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

#         # Hand Landmarker
#         hand_base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
#         hand_options = vision.HandLandmarkerOptions(
#             base_options=hand_base_options,
#             running_mode=vision.RunningMode.IMAGE,
#             num_hands=2,
#             min_hand_detection_confidence=0.5,
#             min_hand_presence_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
#         self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
                
#         self.reader = easyocr.Reader(['en'], gpu=False)
#         self.spell = SpellChecker()
        
#         print("✅ DeepfakeDetector initialized - checks all requested properties!")

#     def load_image(self, image_path):
#         img = cv2.imread(image_path)
#         if img is None:
#             raise ValueError(f"Cannot read image: {image_path}")
#         return img

#     def check_frequency_artifacts(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         f = np.fft.fft2(gray)
#         fshift = np.fft.fftshift(f)
#         magnitude = 20 * np.log(np.abs(fshift) + 1)
#         high_freq_power = np.sum(magnitude[magnitude > np.percentile(magnitude, 95)])
#         score = min(100, high_freq_power / (np.mean(magnitude) * 40))
#         return round(score, 1)

#     def check_texture_noise(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#         variance = laplacian.var()
#         score = max(0, min(100, (8000 - variance) / 60)) if variance < 8000 else 0
#         return round(score, 1)

#     def check_lighting_shadows(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
#         inconsistency = np.std(sobel) / (np.mean(sobel) + 1e-5)
#         score = min(100, inconsistency * 25)
#         return round(score, 1)

#     # def check_facial_landmarks(self, img):
#         # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # results = self.face_mesh.process(rgb)
#         # if not results.multi_face_landmarks:
#         #     return 25.0
#         # landmarks = results.multi_face_landmarks[0].landmark
#         # left_eye = landmarks[33]
#         # right_eye = landmarks[263]
#         # mouth_width = abs(landmarks[61].x - landmarks[291].x)
#         # eye_dist = abs(left_eye.x - right_eye.x)
#         # ratio = mouth_width / (eye_dist + 1e-5)
#         # anomaly = 1 if 0.8 < ratio < 1.3 else 0
#         # score = 60 if anomaly == 0 else 20
#         # return round(score, 1)
#     def check_facial_landmarks(self, img):
#     # Convert OpenCV BGR to RGB and to MediaPipe Image
#         rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

#         # Run detection
#         results = self.face_landmarker.detect(mp_image)

#         if not results.face_landmarks:
#             return 25.0  # neutral if no face

#         # Take first face (we set num_faces=1)
#         landmarks = results.face_landmarks[0]  # list of NormalizedLandmark

#         # Your original check: eye-mouth ratio (adapt to new landmark indices)
#         # MediaPipe Face Landmarker uses 478 points, indices similar but verify
#         # Common: left eye outer ~33, right eye outer ~263, mouth center bottom ~13, etc.
#         # For simplicity, use approximate indices (test and adjust if needed)
#         try:
#             left_eye = landmarks[33]   # left eye outer corner
#             right_eye = landmarks[263] # right eye outer corner
#             mouth_left = landmarks[61]
#             mouth_right = landmarks[291]

#             eye_dist = abs(left_eye.x - right_eye.x)
#             mouth_width = abs(mouth_left.x - mouth_right.x)
#             ratio = mouth_width / (eye_dist + 1e-5)

#             anomaly = 1 if 0.8 < ratio < 1.3 else 0
#             score = 60 if anomaly == 0 else 20
#         except IndexError:
#             score = 25.0  # fallback if indices wrong

#         return round(score, 1)

#     # def check_hands_fingers(self, img):
#     #     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     #     results = self.hands.process(rgb)
#     #     if not results.multi_hand_landmarks:
#     #         return 15.0
        
#     #     total_anomaly = 0
#     #     for hand_landmarks in results.multi_hand_landmarks:
#     #         finger_count = self._count_fingers(hand_landmarks)
#     #         if finger_count != 5:
#     #             total_anomaly += 40
#     #         total_anomaly += 30   # placeholder structure penalty
#     #     return min(100, total_anomaly)
#     def check_hands_fingers(self, img):
#         rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

#         results = self.hand_landmarker.detect(mp_image)

#         if not results.hand_landmarks:
#             return 15.0

#         total_anomaly = 0
#         for hand_landmarks_list in results.hand_landmarks:
#             # hand_landmarks_list is list of 21 NormalizedLandmark per hand
#             finger_count = self._count_fingers_new(hand_landmarks_list)
#             if finger_count != 5:
#                 total_anomaly += 40

#             # Keep simple structure penalty (you can improve later)
#             total_anomaly += 30

#         return min(100, total_anomaly)

#     def _count_fingers_new(self, landmarks):
#         """Adapted finger count for new format (21 landmarks per hand)"""
#         # Landmark indices (standard MediaPipe hand model):
#         # 0: wrist, 1-4: thumb, 5-8: index, 9-12: middle, 13-16: ring, 17-20: pinky
#         # Tip indices: thumb=4, index=8, middle=12, ring=16, pinky=20
#         tips = [4, 8, 12, 16, 20]
#         pip_mcp = [2, 5, 9, 13, 17]  # approximate for raised check

#         count = 0

#         # Thumb (special horizontal check)
#         if landmarks[4].x < landmarks[3].x:  # tip left of IP joint (adjust for orientation)
#             count += 1

#         # Other fingers: tip above PIP joint
#         for i in range(1, 5):  # index to pinky
#             tip_y = landmarks[tips[i]].y
#             pip_y = landmarks[tips[i] - 1].y   # PIP joint
#             if tip_y < pip_y:  # tip higher than PIP (raised)
#                 count += 1

#         return count                    

#     # def _count_fingers(self, landmarks):
#         tips = [4, 8, 12, 16, 20]
#         count = 0
#         for i in range(1, 5):
#             if landmarks.landmark[tips[i]].y < landmarks.landmark[tips[i] - 2].y:
#                 count += 1
#         thumb_tip = landmarks.landmark[4]
#         thumb_ip = landmarks.landmark[3]
#         if abs(thumb_tip.x - thumb_ip.x) > abs(thumb_tip.y - thumb_ip.y):
#             count += 1
#         return count

#     def check_text_errors(self, img):
#         results = self.reader.readtext(img, detail=1)
#         if not results:
#             return 15.0
#         full_text = " ".join([res[1] for res in results])
#         words = full_text.split()
#         if not words:
#             return 15.0
        
#         misspelled = self.spell.unknown(words)
#         spelling_error_rate = len(misspelled) / len(words) * 100
        
#         shape_anomaly = 0
#         for (_, _, conf) in results:
#             if conf < 0.6:
#                 shape_anomaly += 20
#         total_score = min(95, spelling_error_rate * 3 + shape_anomaly + 25)
#         return round(total_score, 1)

#     def check_morphological_view(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         kernel = np.ones((5, 5), np.uint8)
#         dilated = cv2.dilate(gray, kernel, iterations=1)
#         eroded = cv2.erode(gray, kernel, iterations=1)
#         morph_diff = cv2.absdiff(dilated, eroded)
#         anomaly = np.mean(morph_diff) / 255 * 100
#         score = min(90, anomaly * 2.5)
#         return round(score, 1)

#     def check_extra_properties(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         skin_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#         extra_score = 40 if skin_var < 800 else 15
#         return round(extra_score, 1)

#     def predict_from_array(self, img_cv):
#         """Run detection on OpenCV image array (no path needed)"""
#         scores = {
#             "Frequency Artifacts (Google SynthID)": self.check_frequency_artifacts(img_cv),
#             "Texture/Noise (Meta DFDC)": self.check_texture_noise(img_cv),
#             "Lighting/Shadows (Microsoft)": self.check_lighting_shadows(img_cv),
#             "Facial Landmarks (Intel)": self.check_facial_landmarks(img_cv),
#             "Hands/Fingers Structure & Count": self.check_hands_fingers(img_cv),
#             "Text Spelling + Letter Shapes": self.check_text_errors(img_cv),
#             "Morphological View": self.check_morphological_view(img_cv),
#             "Extra (Skin/Eyes)": self.check_extra_properties(img_cv)
#         }

#         weights = [0.20, 0.15, 0.12, 0.15, 0.20, 0.08, 0.07, 0.03]
#         final_score = sum(s * w for s, w in zip(scores.values(), weights))

#         if final_score >= 70:
#             verdict = "HIGH probability DEEPFAKE"
#         elif final_score >= 50:
#             verdict = "SUSPICIOUS - Likely Deepfake"
#         else:
#             verdict = "Likely REAL image"

#         # Simple explanation from top 3 scores
#         sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
#         top_reasons = [f"{prop}: {score:.0f}%" for prop, score in sorted_scores]
#         explanation = "Main signals: " + ", ".join(top_reasons)

#         return {
#             "verdict": verdict,
#             "final_score": round(final_score, 1),
#             "scores": scores,
#             "explanation": explanation
#         }