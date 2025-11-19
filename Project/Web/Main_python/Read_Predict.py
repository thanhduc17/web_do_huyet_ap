import pandas as pd
import cv2
import os
import numpy as np
import time
from scipy.signal import butter, filtfilt, periodogram, find_peaks
from scipy import stats
import mediapipe as mp
import cv2
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.signal import butter, filtfilt, periodogram, find_peaks
from scipy import stats
import mediapipe as mp
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

# ================== HÀM XỬ LÝ TÍN HIỆU ================== #
def chrom_algorithm(rgb_signals):
    rgb = np.array(rgb_signals)
    if rgb.shape[0] < 30:
        return None
    mean_rgb = np.mean(rgb, axis=0)
    normalized = (rgb - mean_rgb) / mean_rgb
    X = 3 * normalized[:, 0] - 2 * normalized[:, 1]
    Y = 1.5 * normalized[:, 0] + normalized[:, 1] - 1.5 * normalized[:, 2]
    std_X = np.std(X)
    std_Y = np.std(Y)
    alpha = std_X / std_Y if std_Y != 0 else 0
    S = X - alpha * Y
    return S

def skin_mask_ycbcr(roi):
    # roi: BGR image crop (may contain black background where masked out)
    if roi is None or roi.size == 0:
        return None
    ycbcr = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycbcr, np.array([0, 133, 77]), np.array([255, 173, 127]))
    skin_pixels = roi[mask > 0]
    if len(skin_pixels) == 0:
        return None
    avg_rgb = np.mean(skin_pixels, axis=0)
    return avg_rgb

def extract_features(ppg_signal, fps=30):
    if ppg_signal is None or len(ppg_signal) < fps * 5:
        return None

    lowcut = 0.8
    highcut = 2.5
    b, a = butter(3, [lowcut / (fps / 2), highcut / (fps / 2)], btype='bandpass')
    filtered = filtfilt(b, a, ppg_signal)

    f, pxx = periodogram(filtered, fs=fps)
    peak_idx = np.argmax(pxx)
    hr_freq = f[peak_idx]
    hr = hr_freq * 60
    power = np.sum(pxx)

    peaks, _ = find_peaks(filtered, distance=fps*0.5)
    rr_intervals = np.diff(peaks) / fps if len(peaks) > 1 else np.array([])

    hrv_std = np.std(rr_intervals) * 1000 if len(rr_intervals) > 0 else 0  # ms
    diff_rr = np.diff(rr_intervals) * 1000 if len(rr_intervals) > 1 else np.array([])
    hrv_rmssd = np.sqrt(np.mean(diff_rr**2)) if len(diff_rr) > 0 else 0
    pnn50 = np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100 if len(diff_rr) > 0 else 0

    peakfreq = f[peak_idx]
    peakpower = pxx[peak_idx]

    mean_val = np.mean(filtered)
    std_val = np.std(filtered)
    skew = stats.skew(filtered)
    kurt = stats.kurtosis(filtered)
    peak_to_peak = np.ptp(filtered)
    snr_val = power / (std_val if std_val != 0 else 1)

    return {
        "HR": hr,
        "HRV_RMSSD": hrv_rmssd,
        "HRV_STD": hrv_std,
        "pNN50": pnn50,
        "Power": power,
        "Peak_Frequency": peakfreq,
        "Peak_Amplitude": peakpower,
        "Mean": mean_val,
        "Std": std_val,
        "Skewness": skew,
        "Kurtosis": kurt,
        "PeakToPeak": peak_to_peak,
        "SNR": snr_val
    }

# =============== Mediapipe FaceMesh ================= #
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_landmark_points(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        landmark_points = []
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            landmark_points.append((x, y))
        return landmark_points
    else:
        return None

# -------------------- HÀM TẠO MASK VÀ VẼ VIỀN -------------------- #
def create_roi_mask(frame, landmarks, roi_point_indices):
    """
    frame: full BGR frame (will be drawn on in-place)
    landmarks: list of (x,y) pixel coordinates (len 468)
    roi_point_indices: list of landmark indices that define polygon for this ROI
    Trả về: mask (h,w) uint8 where polygon region = 255. Ghi viền polygon lên frame (màu xanh).
    """
    h, w, _ = frame.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Prepare polygon points (pixel coords)
    pts = np.array([landmarks[i] for i in roi_point_indices], dtype=np.int32)
    if pts.size == 0:
        return mask

    # Fill polygon in mask
    cv2.fillPoly(mask, [pts], 255)

    # Draw polygon outline on frame (xanh lá)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=1)

    return mask

# -------------------- MAPPING các index cho từng region -------------------- #
# Những bộ index này là chọn mẫu để tạo polygon; bạn có thể tinh chỉnh nếu muốn
ROI_LANDMARKS = {
    "forehead": [10, 338, 297, 332, 284, 10],
    "left_cheek": [234, 93, 132, 58, 172, 234],
    "right_cheek": [454, 323, 361, 288, 397, 454],
    "under_lips": [61, 146, 91, 181, 84, 61]   # near mouth/chin area
}

# ================== ĐỌC CSV VÀ TRÍCH XUẤT ================== #
excel_path = "python_do_ap_huyết/data.xlsx"
df = pd.read_excel(excel_path, engine="openpyxl")

results = []

for index, row in df.iterrows():
    video_file = row['Video']
    sex = row['Sex']
    age = row['Age']
    bmi = row['BMI']
    sbp = row['SBP']
    dbp = row['DBP']
    bpm = row['BPM']

    video_path = os.path.join("Face_dataset/media", video_file)
    print(f"\n📂 Đang xử lý video: {video_file}...")

    if not os.path.exists(video_path):
        print(f"❌ Không tìm thấy video: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    rgb_signals = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape   # dùng kích thước gốc video
        landmarks = get_landmark_points(frame)
        roi_boxes = None

        if landmarks is not None:
            roi_boxes = {}

            # Tạo bounding box khuôn mặt từ landmarks
            lm_x = [pt[0] for pt in landmarks]
            lm_y = [pt[1] for pt in landmarks]
            x_min, x_max = min(lm_x), max(lm_x)
            y_min, y_max = min(lm_y), max(lm_y)
            face_w = x_max - x_min
            face_h = y_max - y_min

            # -------------------- ĐỊNH NGHĨA ROI -------------------- #
            # Forehead (trán)
            fh_w, fh_h = int(face_w * 0.8), int(face_h * 0.25)
            fh_x = x_min + face_w // 2 - fh_w // 2
            fh_y = y_min + int(face_h * 0.05)

            # Left cheek (má trái)
            lc_w, lc_h = int(face_w * 0.35), int(face_h * 0.5)
            lc_x = x_min + int(face_w * 0.15)
            lc_y = y_min + int(face_h * 0.3)

            # Right cheek (má phải)
            rc_w, rc_h = lc_w, lc_h
            rc_x = x_min + int(face_w * 0.5)
            rc_y = lc_y

            # Under lips (dưới môi / cằm trên)
            ul_w, ul_h = int(face_w * 0.5), int(face_h * 0.2)
            ul_x = x_min + face_w // 2 - ul_w // 2
            ul_y = y_min + int(face_h * 0.65)

            # -------------------- CLAMP để không vượt khung hình -------------------- #
            def clamp(v, low, high): 
                return max(low, min(v, high))

            roi_boxes['forehead'] = (clamp(fh_x,0,W), clamp(fh_y,0,H), fh_w, fh_h)
            roi_boxes['left_cheek'] = (clamp(lc_x,0,W), clamp(lc_y,0,H), lc_w, lc_h)
            roi_boxes['right_cheek'] = (clamp(rc_x,0,W), clamp(rc_y,0,H), rc_w, rc_h)
            roi_boxes['under_lips'] = (clamp(ul_x,0,W), clamp(ul_y,0,H), ul_w, ul_h)

        roi_rgbs = []
        if roi_boxes and landmarks is not None:
            # Vẽ lưới landmark toàn bộ (xanh lá)
            try:
                all_pts = np.array(landmarks, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [all_pts], isClosed=True, color=(0, 255, 0), thickness=1)
            except Exception:
                pass

            for name, (rx, ry, rw, rh) in roi_boxes.items():
                # Nếu có polygon định nghĩa → mask
                idxs = ROI_LANDMARKS.get(name, None)
                if idxs is not None:
                    mask_poly = create_roi_mask(frame, landmarks, idxs)
                    masked = cv2.bitwise_and(frame, frame, mask=mask_poly)
                    roi_crop = masked[ry:ry+rh, rx:rx+rw]
                else:
                    roi_crop = frame[ry:ry+rh, rx:rx+rw]

                if roi_crop.size > 0:
                    avg_rgb = skin_mask_ycbcr(roi_crop)
                    if avg_rgb is not None:
                        roi_rgbs.append(avg_rgb)

                # Vẽ khung chữ nhật vàng + label
                # cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 1)
                # cv2.putText(frame, name, (rx, ry - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                #            0.4, (0, 255, 255), 1)

            if roi_rgbs:
                avg_rgb_all = np.mean(roi_rgbs, axis=0)
                rgb_signals.append(avg_rgb_all)
                frame_count += 1

        cv2.imshow("Video Processing", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"📸 Số khung hình đã trích xuất tín hiệu: {frame_count}")

    if len(rgb_signals) < 30:
        print(f"⚠ Video {video_file} không đủ tín hiệu (chỉ có {len(rgb_signals)} khung hình).")
        continue

    raw_ppg = chrom_algorithm(rgb_signals)
    features = extract_features(raw_ppg, fps=fps)

    if features is not None:
        results.append({
            "Sex": sex,
            "Age": age,
            "BMI": bmi,
            "BPM": bpm,
            "SBP": sbp,
            "DBP": dbp,
            **features
        })
        print(f"✅ {video_file} -> HR={features['HR']:.2f}, RMSSD={features['HRV_RMSSD']:.2f}, pNN50={features['pNN50']:.2f}")

# ================== LƯU RA FILE HUẤN LUYỆN ================== #
features_df = pd.DataFrame(results)
features_df.to_csv("features_dataset.csv", index=False)
print("\n💾 Đã lưu đặc trưng vào features_dataset.csv")
