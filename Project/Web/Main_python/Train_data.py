import os
import time
import warnings
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from scipy.signal import butter, filtfilt, periodogram, find_peaks, welch, detrend
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from joblib import dump, load

# ---- warnings / env ----
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')
warnings.filterwarnings("ignore", message="X has feature names, but Ridge was fitted without feature names")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def chrom_algorithm(rgb_signals):
    rgb = np.asarray(rgb_signals, dtype=float)
    if rgb.shape[0] < 30:
        return None
    mean_rgb = np.mean(rgb, axis=0)
    eps = 1e-6
    normalized = (rgb - mean_rgb) / (mean_rgb + eps)
    X = 3 * normalized[:, 0] - 2 * normalized[:, 1]
    Y = 1.5 * normalized[:, 0] + normalized[:, 1] - 1.5 * normalized[:, 2]
    std_X = np.std(X)
    std_Y = np.std(Y)
    alpha = std_X / std_Y if std_Y != 0 else 0.0
    S = X - alpha * Y
    return S

def skin_mask_ycbcr(roi):
    if roi is None or roi.size == 0:
        return None
    ycbcr = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycbcr, np.array([0,133,77]), np.array([255,173,127]))
    skin_pixels = roi[mask > 0]
    if len(skin_pixels) == 0:
        avg_bgr = np.mean(roi.reshape(-1,3), axis=0)
    else:
        avg_bgr = np.mean(skin_pixels, axis=0)
    avg_rgb = avg_bgr[::-1].astype(float)   # BGR -> RGB
    return avg_rgb

def extract_features(ppg_signal, fps=30):
    if ppg_signal is None or len(ppg_signal) < fps * 3:
        return None

    lowcut = 0.7
    highcut = 4.0
    b, a = butter(3, [lowcut / (fps/2), highcut / (fps/2)], btype='bandpass')
    try:
        filtered = filtfilt(b, a, ppg_signal)
    except Exception:
        return None

    if np.isnan(filtered).any() or np.all(filtered == 0):
        return None

    filtered = detrend(filtered)
    f, pxx = welch(filtered, fs=fps, nperseg=min(256, len(filtered)))

    band_mask = (f >= lowcut) & (f <= highcut)
    if not np.any(band_mask):
        return None
    band_f = f[band_mask]
    band_pxx = pxx[band_mask]
    peak_idx = np.argmax(band_pxx)
    hr_freq = band_f[peak_idx]
    hr = hr_freq * 60.0
    power = np.sum(band_pxx)

    peaks, _ = find_peaks(filtered, distance=max(1, int(round(fps*0.4))))
    if len(peaks) <= 1:
        rr_intervals = np.array([])
        hrv_std = 0.0
        hrv_rmssd = 0.0
        pnn50 = 0.0
    else:
        rr_intervals = np.diff(peaks) / fps
        hrv_std = np.std(rr_intervals) * 1000.0
        diff_rr = np.diff(rr_intervals) * 1000.0
        hrv_rmssd = np.sqrt(np.mean(diff_rr**2)) if diff_rr.size > 0 else 0.0
        pnn50 = (np.sum(np.abs(diff_rr) > 50) / diff_rr.size * 100.0) if diff_rr.size > 0 else 0.0

    peakfreq = hr_freq
    peakpower = band_pxx[peak_idx] if band_pxx.size > 0 else 0.0

    mean_val = np.mean(filtered)
    std_val = np.std(filtered)
    skew = stats.skew(filtered)
    kurt = stats.kurtosis(filtered)
    peak_to_peak = np.ptp(filtered)
    snr_val = power / (std_val if std_val != 0 else 1.0)

    return hr, hrv_rmssd, hrv_std, pnn50, power, peakfreq, peakpower, mean_val, std_val, skew, kurt, peak_to_peak, snr_val

def get_landmark_points(frame, face_mesh_instance):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh_instance.process(rgb_frame)
    if results and results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        landmark_points = []
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            landmark_points.append((x, y))
        return landmark_points
    return None

def get_multiple_rois_from_landmarks(landmarks, frame_shape):

    def clamp(val, minv, maxv):
        return max(minv, min(val, maxv))

    if landmarks is None:
        return {}

    rois = {}
    # Các chỉ số landmark theo Mediapipe face mesh
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose_tip = landmarks[1]
    chin = landmarks[152]
    mouth_top = landmarks[13]
    mouth_bottom = landmarks[14]

    # Tính chiều rộng/chều cao khuôn mặt (approx)
    face_width = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
    face_height = np.linalg.norm(np.array(nose_tip) - np.array(chin))

    FRAME_WIDTH = frame_shape[1]
    FRAME_HEIGHT = frame_shape[0]

    # Forehead
    fh_center_x = int((left_eye[0] + right_eye[0]) / 2)
    fh_center_y = int(left_eye[1] - face_height * 0.5)
    fh_w = int(max(10, face_width * 1.2))
    fh_h = int(max(10, face_height * 0.45))
    rois['forehead'] = (
        clamp(fh_center_x - fh_w // 2, 0, FRAME_WIDTH),
        clamp(fh_center_y - fh_h // 2, 0, FRAME_HEIGHT),
        fh_w, fh_h
    )

    # Left cheek
    lc_center_x = int(left_eye[0])
    lc_center_y = int(nose_tip[1] + face_height * -0.1)
    lc_w = int(max(10, face_width * 0.4))
    lc_h = int(max(10, face_height * 0.6))
    rois['left_cheek'] = (
        clamp(lc_center_x - lc_w // 2, 0, FRAME_WIDTH),
        clamp(lc_center_y - lc_h // 2, 0, FRAME_HEIGHT),
        lc_w, lc_h
    )

    # Right cheek
    rc_center_x = int(right_eye[0])
    rc_center_y = lc_center_y
    rc_w, rc_h = lc_w, lc_h
    rois['right_cheek'] = (
        clamp(rc_center_x - rc_w // 2, 0, FRAME_WIDTH),
        clamp(rc_center_y - rc_h // 2, 0, FRAME_HEIGHT),
        rc_w, rc_h
    )

    # Under lips 
    ul_center_x = fh_center_x
    ul_center_y = int(mouth_bottom[1] + face_height * 0.35)
    ul_w = int(max(10, face_width * 0.5))
    ul_h = int(max(10, face_height * 0.25))
    rois['under_lips'] = (
        clamp(ul_center_x - ul_w // 2, 0, FRAME_WIDTH),
        clamp(ul_center_y - ul_h // 2, 0, FRAME_HEIGHT),
        ul_w, ul_h
    )

    return rois

# ========================
# ĐỌC DỮ LIỆU, TRAIN/LOAD MODEL
# ========================
DATA_CSV = "Web/Main_python/fake_data_500.csv"
feature_cols = ["Age", "BMI", "Mean", "Skewness", "Kurtosis", "HRV_RMSSD", "HRV_STD"]

# Cố gắng load dữ liệu (nếu file tồn tại)
if os.path.exists(DATA_CSV):
    df = pd.read_csv(DATA_CSV)
else:
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu '{DATA_CSV}' trong folder làm việc.")

X = df[feature_cols]
y_sbp = df["SBP"]
y_dbp = df["DBP"]

# Tính correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_df.corr()
corr_sbp = corr_matrix["SBP"].sort_values(ascending=False)
corr_dbp = corr_matrix["DBP"].sort_values(ascending=False)
# print(corr_sbp)
# print(corr_dbp)

# Split
X_train, X_test, y_sbp_train, y_sbp_test, y_dbp_train, y_dbp_test = train_test_split(
    X, y_sbp, y_dbp, test_size=0.2, random_state=42
)

MODEL_SBP_FILE = "pipeline_sbp.pkl"
MODEL_DBP_FILE = "pipeline_dbp.pkl"

# Nếu có file model, load; nếu không, train
if os.path.exists(MODEL_SBP_FILE) and os.path.exists(MODEL_DBP_FILE):
    try:
        pipeline_sbp = load(MODEL_SBP_FILE)
        pipeline_dbp = load(MODEL_DBP_FILE)
        print("✔️ Load model từ file .pkl thành công.")
    except Exception as e:
        print("⚠️ Lỗi khi load model .pkl — sẽ train lại. Error:", e)
        pipeline_sbp = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
        pipeline_dbp = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
        pipeline_sbp.fit(X_train, y_sbp_train)
        pipeline_dbp.fit(X_train, y_dbp_train)
        print("✔️ Đã train lại model (không lưu ra file).")
else:
    pipeline_sbp = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    pipeline_dbp = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    pipeline_sbp.fit(X_train, y_sbp_train)
    pipeline_dbp.fit(X_train, y_dbp_train)
    print("✔️ Đã train model từ dữ liệu (không lưu ra file).")

# Evaluate quickly on test set
y_sbp_pred = pipeline_sbp.predict(X_test)
y_dbp_pred = pipeline_dbp.predict(X_test)
print("🔍 MAE kiểm tra mô hình:")
print("MAE SBP:", mean_absolute_error(y_sbp_test, y_sbp_pred))
print("MAE DBP:", mean_absolute_error(y_dbp_test, y_dbp_pred))
print("R2 SBP:", r2_score(y_sbp_test, y_sbp_pred))
print("R2 DBP:", r2_score(y_dbp_test, y_dbp_pred))
print("")

# ========================
# HÀM XỬ LÍ CHUNG CHO WEBCAM / VIDEO
# ========================
def process_capture(cap, pipeline_sbp, pipeline_dbp, age, bmi,
                    window_sec=5, show_window=True, flip_webcam=False, max_seconds=None):
    """
    Xử lý capture (webcam hoặc file). 
    Trả về: list of dict predictions và list rgb_signals (toàn bộ frame avg rgb)
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        fps = float(fps) if fps and fps > 0 else 30.0
    except Exception:
        fps = 30.0

    WINDOW_SIZE = max(30, int(window_sec * fps))
    buffer_rppg = []
    rgb_signals = []
    predictions = []
    frame_count = 0
    start_time = time.time()

    # Tạo face_mesh cục bộ, auto close sau khi xong
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as local_face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if flip_webcam:
                frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape

            landmarks = get_landmark_points(frame, local_face_mesh)
            roi_boxes = get_multiple_rois_from_landmarks(landmarks, frame.shape) if landmarks is not None else None

            roi_rgbs = []
            if roi_boxes:
                for name, (rx, ry, rw, rh) in roi_boxes.items():
                    # clamp to image
                    rx, ry = int(max(0, rx)), int(max(0, ry))
                    rw = int(max(0, min(rw, w - rx)))
                    rh = int(max(0, min(rh, h - ry)))
                    if rw == 0 or rh == 0:
                        continue
                    roi = frame[ry:ry+rh, rx:rx+rw]
                    if roi.size == 0:
                        continue
                    avg_rgb = skin_mask_ycbcr(roi)
                    if avg_rgb is not None:
                        roi_rgbs.append(avg_rgb)

                    if show_window:
                        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 1)
                        cv2.putText(frame, name, (rx, ry - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            if roi_rgbs:
                avg_rgb_all = np.mean(roi_rgbs, axis=0)
                rgb_signals.append(avg_rgb_all)

                buffer_rppg.append(avg_rgb_all)
                if len(buffer_rppg) > WINDOW_SIZE:
                    buffer_rppg.pop(0)

                # Khi đủ window, tính và dự đoán (hạn chế tần suất in để đỡ spam)
                if len(buffer_rppg) == WINDOW_SIZE and (frame_count % max(1, int(round(fps/2))) == 0):
                    raw_ppg = chrom_algorithm(buffer_rppg)
                    features = extract_features(raw_ppg, fps=fps)
                    
                    if features is not None:
                        mean_val = features[7]   # Mean
                        skew = features[9]       # Skewness
                        kurt = features[10]
                        hrv_rmssd = features[1]
                        hrv_std = features[2]
                        new_input = pd.DataFrame([[age, bmi, mean_val, skew, kurt, hrv_rmssd, hrv_std]], columns=feature_cols)

                        try:
                            sbp = float(pipeline_sbp.predict(new_input)[0])
                            dbp = float(pipeline_dbp.predict(new_input)[0])
                        except Exception as e:
                            sbp, dbp = None, None
                            print("⚠️ Lỗi khi predict:", e)

                        timestamp = time.time() - start_time
                        predictions.append({
                            "time_s": timestamp,
                            "frame": frame_count,
                            "sbp": sbp,
                            "dbp": dbp,
                            "mean": float(mean_val),
                            "skew": float(skew)
                        })

                        if frame_count % 10 == 0:
                            print(f"[{frame_count}] t={timestamp:.1f}s -> SBP: {sbp:.2f} | DBP: {dbp:.2f}")

                        # Hiển thị lên frame
                        if show_window:
                            txt1 = f"SBP:{sbp:.1f} DBP:{dbp:.1f}"
                            cv2.putText(frame, txt1, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # Vẽ landmarks (toàn mặt) nếu có
            if landmarks is not None and show_window:
                pts = np.array(landmarks, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=1)

            if show_window:
                cv2.imshow("rPPG - Camera/Video", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("⏹️ User pressed 'q' -> stop.")
                    break

            if max_seconds is not None and (time.time() - start_time) > max_seconds:
                print("⏱️ Đạt thời gian tối đa (max_seconds) -> stop.")
                break

            frame_count += 1
            if frame_count > 300:  # ví dụ dừng sau 300 frame (~10 giây nếu fps=30)
                print("🛑 Đạt số frame tối đa -> stop.")
                break
        if show_window:
            cv2.destroyAllWindows()

    return predictions, rgb_signals

def process_video_file(video_path, pipeline_sbp, pipeline_dbp, age, bmi, **kwargs):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file không tồn tại: {video_path}")
    cap = cv2.VideoCapture(video_path)
    preds, rgbs = process_capture(cap, pipeline_sbp, pipeline_dbp, age, bmi, show_window=kwargs.get("show_window", True),
                                 flip_webcam=False, window_sec=kwargs.get("window_sec", 5),
                                 max_seconds=kwargs.get("max_seconds", None))
    cap.release()
    return preds, rgbs

def process_webcam(pipeline_sbp, pipeline_dbp, age, bmi, cam_id=0, **kwargs):
    cap = cv2.VideoCapture(cam_id)
    preds, rgbs = process_capture(cap, pipeline_sbp, pipeline_dbp, age, bmi,
                                 show_window=kwargs.get("show_window", True),
                                 flip_webcam=kwargs.get("flip_webcam", True),
                                 window_sec=kwargs.get("window_sec", 5),
                                 max_seconds=kwargs.get("max_seconds", None))
    cap.release()
    return preds, rgbs

import tkinter as tk
from tkinter import filedialog

def choose_video_file():
    root = tk.Tk()
    root.withdraw()  # ẩn cửa sổ chính
    file_path = filedialog.askopenfilename(
        title="Chọn file video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    return file_path

# MAIN: chọn chế độ (webcam / video)

if __name__ == "__main__":
    # Thông tin người dùng (bạn có thể sửa trước khi chạy, hoặc sau khi chạy prompt)
    sex = 0      # 0: Nam, 1: Nữ (hiện không dùng)
    age = 21
    bmi = 21.45 # BMI càng cao thì huyết áp càng cao

    print("Chạy chương trình: 1 = webcam, 2 = video file")
    mode = input("Chọn chế độ (1 = webcam, 2 = video): ").strip()

    all_preds = []
    all_rgbs = []

    try:
        if mode == "1":
            print("\n📷 Chạy webcam. Nhấn 'q' trên cửa sổ để dừng.")
            preds, rgbs = process_webcam(pipeline_sbp, pipeline_dbp, age=age, bmi=bmi,
                                        cam_id=0, window_sec=5, show_window=True, flip_webcam=True, max_seconds=None)
            all_preds.extend(preds)
            all_rgbs.extend(rgbs)

        elif mode == "2":
            video_path = choose_video_file()
            if not video_path:
                print("❌ Không chọn file nào.")
            else:
                preds, rgbs = process_video_file(
                video_path,
                pipeline_sbp, pipeline_dbp,
                age, bmi,   # truyền theo vị trí
                window_sec=5,
                max_seconds=None,   # None = chạy đúng toàn bộ video
                show_window=True
)

                all_preds.extend(preds)
                all_rgbs.extend(rgbs)

        else:
            print("⚠️ Lựa chọn không hợp lệ. Thoát.")
    except Exception as e:
        print("⚠️ Lỗi khi chạy capture:", e)

    # Xuất kết quả
    if len(all_preds) == 0:
        print("⚠️ Không có dự đoán nào được tạo (có thể do không đủ tín hiệu).")
    else:
        df_preds = pd.DataFrame(all_preds)
        # Thông số tóm tắt
        valid_sbp = df_preds["sbp"].dropna().values
        valid_dbp = df_preds["dbp"].dropna().values
        if len(valid_sbp) > 0:
            print(f"\n📊 SBP trung bình (trong phiên): {np.mean(valid_sbp):.2f} mmHg  |  STD: {np.std(valid_sbp):.2f}")
        if len(valid_dbp) > 0:
            print(f"📊 DBP trung bình (trong phiên): {np.mean(valid_dbp):.2f} mmHg  |  STD: {np.std(valid_dbp):.2f}")

        # Lưu CSV kết quả
        # out_name = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        # df_preds.to_csv(out_name, index=False)
        # print(f"💾 Đã lưu kết quả dự đoán: {out_name}")

    print("Kết thúc.")
