from flask import Flask, render_template, request, jsonify, Response, session
import os
import cv2
from python_do_ap_huyết.Train_data import process_video_file, process_webcam, pipeline_sbp, pipeline_dbp

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "secret_key_demo"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------ ROUTES GIAO DIỆN ------------------
@app.route("/")
def index_page():
    return render_template("index.html")

@app.route("/welcome")
def welcome_page():
    return render_template("welcome.html")

@app.route("/signup")
def signup_page():
    return render_template("signup.html")

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/information", methods=["GET", "POST"])
def information_page():
    if request.method == "POST":
        data = request.get_json()
        if data:
            session["user_info"] = {
                "fullName": data.get("fullName") or "Người dùng",
                "age": float(data.get("age", 21)),
                "height": float(data.get("height", 0)),
                "weight": float(data.get("weight", 0)),
                "bmi": float(data.get("bmi", 21.5)),
                "gender": data.get("gender") or "Khác",
                "restMin": float(data.get("restMin", 5))
            }
        return jsonify({"status": "ok"})
    return render_template("information.html")


@app.route("/measure")
def measure_page():
    # Lấy dữ liệu từ session, nếu chưa có thì dùng mặc định
    user_info = session.get("user_info", {
        "fullName": "Người dùng",
        "age": 21,
        "height": 0,
        "weight": 0,
        "bmi": 21.5,
        "gender": "Khác",
        "restMin": 5
    })
    return render_template("measure.html", user_info=user_info)



# ------------------ UPLOAD VIDEO ------------------
@app.route("/upload", methods=["POST"])
def upload_file():
    if "video" not in request.files:
        return jsonify({"error": "Không có file"}), 400

    file = request.files["video"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    # Lấy dữ liệu từ session
    user_info = session.get("user_info", {})
    fullName = user_info.get("fullName", "Người dùng")
    age = user_info.get("age", 21)
    height = user_info.get("height", 0)
    weight = user_info.get("weight", 0)
    bmi = user_info.get("bmi", 21.5)
    gender = user_info.get("gender", "Khác")
    restMin = user_info.get("restMin", 5)

    print("📌 [UPLOAD] Đang đo với thông tin:")
    print(f"   Họ tên = {fullName}, Tuổi = {age}, Chiều cao = {height}, Cân nặng = {weight}, BMI = {bmi}, Giới tính = {gender}, Nghỉ = {restMin} phút")

    preds, _ = process_video_file(
        path, pipeline_sbp, pipeline_dbp,
        age=age, bmi=bmi, show_window=False
    )
    if not preds:
        return jsonify({"error": "Không nhận diện được"}), 500

    last = preds[-1]
    print(f"   ➡️ Kết quả đo: SBP = {last['sbp']}, DBP = {last['dbp']}")
    return jsonify({"sbp": last["sbp"], "dbp": last["dbp"]})


# ------------------ WEBCAM ------------------
@app.route("/webcam", methods=["GET"])
def webcam():
    user_info = session.get("user_info", {})
    fullName = user_info.get("fullName", "Người dùng")
    age = user_info.get("age", 21)
    height = user_info.get("height", 0)
    weight = user_info.get("weight", 0)
    bmi = user_info.get("bmi", 21.5)
    gender = user_info.get("gender", "Khác")
    restMin = user_info.get("restMin", 5)

    print("📌 [WEBCAM] Đang đo với thông tin:")
    print(f"   Họ tên = {fullName}, Tuổi = {age}, Chiều cao = {height}, Cân nặng = {weight}, BMI = {bmi}, Giới tính = {gender}, Nghỉ = {restMin} phút")

    preds, _ = process_webcam(
        pipeline_sbp, pipeline_dbp,
        age=age, bmi=bmi,
        show_window=False, flip_webcam=True, max_seconds=10
    )
    if not preds:
        return jsonify({"error": "Không nhận diện được từ webcam"}), 500

    last = preds[-1]
    print(f"   ➡️ Kết quả đo: SBP = {last['sbp']}, DBP = {last['dbp']}")
    return jsonify({"sbp": last["sbp"], "dbp": last["dbp"]})


# ------------------ STREAM WEBCAM ------------------
@app.route("/video_feed")
def video_feed():
    cap = cv2.VideoCapture(0)

    def generate():
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, (640, 480))
                _, buffer = cv2.imencode(".jpg", frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        finally:
            cap.release()

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")



# ------------------ MAIN ------------------
if __name__ == "__main__":
     app.run(port=5000, debug=True)
