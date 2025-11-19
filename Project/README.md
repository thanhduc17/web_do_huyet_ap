# Dự đoán huyết áp dựa vào khuôn mặt

Dự án này là một **Website dự đoán huyết áp** gồm hai phần:
- **Backend:** File `Train_data.py` & `Read_Predict.py` & `app.py` (Python, Flask/FastAPI)
- **Frontend:** Folder `templates` (Giao diện web người dùng)

---

## 📁 Cấu trúc thư mục
```
project/
│
\---Web
    +---Main_python # Code đo huyết áp
    |   \---__pycache__
    +---static 
    |   +---css
    |   +---img
    |   \---js
    +---templates # Giao diện web
    \---uploads
│
\---README.md

```

## ⚙️ Hướng dẫn chạy trên máy (Local)

### 1 Cài Python
Cài Python >= 3.9  
Kiểm tra:
```bash
python --version
```

### 2 Tạo môi trường ảo (khuyến nghị)
```bash
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows
```
### 3 Chạy backend
Nếu bạn dùng **Flask**, chạy:
```bash
python app.py
```

Sau khi chạy thành công, mở trình duyệt và truy cập:
👉 http://127.0.0.1:5000







