
# Hướng dẫn cài đặt và chạy project Vandecongnghethongtin

## 1. Tải code từ GitHub

Bạn có thể lựa chọn một trong các phương pháp sau:

### A. Sử dụng Git (nếu đã cài Git)
```bash
git clone https://github.com/LeBaoLan/Vandecongnghethongtin.git
cd Vandecongnghethongtin
```

### B. Tải thủ công từ trình duyệt
1. Truy cập: https://github.com/LeBaoLan/Vandecongnghethongtin
2. Nhấn **“Code”** → **“Download ZIP”**.
3. Giải nén file ZIP và mở thư mục vừa giải nén.

---

## 2. Thiết lập môi trường Python

### A. Tạo môi trường ảo (khuyến nghị)
```bash
# Trên macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Trên Windows PowerShell
python -m venv venv
.env\Scripts\Activate.ps1
```

### B. Cài các thư viện cần thiết
```bash
pip install matplotlib numpy pandas scikit-learn tensorflow
```

Nếu repo có file `requirements.txt`, chỉ cần:
```bash
pip install -r requirements.txt
```

---

## 3. Chạy code Python

Trong thư mục chứa code:

1. Kích hoạt môi trường ảo (nếu chưa):
   - `source venv/bin/activate` (macOS/Linux)
   - `.env\Scripts\Activate.ps1` (Windows)

2. Chạy file Python:
```bash
python tên_file_của_bạn.py
```

Ví dụ:
```bash
python demo_week_1.py
```

3. Nếu code sử dụng Jupyter Notebook (`.ipynb`):
```bash
pip install notebook
jupyter notebook
```

---
