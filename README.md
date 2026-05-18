# 🎮 Cờ Caro AI – Minimax & Alpha-Beta Pruning (9×9)

**Môn học:** Trí Tuệ Nhân Tạo – INT3401 4
**Trường:** Đại học Công nghệ – ĐHQGHN

**Nhóm sinh viên:**
| Họ và tên | MSSV |
|---|---|
| Nguyễn Vũ Đoàn | 23021243 |
| Đặng Huy Hiệu | 23021265 |

---

## 📋 Mô tả chương trình

Chương trình mô phỏng trò chơi Cờ Caro trên bàn cờ **9×9**,
người chơi đấu với máy tính. Máy sử dụng thuật toán **Minimax**
hoặc **Alpha-Beta Pruning** để chọn nước đi tối ưu.
Điều kiện thắng là có **4 quân liên tiếp**.

---

## ⚙️ Yêu cầu hệ thống

- Python **3.8** trở lên
- Thư viện `numpy`
- `tkinter` (có sẵn trong Python, không cần cài thêm)

---

## 🚀 Hướng dẫn cài đặt và chạy

### Bước 1: Clone repository
```bash
git clone https://github.com/23021265-source/23021243_23021265_CaroAI.git
cd 23021243_23021265_CaroAI
```

### Bước 2: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 3: Chạy chương trình
```bash
cd source_code
python caro_ai.py
```

---

## 🎯 Hướng dẫn sử dụng

| Thành phần | Chức năng |
|---|---|
| Radio **Alpha-Beta/Minimax** | Chọn thuật toán AI |
| Spinbox **Độ sâu** | Điều chỉnh độ sâu (1–6) |
| Nút **Chơi lại** | Reset bàn cờ |
| Nút **Benchmark** | So sánh hai thuật toán |

### Cách chơi
1. Chọn thuật toán và độ sâu
2. Click ô trống để đánh quân **(X – đỏ)**
3. AI tự động đánh **(O – xanh)**
4. Xem thống kê ở bảng bên phải
