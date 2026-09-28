# Hệ Thống Agent Đặt Vé Máy Bay & Lớp Harness Kiểm Soát An Toàn

## 📑 BÁO CÁO KỸ THUẬT CHUYÊN SÂU
Toàn bộ phân tích kiến trúc, vai trò từng file, các hàm chốt chặn trong Harness và đánh giá sự đánh đổi kỹ thuật (Engineering Trade-offs) giữa 3 mẫu Agent được trình bày chi tiết tại:  
👉 **[Xem Báo cáo Chi tiết: baocao_BTVN3.md](baocao_BTVN3.md)**

---

## 🌟 Giới Thiệu Dự Án
Dự án cài đặt và kiểm thử hệ thống Agent đặt vé máy bay tự động bằng LangChain, phân định rạch ròi giữa **Bộ não suy luận (Model)** và **Khung giàn điều phối & an toàn (Harness)**.

### 1. Ba mẫu thiết kế suy luận của Agent
* **ReAct** (`agent/react_agent.py`): Suy luận đan xen hành động từng bước (`Thought -> Action -> Observation`), thích ứng cực nhạy với biến động môi trường.
* **Plan-then-Execute** (`agent/plan_execute_agent.py`): Lập kế hoạch tĩnh dạng JSON và thực thi tuần tự bằng code, tối ưu chi phí token.
* **Mẫu Lai - Hybrid** (`agent/hybrid_agent.py`): Kết hợp ưu điểm của cả hai, trang bị cảm biến phát hiện biến cố để tự động tái lập kế hoạch (*Dynamic Re-planning*).

### 2. Khung giàn Harness kiểm soát an toàn đa tầng
* **Constraints as Data**: Đóng gói yêu cầu người dùng thành dữ liệu bất biến, chống trôi dạt mục tiêu (*Goal Drift*).
* **Pre-tool Action Authorization**: Tiền kiểm quyền hạn trước khi gọi tool (chặn thanh toán vượt trần, chặn vé không hoàn tiền).
* **Computational Sensor**: Nghiệm thu kết quả bằng code thuần trong CSDL thực tế, không tin vào lời tuyên bố chủ quan của LLM.
* **Loop & Stall Detection**: Bắt lặp thao tác qua cửa sổ trượt và bắt bế tắc tiến trình.
* **Handoff Protocol**: Bàn giao con người chuẩn 30 giây khi vượt thẩm quyền.
* **Anti-Hallucination Synthesizer**: Trích xuất dữ liệu vé thật trong CSDL để trả lời khách hàng.

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Dự Án

### 1. Clone repository từ GitHub
```bash
git clone https://github.com/phatcao026/agent-dat-ve-may-bay.git
cd agent-dat-ve-may-bay
```

### 2. Khởi tạo môi trường ảo (Khuyến nghị)
* **Trên Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
* **Trên Linux / macOS**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

---

## 🎮 Cách Thực Thi Hệ Thống

### Cách 1: Giao diện CLI Tương tác Dòng lệnh
```bash
python main.py
```
Menu hiển thị 3 chế độ tương tác:
* **[1] Đặt vé tương tác từ bàn phím**: Tự do nhập điểm đi, điểm đến, ngày bay, ngân sách và chọn 1 trong 3 Agent để quan sát chu trình suy luận từng bước và phản hồi tự nhiên ở cuối.
* **[2] Chạy tự động Benchmark 4 Kịch bản**: Chạy kiểm thử tự động toàn bộ ma trận 12 lượt đo đạc.
* **[3] Xem lịch sử benchmark gần nhất**: In bảng tổng hợp số liệu (Token, Latency, Mã dừng) trích xuất trực tiếp từ file `benchmark_history.json` trong 0.1 giây.

---

### Cách 2: Chạy Ma trận Benchmark Đánh giá (12 Ca kiểm thử)
```bash
python evaluate.py
```
Lệnh này sẽ tự động:
1. Đưa 3 mẫu Agent qua 4 kịch bản biên (Happy Path, Cháy vé rẻ nhất, Vượt trần ngân sách, Vé không hoàn tiền).
2. Đo đếm số lượt gọi tool, chi phí token, thời gian thực thi.
3. Tự động lưu vết dữ liệu vào file `benchmark_history.json`.
4. Xuất bảng tổng hợp Markdown trực tiếp ra màn hình.

---

## ⚙️ Cấu Hình Mô Hình Ngôn Ngữ (LLM Configuration)

Mặc định hệ thống sử dụng **Mock LLM Offline** tích hợp sẵn (chạy tức thì, 0 đồng, không cần internet hay API Key).

Nếu muốn kết nối với **Mô hình AI Thật**, tạo file `.env` tại thư mục gốc của dự án:

### Lựa chọn A: Máy chủ GPU Nội bộ UIT (`llm.uit.edu.vn`)
*(Yêu cầu kết nối mạng Wifi trường UIT hoặc đang bật VPN nội bộ trường)*:
```env
OPENAI_API_KEY="ma-api-key-cua-ban"
OPENAI_API_BASE="https://llm.uit.edu.vn/qwen/v1"
SE373_MODEL="qwen3.8-27b"
```
*(Hệ thống đã cài sẵn cấu hình tự động nhận diện mô hình Qwen của trường và tắt thinking `enable_thinking: False` để tối ưu hóa tốc độ gọi Tool).*

### Lựa chọn B: OpenAI API
```env
OPENAI_API_KEY="sk-proj-..."
SE373_MODEL="gpt-4o-mini"
```

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
agent-dat-ve-may-bay/
├── README.md               # Hướng dẫn cài đặt, thực thi và giới thiệu dự án
├── baocao_BTVN3.md         # Báo cáo kỹ thuật phân tích chuyên sâu
├── mock_env.py             # CSDL giả lập trong bộ nhớ & 5 LangChain Tools
├── harness.py              # Khung giàn kiểm soát Harness 4 chốt chặn & Loop/Stall
├── evaluate.py             # Kịch bản benchmark 12 ca chạy thực nghiệm
├── main.py                 # Giao diện CLI tương tác đa chế độ
├── benchmark_history.json  # Dữ liệu vết thực nghiệm gần nhất
├── requirements.txt        # Danh sách thư viện phụ thuộc
├── .gitignore              # Bỏ qua file rác và file .env
└── agent/                  # Các mô hình suy luận
    ├── __init__.py
    ├── llm_setup.py        # Quản lý kết nối LLM (OpenAI / GPU UIT / Mock)
    ├── react_agent.py      # ReAct Agent
    ├── plan_execute_agent.py# Plan-then-Execute Agent
    └── hybrid_agent.py     # Hybrid Agent với Dynamic Re-planning
```