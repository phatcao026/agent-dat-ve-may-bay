# BÁO CÁO BÀI TẬP VỀ NHÀ #3: KỸ THUẬT XÂY DỰNG HỆ THỐNG AGENTIC AI

### THÔNG TIN CÁ NHÂN:
* **Họ và tên**: Cao Tiến Phát
* **MSSV**: 24521289
* **Lớp**: Kỹ thuật xây dựng hệ thống Agentic AI - SE373.R11
* **Khoa**: Công nghệ Phần mềm – Trường Đại học Công nghệ Thông tin (UIT - ĐHQG-HCM)

### ĐỀ BÀI:
> **BTVN#3 · Dựng agent đặt vé máy bay bằng LangChain**  
> *Tìm hiểu Langchain, LangGraph → Tạo tool mockup → Viết lớp harness cho Agent này. Nộp .py kèm báo cáo.*  
> 1. Cài đặt đủ các lớp harness: ràng buộc là dữ liệu, tiêu chí hoàn thành kiểm bằng code, kiểm quyền, bàn giao;  
> 2. Cài đặt Agent với 3 mẫu thiết kế: ReAct, Plan-then-Execute, Lai;  
> 3. Đánh giá hiệu quả của Agent với 3 mẫu thiết kế khác nhau.

---

## PHẦN 1: TỔNG QUAN KIẾN TRÚC & PHÂN TẦNG HỆ THỐNG (SYSTEM ARCHITECTURE)

### 1.1. Triết lý thiết kế và Ranh giới Phân định Model – Harness
Hệ thống được phát triển bám sát công thức nền tảng của kỹ nghệ hệ thống Agentic AI:
$$\textbf{Agent} = \textbf{Goal} + \textbf{Tools} + \textbf{Loop} + \textbf{Termination}$$

Trong kiến trúc này, ranh giới trách nhiệm giữa **Model (LLM)** và **Harness** được phân định tuyệt đối rõ ràng:
* **Model (LLM)**: Đóng vai trò là **"Bộ não suy luận"**. Mô hình chỉ có nhiệm vụ đọc hiểu ngữ cảnh hiện tại và **đề xuất gọi công cụ nào với tham số gì** (`tool_calls`). Mô hình hoàn toàn **không** được tự ý thực thi các thay đổi vào cơ sở dữ liệu và **không** có quyền tự tuyên bố hoàn thành tác vụ.
* **Harness**: Đóng vai trò là **"Khung giàn điều phối và Giám sát an toàn"** bọc bên ngoài toàn bộ chu trình. Harness chịu trách nhiệm 100% trong việc: duy trì mục tiêu bất biến, tiền kiểm quyền hạn trước khi chạy tool, thực thi gọi tool, phát hiện vòng lặp/bế tắc, nghiệm thu kết quả thực tế bằng code và tự động bàn giao cho con người khi vượt thẩm quyền.

```
[Yêu cầu đặt vé của Người dùng]
              │
              ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      LỚP GIÁM SÁT HARNESS                              │
 │  1. Chuẩn hóa UserConstraints (Ràng buộc là dữ liệu cứng)              │
 └──────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
       ┌───────────────────► [VÒNG LẶP SUY LUẬN AGENT] ◄──────────────────┐
       │                            │                                     │
       │                            ▼                                     │
       │               [1. Model đề xuất Tool Call]                       │
       │                            │                                     │
       │                            ▼                                     │
       │             [TRẠM 0: KIỂM QUYỀN (Pre-tool Gate)]                 │
       │              ├── Vượt quyền ──► BÀN GIAO (Handoff 30s)           │
       │              └── Hợp lệ ──────► Thực thi Tool vào Mock Sandbox   │
       │                                     │                            │
       │                                     ▼                            │
       │                            [Nhận Observation]                    │
       │                                     │                            │
       │                                     ▼                            │
       │             [TRẠM 1-4: XÉT ĐIỀU KIỆN DỪNG (Post-tool Checks)]    │
       │              ├── #1. Nghiệm thu bằng Code ──► DỪNG THÀNH CÔNG   │
       │              ├── #2. Bắt Lặp (LOOP) ────────► DỪNG BẤT THƯỜNG    │
       │              ├── #3. Bắt Bế tắc (STALL) ────► DỪNG BẤT THƯỜNG    │
       │              └── #4. Hết Ngân sách Vòng ────► DỪNG BẤT THƯỜNG    │
       │                                     │                            │
       └──────── (Chưa xong: Nạp Observation mới rồi lặp tiếp) ───────────┘
                                    │
                                    ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │        HARNESS TỰ ĐỘNG XUẤT CÂU TRẢ LỜI TỰ NHIÊN CHO KHÁCH HÀNG        │
 │   (Trích xuất dữ liệu vé thật trong CSDL để chống ảo giác Hallucination)│
 └────────────────────────────────────────────────────────────────────────┘
```

---

### 1.2. Phân rã Cấu trúc File theo Kiến trúc Phân tầng Sạch (Clean Architecture)
Toàn bộ mã nguồn được tổ chức theo từng tầng độc lập, tuân thủ chặt chẽ nguyên lý **Đơn nhiệm (Single Responsibility Principle - SRP)**:

```text
agent-dat-ve-may-bay/
│
├── mock_env.py               # [Tầng 0: Environment] CSDL giả lập trong bộ nhớ & 5 LangChain Tools
├── harness.py                # [Tầng 1: Safety & Control] Khung giàn kiểm soát Harness 4 chốt chặn
│
├── agent/                    # [Tầng 2: Reasoning Models] Cài đặt 3 Mẫu thiết kế Agent suy luận
│   ├── __init__.py           # Package khởi tạo
│   ├── llm_setup.py          # Factory LLM: switch giữa OpenAI API, GPU UIT và Mock Offline
│   ├── react_agent.py        # Mẫu 1: ReAct (Reasoning + Acting)
│   ├── plan_execute_agent.py # Mẫu 2: Plan-then-Execute (Kế hoạch tĩnh)
│   └── hybrid_agent.py       # Mẫu 3: Mẫu Lai (Plan + ReAct with Replanning)
│
├── evaluate.py               # [Tầng 3: Evaluation] Ma trận Benchmark 12 lượt chạy (3 Agent x 4 Edge Cases)
├── main.py                   # [Tầng 4: Presentation] Giao diện CLI tương tác dòng lệnh
├── benchmark_history.json    # Artifact lưu vết số liệu thực nghiệm
├── requirements.txt          # Thư viện phụ thuộc
└── .gitignore                # Cấu hình an toàn git
```

### Bảng phân tích chi tiết vai trò từng file (Why - How - Impact):

| Tên File / Thư mục | Mục đích thiết kế (Why) | Cơ chế hiện thực (How) | Rủi ro kỹ thuật giải quyết (Impact) |
| :--- | :--- | :--- | :--- |
| **`mock_env.py`** | Tách rời môi trường hàng không giả lập khỏi logic suy luận của Agent. | Cung cấp CSDL bộ nhớ (`FLIGHTS_DB`, `BOOKINGS_DB`) và 5 `@tool` chuẩn hóa: 3 tool đọc dữ liệu (`search_flights`, `check_seat`, `get_booking`) và 2 tool tác động ghi (`book_seat`, `pay`). | Ngăn chặn Agent gọi nhầm API môi trường thật; dữ liệu trả về chuẩn JSON giúp Model dễ phân tích cú pháp. |
| **`harness.py`** | Đóng vai trò lớp phòng thủ và điều phối an toàn bọc bên ngoài Agent. | Triển khai 4 chốt chặn: Ràng buộc là dữ liệu cứng (`UserConstraints`), Kiểm quyền trước (`check_permission`), Nghiệm thu trạng thái bằng code (`verify_completion`), Bắt lặp/bế tắc (`LoopDetector`), Bàn giao 30s (`handoff`) và Chống ảo giác (`generate_final_response`). | Chặn đứng việc Model tự ý trừ tiền tài khoản, chống ảo giác (hallucination) và triệt tiêu nguy cơ lặp vô tận đốt token. |
| **`agent/react_agent.py`** | Triển khai trường phái suy luận phản xạ từng bước (ReAct: Thought-Action-Observation). | Vòng lặp `while not harness.is_terminated`. Mỗi bước gửi toàn bộ lịch sử kèm `ToolMessage` cho LLM để quyết định hành động kế tiếp. | Xử lý linh hoạt và cực nhạy khi môi trường đổi bất ngờ (như vé hết chỗ thì lập tức chuyển sang chuyến khác). |
| **`agent/plan_execute_agent.py`** | Triển khai trường phái lập kế hoạch tĩnh tách rời (Plan-then-Execute). | Pha 1: Planner gọi LLM 1 lần sinh toàn bộ danh sách bước JSON; Pha 2: Executor duyệt tuần tự bằng code Python mà không gọi thêm LLM. | Tiết kiệm token tối đa và cho phép con người duyệt toàn bộ kế hoạch trước khi chạy; nhưng dễ gãy đổ khi gặp kế hoạch cũ (*stale plan*). |
| **`agent/hybrid_agent.py`** | Triển khai mẫu lai tối ưu: Kết hợp định hướng của Plan và sự linh hoạt của ReAct. | Sinh Plan ban đầu, thực thi từng bước; trang bị cảm biến `is_observation_significantly_changed()` để kích hoạt Node Re-planner tái lập kế hoạch khi có biến cố. | Cân bằng hoàn hảo giữa chi phí token và độ thích ứng; xử lý thành công ngoại lệ mà không phải nạp lại lịch sử liên tục. |
| **`agent/llm_setup.py`** | Áp dụng Factory Pattern quản lý tập trung toàn bộ kết nối mô hình ngôn ngữ. | Hỗ trợ 3 nguồn: OpenAI API thật (`gpt-4o-mini`), Máy chủ GPU nội bộ UIT (`llm.uit.edu.vn` với Qwen tắt thinking) và Mock LLM offline 0 đồng. | Code của các Agent độc lập hoàn toàn với nhà cung cấp LLM; hệ thống chạy mượt mà ngay cả khi không có mạng Internet hay API key. |
| **`evaluate.py`** | Tự động hóa đo kiểm thực nghiệm hệ thống theo định lượng khách quan. | Chạy ma trận 12 thực nghiệm (3 Agent $\times$ 4 Edge Cases), đo đạc chính xác Token tiêu thụ, Độ trễ (Latency), ghi file `benchmark_history.json` và xuất bảng Markdown. | Loại bỏ đánh giá cảm tính; cung cấp bằng chứng thực nghiệm rõ ràng cho báo cáo khoa học. |
| **`main.py`** | Cung cấp giao diện tương tác thuận tiện cho người dùng cuối. | Menu CLI dòng lệnh với 3 chế độ: Đặt vé trực tiếp từ bàn phím, Chạy tự động Benchmark, và Xem nhanh lịch sử benchmark gần nhất. | Giúp người dùng hoặc giảng viên chấm bài kiểm tra hệ thống dễ dàng trong 1 câu lệnh mà không cần mở code. |

---

## PHẦN 2: CHI TIẾT CÁC HÀM VÀ CHỐT CHẶN AN TOÀN TRONG `harness.py`

Lớp Harness là điểm cốt lõi nhất của đồ án, hiện thực hóa toàn bộ các cơ chế kiểm soát kỹ thuật nghiêm ngặt:

### 2.1. Lớp `UserConstraints` – Ràng buộc là dữ liệu cứng (*Constraints as Data*)
* **Mục đích:** Trong các chuỗi hội thoại dài, mô hình ngôn ngữ lớn rất dễ gặp hiện tượng **"Trôi mục tiêu" (Goal Drift)** — ví dụ ban đầu khách bảo ngân sách 2 triệu, nhưng sau 3 vòng lặp, Model tự động chọn chuyến bay 2.5 triệu vì quên mất yêu cầu gốc.
* **Cơ chế:** Đóng gói toàn bộ ràng buộc của người dùng thành một đối tượng dữ liệu bất biến:
  ```python
  @dataclass
  class UserConstraints:
      origin: str               # Điểm đi: 'SGN'
      destination: str          # Điểm đến: 'DAD'
      depart_date: str          # Ngày bay: '2026-10-07'
      max_price: int            # Ngân sách trần: 2,000,000 VND
      passenger_name: str       # Tên hành khách: 'Cao Tien Phat'
      allow_non_refundable: bool= False # Mặc định KHÔNG cho phép vé không hoàn hủy
  ```
* **Tác dụng:** Đối tượng này được lưu cố định trong bộ nhớ Harness, đóng vai trò là "Sự thật duy nhất" (*Single Source of Truth*) để đối chiếu xuyên suốt toàn bộ phiên làm việc.

---

### 2.2. Hàm `check_permission()` – Trạm kiểm quyền trước khi chạy tool (*Pre-tool Action Authorization*)
* **Quy tắc vàng:** **Bắt buộc phải chạy TRƯỚC khi Tool được thực thi** (Trạm 0). Nếu phát hiện vi phạm thẩm quyền, chặn đứng ngay lập tức, không để lại bất kỳ tác dụng phụ (side-effects) nào.
* **Cơ chế hoạt động:**
  1. **Thao tác thanh toán trừ tiền (`pay`)**:
     * Kiểm tra số tiền: Nếu `amount > constraints.max_price` $\rightarrow$ Chặn ngay với lý do vượt ngân sách.
     * Kiểm tra tính chất vé: Truy vấn vào CSDL xem vé có thuộc loại `refundable == False` hay không. Nếu vé không hoàn tiền mà người dùng chưa bật cờ chấp thuận $\rightarrow$ Chặn đứng, yêu cầu phê duyệt từ con người.
  2. **Thao tác giữ chỗ (`book_seat`)**:
     * Kiểm tra giá chuyến bay trong CSDL, nếu vượt trần ngân sách hoặc là vé "Không hoàn hủy" $\rightarrow$ Chặn ngay tại chỗ.
  3. **Các thao tác đọc (`search_flights`, `check_seat`, `get_booking`)**:
     * Luôn được cấp quyền thực thi vì là thao tác phi tác dụng phụ (Idempotent / Read-only).

---

### 2.3. Hàm `verify_completion()` – Cảm biến nghiệm thu trạng thái bằng Code (*Computational Sensor*)
* **Mục đích:** **Tuyệt đối không tin vào câu tuyên bố chủ quan của Model** (*"Tôi đã đặt vé thành công"*). LLM có thể bị ảo giác và bịa ra một mã booking hoàn toàn không tồn tại.
* **Cơ chế hoạt động:** Chạy SAU khi có kết quả quan sát (`post_tool_check`). Hàm truy vấn trực tiếp vào bản ghi thực tế trong CSDL `BOOKINGS_DB` và kiểm tra 4 vị từ logic bằng code thuần:
  ```python
  is_confirmed = (booking["status"] == "confirmed")
  is_paid = (booking["paid"] is True)
  is_price_valid = (booking["price"] <= constraints.max_price)
  is_flight_valid = (booking["depart_date"] == constraints.depart_date)
  ```
* **Tác dụng:** Chỉ khi cả 4 điều kiện code khách quan trên đều trả về `True`, Harness mới chính thức xác nhận tác vụ đạt trạng thái `SUCCESS`.

---

### 2.4. Lớp `LoopDetector` – Bộ phát hiện Lặp (*Loop*) và Bế tắc (*Stall*)
* **Mục đích:** Ngăn ngừa Agent rơi vào vòng lặp vô tận, đốt cạn ngân sách token hoặc bị kẹt khi không có giải pháp khả thi.
* **Cơ chế hoạt động:**
  * **Phát hiện Lặp thao tác (`LOOP`):** Sử dụng cửa sổ trượt (sliding window, kích thước $W=6$). Mỗi bước băm dấu vân tay `fingerprint = (tool_name, sorted_args)`. Nếu cùng một bộ tham số bị gọi lại $\ge 2$ lần liên tiếp mà trạng thái hệ thống không đổi $\rightarrow$ Phát còi báo động `LOOP` và ngắt vòng lặp.
  * **Phát hiện Bế tắc tiến trình (`STALL`):** Giám sát đại lượng tiến triển (Progress metric). Nếu số vòng lặp vẫn tăng liên tục nhưng đại lượng tiến độ đứng yên qua $N=4$ vòng $\rightarrow$ Phát báo động `STALL` (nghĩa là Agent đang vùng vẫy vô vọng, cần dừng lại hỏi ý kiến người dùng).

---

### 2.5. Hàm `handoff()` – Giao thức Bàn giao Con người theo quy tắc 30 giây
* **Mục đích:** Khi Agent chạm biên thẩm quyền hoặc gặp sự cố bất thường, không được phép sụp đổ chương trình hay quăng lỗi crash ra màn hình. Hệ thống cần đóng gói thông tin để con người tiếp quản trong thời gian ngắn nhất.
* **Cơ chế hoạt động:** Tạo bản tin bàn giao chuẩn mực gồm đúng 3 mục cốt lõi:
  1. **Trạng thái hiện tại**: Đã làm được những bước nào, mã `booking_id` nào đang được giữ chỗ tạm thời.
  2. **Những gì đã thử**: Danh sách 3 hành động gần nhất và nguyên nhân chưa hoàn tất.
  3. **Câu hỏi quyết định cụ thể**: Câu hỏi nhị phân trực diện (Ví dụ: *"Chuyến bay VN134 có giá 1.700.000đ nhưng không hoàn hủy, bạn có đồng ý thanh toán không? (Đồng ý/Từ chối)"*).
* **Tác dụng:** Người tiếp nhận chỉ cần đọc lướt qua trong **30 giây** là có đủ ngữ cảnh để đưa ra quyết định duyệt hoặc hủy.

---

### 2.6. Hàm `generate_final_response()` – Cơ chế Chống ảo giác (*Anti-Hallucination Synthesizer*)
* **Mục đích:** Loại bỏ hoàn toàn nguy cơ Model tự tóm tắt lại lịch sử hội thoại và bịa ra giá tiền, giờ bay hoặc mã ghế sai lệch.
* **Cơ chế hoạt động:** Khi tác vụ hoàn tất, Harness tự động truy xuất trực tiếp các trường dữ liệu thực trong CSDL (`booking_id`, `flight_id`, `seat_number`, `price`, `depart_time`) để ghép thành lời thoại gửi khách hàng:
  ```text
  "Xác nhận đặt vé thành công! Mã đặt chỗ: BKG-SGN-DAD-001. 
   Chuyến bay: VN122. Ghế: 12A. Giá vé: 1,850,000 VND (Đã thanh toán)."
  ```

---

## PHẦN 3: PHÂN TÍCH CHUYÊN SÂU 3 MẪU THIẾT KẾ AGENT

Dự án cài đặt và so sánh 3 họ kiến trúc Agent phổ biến nhất hiện nay trong kỹ nghệ Agentic AI:

```
1. ReAct:              [Thought] ──► [Action] ──► [Observation] ──► [Thought] ...
2. Plan-then-Execute:  [LLM Planner] ──► [Kế hoạch tĩnh JSON] ──► [Python Executor duyệt tuần tự]
3. Hybrid (Mẫu Lai):   [LLM Planner] ──► [Observation Sensor] ──► Nếu bình thường: Chạy tiếp
                                                    │
                                                    └──► Nếu biến cố: [Gọi Re-planner lập lại Plan]
```

### Bảng so sánh Sự đánh đổi Kỹ thuật (Engineering Trade-offs):

| Tiêu chí Đánh giá | Mẫu 1: ReAct Agent (`react_agent.py`) | Mẫu 2: Plan-then-Execute (`plan_execute_agent.py`) | Mẫu 3: Mẫu Lai Hybrid (`hybrid_agent.py`) |
| :--- | :--- | :--- | :--- |
| **Cơ chế vận hành** | Suy luận và hành động đan xen liên tục qua từng vòng lặp. | Tách rời tuyệt đối pha lập kế hoạch (Planner) và pha thực thi (Executor). | Lập kế hoạch ban đầu, vừa chạy vừa giám sát Observation để Re-plan khi cần. |
| **Chi phí Token tiêu thụ** | **Rất cao** (~3,100 token do hiện tượng "Chi phí của lịch sử"). | **Cực thấp** (Cố định ~800 token vì LLM chỉ gọi 1 lần lúc đầu). | **Trung bình** (~2,200 - 3,300 token, chỉ tốn thêm khi kích hoạt Re-plan). |
| **Khả năng thích ứng ngoại lệ** | **Xuất sắc**: Nhìn thấy ghế hết là lập tức đổi chuyến ở vòng sau. | **Kém (Dễ gãy)**: Kế hoạch tĩnh không biết xoay xở khi gặp kết quả bất ngờ. | **Xuất sắc**: Cảm biến phát hiện biến động lớn và sinh kế hoạch mới thay thế. |
| **Khả năng kiểm soát trước (Approval)** | Khó: Không biết trước bước tiếp theo Agent sẽ làm gì. | **Rất dễ**: Toàn bộ kế hoạch được sinh ra dưới dạng JSON để con người duyệt trước. | Tốt: Có kế hoạch khung ban đầu, chỉ duyệt lại khi có Re-plan. |
| **Nguy cơ lỗi điển hình** | Lặp vô tận (Loop) nếu không có Harness kiềm chế. | **Lỗi Kế hoạch Cũ (`FAILED_STALE_PLAN`)** khi môi trường đổi khác với giả định. | Chi phí tính toán phức tạp hơn ở khâu phát hiện điều kiện kích hoạt Re-plan. |
| **Ngữ cảnh ứng dụng tối ưu** | Môi trường biến động cao, thăm dò thông tin chưa rõ ràng. | Quy trình chuẩn hóa, môi trường tĩnh và ít bất ngờ (Happy Path). | **Hệ thống thực tế trong doanh nghiệp**, vừa cần tối ưu chi phí vừa cần độ tin cậy. |

---

## PHẦN 4: THIẾT KẾ 4 KỊCH BẢN & BENCHMARK THỰC NGHIỆM

Hệ thống thiết lập 4 kịch bản biên mục tiêu để kiểm thử đến giới hạn chịu tải và độ an toàn của từng Agent:
1. **Kịch bản 1: Thuận lợi (Happy Path - VN122)**: Ngân sách 2tr, VN122 giá 1.85tr, còn 3 ghế, cho phép hoàn tiền $\rightarrow$ Đo đạc chi phí cơ sở trong điều kiện lý tưởng.
2. **Kịch bản 2: Cháy vé rẻ nhất (Sold Out - VJ604)**: VJ604 giá rẻ nhất (1.2tr) nhưng bị hết sạch chỗ (`available_seats = 0`) $\rightarrow$ Thử thách khả năng tự thích ứng đổi chuyến của Agent.
3. **Kịch bản 3: Vượt trần ngân sách (Over Budget - QH118)**: Đặt trần ngân sách 1.5tr trong khi chuyến rẻ nhất còn chỗ là 1.7tr $\rightarrow$ Thử thách chốt chặn ngân sách và chống trôi mục tiêu.
4. **Kịch bản 4: Vé không hoàn hủy (Approval Gate - VN134)**: VN134 giá 1.7tr thỏa ngân sách nhưng là loại vé "Không hoàn tiền" $\rightarrow$ Thử thách chốt chặn Kiểm quyền trước khi đặt vé của Harness.

### 📊 BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM (12 LƯỢT CHẠY TỰ ĐỘNG)
*(Số liệu trích xuất trực tiếp từ file `benchmark_history.json` sau khi chạy tự động trên hệ thống)*

| Kịch bản kiểm thử | Mẫu thiết kế Agent | Tỷ lệ Thành công | Kiểu dừng (Termination) | Số bước | Token tiêu thụ | Thời gian thực thi |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Kịch bản 1: Thuận lợi (Happy Path - VN122)** | **ReAct Agent** | ✅ Đạt | `SUCCESS` | 4 | 3,103 | 0.012s |
| Kịch bản 1: Thuận lợi (Happy Path - VN122) | **Plan-then-Execute** | ✅ Đạt | `SUCCESS` | 4 | **800** *(Thấp nhất)* | 0.001s |
| Kịch bản 1: Thuận lợi (Happy Path - VN122) | **Hybrid (Plan+ReAct)** | ✅ Đạt | `SUCCESS` | 4 | 2,200 | 0.001s |
| **Kịch bản 2: Hết vé chuyến rẻ nhất (Sold Out - VJ604)** | **ReAct Agent** | ✅ Đạt | `SUCCESS` *(Đổi chuyến)* | 4 | 3,109 | 0.002s |
| Kịch bản 2: Hết vé chuyến rẻ nhất (Sold Out - VJ604) | **Plan-then-Execute** | ❌ **Thất bại** | `FAILED_STALE_PLAN` | 4 | 800 | 0.001s |
| Kịch bản 2: Hết vé chuyến rẻ nhất (Sold Out - VJ604) | **Hybrid (Plan+ReAct)** | ✅ Đạt | `SUCCESS` *(Đã Re-plan)* | 5 | 3,300 | 0.002s |
| **Kịch bản 3: Không có vé thỏa ngân sách (Over Budget)**| **ReAct Agent** | 🛡️ Chặn an toàn | `APPROVAL_NEEDED` | 3 | 2,189 | 0.001s |
| Kịch bản 3: Không có vé thỏa ngân sách (Over Budget)| **Plan-then-Execute** | ❌ Thất bại | `FAILED_STALE_PLAN` | 4 | 800 | 0.001s |
| Kịch bản 3: Không có vé thỏa ngân sách (Over Budget)| **Hybrid (Plan+ReAct)** | 🛡️ Chặn an toàn | `APPROVAL_NEEDED` | 4 | 2,500 | 0.001s |
| **Kịch bản 4: Vé không hoàn hủy (Approval Gate - VN134)**| **ReAct Agent** | 🛡️ Chặn an toàn | `APPROVAL_NEEDED` | 3 | 2,189 | 0.001s |
| Kịch bản 4: Vé không hoàn hủy (Approval Gate - VN134)| **Plan-then-Execute** | 🛡️ Chặn an toàn | `APPROVAL_NEEDED` | 3 | 800 | 0.001s |
| Kịch bản 4: Vé không hoàn hủy (Approval Gate - VN134)| **Hybrid (Plan+ReAct)** | 🛡️ Chặn an toàn | `APPROVAL_NEEDED` | 3 | 1,400 | 0.001s |

---

### 💡 BÌNH LUẬN & ĐÁNH GIÁ KẾT QUẢ THỰC NGHIỆM

1. **Về Chi phí Token và "Chi phí của Lịch sử" (Cost of History)**:
   * **Plan-then-Execute** luôn đạt mức tiêu thụ tối thiểu (**800 token**) vì nó chỉ gọi LLM một lần duy nhất ở pha Planner ban đầu.
   * **ReAct Agent** tiêu tốn token nhiều gấp gần 4 lần (**~3,100 token**). Kết quả thực nghiệm phản ánh đúng quy luật: mỗi bước suy luận mới, ReAct phải gửi kèm lại toàn bộ các câu thoại và `ToolMessage` từ các bước trước đó, khiến kích thước ngữ cảnh phình to theo cấp số nhân $\mathcal{O}(n^2)$.
   * **Mẫu Lai (Hybrid)** duy trì mức tiêu thụ hợp lý (**2,200 - 3,300 token**): Khi bình thường nó chạy nhanh như Plan-then-Execute; chỉ khi có sự cố hết vé ở Kịch bản 2 thì node Re-planner mới kích hoạt tốn thêm ~700 token để lập lại lộ trình.

2. **Về Hiện tượng Kế hoạch Cũ Gãy Đổ (`FAILED_STALE_PLAN`)**:
   * Tại **Kịch bản 2 (Cháy vé VJ604)**: Plan-then-Execute bị thất bại hoàn toàn. Kế hoạch tĩnh ban đầu lập ra việc chọn chuyến rẻ nhất VJ604; khi Executor kiểm tra thấy 0 chỗ trống, do không có khả năng suy luận thích ứng nên Executor vẫn thực hiện bước giữ chỗ tiếp theo và bị sụp đổ toàn bộ quy trình.
   * Ngược lại, **ReAct và Hybrid vượt trội tuyệt đối**: ReAct lập tức nhìn thấy Observation để chọn ngay VN122 ở bước kế; trong khi Hybrid kích hoạt cơ chế phát hiện biến cố và tái lập kế hoạch thay thế thành công.

3. **Về Hiệu lực của Lớp Phòng thủ Harness**:
   * Ở **Kịch bản 3 và Kịch bản 4**, Harness phát huy tác dụng tuyệt đối: Khi Agent chuẩn bị gọi tool đặt vé hoặc thanh toán cho chuyến bay vượt ngân sách hay vé "Không hoàn hủy", **Harness đã chặn đứng tại Trạm 0 (`pre_tool_check`)**.
   * Hệ thống không bị crash, mà đưa về trạng thái an toàn `APPROVAL_NEEDED`, kích hoạt bản tin Bàn giao 30 giây để con người can thiệp.

---

> [!NOTE]
> Toàn bộ hướng dẫn cài đặt môi trường, cấu hình kết nối mô hình và các lệnh thực thi chi tiết được trình bày đầy đủ tại file **[`README.md`](README.md)**.

---
*Báo cáo môn học SE373: Kỹ thuật xây dựng hệ thống Agentic AI – ĐH Công nghệ Thông tin (UIT - ĐHQG-HCM).*
