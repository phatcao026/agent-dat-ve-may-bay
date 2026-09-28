"""
harness.py - Khung giàn Harness kiểm soát an toàn và điều kiện dừng cho Agent Đặt Vé Máy Bay.
Môn học: Kỹ thuật xây dựng hệ thống Agentic AI (SE373) - Buổi 03: Agent Fundamentals.

Bao gồm đủ 4 chốt chặn kỹ thuật:
1. Ràng buộc là dữ liệu (UserConstraints).
2. Kiểm quyền (check_permission) chạy TRƯỚC khi gọi tool.
3. Tiêu chí hoàn thành kiểm bằng code (verify_completion) chạy SAU khi có observation.
4. Bàn giao cho con người (handoff) tuân thủ quy tắc 30 giây.
Kèm bộ phát hiện lặp và bế tắc (LoopDetector) & Ngân sách cứng.
"""

import sys
import json
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

# Cấu hình encoding stdout để chạy êm trên terminal Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from mock_env import FLIGHTS_DB, BOOKINGS_DB, reset_db, search_flights, book_seat, pay, check_seat


# ==============================================================================
# CHỐT CHẶN 1: RÀNG BUỘC LÀ DỮ LIỆU (CONSTRAINTS AS DATA)
# ==============================================================================
@dataclass
class UserConstraints:
    """
    Ràng buộc và yêu cầu của người dùng được chuẩn hóa thành dữ liệu có cấu trúc.
    Khắc phục lỗi 'Quên yêu cầu ban đầu' (Goal Drift) khi chuỗi hội thoại dài ra.
    """
    origin: str                         # Điểm đi, VD: 'SGN'
    destination: str                    # Điểm đến, VD: 'DAD'
    depart_date: str                    # Ngày bay 'YYYY-MM-DD', VD: '2026-10-07'
    max_price: int                      # Trần ngân sách cho 1 vé (VND), VD: 2_000_000
    passenger_name: str                 # Tên hành khách, VD: 'Cao Tien Phat'
    preferred_time: Optional[str] = None # 'morning', 'afternoon', 'any'
    allow_non_refundable: bool = False  # Mặc định KHÔNG cho phép tự ý mua vé không hoàn hủy


# ==============================================================================
# CHỐT CHẶN 2: BỘ PHÁT HIỆN LẶP VÀ BẾ TẮC (LOOP & STALL DETECTOR)
# ==============================================================================
class LoopDetector:
    """
    Cài đặt thuật toán theo dõi lặp và bế tắc:
    - So sánh (tool, args) trong cửa sổ trượt (window).
    - So sánh đại lượng tiến triển (progress) qua các vòng.
    """
    def __init__(self, window: int = 6, repeat_k: int = 2, stall_n: int = 4):
        self.recent = deque(maxlen=window)  # Chỉ so cửa sổ gần
        self.k = repeat_k                    # Số lần lặp cùng (tool, args) để báo động
        self.n = stall_n                    # Số vòng progress không nhúc nhích để báo STALL
        self.last_progress = None
        self.stall_count = 0

    def check(self, tool_name: str, args: Dict[str, Any], progress: Any) -> Optional[str]:
        """
        Kiểm tra dấu vết hành động xem có bị rơi vào vòng lặp hay bế tắc không.
        
        Args:
            tool_name: Tên tool sắp gọi
            args: Dict tham số truyền vào tool
            progress: Đại lượng tiến triển của bài toán (ví dụ: số chuyến bay khả dĩ đã lọc)
            
        Returns:
            'LOOP' nếu lặp lại cùng action, 'STALL' nếu bế tắc, hoặc None nếu bình thường.
        """
        # Tạo dấu vân tay (fingerprint) chuẩn hóa từ tool và args đã sắp xếp
        fp = (tool_name, repr(sorted(args.items())))

        # 1. Kiểm tra lặp hành động (Trùng action)
        if self.recent.count(fp) + 1 >= self.k:
            return "LOOP"
        self.recent.append(fp)

        # 2. Kiểm tra bế tắc (Đại lượng tiến triển không đổi)
        if progress is not None and progress == self.last_progress:
            self.stall_count += 1
        else:
            self.stall_count = 0

        self.last_progress = progress

        if self.stall_count >= self.n:
            return "STALL"

        return None


# ==============================================================================
# CHỐT CHẶN 3: KIỂM QUYỀN (PERMISSION / HUMAN-IN-THE-LOOP)
# Quy tắc vàng: Chạy TRƯỚC khi thực thi tool!
# ==============================================================================
def check_permission(
    tool_name: str,
    args: Dict[str, Any],
    constraints: UserConstraints
) -> Tuple[bool, Optional[str]]:
    """
    Kiểm tra thẩm quyền của Agent trước khi cho phép gọi Tool.
    Chặn các hành động có tác dụng phụ (Side-effects) hoặc tiềm ẩn rủi ro tài chính.
    
    Returns:
        (is_allowed: bool, reason_if_denied: str)
    """
    # 1. Kiểm tra thao tác thanh toán trừ tiền (pay)
    if tool_name == "pay":
        booking_id = args.get("booking_id")
        amount = args.get("amount", 0)

        # Kiểm tra vượt ngân sách cho phép
        if amount > constraints.max_price:
            return False, f"Hành động 'pay' bị chặn: Số tiền {amount:,}đ vượt ngân sách tối đa ({constraints.max_price:,}đ)."

        # Kiểm tra tính chất vé của booking trong CSDL
        booking = BOOKINGS_DB.get(booking_id)
        if booking and not booking.get("refundable", True) and not constraints.allow_non_refundable:
            return False, f"Hành động 'pay' bị chặn: Vé {booking.get('flight_id')} là loại KHÔNG HOÀN HỦY. Cần người dùng phê duyệt trước khi trừ tiền."

        # Mọi thao tác thanh toán đều thuộc nhóm nhạy cảm
        # Trong chế độ tự động, có thể cho phép nếu hoàn toàn thỏa ngân sách & hoàn tiền được:
        return True, None

    # 2. Kiểm tra thao tác giữ chỗ (book_seat) với chuyến bay không hoàn tiền
    if tool_name == "book_seat":
        flight_id = args.get("flight_id", "").strip().upper()
        flight = FLIGHTS_DB.get(flight_id)
        if flight:
            # Nếu vé vượt trần ngân sách
            if flight["price"] > constraints.max_price:
                return False, f"Hành động 'book_seat' bị chặn: Chuyến bay {flight_id} có giá {flight['price']:,}đ vượt trần {constraints.max_price:,}đ."
            # Nếu vé không hoàn hủy mà người dùng chưa bật cờ cho phép
            if not flight["refundable"] and not constraints.allow_non_refundable:
                return False, f"Hành động 'book_seat' bị chặn: Chuyến {flight_id} có chính sách 'Không hoàn hủy'. Cần xin ý kiến người dùng."

    # Các tool đọc dữ liệu (search_flights, check_seat, get_booking) luôn được phép
    return True, None


# ==============================================================================
# CHỐT CHẶN 4: TIÊU CHÍ HOÀN THÀNH KIỂM BẰNG CODE (COMPUTATIONAL SENSOR)
# Quy tắc vàng: Chạy SAU khi có observation, không nghe lời tuyên bố của LLM!
# ==============================================================================
def verify_completion(
    booking_id: Optional[str],
    constraints: UserConstraints
) -> Tuple[bool, str]:
    """
    Tiêu chí hoàn thành là quy tắc lập trình khách quan được kiểm chứng trực tiếp
    vào CSDL thực tế, hoàn toàn độc lập với phán đoán chủ quan của model.
    
    Returns:
        (is_completed: bool, reason: str)
    """
    if not booking_id:
        return False, "Chưa có mã đặt chỗ (booking_id) nào được ghi nhận."

    booking = BOOKINGS_DB.get(booking_id.strip())
    if not booking:
        return False, f"Mã đặt chỗ '{booking_id}' không tồn tại trong CSDL thực tế."

    # Kiểm tra các vị từ logic cứng bằng code:
    is_confirmed = (booking.get("status") == "confirmed")
    is_paid = (booking.get("paid") is True)
    is_price_valid = (booking.get("price", 0) <= constraints.max_price)
    is_date_valid = (booking.get("depart_date") == constraints.depart_date)
    is_passenger_valid = (booking.get("passenger_name", "").strip().lower() == constraints.passenger_name.strip().lower())

    if not is_confirmed:
        return False, f"Trạng thái vé chưa hoàn tất: status='{booking.get('status')}', cần 'confirmed'."
    if not is_paid:
        return False, "Vé chưa được thanh toán thành công (paid=False)."
    if not is_price_valid:
        return False, f"Giá vé thực tế ({booking.get('price'):,}đ) vượt ngân sách ({constraints.max_price:,}đ)."
    if not is_date_valid:
        return False, f"Ngày bay ({booking.get('depart_date')}) không khớp yêu cầu ({constraints.depart_date})."
    if not is_passenger_valid:
        return False, f"Tên hành khách ({booking.get('passenger_name')}) không khớp ({constraints.passenger_name})."

    return True, f"XÁC MINH THÀNH CÔNG: Vé {booking_id} (Chuyến {booking.get('flight_id')}) đã thanh toán và đúng mọi ràng buộc!"


# ==============================================================================
# CHỐT CHẶN 5: BÀN GIAO CHO CON NGƯỜI (HANDOFF PROTOCOL)
# Tiêu chuẩn: Người nhận đọc và ra quyết định được trong vòng 30 giây!
# ==============================================================================
def handoff(
    reason: str,
    current_state: Dict[str, Any],
    tried_actions: List[str],
    specific_question: str
) -> Dict[str, Any]:
    """
    Tạo bản tin bàn giao chuẩn mực 3 thành phần khi dừng bất thường hoặc chờ duyệt:
    1. Trạng thái hiện tại (Đã làm tới đâu, tác dụng phụ nào đã phát sinh).
    2. Những gì đã thử (Hướng nào đã thử và vì sao hỏng).
    3. Câu hỏi cụ thể (Người nhận trả lời được trong 30 giây).
    """
    report = {
        "handoff_required": True,
        "reason": reason,
        "1_current_state": current_state,
        "2_what_was_tried": tried_actions,
        "3_specific_question": specific_question
    }

    # Format hiển thị ra màn hình chuẩn trực quan
    print("\n" + "="*60)
    print("🔔 [HARNESS HANDOFF] BÀN GIAO CHO CON NGƯỜI (RULE 30 GIÂY)")
    print("="*60)
    print(f"📌 LÝ DO DỪNG        : {reason}")
    print(f"📍 TRẠNG THÁI HIỆN TẠI : {json.dumps(current_state, ensure_ascii=False)}")
    print("🔍 NHỮNG GÌ ĐÃ THỬ    :")
    for idx, act in enumerate(tried_actions, 1):
        print(f"   {idx}. {act}")
    print(f"❓ CÂU HỎI QUYẾT ĐỊNH : 👉 {specific_question} 👈")
    print("="*60 + "\n")

    return report


# ==============================================================================
# LỚP ĐIỀU PHỐI VÒNG LẶP HARNESS (AGENT HARNESS COORDINATOR)
# ==============================================================================
class FlightBookingHarness:
    """
    Bộ điều phối toàn diện cho vòng lặp Agent.
    Quản lý ngân sách, kiểm quyền, bắt lỗi lặp và kiểm chứng hoàn thành.
    """
    def __init__(self, constraints: UserConstraints, max_budget_steps: int = 8):
        self.constraints = constraints
        self.max_steps = max_budget_steps
        self.current_step = 0
        self.loop_detector = LoopDetector(window=6, repeat_k=2, stall_n=4)
        
        # Nhật ký vết thực thi
        self.action_history: List[str] = []
        self.latest_booking_id: Optional[str] = None
        self.is_terminated = False
        self.termination_type: Optional[str] = None  # SUCCESS, APPROVAL_NEEDED, LOOP, STALL, BUDGET_EXCEEDED
        self.handoff_data: Optional[Dict[str, Any]] = None

    def pre_tool_check(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """
        Bước 0: Chạy TRƯỚC khi thực thi tool.
        Kiểm tra quyền hạn. Nếu vi phạm -> Bàn giao ngay.
        """
        self.current_step += 1
        allowed, reason = check_permission(tool_name, args, self.constraints)

        if not allowed:
            self.is_terminated = True
            self.termination_type = "APPROVAL_NEEDED"
            self.handoff_data = handoff(
                reason=reason or "Hành động vượt thẩm quyền",
                current_state={
                    "step": self.current_step,
                    "attempted_tool": tool_name,
                    "args": args,
                    "latest_booking_id": self.latest_booking_id
                },
                tried_actions=self.action_history[-3:],
                specific_question=f"Bạn có đồng ý phê duyệt thực thi hành động '{tool_name}' với tham số {args} không? (Đồng ý/Từ chối)"
            )
            return False

        return True

    def post_tool_check(
        self,
        tool_name: str,
        args: Dict[str, Any],
        observation: str,
        progress_metric: Any = None
    ) -> bool:
        """
        Các bước hậu kiểm: Chạy SAU khi có observation.
        Quy tắc thứ tự:
        1. Tiêu chí hoàn thành (Đạt mục tiêu) -> Dừng bình thường.
        2. Phát hiện lặp (LOOP) -> Dừng bất thường.
        3. Phát hiện bế tắc (STALL) -> Dừng bất thường.
        4. Hết ngân sách số vòng (BUDGET) -> Dừng bất thường (kiểm tra cuối cùng!).
        
        Returns:
            continue_running: True nếu cần chạy tiếp, False nếu vòng lặp phải dừng lại.
        """
        # Ghi nhận lịch sử
        log_entry = f"Bước {self.current_step}: Gọi {tool_name}({args})"
        self.action_history.append(log_entry)

        # Cập nhật booking_id nếu có
        try:
            obs_json = json.loads(observation)
            if "booking_id" in obs_json:
                self.latest_booking_id = obs_json["booking_id"]
        except Exception:
            pass

        # 1. KIỂM TRA TIÊU CHÍ HOÀN THÀNH (ĐẠT MỤC TIÊU)
        completed, verify_msg = verify_completion(self.latest_booking_id, self.constraints)
        if completed:
            self.is_terminated = True
            self.termination_type = "SUCCESS"
            print(f"✅ [HARNESS] {verify_msg}")
            return False  # Dừng thành công!

        # 2. KIỂM TRA PHÁT HIỆN LẶP
        loop_status = self.loop_detector.check(tool_name, args, progress_metric)
        if loop_status == "LOOP":
            self.is_terminated = True
            self.termination_type = "LOOP"
            self.handoff_data = handoff(
                reason=f"Phát hiện vòng lặp vô ích: Công cụ '{tool_name}' bị gọi lại với cùng tham số.",
                current_state={"step": self.current_step, "booking_id": self.latest_booking_id},
                tried_actions=self.action_history[-4:],
                specific_question=f"Agent đang lặp lại lệnh '{tool_name}'. Bạn có muốn hủy tác vụ hay cung cấp thông tin mới?"
            )
            return False

        # 3. KIỂM TRA BẾ TẮC
        if loop_status == "STALL":
            self.is_terminated = True
            self.termination_type = "STALL"
            self.handoff_data = handoff(
                reason="Agent rơi vào bế tắc: Đã thử nhiều tool khác nhau nhưng tiến độ không nhúc nhích qua nhiều vòng.",
                current_state={"step": self.current_step, "booking_id": self.latest_booking_id},
                tried_actions=self.action_history[-4:],
                specific_question="Không tìm thấy chuyến bay nào thỏa điều kiện. Bạn có muốn nâng mức ngân sách hoặc đổi ngày bay không?"
            )
            return False

        # 4. KIỂM TRA NGÂN SÁCH CỨNG - Luôn kiểm tra cuối cùng!
        if self.current_step >= self.max_steps:
            self.is_terminated = True
            self.termination_type = "BUDGET_EXCEEDED"
            self.handoff_data = handoff(
                reason=f"Hết ngân sách vòng lặp ({self.max_steps} bước tối đa) mà chưa đạt mục tiêu.",
                current_state={"step": self.current_step, "booking_id": self.latest_booking_id},
                tried_actions=self.action_history[-4:],
                specific_question=f"Agent đã chạm giới hạn {self.max_steps} bước. Bạn có muốn cấp thêm 5 bước nữa không?"
            )
            return False

        return True  # Tiếp tục vòng lặp bình thường

    def generate_final_response(self) -> str:
        """
        Sinh câu trả lời tổng kết tự nhiên (Conversational Final Response) 
        gửi lại cho khách hàng sau khi vòng lặp kết thúc.
        """
        if self.termination_type == "SUCCESS" and self.latest_booking_id:
            booking = BOOKINGS_DB.get(self.latest_booking_id, {})
            return (
                f"Chào bạn {self.constraints.passenger_name}!\n"
                f"Tôi đã đặt và thanh toán vé máy bay thành công cho bạn theo đúng yêu cầu.\n"
                f"📋 THÔNG TIN VÉ CHI TIẾT:\n"
                f"  - Chặng bay       : {self.constraints.origin} ✈️ {self.constraints.destination}\n"
                f"  - Mã chuyến bay   : {booking.get('flight_id')} (Khởi hành: {booking.get('depart_time')})\n"
                f"  - Ngày bay        : {booking.get('depart_date')}\n"
                f"  - Số ghế          : {booking.get('seat_number')}\n"
                f"  - Mã đặt chỗ (PNR): {booking.get('booking_id')}\n"
                f"  - Tổng tiền đã trả: {booking.get('price', 0):,} VND\n"
                f"  - Trạng thái vé   : ĐÃ XÁC NHẬN (CONFIRMED) & ĐÃ THANH TOÁN (PAID)\n"
                f"Chúc bạn có một chuyến bay an toàn và vui vẻ!"
            )

        if self.termination_type == "APPROVAL_NEEDED":
            reason = self.handoff_data.get("reason", "") if self.handoff_data else "Hành động vượt thẩm quyền"
            question = self.handoff_data.get("3_specific_question", "") if self.handoff_data else "Bạn có đồng ý không?"
            return (
                f"Chào bạn {self.constraints.passenger_name}!\n"
                f"Tôi đã tìm kiếm chuyến bay nhưng cần xin ý kiến phê duyệt của bạn trước khi thực hiện:\n"
                f"⚠️ VẤN ĐỀ: {reason}\n"
                f"👉 CÂU HỎI: {question}"
            )

        if self.termination_type == "FAILED_STALE_PLAN":
            return (
                f"Chào bạn {self.constraints.passenger_name}!\n"
                f"Rất tiếc, kế hoạch đặt vé ban đầu không thể hoàn tất do chuyến bay dự kiến đã hết vé hoặc gặp sự cố.\n"
                f"Mẫu thiết kế Plan-then-Execute sử dụng kế hoạch tĩnh nên không thể tự động đổi hướng khi gặp biến cố."
            )

        if self.termination_type in ["STALL", "BUDGET_EXCEEDED"]:
            return (
                f"Chào bạn {self.constraints.passenger_name}!\n"
                f"Tôi đã tìm kiếm nhiều lần nhưng không tìm thấy chuyến bay nào từ {self.constraints.origin} "
                f"đi {self.constraints.destination} ngày {self.constraints.depart_date} "
                f"có giá dưới ngân sách {self.constraints.max_price:,} VND của bạn.\n"
                f"Bạn có muốn tăng ngân sách hoặc dời sang ngày khác không?"
            )

        if self.termination_type == "LOOP":
            return (
                f"Chào bạn {self.constraints.passenger_name}!\n"
                f"Hệ thống phát hiện tác vụ đang bị lặp lại nhiều lần không tiến triển nên đã tạm dừng "
                f"để bảo vệ an toàn cho bạn."
            )

        return "Tác vụ đặt vé đã dừng lại. Hãy kiểm tra lại thông tin yêu cầu của bạn."


# ==============================================================================
# SMOKE TEST KIỂM TRA IMPORT VÀ CẤU TRÚC
# ==============================================================================
if __name__ == "__main__":
    test_c = UserConstraints(
        origin="SGN", destination="DAD", depart_date="2026-10-07",
        max_price=2_000_000, passenger_name="Cao Tien Phat"
    )
    h = FlightBookingHarness(constraints=test_c)
    print("✅ [harness] Module sẵn sàng với 4 chốt chặn kiểm soát vòng lặp.")
