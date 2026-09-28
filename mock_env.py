"""
mock_env.py - Môi trường giả lập (Mock Environment) và các Tool cho Agent Đặt Vé Máy Bay.
Môn học: Kỹ thuật xây dựng hệ thống Agentic AI (SE373) - Buổi 03: Agent Fundamentals.

Chức năng:
1. Quản lý cơ sở dữ liệu giả lập trong bộ nhớ (In-memory Mock Database): Chuyến bay & Đặt chỗ.
2. Cung cấp 5 LangChain Tools chuẩn hóa dữ liệu đầu ra JSON:
   - search_flights: Tìm kiếm chuyến bay theo điểm đi, điểm đến, ngày.
   - check_seat: Kiểm tra giá vé, số ghế trống, chính sách hoàn hủy.
   - book_seat: Giữ chỗ tạm thời (status: held).
   - pay: Thanh toán tiền vé (status: confirmed, paid: True).
   - get_booking: Truy vấn trạng thái booking từ DB (phục vụ Harness kiểm chứng chéo).
"""

import json
import uuid
import copy
import sys
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool

# Cấu hình encoding stdout để chạy êm trên terminal Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ==============================================================================
# 1. DỮ LIỆU GỐC BAN ĐẦU (SEED DATA)
# Gài các tình huống biên (Edge cases) phục vụ kiểm thử Agent & Harness:
# - VN122: Thuận lợi (Happy path) - giá tốt, còn ghế, hoàn hủy được.
# - VJ604: Hết vé (Sold out) - giá rẻ nhất nhưng available_seats = 0.
# - QH118: Quá ngân sách (Over budget) - giá 2.450.000đ > trần 2.000.000đ.
# - VN134: Cần kiểm quyền (Approval needed) - giá tốt nhưng vé không hoàn hủy (refundable = False).
# ==============================================================================
INITIAL_FLIGHTS_DATA: List[Dict[str, Any]] = [
    {
        "flight_id": "VN122",
        "airline": "Vietnam Airlines",
        "origin": "SGN",
        "destination": "DAD",
        "depart_date": "2026-10-07",
        "depart_time": "08:30",
        "price": 1_850_000,
        "available_seats": 3,
        "seats": ["12A", "12B", "12C"],
        "refundable": True,
    },
    {
        "flight_id": "VJ604",
        "airline": "Vietjet Air",
        "origin": "SGN",
        "destination": "DAD",
        "depart_date": "2026-10-07",
        "depart_time": "06:15",
        "price": 1_200_000,
        "available_seats": 2,
        "seats": ["08A", "08B"],
        "refundable": True,
    },
    {
        "flight_id": "QH118",
        "airline": "Bamboo Airways",
        "origin": "SGN",
        "destination": "DAD",
        "depart_date": "2026-10-07",
        "depart_time": "10:00",
        "price": 2_450_000,
        "available_seats": 5,
        "seats": ["05A", "05B", "06A", "06B", "06C"],
        "refundable": True,
    },
    {
        "flight_id": "VN134",
        "airline": "Vietnam Airlines",
        "origin": "SGN",
        "destination": "DAD",
        "depart_date": "2026-10-07",
        "depart_time": "11:30",
        "price": 1_700_000,
        "available_seats": 2,
        "seats": ["14A", "14B"],
        "refundable": False,  # Vé khuyến mãi không hoàn tiền -> Cần Human Approval
    },
    {
        "flight_id": "VJ150",
        "airline": "Vietjet Air",
        "origin": "HAN",
        "destination": "SGN",
        "depart_date": "2026-10-07",
        "depart_time": "14:00",
        "price": 1_650_000,
        "available_seats": 4,
        "seats": ["15A", "15B", "16A", "16B"],
        "refundable": True,
    }
]

# Cơ sở dữ liệu Runtime trong bộ nhớ
FLIGHTS_DB: Dict[str, Dict[str, Any]] = {}
BOOKINGS_DB: Dict[str, Dict[str, Any]] = {}


def reset_db() -> None:
    """Khôi phục trạng thái CSDL giả lập về ban đầu để phục vụ lặp lại các bài test."""
    global FLIGHTS_DB, BOOKINGS_DB
    FLIGHTS_DB.clear()
    BOOKINGS_DB.clear()
    for flight in copy.deepcopy(INITIAL_FLIGHTS_DATA):
        FLIGHTS_DB[flight["flight_id"]] = flight


# Khởi tạo DB lần đầu khi import module
reset_db()


# ==============================================================================
# 2. CÁC LANGCHAIN TOOLS CHUẨN HÓA (STRUCTURED JSON OUTPUT)
# ==============================================================================

@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """
    Tìm kiếm danh sách chuyến bay theo điểm đi, điểm đến và ngày bay.
    
    Args:
        origin (str): Mã sân bay khởi hành (VD: 'SGN', 'HAN').
        destination (str): Mã sân bay đến (VD: 'DAD', 'SGN').
        date (str): Ngày bay theo định dạng 'YYYY-MM-DD' (VD: '2026-10-07').
    
    Returns:
        str: Chuỗi JSON danh sách chuyến bay phù hợp và số lượng tìm thấy.
    """
    matches = []
    for flight in FLIGHTS_DB.values():
        if (
            flight["origin"].upper() == origin.strip().upper()
            and flight["destination"].upper() == destination.strip().upper()
            and flight["depart_date"] == date.strip()
        ):
            matches.append({
                "flight_id": flight["flight_id"],
                "airline": flight["airline"],
                "depart_time": flight["depart_time"],
                "price": flight["price"],
                "available_seats": flight["available_seats"],
                "refundable": flight["refundable"]
            })

    if not matches:
        return json.dumps({
            "status": "not_found",
            "message": f"Không có chuyến bay nào từ {origin} đến {destination} vào ngày {date}.",
            "count": 0,
            "flights": []
        }, ensure_ascii=False)

    return json.dumps({
        "status": "success",
        "count": len(matches),
        "flights": matches
    }, ensure_ascii=False)


@tool
def check_seat(flight_id: str) -> str:
    """
    Kiểm tra chi tiết chuyến bay, bao gồm danh sách ghế trống cụ thể, giá vé và điều kiện hoàn vé.
    
    Args:
        flight_id (str): Mã chuyến bay (VD: 'VN122', 'VJ604').
        
    Returns:
        str: Chuỗi JSON chứa thông tin ghế và điều kiện vé.
    """
    flight_id_clean = flight_id.strip().upper()
    flight = FLIGHTS_DB.get(flight_id_clean)

    if not flight:
        return json.dumps({
            "status": "error",
            "error_code": "FLIGHT_NOT_FOUND",
            "message": f"Mã chuyến bay '{flight_id}' không tồn tại trong hệ thống."
        }, ensure_ascii=False)

    return json.dumps({
        "status": "success",
        "flight_id": flight["flight_id"],
        "airline": flight["airline"],
        "origin": flight["origin"],
        "destination": flight["destination"],
        "depart_date": flight["depart_date"],
        "depart_time": flight["depart_time"],
        "price": flight["price"],
        "available_seats_count": flight["available_seats"],
        "available_seats": flight["seats"],
        "refundable": flight["refundable"]
    }, ensure_ascii=False)


@tool
def book_seat(flight_id: str, seat_number: str, passenger_name: str) -> str:
    """
    Giữ chỗ tạm thời cho một hành khách trên chuyến bay xác định.
    Lưu ý: Hành động này chỉ giữ chỗ (status: held), chưa phải là vé đã thanh toán.
    
    Args:
        flight_id (str): Mã chuyến bay (VD: 'VN122').
        seat_number (str): Mã số ghế cần đặt (VD: '12A').
        passenger_name (str): Tên hành khách (VD: 'Nguyen Van A').
        
    Returns:
        str: Chuỗi JSON chứa booking_id và trạng thái 'held'.
    """
    flight_id_clean = flight_id.strip().upper()
    seat_clean = seat_number.strip().upper()
    flight = FLIGHTS_DB.get(flight_id_clean)

    if not flight:
        return json.dumps({
            "status": "error",
            "error_code": "FLIGHT_NOT_FOUND",
            "message": f"Chuyến bay '{flight_id}' không tồn tại."
        }, ensure_ascii=False)

    if seat_clean not in flight["seats"]:
        return json.dumps({
            "status": "error",
            "error_code": "SEAT_NOT_AVAILABLE",
            "message": f"Ghế {seat_number} không còn trống trên chuyến bay {flight_id}. Ghế còn: {flight['seats']}"
        }, ensure_ascii=False)

    # Trừ ghế trống trong kho
    flight["seats"].remove(seat_clean)
    flight["available_seats"] = len(flight["seats"])

    # Tạo bản ghi đặt chỗ trạng thái 'held'
    booking_id = f"BK-{uuid.uuid4().hex[:6].upper()}"
    BOOKINGS_DB[booking_id] = {
        "booking_id": booking_id,
        "flight_id": flight["flight_id"],
        "seat_number": seat_clean,
        "passenger_name": passenger_name.strip(),
        "price": flight["price"],
        "depart_date": flight["depart_date"],
        "depart_time": flight["depart_time"],
        "refundable": flight["refundable"],
        "status": "held",
        "paid": False
    }

    return json.dumps({
        "status": "success",
        "booking_id": booking_id,
        "flight_id": flight["flight_id"],
        "seat_number": seat_clean,
        "passenger_name": passenger_name,
        "price": flight["price"],
        "booking_status": "held",
        "refundable": flight["refundable"],
        "message": f"Đã giữ chỗ thành công mã {booking_id}. Vui lòng thanh toán để xác nhận."
    }, ensure_ascii=False)


@tool
def pay(booking_id: str, payment_method: str, amount: int) -> str:
    """
    Thanh toán cho mã đặt chỗ (booking_id).
    CẢNH BÁO: Đây là hành động có tác dụng phụ (Side-effect) làm thay đổi tiền và trạng thái giao dịch.
    
    Args:
        booking_id (str): Mã đặt chỗ nhận được từ bước book_seat (VD: 'BK-AB12CD').
        payment_method (str): Phương thức thanh toán (VD: 'corp_card', 'bank_transfer').
        amount (int): Số tiền thanh toán thực tế (bằng số nguyên VND, VD: 1850000).
        
    Returns:
        str: Chuỗi JSON xác nhận thanh toán thành công và vé chuyển sang 'confirmed'.
    """
    b_id = booking_id.strip()
    booking = BOOKINGS_DB.get(b_id)

    if not booking:
        return json.dumps({
            "status": "error",
            "error_code": "BOOKING_NOT_FOUND",
            "message": f"Mã đặt chỗ '{booking_id}' không tồn tại."
        }, ensure_ascii=False)

    if booking["status"] == "confirmed" and booking["paid"]:
        return json.dumps({
            "status": "warning",
            "message": f"Mã đặt chỗ '{booking_id}' đã được thanh toán trước đó rồi.",
            "booking": booking
        }, ensure_ascii=False)

    # Kiểm tra số tiền thanh toán có khớp giá vé
    if amount != booking["price"]:
        return json.dumps({
            "status": "error",
            "error_code": "AMOUNT_MISMATCH",
            "message": f"Số tiền thanh toán ({amount:,}đ) không khớp giá vé ({booking['price']:,}đ)."
        }, ensure_ascii=False)

    # Cập nhật trạng thái thành công
    booking["status"] = "confirmed"
    booking["paid"] = True
    booking["payment_method"] = payment_method

    return json.dumps({
        "status": "success",
        "booking_id": b_id,
        "flight_id": booking["flight_id"],
        "seat_number": booking["seat_number"],
        "amount_paid": amount,
        "booking_status": "confirmed",
        "paid": True,
        "message": "Thanh toán thành công! Vé đã được xác nhận."
    }, ensure_ascii=False)


@tool
def get_booking(booking_id: str) -> str:
    """
    Tra cứu trạng thái chi tiết của một mã đặt chỗ từ cơ sở dữ liệu.
    Dùng cho Harness và Agent để kiểm chứng chéo trạng thái thực tế.
    
    Args:
        booking_id (str): Mã đặt chỗ (VD: 'BK-AB12CD').
        
    Returns:
        str: Chuỗi JSON chứa toàn bộ trạng thái đơn đặt chỗ hiện tại trong CSDL.
    """
    b_id = booking_id.strip()
    booking = BOOKINGS_DB.get(b_id)

    if not booking:
        return json.dumps({
            "status": "error",
            "error_code": "BOOKING_NOT_FOUND",
            "message": f"Không tìm thấy booking '{booking_id}'."
        }, ensure_ascii=False)

    return json.dumps({
        "status": "success",
        "booking": booking
    }, ensure_ascii=False)


ALL_TOOLS = [search_flights, check_seat, book_seat, pay, get_booking]


# ==============================================================================
# 3. SMOKE TEST KIỂM TRA IMPORT VÀ CSDL BAN ĐẦU
# ==============================================================================
if __name__ == "__main__":
    reset_db()
    print(f"✅ [mock_env] Khởi tạo thành công CSDL: {len(FLIGHTS_DB)} chuyến bay sẵn sàng.")
