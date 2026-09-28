"""
main.py - Cổng giao tiếp chính (Interactive CLI) cho Hệ thống Agent Đặt Vé Máy Bay.
Môn học: Kỹ thuật xây dựng hệ thống Agentic AI (SE373) - BTVN#3.

Cung cấp 3 chế độ hoạt động:
1. Đặt vé tương tác từ bàn phím (Người dùng tự nhập thông tin và chọn Agent).
2. Chạy tự động trọn bộ Benchmark (4 Kịch bản Biên trên cả 3 Agent).
3. Xem lại bảng kết quả và lịch sử đo đạc Benchmark từ file benchmark_history.json.
"""

import sys
import os
import json

# Cấu hình encoding Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from mock_env import reset_db
from harness import UserConstraints, FlightBookingHarness
from agent import run_react_agent, run_plan_execute_agent, run_hybrid_agent
from evaluate import run_benchmark, print_markdown_report, HISTORY_FILE


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    print("="*65)
    print("✈️  HỆ THỐNG AGENTIC AI ĐẶT VÉ MÁY BAY (SE373 - UIT)")
    print("    Kiến trúc 4 Chốt chặn Harness & 3 Mẫu thiết kế Suy luận")
    print("="*65)


def mode_interactive_booking():
    """Chế độ 1: Người dùng tự gõ thông tin từ bàn phím để chạy thử Agent."""
    print("\n--- [CHẾ ĐỘ 1] ĐẶT VÉ TƯƠNG TÁC TỪ BÀN PHÍM ---")
    print("👉 Nhấn [Enter] để sử dụng giá trị mặc định trong ngoặc vuông [ ].\n")

    origin = input("1. Điểm khởi hành (mã IATA) [SGN]: ").strip().upper() or "SGN"
    destination = input("2. Điểm đến (mã IATA) [DAD]: ").strip().upper() or "DAD"
    depart_date = input("3. Ngày khởi hành (YYYY-MM-DD) [2026-10-07]: ").strip() or "2026-10-07"
    
    price_str = input("4. Trần ngân sách tối đa (VND) [2000000]: ").strip() or "2000000"
    try:
        max_price = int(price_str.replace(".", "").replace(",", ""))
    except ValueError:
        max_price = 2_000_000

    passenger_name = input("5. Họ và tên hành khách [Cao Tien Phat]: ").strip() or "Cao Tien Phat"
    
    non_ref_str = input("6. Cho phép tự động mua vé Không Hoàn Hủy? (y/N) [N]: ").strip().lower()
    allow_non_refundable = (non_ref_str == "y")

    print("\nChọn Mẫu thiết kế Agent muốn sử dụng:")
    print("  [1] ReAct Agent (Suy luận linh hoạt từng bước)")
    print("  [2] Plan-then-Execute Agent (Lập kế hoạch tổng thể rồi chạy)")
    print("  [3] Mẫu Lai (Hybrid: Plan + ReAct có Re-planning)")
    choice = input("👉 Lựa chọn của bạn (1-3) [1]: ").strip() or "1"

    # Đóng gói dữ liệu yêu cầu vào UserConstraints (Chốt chặn 1)
    constraints = UserConstraints(
        origin=origin,
        destination=destination,
        depart_date=depart_date,
        max_price=max_price,
        passenger_name=passenger_name,
        allow_non_refundable=allow_non_refundable
    )

    reset_db()
    harness = FlightBookingHarness(constraints=constraints, max_budget_steps=8)

    print(f"\n🚀 Đang khởi chạy Agent với yêu cầu: {origin} -> {destination} | Ngân sách: {max_price:,}đ...")

    if choice == "2":
        result = run_plan_execute_agent(constraints, harness, verbose=True)
    elif choice == "3":
        result = run_hybrid_agent(constraints, harness, verbose=True)
    else:
        result = run_react_agent(constraints, harness, verbose=True)

    input("\nNhấn [Enter] để quay lại Menu chính...")


def mode_run_benchmark():
    """Chế độ 2: Tự động chạy toàn bộ 4 kịch bản trên cả 3 Agent."""
    print("\n--- [CHẾ ĐỘ 2] CHẠY BENCHMARK ĐÁNH GIÁ 4 KỊCH BẢN BIÊN ---")
    data = run_benchmark(verbose_agent=False)
    print_markdown_report(data)
    input("\nNhấn [Enter] để quay lại Menu chính...")


def mode_view_history():
    """Chế độ 3: Đọc và hiển thị lại lịch sử các lần đo đạc từ file JSON."""
    print("\n--- [CHẾ ĐỘ 3] XEM LỊCH SỬ KẾT QUẢ BENCHMARK ĐÃ LƯU ---")
    if not os.path.exists(HISTORY_FILE):
        print("⚠️ Chưa có dữ liệu lịch sử nào. Hãy chạy Chế độ [2] trước!")
    else:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
        print(f"📁 Đã tìm thấy {len(records)} bản ghi trong '{HISTORY_FILE}'.")
        print_markdown_report(records[-12:])  # Hiển thị 12 bản ghi gần nhất (1 phiên benchmark đầy đủ)
    input("\nNhấn [Enter] để quay lại Menu chính...")


def main():
    while True:
        print_banner()
        print("  [1] Đặt vé tương tác từ bàn phím (Chạy thử nghiệm Agent)")
        print("  [2] Chạy tự động Benchmark 4 Kịch bản Biên (Xuất số liệu Báo cáo)")
        print("  [3] Xem lại Lịch sử Kết quả Benchmark đã lưu")
        print("  [4] Thoát chương trình")
        print("="*65)
        
        choice = input("👉 Nhập lựa chọn của bạn (1-4): ").strip()

        if choice == "1":
            mode_interactive_booking()
        elif choice == "2":
            mode_run_benchmark()
        elif choice == "3":
            mode_view_history()
        elif choice == "4":
            print("\n👋 Cảm ơn bạn đã sử dụng hệ thống! Chúc bạn đạt điểm cao trong BTVN#3.")
            break
        else:
            print("❌ Lựa chọn không hợp lệ, vui lòng chọn từ 1 đến 4.")


if __name__ == "__main__":
    main()
