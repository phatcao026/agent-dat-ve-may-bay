"""
agent/plan_execute_agent.py - Cài đặt Agent theo mẫu thiết kế Plan-then-Execute.
Môn học: Kỹ thuật xây dựng hệ thống Agentic AI (SE373).

Nguyên lý hoạt động:
1. Giai đoạn Planner: Gọi LLM đúng 1 lần để sinh trọn vẹn kế hoạch gồm danh sách các bước tĩnh.
2. Giai đoạn Executor: Duyệt tuần tự qua danh sách các bước và gọi tool.
Đặc điểm: Tiết kiệm token, kế hoạch nhìn thấy trước nên duyệt được; nhưng thiếu linh hoạt
(khi bước đầu lỗi hoặc hết vé thì toàn bộ kế hoạch phía sau sẽ gãy đổ).
"""

import sys
import json
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Cấu hình encoding Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from mock_env import ALL_TOOLS
from harness import UserConstraints, FlightBookingHarness
from agent.llm_setup import get_llm

TOOL_MAP = {t.name: t for t in ALL_TOOLS}


def run_plan_execute_agent(
    constraints: UserConstraints,
    harness: FlightBookingHarness,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Chạy Agent theo mẫu Plan-then-Execute qua 2 giai đoạn tách biệt.
    """
    llm = get_llm(agent_type="planner")

    if verbose:
        print("\n" + "="*60)
        print("📋 [START] KHỞI ĐỘNG AGENT PLAN-THEN-EXECUTE")
        print("="*60)

    # --------------------------------------------------------------------------
    # GIAI ĐOẠN 1: PLANNER (LẬP KẾ HOẠCH TỔNG THỂ)
    # --------------------------------------------------------------------------
    planner_prompt = (
        f"Bạn là AI Planner chuyên lập kế hoạch đặt vé máy bay.\n"
        f"Nhiệm vụ: Hãy lập danh sách trọn vẹn 4 bước để đặt và thanh toán vé hoàn tất:\n"
        f"1. search_flights: tìm chuyến bay từ {constraints.origin} đi {constraints.destination} ngày {constraints.depart_date}\n"
        f"2. check_seat: kiểm tra ghế chuyến bay rẻ nhất thỏa trần giá {constraints.max_price} VND (VD: VN122 hoặc VJ604)\n"
        f"3. book_seat: giữ chỗ ghế 12A cho khách {constraints.passenger_name}\n"
        f"4. pay: thanh toán mã booking vừa giữ\n\n"
        f"Ví dụ định dạng trả về:\n"
        f"[\n"
        f"  {{\"step\": 1, \"tool\": \"search_flights\", \"args\": {{\"origin\": \"{constraints.origin}\", \"destination\": \"{constraints.destination}\", \"date\": \"{constraints.depart_date}\"}}}},\n"
        f"  {{\"step\": 2, \"tool\": \"check_seat\", \"args\": {{\"flight_id\": \"VN122\"}}}},\n"
        f"  {{\"step\": 3, \"tool\": \"book_seat\", \"args\": {{\"flight_id\": \"VN122\", \"seat_number\": \"12A\", \"passenger_name\": \"{constraints.passenger_name}\"}}}},\n"
        f"  {{\"step\": 4, \"tool\": \"pay\", \"args\": {{\"booking_id\": \"$PREV_BOOKING_ID\", \"payment_method\": \"corp_card\", \"amount\": 1850000}}}}\n"
        f"]\n"
        f"Chỉ trả lời bằng JSON thuần, không thêm văn bản giải thích."
    )

    messages = [
        SystemMessage(content=planner_prompt),
        HumanMessage(content="Lập kế hoạch đặt vé ngay.")
    ]

    total_tokens_estimated = 800  # Token cố định cho lần gọi planner ban đầu
    
    if verbose:
        print("🧠 [Phase 1: Planner] Đang yêu cầu Model sinh kế hoạch hoàn chỉnh...")

    try:
        plan_response: AIMessage = llm.invoke(messages)
        # Trích xuất Token thật nếu có API Key
        meta_usage = getattr(plan_response, "response_metadata", {}).get("token_usage")
        if meta_usage and "total_tokens" in meta_usage:
            total_tokens_estimated = meta_usage["total_tokens"]

        plan_content = plan_response.content
        # Parse JSON plan
        if isinstance(plan_content, str):
            # Lọc bỏ markdown code fences nếu có
            cleaned_json = plan_content.strip()
            if cleaned_json.startswith("```json"):
                cleaned_json = cleaned_json[7:]
            if cleaned_json.startswith("```"):
                cleaned_json = cleaned_json[3:]
            parsed_data = json.loads(cleaned_json.strip())
            if isinstance(parsed_data, dict):
                for val in parsed_data.values():
                    if isinstance(val, list):
                        parsed_data = val
                        break
                if not isinstance(parsed_data, list):
                    parsed_data = [parsed_data]
            plan = parsed_data if isinstance(parsed_data, list) else []
        else:
            plan = []
    except Exception as e:
        if verbose:
            print(f"❌ Lỗi khi sinh hoặc parse kế hoạch: {e}")
        plan = []

    if verbose:
        print(f"📝 [Kế hoạch sinh ra]: Gồm {len(plan)} bước:")
        for p in plan:
            print(f"   Bước {p.get('step')}: Tool '{p.get('tool')}' với args: {p.get('args')}")
        print("👀 [Người duyệt]: Đồng ý kế hoạch -> Bắt đầu chuyển sang Executor.\n")

    # --------------------------------------------------------------------------
    # GIAI ĐOẠN 2: EXECUTOR (THỰC THI TỪNG BƯỚC TUẦN TỰ)
    # Không gọi lại LLM nữa, chạy thuần túy bằng code thực thi
    # --------------------------------------------------------------------------
    runtime_booking_id: Optional[str] = None

    for step_item in plan:
        if harness.is_terminated:
            break

        tool_name = step_item.get("tool")
        tool_args = step_item.get("args", {}).copy()

        # Thay thế biến động từ bước trước nếu có ($PREV_BOOKING_ID)
        for k, v in tool_args.items():
            if str(v) in ["$PREV_BOOKING_ID", "FROM_PREV_STEP"] and runtime_booking_id:
                tool_args[k] = runtime_booking_id

        if verbose:
            print(f"⚙️ [Phase 2: Executor] Đang chạy Bước {step_item.get('step')}: {tool_name}({tool_args})")

        # --- TRẠM 1: KIỂM QUYỀN TRƯỚC KHI THỰC THI ---
        is_permitted = harness.pre_tool_check(tool_name, tool_args)
        if not is_permitted:
            if verbose:
                print(f"⛔ [HARNESS CHẶN] Hành động '{tool_name}' bị từ chối quyền. Tạm dừng.")
            break

        # THỰC THI TOOL
        target_tool = TOOL_MAP.get(tool_name)
        if not target_tool:
            obs = json.dumps({"status": "error", "message": f"Không có tool '{tool_name}'"})
        else:
            try:
                obs = target_tool.invoke(tool_args)
            except Exception as e:
                obs = json.dumps({"status": "error", "message": f"Lỗi tham số khi gọi tool '{tool_name}': {e}"})

        if verbose:
            print(f"👁️ [Observation]: {obs}")

        # Trích xuất booking_id nếu có
        try:
            obs_json = json.loads(obs)
            if "booking_id" in obs_json:
                runtime_booking_id = obs_json["booking_id"]
        except Exception:
            pass

        # --- TRẠM 2: KIỂM TIÊU CHÍ HOÀN THÀNH VÀ DỪNG ---
        should_continue = harness.post_tool_check(
            tool_name=tool_name,
            args=tool_args,
            observation=obs,
            progress_metric=step_item.get("step")
        )

        if not should_continue or harness.is_terminated:
            break

    term_type = harness.termination_type
    is_success = (term_type == "SUCCESS")
    if not is_success and not term_type:
        term_type = "FAILED_STALE_PLAN"
        harness.termination_type = term_type

    final_answer = harness.generate_final_response()

    summary = {
        "agent_name": "Plan-then-Execute Agent",
        "steps": harness.current_step,
        "tokens_estimated": total_tokens_estimated,
        "is_success": is_success,
        "termination_type": term_type,
        "booking_id": harness.latest_booking_id,
        "handoff_data": harness.handoff_data,
        "final_response": final_answer
    }

    if verbose:
        print("\n" + "="*60)
        print("🤖 [AI PHẢN HỒI KHÁCH HÀNG]:")
        print(final_answer)
        print("="*60)
        print(f"🏁 [KẾT QUẢ KỸ THUẬT] Hoàn thành: {summary['is_success']} | Kiểu dừng: {summary['termination_type']} | Tổng bước: {summary['steps']} | Token: {summary['tokens_estimated']:,}")
        print("-"*60 + "\n")

    return summary


if __name__ == "__main__":
    print("✅ [plan_execute_agent] Module sẵn sàng.")
