"""
agent/hybrid_agent.py - Cài đặt Agent theo Mẫu Lai (Hybrid: Plan + ReAct with Replanning).
Môn học: Kỹ thuật xây dựng hệ thống Agentic AI (SE373).

Nguyên lý hoạt động:
1. Lập kế hoạch tổng thể ban đầu (Plan).
2. Thực thi từng bước có giám sát theo phong cách ReAct.
3. Chốt kiểm tra thích ứng: "Observation đổi đáng kể?".
   - Nếu bình thường: Chạy tiếp bước kế tiếp.
   - Nếu gặp biến cố (hết vé, lỗi tham số, quá ngân sách): LẬP TỨC GỌI LẠI MODEL ĐỂ TÁI LẬP KẾ HOẠCH (RE-PLANNING)!
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


def is_observation_significantly_changed(observation_str: str) -> bool:
    """
    Hiện thực hóa câu hỏi kiểm tra thích ứng: "Observation đổi đáng kể?"
    Phát hiện xem kết quả vừa nhận có phá vỡ giả định ban đầu hay không
    (VD: chuyến bay hết sạch ghế, hoặc tool trả về lỗi).
    """
    try:
        data = json.loads(observation_str)
        # 1. Nếu tool trả lỗi
        if data.get("status") == "error":
            return True
        # 2. Nếu kiểm tra ghế mà số ghế trống bằng 0
        if "available_seats_count" in data and data["available_seats_count"] == 0:
            return True
        if "available_seats" in data and len(data["available_seats"]) == 0:
            return True
    except Exception:
        pass
    return False


def run_hybrid_agent(
    constraints: UserConstraints,
    harness: FlightBookingHarness,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Chạy Agent theo Mẫu Lai: Lập kế hoạch + ReAct có Re-planning linh hoạt.
    """
    planner_llm = get_llm(agent_type="planner")
    replanner_llm = get_llm(agent_type="replanner")

    if verbose:
        print("\n" + "="*60)
        print("🔀 [START] KHỞI ĐỘNG AGENT MẪU LAI (HYBRID: PLAN + REACT + REPLAN)")
        print("="*60)

    # 1. LẬP KẾ HOẠCH BAN ĐẦU (INITIAL PLAN)
    planner_prompt = (
        f"Lập danh sách JSON các bước đặt vé từ {constraints.origin} đến {constraints.destination} "
        f"ngày {constraints.depart_date}, ngân sách {constraints.max_price}."
    )
    
    total_tokens_estimated = 600

    try:
        res: AIMessage = planner_llm.invoke([
            SystemMessage(content="Bạn là Planner. Hãy sinh danh sách JSON các bước đặt vé."),
            HumanMessage(content=planner_prompt)
        ])
        content = res.content.strip()
        if content.startswith("```json"): content = content[7:]
        if content.startswith("```"): content = content[3:]
        if content.endswith("```"): content = content[:-3]
        current_plan: List[Dict[str, Any]] = json.loads(content.strip())
    except Exception:
        # Fallback plan mặc định
        current_plan = [
            {"step": 1, "tool": "search_flights", "args": {"origin": constraints.origin, "destination": constraints.destination, "date": constraints.depart_date}},
            {"step": 2, "tool": "check_seat", "args": {"flight_id": "VN122"}},
            {"step": 3, "tool": "book_seat", "args": {"flight_id": "VN122", "seat_number": "12A", "passenger_name": constraints.passenger_name}},
            {"step": 4, "tool": "pay", "args": {"booking_id": "$PREV_BOOKING_ID", "payment_method": "corp_card", "amount": 1850000}}
        ]

    if verbose:
        print(f"📋 [Initial Plan] Kế hoạch ban đầu gồm {len(current_plan)} bước.")

    runtime_booking_id: Optional[str] = None
    step_pointer = 0
    max_replans = 2
    replan_count = 0

    # VÒNG LẶP THỰC THI & TÁI LẬP KẾ HOẠCH
    while step_pointer < len(current_plan) and not harness.is_terminated:
        step_item = current_plan[step_pointer]
        tool_name = step_item.get("tool")
        tool_args = step_item.get("args", {}).copy()

        # Cập nhật biến động từ bước trước
        for k, v in tool_args.items():
            if str(v) in ["$PREV_BOOKING_ID", "FROM_PREV_STEP"] and runtime_booking_id:
                tool_args[k] = runtime_booking_id

        if verbose:
            print(f"\n⚡ [Mẫu Lai - Thực thi] Bước {step_pointer + 1}: {tool_name}({tool_args})")

        # --- TRẠM 1: KIỂM QUYỀN (PRE-TOOL) ---
        is_permitted = harness.pre_tool_check(tool_name, tool_args)
        if not is_permitted:
            if verbose:
                print(f"⛔ [HARNESS CHẶN] Hành động '{tool_name}' vượt thẩm quyền.")
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

        total_tokens_estimated += 400

        if verbose:
            print(f"👁️ [Observation]: {obs}")

        # Trích xuất booking_id nếu có
        try:
            obs_json = json.loads(obs)
            if "booking_id" in obs_json:
                runtime_booking_id = obs_json["booking_id"]
        except Exception:
            pass

        # --- TRẠM 2: KIỂM TIÊU CHÍ HOÀN THÀNH & LẶP (POST-TOOL) ---
        should_continue = harness.post_tool_check(
            tool_name=tool_name,
            args=tool_args,
            observation=obs,
            progress_metric=step_pointer
        )

        if not should_continue or harness.is_terminated:
            break

        # --- KIỂM TRA ĐIỀU KIỆN TÁI LẬP KẾ HOẠCH (OBSERVATION ĐỔI ĐÁNG KỂ?) ---
        if is_observation_significantly_changed(obs) and replan_count < max_replans:
            replan_count += 1
            total_tokens_estimated += 700
            if verbose:
                print("\n🚨 [PHÁT HIỆN BIẾN ĐỘNG LỚN] Kế hoạch hiện tại bị cản trở bởi kết quả quan sát!")
                print(f"🔄 [Kích hoạt Re-planner (Lần {replan_count})] Đang tái lập kế hoạch mới dựa trên môi trường thực tế...")

            replan_prompt = (
                f"Kế hoạch bước '{tool_name}' đã thất bại với kết quả: {obs}.\n"
                f"Hãy tái lập danh sách các bước tiếp theo từ trạng thái hiện tại để đạt mục tiêu đặt vé."
            )
            try:
                replan_res: AIMessage = replanner_llm.invoke([
                    SystemMessage(content="Bạn là Re-planner. Lập lại kế hoạch dự phòng dạng JSON."),
                    HumanMessage(content=replan_prompt)
                ])
                # Trích xuất token replanner nếu có
                rp_usage = getattr(replan_res, "response_metadata", {}).get("token_usage")
                if rp_usage and "total_tokens" in rp_usage:
                    total_tokens_estimated += rp_usage["total_tokens"]

                rp_content = replan_res.content.strip()
                if rp_content.startswith("```json"): rp_content = rp_content[7:]
                if rp_content.startswith("```"): rp_content = rp_content[3:]
                parsed_plan = json.loads(rp_content.strip())
                if isinstance(parsed_plan, dict):
                    for val in parsed_plan.values():
                        if isinstance(val, list):
                            parsed_plan = val
                            break
                    if not isinstance(parsed_plan, list):
                        parsed_plan = [parsed_plan]
                
                if isinstance(parsed_plan, list) and len(parsed_plan) > 0:
                    current_plan = parsed_plan
                    step_pointer = 0
                    if verbose:
                        print(f"✅ [Kế hoạch mới đã sẵn sàng]: Gồm {len(current_plan)} bước tiếp theo.")
                    continue
            except Exception as e:
                if verbose:
                    print(f"⚠️ Re-planning thất bại: {e}. Tiếp tục kế hoạch cũ.")

        step_pointer += 1

    final_answer = harness.generate_final_response()

    summary = {
        "agent_name": "Hybrid Agent (Plan + ReAct)",
        "steps": harness.current_step,
        "tokens_estimated": total_tokens_estimated,
        "is_success": (harness.termination_type == "SUCCESS"),
        "termination_type": harness.termination_type,
        "booking_id": harness.latest_booking_id,
        "replan_count": replan_count,
        "handoff_data": harness.handoff_data,
        "final_response": final_answer
    }

    if verbose:
        print("\n" + "="*60)
        print("🤖 [AI PHẢN HỒI KHÁCH HÀNG]:")
        print(final_answer)
        print("="*60)
        print(f"🏁 [KẾT QUẢ KỸ THUẬT] Hoàn thành: {summary['is_success']} | Kiểu dừng: {summary['termination_type']} | Tổng bước: {summary['steps']} | Số lần Re-plan: {summary['replan_count']} | Token: {summary['tokens_estimated']:,}")
        print("-"*60 + "\n")

    return summary


if __name__ == "__main__":
    print("✅ [hybrid_agent] Module sẵn sàng.")
