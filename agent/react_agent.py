"""
agent/react_agent.py - Cài đặt Agent theo mẫu thiết kế ReAct (Reasoning + Acting).
Môn học: Kỹ thuật xây dựng hệ thống Agentic AI (SE373).

Nguyên lý hoạt động:
Vòng lặp tương tác liên tục: Suy luận -> Hành động -> Quan sát -> Suy luận tiếp.
Mỗi quyết định bước tiếp theo phụ thuộc hoàn toàn vào Observation của bước trước đó.
Toàn bộ hành động đều chịu sự kiểm soát của FlightBookingHarness.
"""

import sys
import json
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# Cấu hình encoding Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from mock_env import search_flights, check_seat, book_seat, pay, get_booking, ALL_TOOLS
from harness import UserConstraints, FlightBookingHarness
from agent.llm_setup import get_llm

# Bản đồ tên tool sang hàm thực thi
TOOL_MAP = {t.name: t for t in ALL_TOOLS}


def run_react_agent(
    constraints: UserConstraints,
    harness: FlightBookingHarness,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Chạy Agent theo mẫu thiết kế ReAct có giám sát qua Harness.
    
    Args:
        constraints: Bộ ràng buộc dữ liệu gốc của người dùng (Chốt chặn 1).
        harness: Lớp điều phối kiểm soát vòng lặp và an toàn.
        verbose: In chi tiết từng bước nếu True.
        
    Returns:
        Dict chứa số liệu thống kê: số bước, chi phí token ước tính, trạng thái thành công.
    """
    llm = get_llm(agent_type="react")
    
    # Ràng buộc là dữ liệu: Ép chặt vào System Prompt
    system_prompt = (
        f"Bạn là trợ lý AI chuyên đặt vé máy bay theo chu trình ReAct.\n"
        f"MỤC TIÊU & RÀNG BUỘC CỐ ĐỊNH:\n"
        f"- Chặng bay: từ {constraints.origin} đến {constraints.destination}\n"
        f"- Ngày bay: {constraints.depart_date}\n"
        f"- Ngân sách tối đa: {constraints.max_price:,} VND\n"
        f"- Hành khách: {constraints.passenger_name}\n"
        f"- Cho phép vé không hoàn tiền: {constraints.allow_non_refundable}\n"
        f"QUY TẮC: Luôn kiểm tra ghế và giá trước khi đặt. Không được vượt ngân sách."
    )

    user_prompt = f"Hãy tìm và đặt giúp tôi vé máy bay từ {constraints.origin} đi {constraints.destination} ngày {constraints.depart_date}."

    messages: List[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    total_tokens_estimated = 0
    step_count = 0

    if verbose:
        print("\n" + "="*60)
        print("🤖 [START] KHỞI ĐỘNG AGENT REACT (REASONING + ACTING)")
        print("="*60)

    # VÒNG LẶP SUY LUẬN REACT
    while not harness.is_terminated:
        step_count += 1
        try:
            if hasattr(llm, "bind_tools") and not isinstance(llm, type):
                llm_with_tools = llm.bind_tools(ALL_TOOLS)
                response: AIMessage = llm_with_tools.invoke(messages)
            else:
                response = llm.invoke(messages)
        except Exception as e:
            if verbose:
                print(f"❌ Lỗi khi gọi LLM: {e}")
            break

        # Trích xuất Token thật nếu có API Key, ngược lại dùng ước tính heuristic
        meta_usage = getattr(response, "response_metadata", {}).get("token_usage")
        if meta_usage and "total_tokens" in meta_usage:
            total_tokens_estimated += meta_usage["total_tokens"]
        else:
            history_len = sum(len(str(m.content)) for m in messages)
            total_tokens_estimated += 500 + int(history_len / 4)

        messages.append(response)

        # Kiểm tra xem Model có gọi Tool không
        tool_calls = getattr(response, "tool_calls", [])

        if not tool_calls:
            # Model không gọi tool nữa mà tự trả lời (hoặc dừng)
            if verbose:
                print(f"💬 [Model Response]: {response.content}")
            break

        # 2. XỬ LÝ LỜI GỌI TOOL QUA BỘ GIÁM SÁT HARNESS
        for tc in tool_calls:
            tool_name = tc.get("name")
            tool_args = tc.get("args", {})
            call_id = tc.get("id", f"call_{step_count}")

            if verbose:
                print(f"\n👉 [Vòng {step_count} - ReAct] Model đề xuất: {tool_name}({tool_args})")

            # --- TRẠM 1: KIỂM QUYỀN TRƯỚC KHI THỰC THI ---
            is_permitted = harness.pre_tool_check(tool_name, tool_args)
            if not is_permitted:
                if verbose:
                    print(f"⛔ [HARNESS CHẶN] Hành động '{tool_name}' vượt thẩm quyền! Tạm dừng chờ duyệt.")
                break

            # THỰC THI TOOL TỪ MOCK ENVIRONMENT
            target_tool = TOOL_MAP.get(tool_name)
            if not target_tool:
                obs_content = json.dumps({"status": "error", "message": f"Tool '{tool_name}' không tồn tại."})
            else:
                obs_content = target_tool.invoke(tool_args)

            if verbose:
                print(f"👁️ [Observation]: {obs_content}")

            # Đưa Observation vào lịch sử chat (ToolMessage)
            messages.append(ToolMessage(content=obs_content, tool_call_id=call_id))

            # --- TRẠM 2: KIỂM ĐIỀU KIỆN DỪNG SAU KHI CÓ OBSERVATION ---
            should_continue = harness.post_tool_check(
                tool_name=tool_name,
                args=tool_args,
                observation=obs_content,
                progress_metric=step_count
            )

            if not should_continue or harness.is_terminated:
                break

    # Tạo câu trả lời tự nhiên cuối cùng gửi khách hàng
    final_answer = harness.generate_final_response()

    summary = {
        "agent_name": "ReAct Agent",
        "steps": harness.current_step,
        "tokens_estimated": total_tokens_estimated,
        "is_success": (harness.termination_type == "SUCCESS"),
        "termination_type": harness.termination_type,
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
    print("✅ [react_agent] Module sẵn sàng.")
