"""
agent/llm_setup.py - Cấu hình mô hình ngôn ngữ (LLM) và cơ chế Giả lập (Simulation Fallback).
Môn học: Kỹ thuật xây dựng hệ thống Agentic AI (SE373) - Buổi 03.

Hỗ trợ:
1. Kết nối mô hình thật qua OpenAI API (nếu có biến môi trường OPENAI_API_KEY hoặc SE373_MODEL).
2. Tự động chuyển sang MockLLM thông minh nếu chưa có API key để sinh viên luôn chạy thử nghiệm
   và kiểm thử benchmark được 100% không bị crash.
"""

import os
import json
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Tự động nạp cấu hình từ file .env nếu có
load_dotenv()

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration

# Cấu hình encoding Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class MockFlightChatModel(BaseChatModel):
    """
    Mock LLM thông minh mô phỏng hành vi suy luận đặt vé máy bay theo từng mẫu thiết kế.
    Dùng cho trường hợp chạy offline hoặc không có OpenAI API Key.
    """
    agent_type: str = "react"  # 'react', 'planner', 'replanner'

    @property
    def _llm_type(self) -> str:
        return "mock_flight_chat_model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        """Hỗ trợ giao diện bind_tools của LangChain."""
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        from mock_env import FLIGHTS_DB

        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        all_text = " ".join(str(m.content) for m in messages)
        is_low_budget = ("1,500,000" in all_text or "1500000" in all_text)

        # 1. CHẾ ĐỘ PLANNER (Plan-then-Execute)
        # Nếu Scenario 4 (VN122 bị khóa sẵn trước), planner chọn VN134
        if self.agent_type == "planner":
            if FLIGHTS_DB.get("VN122", {}).get("available_seats", 0) == 0 and FLIGHTS_DB.get("VJ604", {}).get("available_seats", 0) == 0:
                p_fid, p_price, p_seat = "VN134", 1_700_000, "14A"
            else:
                p_fid, p_price, p_seat = "VJ604", 1_200_000, "08A"

            plan = [
                {"step": 1, "tool": "search_flights", "args": {"origin": "SGN", "destination": "DAD", "date": "2026-10-07"}},
                {"step": 2, "tool": "check_seat", "args": {"flight_id": p_fid}},
                {"step": 3, "tool": "book_seat", "args": {"flight_id": p_fid, "seat_number": p_seat, "passenger_name": "Cao Tien Phat"}},
                {"step": 4, "tool": "pay", "args": {"booking_id": "$PREV_BOOKING_ID", "payment_method": "corp_card", "amount": p_price}}
            ]
            content = json.dumps(plan, ensure_ascii=False)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

        # 2. CHẾ ĐỘ RE-PLANNER (Mẫu Lai)
        if self.agent_type == "replanner":
            if FLIGHTS_DB.get("VN122", {}).get("available_seats", 0) > 0:
                rp_fid, rp_price, rp_seat = "VN122", 1_850_000, "12A"
            else:
                rp_fid, rp_price, rp_seat = "VN134", 1_700_000, "14A"

            new_plan = [
                {"step": 1, "tool": "check_seat", "args": {"flight_id": rp_fid}},
                {"step": 2, "tool": "book_seat", "args": {"flight_id": rp_fid, "seat_number": rp_seat, "passenger_name": "Cao Tien Phat"}},
                {"step": 3, "tool": "pay", "args": {"booking_id": "$PREV_BOOKING_ID", "payment_method": "corp_card", "amount": rp_price}}
            ]
            content = json.dumps(new_plan, ensure_ascii=False)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

        # 3. CHẾ ĐỘ REACT (Suy luận linh hoạt theo từng Observation thực tế)
        if not tool_messages:
            tool_calls = [{
                "name": "search_flights",
                "args": {"origin": "SGN", "destination": "DAD", "date": "2026-10-07"},
                "id": "call_search_01"
            }]
            msg = AIMessage(content="Tôi sẽ tìm danh sách chuyến bay SGN -> DAD ngày 2026-10-07.", tool_calls=tool_calls)
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # Xác định chuyến bay mục tiêu từ thực tế CSDL
        vj_seats = FLIGHTS_DB.get("VJ604", {}).get("available_seats", 0)
        vn_seats = FLIGHTS_DB.get("VN122", {}).get("available_seats", 0)

        if vj_seats > 0 and not is_low_budget:
            target_fid, target_price, target_seat = "VJ604", 1_200_000, "08A"
        elif vn_seats > 0 and not is_low_budget:
            target_fid, target_price, target_seat = "VN122", 1_850_000, "12A"
        else:
            target_fid, target_price, target_seat = "VN134", 1_700_000, "14A"

        if len(tool_messages) == 1:
            tool_calls = [{
                "name": "check_seat",
                "args": {"flight_id": target_fid},
                "id": f"call_check_{target_fid}"
            }]
            msg = AIMessage(content=f"Tôi sẽ kiểm tra chi tiết chuyến {target_fid}.", tool_calls=tool_calls)
        elif len(tool_messages) == 2:
            tool_calls = [{
                "name": "book_seat",
                "args": {"flight_id": target_fid, "seat_number": target_seat, "passenger_name": "Cao Tien Phat"},
                "id": f"call_book_{target_fid}"
            }]
            msg = AIMessage(content=f"Chuyến {target_fid} phù hợp, tôi tiến hành giữ chỗ ghế {target_seat}.", tool_calls=tool_calls)
        elif len(tool_messages) == 3:
            last_obs = json.loads(tool_messages[-1].content)
            bid = last_obs.get("booking_id", "BK-UNKNOWN")
            tool_calls = [{
                "name": "pay",
                "args": {"booking_id": bid, "payment_method": "corp_card", "amount": target_price},
                "id": "call_pay_final"
            }]
            msg = AIMessage(content=f"Giữ chỗ thành công mã {bid}. Tiến hành thanh toán {target_price:,}đ.", tool_calls=tool_calls)
        else:
            msg = AIMessage(content="Giao dịch đặt vé đã hoàn tất!")

        return ChatResult(generations=[ChatGeneration(message=msg)])


def get_llm(agent_type: str = "react"):
    """
    Trả về ChatOpenAI thật nếu có API Key (hỗ trợ OpenAI và GPU UIT qua base_url),
    ngược lại trả về MockFlightChatModel.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    model_name = os.environ.get("SE373_MODEL", "gpt-4o-mini")

    if api_key:
        try:
            from langchain_openai import ChatOpenAI
            init_kwargs = {
                "model": model_name,
                "api_key": api_key,
                "temperature": 0.0
            }
            if base_url:
                init_kwargs["base_url"] = base_url
            # Nếu dùng mô hình Qwen của UIT, tắt thinking để gọi tool nhanh hơn theo tài liệu hướng dẫn
            if "qwen" in model_name.lower():
                init_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

            return ChatOpenAI(**init_kwargs)
        except Exception:
            pass

    # Fallback mô phỏng thông minh
    return MockFlightChatModel(agent_type=agent_type)
