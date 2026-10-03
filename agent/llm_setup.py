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

# Tự động nạp cấu hình từ file .env nếu có (tìm cả thư mục gốc và thư mục agent)
load_dotenv()
agent_env = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(agent_env):
    load_dotenv(agent_env)
root_env = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(root_env):
    load_dotenv(root_env)

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage, SystemMessage
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


class UITOpenAIChatModel(BaseChatModel):
    """
    Client LangChain thuần túy kết nối GPU UIT hoặc OpenAI bằng thư viện openai chính thức.
    Khắc phục triệt để lỗi DLL load failed của tiktoken trên môi trường Windows bị chặn policy.
    """
    model_name: str
    api_key: str
    base_url: Optional[str] = None
    tools_list: Optional[List[Any]] = None
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "uit_openai_chat_model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return UITOpenAIChatModel(
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            tools_list=list(tools) if tools else None,
            temperature=self.temperature
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        import openai
        import httpx
        from langchain_core.utils.function_calling import convert_to_openai_tool

        client_kwargs = {
            "api_key": self.api_key,
            "http_client": httpx.Client(verify=False)
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = openai.OpenAI(**client_kwargs)

        formatted_messages = []
        for m in messages:
            if isinstance(m, HumanMessage):
                formatted_messages.append({"role": "user", "content": str(m.content)})
            elif isinstance(m, SystemMessage):
                formatted_messages.append({"role": "system", "content": str(m.content)})
            elif isinstance(m, ToolMessage):
                formatted_messages.append({
                    "role": "tool",
                    "content": str(m.content),
                    "tool_call_id": getattr(m, "tool_call_id", "call_default")
                })
            elif isinstance(m, AIMessage):
                msg_dict = {"role": "assistant", "content": str(m.content or "")}
                if getattr(m, "tool_calls", None):
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc.get("id", f"call_{i}"),
                            "type": "function",
                            "function": {
                                "name": tc.get("name"),
                                "arguments": json.dumps(tc.get("args", {}))
                            }
                        }
                        for i, tc in enumerate(m.tool_calls)
                    ]
                formatted_messages.append(msg_dict)
            else:
                formatted_messages.append({"role": "user", "content": str(m.content)})

        create_kwargs = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": self.temperature
        }
        if "qwen" in self.model_name.lower():
            create_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        res = None
        if self.tools_list:
            create_kwargs["tools"] = [convert_to_openai_tool(t) for t in self.tools_list]
            try:
                res = client.chat.completions.create(**create_kwargs)
            except Exception as e:
                # Nếu server vLLM chưa bật --enable-auto-tool-choice, tự động chuyển sang Prompt-based Tool Calling
                if "tool" in str(e).lower() or "400" in str(e):
                    create_kwargs.pop("tools", None)
                    tools_desc = "\n".join([f"- {t.name}: {t.description}" for t in self.tools_list])
                    injection = (
                        f"\n\nBẠN CÓ CÁC CÔNG CỤ SAU:\n{tools_desc}\n"
                        f"KHI CẦN GỌI CÔNG CỤ, HÃY TRẢ VỀ DUY NHẤT ĐỊNH DẠNG JSON:\n"
                        f'{{"tool": "tên_công_cụ", "args": {{...}}}}\n'
                        f"KHÔNG THÊM BẤT KỲ VĂN BẢN NÀO NGOÀI JSON."
                    )
                    if formatted_messages and formatted_messages[0]["role"] == "system":
                        formatted_messages[0]["content"] += injection
                    else:
                        formatted_messages.insert(0, {"role": "system", "content": injection})
                    res = client.chat.completions.create(**create_kwargs)
                else:
                    raise e
        else:
            res = client.chat.completions.create(**create_kwargs)

        choice = res.choices[0]

        parsed_tool_calls = []
        if getattr(choice.message, "tool_calls", None):
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except Exception:
                    args = {}
                parsed_tool_calls.append({
                    "name": tc.function.name,
                    "args": args,
                    "id": tc.id
                })
        elif choice.message.content:
            import re
            text = choice.message.content.strip()
            # 1. Bắt cú pháp call:tool_name{args} đặc thù của Gemma
            gemma_match = re.search(r'call:(\w+)\{(.*?)\}', text)
            if gemma_match:
                tool_name = gemma_match.group(1)
                raw_args = gemma_match.group(2)
                parsed_args = {}
                for pair in raw_args.split(","):
                    if ":" in pair:
                        k, v = pair.split(":", 1)
                        k = k.strip().strip("'\"")
                        v = v.strip().strip("'\"")
                        try:
                            v = int(v)
                        except ValueError:
                            pass
                        parsed_args[k] = v
                parsed_tool_calls.append({
                    "name": tool_name,
                    "args": parsed_args,
                    "id": f"call_gemma_1"
                })
            elif "tool" in text:
                raw = text
                if raw.startswith("```json"): raw = raw[7:]
                if raw.startswith("```"): raw = raw[3:]
                if raw.endswith("```"): raw = raw[:-3]
                try:
                    data = json.loads(raw.strip())
                    if isinstance(data, dict) and "tool" in data:
                        parsed_tool_calls.append({
                            "name": data["tool"],
                            "args": data.get("args", {}),
                            "id": f"call_1"
                        })
                except Exception:
                    pass

        token_usage = {
            "total_tokens": getattr(res.usage, "total_tokens", 0),
            "prompt_tokens": getattr(res.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(res.usage, "completion_tokens", 0),
        }

        ai_msg = AIMessage(
            content=choice.message.content or "",
            tool_calls=parsed_tool_calls,
            response_metadata={"token_usage": token_usage}
        )
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])


def get_llm(agent_type: str = "react"):
    """
    Trả về mô hình LLM thật (UITOpenAIChatModel) nếu có API Key,
    ngược lại fallback về MockFlightChatModel.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    model_name = os.environ.get("SE373_MODEL", "gpt-4o-mini")

    if api_key and api_key.strip():
        try:
            return UITOpenAIChatModel(
                model_name=model_name,
                api_key=api_key.strip(),
                base_url=base_url.strip() if base_url else None,
                temperature=0.0
            )
        except Exception as e:
            print(f"⚠️ [LLM Setup] Lỗi khởi tạo client: {e}. Đang chuyển về MockLLM.")

    # Fallback mô phỏng thông minh
    return MockFlightChatModel(agent_type=agent_type)
