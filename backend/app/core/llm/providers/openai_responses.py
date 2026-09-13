"""OpenAI Responses API Provider。"""

from openai import AsyncOpenAI
from app.core.llm.providers.base import BaseProvider
from app.core.llm.types import StandardResponse, ToolCall, Usage


class OpenAIResponsesProvider(BaseProvider):
    """OpenAI Responses API (/v1/responses) 实现。"""

    async def call(
        self,
        messages: list[dict],
        model: str,
        api_key: str,
        base_url: str | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        disable_thinking: bool = False,
    ) -> StandardResponse:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        input_items = self._messages_to_input(messages)

        kwargs: dict = {"model": model, "input": input_items}
        if max_tokens:
            kwargs["max_output_tokens"] = max_tokens
        if top_p is not None:
            kwargs["top_p"] = top_p
        if disable_thinking:
            # Responses 格式的思考模式逃生口：effort=none 表示关闭思考
            kwargs["reasoning"] = {"effort": "none"}
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            if tool_choice:
                kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)

        response = await client.responses.create(**kwargs)

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        reasoning_parts: list[str] = []

        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    if part.type == "output_text":
                        content_parts.append(part.text)
            elif item.type == "function_call":
                tool_calls.append(ToolCall(
                    id=item.call_id,
                    name=item.name,
                    arguments=item.arguments,
                ))
            elif item.type == "reasoning":
                # 思考模式：思维链必须原样回传，否则下一轮请求会 400
                text = self._extract_reasoning_text(item)
                if text:
                    reasoning_parts.append(text)

        content = "".join(content_parts) if content_parts else None
        reasoning_content = "\n".join(reasoning_parts) if reasoning_parts else None

        usage = Usage(
            prompt_tokens=response.usage.input_tokens if response.usage else 0,
            completion_tokens=response.usage.output_tokens if response.usage else 0,
        )

        return StandardResponse(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            usage=usage,
        )

    @staticmethod
    def _extract_reasoning_text(item: object) -> str:
        """从 reasoning item 里提取思维链明文。

        需要兼容多种返回形态（不同厂商/中转对 Responses 规范的实现不一致）：

        - DeepSeek / 规范形态：``content=[{"type":"reasoning_text","text":"..."}]``
          （openai SDK 1.75 的 ResponseReasoningItem 只有 id/summary/type/status，
          ``content`` 靠 ``extra="allow"`` 透传，所以拿到的是原始 list，必须按列表解析）
        - 部分中转：item 级别直接给 ``reasoning_text`` / ``text`` 字符串
        - OpenAI 官方：``summary=[{"type":"summary_text","text":"..."}]``
          （只有显式请求 summary 时才有值，默认拿不到思维链正文）
        """
        for attr in ("reasoning_text", "text", "content", "summary"):
            parts = OpenAIResponsesProvider._collect_text(getattr(item, attr, None))
            if parts:
                return "".join(parts)
        return ""

    @staticmethod
    def _collect_text(value: object) -> list[str]:
        """把「字符串 / 内容块列表 / 单个内容块」统一收集为文本片段列表。

        内容块既可能是 dict（raw 透传），也可能是 SDK 模型对象（新版 SDK 把
        reasoning 的 content 声明成了真实字段），两种都要能取到 text。
        """
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, (list, tuple)):
            text = getattr(value, "text", None)
            return [text] if isinstance(text, str) and text.strip() else []

        parts: list[str] = []
        for block in value:
            if isinstance(block, str):
                if block.strip():
                    parts.append(block)
                continue
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text)
        return parts

    def _messages_to_input(self, messages: list[dict]) -> list[dict]:
        """将 Chat Completions messages 格式转为 Responses input 格式。

        注意 Responses API 的硬性约束：`function_call` 必须**紧邻**它的
        `function_call_output`。若两者之间插入任何 item（哪怕是一条 assistant
        文本消息），服务端会判定该工具调用没有输出，直接返回
        400「No tool output found for tool call xxx.」。
        所以 assistant 的文本必须排在 `function_call` **之前**。
        """
        input_items = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                input_items.append({"role": "developer", "content": msg.get("content") or ""})
            elif role == "tool":
                output = msg.get("content", "")
                input_items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    # output 必须是字符串，None / list 都会被服务端拒绝
                    "output": output if isinstance(output, str) else str(output),
                })
            elif role == "assistant":
                # 思考模式 + 带 tools 的请求：思维链必须原样回传，否则 400
                # （The `reasoning_text` in the thinking mode must be passed back to the API.）
                # 回传格式按规范：content 为 reasoning_text 内容块列表，而不是
                # item 级别的 {"reasoning_text": "..."}（后者不被识别，等于没传）。
                reasoning = msg.get("reasoning_content") or msg.get("reasoning_text")
                reasoning_blocks = self._to_reasoning_blocks(reasoning)
                if reasoning_blocks:
                    input_items.append({
                        "type": "reasoning",
                        "content": reasoning_blocks,
                    })

                if msg.get("tool_calls"):
                    # 文本必须排在 function_call 之前，保证 call 与 output 相邻
                    if msg.get("content"):
                        input_items.append({"role": "assistant", "content": msg["content"]})
                    for tc in msg["tool_calls"]:
                        input_items.append({
                            "type": "function_call",
                            "call_id": tc["id"],
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        })
                else:
                    input_items.append({"role": "assistant", "content": msg.get("content", "")})
            else:
                input_items.append({"role": role, "content": msg.get("content", "")})
        return input_items

    @staticmethod
    def _to_reasoning_blocks(reasoning: object) -> list[dict]:
        """把历史里存的思维链还原成规范的 reasoning_text 内容块列表。"""
        parts = OpenAIResponsesProvider._collect_text(reasoning)
        return [{"type": "reasoning_text", "text": p} for p in parts]

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """将 Chat Completions tools 格式转为 Responses tools 格式。"""
        converted = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                converted.append({
                    "type": "function",
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                    "strict": func.get("strict", True),
                })
        return converted

    def _convert_tool_choice(self, tool_choice: str) -> str | dict:
        """转换 tool_choice 格式。"""
        if tool_choice == "auto":
            return "auto"
        if tool_choice == "none":
            return "none"
        if tool_choice == "required":
            return {"type": "function"}
        return tool_choice
