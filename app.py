"""
AI 调用服务 —— 模块化 Agent 工具调度平台

基于 FastAPI + DeepSeek API 的智能 Agent 助手。
核心能力：Tool Calling（手写实现）、工具热插拔、SSE 流式输出。

接口：
  GET  /health      健康检查
  GET  /tools       查看可用工具列表
  POST /chat        非流式对话
  POST /chat/stream SSE 流式对话
"""

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import requests
import json
import datetime
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY")
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL")

# ============================================================
# 工具定义
# ============================================================

def get_current_time() -> str:
    now = datetime.datetime.now()
    return json.dumps({
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": now.strftime("%A"),
    }, ensure_ascii=False)


def calculate(expression: str) -> str:
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return "表达式包含不允许的符号"
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算出错：{e}"


def get_weather(city: str) -> str:
    url = f"https://wttr.in/{city}"
    params = {"format": "j1"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        current = data["current_condition"][0]
        weather_info = {
            "city": city,
            "temperature": f"{current['temp_C']}°C",
            "condition": current["weatherDesc"][0]["value"],
            "humidity": f"{current['humidity']}%",
        }
        return json.dumps(weather_info, ensure_ascii=False)
    except requests.exceptions.Timeout:
        return json.dumps({"error": f"查询 {city} 天气超时"}, ensure_ascii=False)
    except requests.exceptions.HTTPError:
        return json.dumps({"error": f"未找到 {city} 的天气数据"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"查询天气失败：{e}"}, ensure_ascii=False)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '100+200'",
                    }
                },
                "required": ["expression"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市实时天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称（中文）",
                    }
                },
                "required": ["city"],
            }
        }
    },
]

TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "get_weather": get_weather,
}

# ============================================================
# Tool Calling 核心调度
# ============================================================

def call_llm_with_tools(messages: list) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    body = {"model": "deepseek-chat", "messages": messages, "tools": TOOLS}
    resp = requests.post(DEEPSEEK_URL, json=body, headers=headers, timeout=30)
    resp.raise_for_status()
    result = resp.json()

    choice = result["choices"][0]
    finish_reason = choice["finish_reason"]

    if finish_reason == "tool_calls":
        messages.append(choice["message"])

        for tool_call in choice["message"]["tool_calls"]:
            func_name = tool_call["function"]["name"]
            func_args = json.loads(tool_call["function"]["arguments"])
            func_result = TOOL_FUNCTIONS[func_name](**func_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": func_result,
            })

        body2 = {"model": "deepseek-chat", "messages": messages}
        resp2 = requests.post(DEEPSEEK_URL, json=body2, headers=headers, timeout=30)
        resp2.raise_for_status()
        reply = resp2.json()["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": reply})
        return reply
    else:
        reply = choice["message"]["content"]
        messages.append({"role": "assistant", "content": reply})
        return reply


# ============================================================
# FastAPI 应用
# ============================================================

from fastapi import Depends, HTTPException

def verify_api_key(x_api_key: str = None):
    """API Key 认证：请求头必须带 X-API-Key"""
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key 无效或缺失")
    return True

from fastapi import Depends, HTTPException

def verify_api_key(x_api_key: str = None):
    """API Key 认证：请求头必须带 X-API-Key"""
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key 无效或缺失")
    return True

app = FastAPI(
    title="多功能 Agent 助手",
    description="集成时间、计算器、天气查询的智能 Agent，手写 Tool Calling + SSE 流式",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="用户消息")
    history: list = Field(default=[], description="对话历史")


class ChatResponse(BaseModel):
    reply: str
    history: list
    tools_used: list = Field(default=[])


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    if request.history:
        messages = list(request.history)
    else:
        messages = [{
            "role": "system",
            "content": "你是一个智能助手，可以查询时间、计算数学、查询天气。回答用纯文本。"
        }]
    messages.append({"role": "user", "content": request.message.strip()})

    try:
        reply = call_llm_with_tools(messages)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM 调用失败：{e}")

    return ChatResponse(reply=reply, history=messages)


@app.post("/chat/stream", dependencies=[Depends(verify_api_key)])
async def chat_stream(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    if request.history:
        messages = list(request.history)
    else:
        messages = [{
            "role": "system",
            "content": "你是一个智能助手，可以查询时间、计算数学、查询天气。回答用纯文本。"
        }]
    messages.append({"role": "user", "content": request.message.strip()})

    async def generate():
        try:
            reply = call_llm_with_tools(messages)
            for char in reply:
                chunk = json.dumps({"type": "text", "content": char}, ensure_ascii=False)
                yield f"data: {chunk}\n\n"
                import asyncio
                await asyncio.sleep(0.02)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            error_chunk = json.dumps({"type": "error", "content": str(e)}, ensure_ascii=False)
            yield f"data: {error_chunk}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/tools")
def list_tools():
    tool_list = [{"name": t["function"]["name"], "description": t["function"]["description"]} for t in TOOLS]
    return {"total": len(tool_list), "tools": tool_list}


@app.get("/health")
def health():
    try:
        resp = requests.get(DEEPSEEK_URL.replace("/chat/completions", ""), timeout=5)
        api_status = "connected" if resp.status_code < 500 else "error"
    except:
        api_status = "unreachable"
    return {"status": "ok", "api": api_status}


@app.exception_handler(Exception)
def global_exception(request: Request, exc: Exception):
    logger.error(f"未捕获异常: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "服务器内部错误"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
