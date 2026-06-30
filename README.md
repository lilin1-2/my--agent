# AI 调用服务 —— 模块化 Agent 工具调度平台

## 简介

基于 FastAPI + DeepSeek API 的智能 Agent 助手。**手写 Tool Calling 全流程**，不依赖 LangChain 等高层封装。

## 功能

- 时间查询 —— 获取当前日期、时间、星期
- 安全计算器 —— 计算数学表达式（含符号校验）
- 天气查询 —— 对接 [wttr.in](https://wttr.in) 免费 API，获取真实天气数据
- 工具热插拔 —— 新增工具只需 3 步：定义函数 → 添加 JSON Schema → 注册映射表

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/tools` | 查看可用工具列表 |
| POST | `/chat` | 非流式对话 |
| POST | `/chat/stream` | SSE 流式对话 |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp config_example.py config.py
# 编辑 config.py，填入你的 DeepSeek API Key

# 3. 启动服务
python app.py

# 4. 测试
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message":"北京天气怎么样"}'
```

## 技术栈

Python · FastAPI · DeepSeek API · Tool Calling · Pydantic · SSE
