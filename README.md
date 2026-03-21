# LangGraph Trip Planner

基于 FastAPI + LangGraph + 高德地图 MCP 的旅行规划项目。

输入目的地、日期、出行偏好后，系统会自动生成多日行程（景点、酒店、餐饮、预算、天气建议），并在前端可视化展示。

## 功能简介

- 旅行计划生成：`POST /api/trip/plan`
- 地图能力：POI 搜索、天气查询、路线规划
- 餐饮输出：在 `plan` 节点直接生成三餐，前端可显示餐厅名与地址
- 日志追踪：按 `req` / `run` 打点，便于定位慢请求

## 项目结构

```text
.
├─ backend/
│  ├─ app/
│  │  ├─ api/                 # FastAPI 路由
│  │  ├─ workflows/           # LangGraph 旅行工作流
│  │  ├─ tools/               # AMap MCP 工具适配
│  │  ├─ services/            # LLM / 地图 / 图片服务
│  │  └─ models/              # Pydantic 数据模型
│  ├─ logs/                   # 运行日志（自动创建）
│  ├─ requirements.txt
│  └─ .env.example
├─ frontend/
│  ├─ src/
│  ├─ package.json
│  └─ .env.example
└─ README.md
```

## 环境要求

- Python 3.10+
- Node.js 18+（建议）
- 可用的 LLM API（OpenAI / DeepSeek 兼容接口）
- 高德 API Key（后端 Web Service Key + 前端 JS Key）

## 快速启动

### 1) 启动后端

```bash
cd backend
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

复制并编辑环境变量：

```bash
# Windows
copy .env.example .env

# macOS / Linux
# cp .env.example .env
```

启动服务：

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

后端文档地址：

- Swagger: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>

### 2) 启动前端

新开一个终端：

```bash
cd frontend
npm install
```

复制并编辑前端环境变量：

```bash
# Windows
copy .env.example .env

# macOS / Linux
# cp .env.example .env
```

启动开发服务器：

```bash
npm run dev
```

访问：<http://localhost:5173>

## 环境变量说明

### backend/.env

必填：

- `AMAP_API_KEY`：高德 Web Service Key（后端地图服务）
- `OPENAI_API_KEY` 或 `LLM_API_KEY`：LLM Key（二选一）

常用可选：

- `OPENAI_BASE_URL` 或 `LLM_BASE_URL`
- `OPENAI_MODEL` 或 `LLM_MODEL_ID`
- `LLM_TIMEOUT`
- `CORS_ORIGINS`
- `LOG_LEVEL`

说明：项目支持 `OPENAI_*` 与 `LLM_*` 两套别名配置。

### frontend/.env

- `VITE_API_BASE_URL`：后端地址（开发建议 `http://localhost:8000` 或 `/`）
- `VITE_AMAP_WEB_JS_KEY`：前端地图 JS Key
- `VITE_AMAP_WEB_KEY`：可选，若前端有直接调用 Web API 场景可用

## 常用接口示例

### 生成行程

```bash
curl -X POST "http://localhost:8000/api/trip/plan" \
  -H "Content-Type: application/json" \
  -d '{
    "city":"洛阳",
    "start_date":"2026-03-26",
    "end_date":"2026-03-27",
    "travel_days":2,
    "transportation":"地铁+步行",
    "accommodation":"经济型酒店",
    "preferences":["历史文化","美食"],
    "free_text_input":"行程轻松一点，餐饮给具体店名"
  }'
```

### 搜索 POI

```bash
curl "http://localhost:8000/api/map/poi?keywords=餐厅&city=洛阳&citylimit=true"
```

## 日志与排查

日志文件：

- `backend/logs/backend.out.log`
- `backend/logs/backend.err.log`

常用排查命令：

```bash
# 查看最近的 trip 请求日志
rg "trip_request_start|trip_request_success|node_done" backend/logs/backend.out.log
```

## 依赖说明（requirements）

当前 `backend/requirements.txt` 是“锁定版本”策略（含部分传递依赖），优点是复现稳定，缺点是体积偏大。

本次已补充跨平台兼容（Windows 专属包增加平台标记），避免 Linux/macOS 安装失败。

---

如果你准备对外开源，建议后续补一个 `requirements-min.txt`（仅直接依赖），保留当前 `requirements.txt` 作为锁定版本。
