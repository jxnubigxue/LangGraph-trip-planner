# 离线评测说明

这个目录用于旅行规划工作流的离线回归评测，目标是做到“改动可量化、效果可对比、质量可门禁”。

## 目录结构
- `eval_cases.jsonl`：评测用例（JSONL，每行一个 case）。
- `eval_runner.py`：批量执行脚本，输出指标、门禁结果、对比报告。
- `reports/`：生成的评测报告（`.json` + `.md`）。

## 用例格式
`eval_cases.jsonl` 每行一个 JSON，对应一个完整评测样本。结构如下：

```json
{
  "id": "hz_culture_2d_001",
  "input": {
    "city": "杭州",
    "start_date": "2026-04-10",
    "end_date": "2026-04-11",
    "travel_days": 2,
    "transportation": "地铁+步行",
    "accommodation": "舒适型酒店",
    "preferences": ["历史文化", "美食"],
    "free_text_input": "希望每天不要安排太赶"
  },
  "constraints": {
    "min_attractions_per_day": 2,
    "max_attractions_per_day": 3,
    "required_meal_types": ["breakfast", "lunch", "dinner"],
    "avoid_outdoor_on_rain": true
  }
}
```

`constraints` 字段可选，默认值如下：
- `min_attractions_per_day=2`（每天最少景点数）
- `max_attractions_per_day=3`（每天最多景点数）
- `required_meal_types=["breakfast","lunch","dinner"]`（必含餐食类型）
- `avoid_outdoor_on_rain=true`（雨天避免高暴露户外项目）

## 评价指标（中文说明）
- `成功率`：成功返回行程结果的 case 占比。
- `失败率`：运行异常 case 占比。
- `约束满足率`：满足所有约束（景点数、餐食、雨天规则等）的 case 占比。
- `MCP命中率`：检索节点中直接命中 `source=mcp` 的占比。
- `回退率`：检索节点走 `llm_fallback` 的占比。
- `平均总耗时(ms)`：整体工作流平均耗时。
- `平均行程生成耗时(ms)`：`plan_itinerary` 节点平均耗时。
- `平均景点检索耗时(ms)`：`search_attractions` 节点平均耗时。
- `平均景点解析耗时(ms)`：`parse_attractions` 阶段平均耗时。

## 运行评测
在 `backend/` 目录执行：

```bash
python evals/eval_runner.py \
  --cases evals/eval_cases.jsonl \
  --gate \
  --min-constraint-pass-rate 0.95 \
  --max-fallback-rate 0.20 \
  --max-failure-rate 0.05
```

## 与基线报告对比
```bash
python evals/eval_runner.py \
  --cases evals/eval_cases.jsonl \
  --baseline evals/reports/eval_report_20260318_120000.json
```

## 门禁开关（可选）
- 默认就是关闭门禁（不加 `--gate` 即可）。
- 你也可以显式写 `--no-gate`，效果与默认一致。
- 只有传 `--gate` 时，才会按阈值判定并在不达标时返回非零退出码。

## 给“其他大模型”生成数据的提示词
把下面这段直接给模型，让它输出 50-100 条 JSONL：

```text
你是旅行规划离线评测数据生成器。请直接输出 JSONL（每行一个 JSON 对象），不要加解释，不要 markdown。

目标：生成 80 条中国城市短途旅行评测用例，覆盖 1-5 天行程，城市分布均衡（一线/新一线/旅游城市），人群偏好多样（亲子、情侣、历史文化、美食、自然、夜游、预算敏感）。

每条 JSON 必须包含字段：
- id: 全局唯一字符串，如 case_0001
- input: {
  city, start_date, end_date, travel_days, transportation, accommodation, preferences, free_text_input
}
- constraints: {
  min_attractions_per_day, max_attractions_per_day, required_meal_types, avoid_outdoor_on_rain
}

规则：
1) start_date 和 end_date 必须匹配 travel_days。
2) travel_days 范围 1-5。
3) transportation 只从 ["地铁+步行","公共交通","打车+步行","自驾"] 选。
4) accommodation 只从 ["经济型酒店","舒适型酒店","豪华型酒店"] 选。
5) preferences 2-4 个标签，从 ["历史文化","美食","博物馆","亲子","自然风光","夜游","购物","摄影","轻松慢游"] 选。
6) free_text_input 给出1句自然语言额外要求（如“下雨天多安排室内活动”）。
7) constraints 固定为：
   - min_attractions_per_day: 2
   - max_attractions_per_day: 3
   - required_meal_types: ["breakfast","lunch","dinner"]
   - avoid_outdoor_on_rain: true
8) 不要出现空字段；输出必须是合法 JSONL。
```
