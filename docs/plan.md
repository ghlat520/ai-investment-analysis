# AI投研助手系统 — 设计方案与实施计划

## Context

**项目位置**：`/Applications/soft/CodeSpace/ai-investment-analysis/`
**参考项目**：`/Applications/soft/CodeSpace/daily_stock_analysis/`（已有成熟的单AI股票分析系统）

**目标**：在 daily_stock_analysis 的基础上，构建一个全维度多Agent协作的AI投研助手系统。继承参考项目的成熟模式（auto-fallback数据源、结构化LLM输出、回测验证），扩展为多Agent架构，覆盖6个分析维度。

**关键差异**（新系统 vs 参考项目）：

| 维度 | daily_stock_analysis | ai-investment-analysis |
|------|---------------------|----------------------|
| 分析模式 | 单LLM一次性分析 | 多Agent分维度协作 |
| 覆盖维度 | 技术面+新闻 | 技术+基本面+估值+资金+情绪+产业链 |
| 市场 | A股为主 | A股+港股+美股 |
| 筛选 | 手动指定股票列表 | 量化筛选全市场→Top N |
| Agent框架 | 无（直接调LLM） | LangGraph |
| 数据库 | SQLite | PostgreSQL + TimescaleDB |

---

## 从参考项目复用的模式

### 1. Auto-Fallback数据采集
参考：`daily_stock_analysis/data_provider/base.py` — DataFetcherManager
- 6个数据源按优先级自动切换（efinance → akshare → tushare → pytdx → baostock → yfinance）
- Circuit breaker：失败源5分钟冷却
- 新系统改进：增加港股/美股数据源适配，统一Market Adapter接口

### 2. 结构化LLM输出
参考：`daily_stock_analysis/src/analyzer.py` — GeminiAnalyzer
- System prompt嵌入交易理念
- 强制JSON结构化输出 + json_repair容错
- 新系统改进：每个Agent有独立prompt模板，输出统一为signal_score + confidence + reasoning

### 3. 技术面分析
参考：`daily_stock_analysis/src/stock_analyzer.py` — StockTrendAnalyzer
- MA/MACD/RSI信号评分系统（0-100分）
- 均线排列、乖离率、量能分析
- 新系统改进：扩展为独立的技术面Agent，增加形态识别

### 4. 新闻搜索
参考：`daily_stock_analysis/src/search_service.py` — SearchService
- 5维度并行搜索（最新新闻/市场分析/风险检查/业绩/行业）
- 多搜索引擎fallback（Bocha → Tavily → Brave → SerpAPI）
- 多API Key轮换

### 5. 回测验证
参考：`daily_stock_analysis/src/core/backtest_engine.py`
- 分析结果 vs 实际走势验证
- 新系统改进：多维度回测（各Agent准确率 + 融合决策准确率）

### 6. 通知推送
参考：`daily_stock_analysis/src/notification.py`
- 企业微信/飞书/Telegram/邮件多通道

---

## 核心架构

### 两层分离设计

```
Layer 1: 量化筛选引擎（纯代码，全市场覆盖 5000+标的）
    ↓ Top 50 候选标的
Layer 2: LLM多Agent深度分析（LangGraph，仅候选标的）
    ↓
研报 + 投资建议 + 通知推送
```

### LangGraph Agent编排

```
DataLoader → [技术面|基本面|估值|资金面|情绪面|产业链] (并行) → 决策融合 → 研报生成
```

### 8个Agent

| Agent | 核心任务 | LLM模型 | 代码/LLM |
|-------|---------|---------|---------|
| 技术面 | 趋势+形态+指标 | gpt-4o-mini | 80/20 |
| 基本面 | 财报质量+盈利+增长 | claude-sonnet | 60/40 |
| 估值 | PE/PB分位+DCF+行业对比 | gpt-4o-mini | 70/30 |
| 资金面 | 主力资金+北向+融资融券 | gpt-4o-mini | 80/20 |
| 情绪面 | 新闻舆情+分析师评级 | gpt-4o-mini | 30/70 |
| 产业链 | 行业景气度+上下游+政策 | claude-sonnet | 40/60 |
| 决策融合 | 信号仲裁+综合决策 | claude-sonnet | 20/80 |
| 研报生成 | 结构化投研报告 | claude-sonnet | 10/90 |

---

## 分阶段建设

### Phase 1 — MVP（6周）- 仅A股，3个Agent
- W1：项目骨架 + 数据库 + akshare/efinance数据源
- W2：行情+财务数据采集器 + 数据入库
- W3：量化筛选引擎（技术+估值因子）
- W4：技术面Agent + 基本面Agent
- W5：估值Agent + 简版决策融合（固定权重）
- W6：研报生成 + CLI + 每日调度 + 通知

### Phase 2 — 增强（6周）- 加维度+加市场
- W7-8：资金面Agent + 情绪面Agent
- W9：港股支持
- W10：动态权重 + LLM信号仲裁
- W11：Streamlit Web UI
- W12：Redis缓存 + 成本优化

### Phase 3 — 完善（6周）- 全市场+全维度
- W13-14：美股支持 + 产业链Agent
- W15：HTML/PDF研报
- W16：回测框架
- W17：通知系统增强
- W18：性能优化 + 监控

---

## 每日分析流水线

```
16:00  A股收盘
16:30  数据采集（行情+资金+新闻+融资融券）
17:30  量化筛选（全A股 → Top 50）
18:00  AI深度分析（50只股票 × 6Agent并行）
19:30  输出（研报入库 + 信号推送）
```

## LLM成本：50只/天 ≈ $2-3（人民币15-20元）
