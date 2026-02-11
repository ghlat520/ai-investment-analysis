# 数据库设计

## 技术选型

- **PostgreSQL 15+**：主数据库
- **TimescaleDB**：时序数据扩展（日线行情）
- **SQLAlchemy 2.0**：ORM + 异步支持
- **Alembic**：数据库迁移

## 表结构（10张核心表）

### 1. stock_info — 股票基本信息

```sql
CREATE TABLE stock_info (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,          -- 股票代码（如 000001.SZ）
    name VARCHAR(100) NOT NULL,           -- 股票名称
    market VARCHAR(10) NOT NULL,          -- 市场（A/HK/US）
    exchange VARCHAR(20),                 -- 交易所（SZ/SH/HKEX/NYSE/NASDAQ）
    industry VARCHAR(100),               -- 行业分类
    sector VARCHAR(100),                 -- 板块
    list_date DATE,                      -- 上市日期
    is_st BOOLEAN DEFAULT FALSE,         -- 是否ST
    is_delisting BOOLEAN DEFAULT FALSE,  -- 是否退市风险
    total_shares BIGINT,                 -- 总股本
    float_shares BIGINT,                 -- 流通股本
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(symbol)
);

CREATE INDEX idx_stock_info_market ON stock_info(market);
CREATE INDEX idx_stock_info_industry ON stock_info(industry);
```

### 2. stock_daily_quotes — 日线行情（TimescaleDB hypertable）

```sql
CREATE TABLE stock_daily_quotes (
    time DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open NUMERIC(12,4),
    high NUMERIC(12,4),
    low NUMERIC(12,4),
    close NUMERIC(12,4),
    volume BIGINT,
    amount NUMERIC(20,4),
    turnover_rate NUMERIC(8,4),
    amplitude NUMERIC(8,4),            -- 振幅 %
    change_pct NUMERIC(8,4),           -- 涨跌幅 %
    change_amount NUMERIC(12,4),       -- 涨跌额
    PRIMARY KEY (time, symbol)
);

-- 转为 TimescaleDB 超级表
SELECT create_hypertable('stock_daily_quotes', 'time');

CREATE INDEX idx_quotes_symbol ON stock_daily_quotes(symbol, time DESC);
```

### 3. stock_financial_data — 财务数据

```sql
CREATE TABLE stock_financial_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    report_date DATE NOT NULL,           -- 报告期（如 2025-09-30）
    report_type VARCHAR(10),             -- Q1/Q2/Q3/Annual

    -- 利润表
    revenue NUMERIC(20,4),               -- 营业收入
    net_profit NUMERIC(20,4),            -- 净利润
    net_profit_deducted NUMERIC(20,4),   -- 扣非净利润

    -- 盈利能力
    roe NUMERIC(8,4),                    -- 净资产收益率 %
    roa NUMERIC(8,4),                    -- 总资产收益率 %
    gross_margin NUMERIC(8,4),           -- 毛利率 %
    net_margin NUMERIC(8,4),             -- 净利率 %

    -- 成长性
    revenue_yoy NUMERIC(8,4),            -- 营收同比增速 %
    profit_yoy NUMERIC(8,4),             -- 净利同比增速 %

    -- 财务健康
    debt_ratio NUMERIC(8,4),             -- 资产负债率 %
    current_ratio NUMERIC(8,4),          -- 流动比率
    quick_ratio NUMERIC(8,4),            -- 速动比率

    -- 现金流
    operating_cashflow NUMERIC(20,4),    -- 经营现金流
    free_cashflow NUMERIC(20,4),         -- 自由现金流

    -- 估值相关
    pe_ttm NUMERIC(12,4),               -- 市盈率TTM
    pb NUMERIC(12,4),                   -- 市净率
    ps_ttm NUMERIC(12,4),               -- 市销率TTM
    total_market_cap NUMERIC(20,4),     -- 总市值

    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(symbol, report_date)
);

CREATE INDEX idx_financial_symbol ON stock_financial_data(symbol, report_date DESC);
```

### 4. stock_money_flow — 资金流向

```sql
CREATE TABLE stock_money_flow (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,

    -- 主力资金
    main_net_inflow NUMERIC(20,4),       -- 主力净流入（元）
    main_net_inflow_pct NUMERIC(8,4),    -- 主力净流入占比 %
    huge_net_inflow NUMERIC(20,4),       -- 超大单净流入
    large_net_inflow NUMERIC(20,4),      -- 大单净流入
    medium_net_inflow NUMERIC(20,4),     -- 中单净流入
    small_net_inflow NUMERIC(20,4),      -- 小单净流入

    -- 北向资金（A股特有）
    northbound_holding BIGINT,           -- 北向持仓数量（股）
    northbound_holding_pct NUMERIC(8,4), -- 北向持仓占比 %
    northbound_change BIGINT,            -- 北向持仓变化

    -- 融资融券
    margin_balance NUMERIC(20,4),        -- 融资余额
    short_balance NUMERIC(20,4),         -- 融券余额

    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(symbol, date)
);

CREATE INDEX idx_money_flow_symbol ON stock_money_flow(symbol, date DESC);
```

### 5. stock_news — 新闻数据

```sql
CREATE TABLE stock_news (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    title TEXT NOT NULL,
    content TEXT,
    source VARCHAR(100),
    url TEXT,
    published_at TIMESTAMP,
    search_dimension VARCHAR(50),        -- 搜索维度（news/risk/earnings/industry/analyst）
    sentiment VARCHAR(20),               -- 情绪标注（positive/negative/neutral）
    sentiment_score NUMERIC(4,2),        -- 情绪分数 -1.0 ~ 1.0
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_news_symbol ON stock_news(symbol, published_at DESC);
CREATE INDEX idx_news_dimension ON stock_news(search_dimension);
```

### 6. analysis_results — Agent分析结果

```sql
CREATE TABLE analysis_results (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL,                -- 本次分析运行ID
    symbol VARCHAR(20) NOT NULL,
    agent_name VARCHAR(50) NOT NULL,     -- technical/fundamental/valuation/money_flow/sentiment/industry
    signal_score INTEGER NOT NULL,       -- -100 ~ +100
    confidence NUMERIC(4,3) NOT NULL,    -- 0.000 ~ 1.000
    reasoning TEXT,
    key_factors JSONB,                   -- ["因素1", "因素2"]
    risks JSONB,                         -- ["风险1", "风险2"]
    data_quality NUMERIC(4,3),           -- 数据质量评分
    metadata JSONB,                      -- Agent特有的结构化数据
    llm_model VARCHAR(100),             -- 使用的LLM模型
    llm_tokens_used INTEGER,            -- Token消耗
    llm_cost NUMERIC(8,4),             -- LLM成本（USD）
    execution_time_ms INTEGER,          -- 执行耗时（毫秒）
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_analysis_run ON analysis_results(run_id);
CREATE INDEX idx_analysis_symbol ON analysis_results(symbol, created_at DESC);
CREATE INDEX idx_analysis_agent ON analysis_results(agent_name);
```

### 7. fusion_decisions — 融合决策

```sql
CREATE TABLE fusion_decisions (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    final_score INTEGER NOT NULL,        -- -100 ~ +100
    final_action VARCHAR(20) NOT NULL,   -- 建仓/加仓/持有/减仓/清仓/观望
    confidence NUMERIC(4,3) NOT NULL,
    position_pct INTEGER,                -- 建议仓位 %
    stop_loss_pct NUMERIC(6,2),         -- 止损比例 %
    take_profit_pct NUMERIC(6,2),       -- 止盈比例 %
    reasoning TEXT,
    signal_summary JSONB,                -- 各Agent信号汇总
    conflicts JSONB,                     -- 矛盾列表
    conflict_resolution TEXT,            -- 矛盾解决说明
    market_regime VARCHAR(20),           -- 市场环境（bull/bear/neutral）
    weights_used JSONB,                  -- 使用的权重方案
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_fusion_run ON fusion_decisions(run_id);
CREATE INDEX idx_fusion_symbol ON fusion_decisions(symbol, created_at DESC);
```

### 8. investment_reports — 投研报告

```sql
CREATE TABLE investment_reports (
    id SERIAL PRIMARY KEY,
    run_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    report_format VARCHAR(20) DEFAULT 'markdown',  -- markdown/html/pdf
    report_content TEXT NOT NULL,
    summary TEXT,                         -- 摘要
    rating VARCHAR(20),                  -- 评级
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_report_run ON investment_reports(run_id);
CREATE INDEX idx_report_symbol ON investment_reports(symbol, created_at DESC);
```

### 9. screening_results — 量化筛选结果

```sql
CREATE TABLE screening_results (
    id SERIAL PRIMARY KEY,
    run_date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    rank INTEGER NOT NULL,
    composite_score NUMERIC(8,4),        -- 综合评分
    factor_scores JSONB,                 -- 各因子得分明细
    market VARCHAR(10),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(run_date, symbol)
);

CREATE INDEX idx_screening_date ON screening_results(run_date DESC);
CREATE INDEX idx_screening_rank ON screening_results(run_date, rank);
```

### 10. llm_cost_log — LLM成本追踪

```sql
CREATE TABLE llm_cost_log (
    id SERIAL PRIMARY KEY,
    run_id UUID,
    agent_name VARCHAR(50),
    llm_provider VARCHAR(50),            -- openai/anthropic/google
    llm_model VARCHAR(100),
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    cost_usd NUMERIC(8,6),               -- 成本（USD）
    latency_ms INTEGER,                  -- 延迟（毫秒）
    is_cached BOOLEAN DEFAULT FALSE,     -- 是否命中缓存
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_cost_run ON llm_cost_log(run_id);
CREATE INDEX idx_cost_date ON llm_cost_log(created_at);
CREATE INDEX idx_cost_model ON llm_cost_log(llm_model);
```

## 数据保留策略

| 表 | 保留期 | 说明 |
|----|--------|------|
| stock_daily_quotes | 永久 | 历史行情不可再生 |
| stock_financial_data | 永久 | 历史财报不可再生 |
| stock_money_flow | 2年 | 资金流向时效性有限 |
| stock_news | 6个月 | 新闻时效性短 |
| analysis_results | 1年 | 用于回测验证 |
| fusion_decisions | 1年 | 用于回测验证 |
| investment_reports | 1年 | 历史研报 |
| screening_results | 6个月 | 筛选快照 |
| llm_cost_log | 3个月 | 成本统计 |
