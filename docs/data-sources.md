# 数据源规划

## Auto-Fallback 架构

复用 daily_stock_analysis 的 DataFetcherManager 模式：
- 多数据源按优先级注册
- 失败自动切换下一个
- Circuit breaker：失败源冷却5分钟
- 统一输出格式（DataFrame）

```python
class DataSourceManager:
    """数据源管理器（auto-fallback模式）"""

    def fetch(self, method: str, **kwargs) -> pd.DataFrame:
        for source in self.sources:
            if source.is_available():
                try:
                    return source.call(method, **kwargs)
                except Exception:
                    source.mark_failed()
        raise AllSourcesFailedError()
```

## A股数据源

### 行情数据

| 优先级 | 数据源 | 库 | 覆盖 | 限制 |
|--------|--------|-----|------|------|
| 0 | 东方财富 | efinance | 实时+历史日线 | 无限制 |
| 1 | AKShare | akshare | 实时+历史日线 | 频率限制 |
| 2 | Tushare Pro | tushare | 历史日线 | 每日5000次 |
| 3 | 通达信 | pytdx | 实时+历史 | 服务器不稳定 |
| 4 | BaoStock | baostock | 历史日线 | 延迟1天 |
| 5 | Yahoo Finance | yfinance | 历史日线 | A股数据有限 |

### 财务数据

| 数据 | 主源 | 备源 | 说明 |
|------|------|------|------|
| 财务报表 | akshare | tushare | 资产负债表/利润表/现金流量表 |
| 主要财务指标 | akshare | efinance | ROE/ROA/毛利率等 |
| 估值指标 | akshare | tushare | PE/PB/PS |
| 股本信息 | akshare | tushare | 总股本/流通股 |

### 资金数据

| 数据 | 主源 | 备源 |
|------|------|------|
| 主力资金流向 | efinance | akshare |
| 北向资金 | akshare | tushare |
| 融资融券 | akshare | tushare |
| 大宗交易 | akshare | — |

### 新闻数据

复用 daily_stock_analysis 的 SearchService：

| 优先级 | 搜索引擎 | 免费额度 |
|--------|---------|---------|
| 0 | Bocha | 付费 |
| 1 | Tavily | 1000次/月 |
| 2 | Brave Search | 2000次/月 |
| 3 | SerpAPI | 100次/月 |

## 港股数据源（Phase 2）

| 数据 | 数据源 | 说明 |
|------|--------|------|
| 日线行情 | yfinance | .HK 后缀 |
| 财务数据 | akshare(港股) | 部分支持 |
| 资金数据 | 南向资金(akshare) | 反向追踪 |
| 新闻 | SearchService | 搜索引擎通用 |

## 美股数据源（Phase 3）

| 数据 | 数据源 | 说明 |
|------|--------|------|
| 日线行情 | yfinance | 主力数据源 |
| 财务数据 | yfinance | Quarterly financials |
| 机构持仓 | SEC 13F | 季度更新 |
| 新闻 | SearchService | 搜索引擎通用 |

## 数据采集策略

### 每日采集（交易日）

```
时间       任务              数据量
16:30     A股日线行情         ~5000条
16:35     资金流向数据         ~5000条
16:40     融资融券数据         ~3000条
16:45     北向资金数据         ~2000条
17:00     新闻搜索(Top50)     ~250条（50只×5维度）
```

### 定期采集

| 频率 | 数据 | 说明 |
|------|------|------|
| 每日 | 日线行情、资金流向 | 收盘后采集 |
| 每周 | 股票列表更新 | 新股/退市 |
| 每季 | 财务数据 | 季报发布后 |
| 每月 | 估值分位数 | 重新计算历史分位 |

## 数据标准化

所有数据源输出统一为标准 DataFrame 格式：

```python
# 日线行情标准格式
columns = ["date", "open", "high", "low", "close", "volume", "amount", "turnover_rate"]

# 财务数据标准格式
columns = ["report_date", "revenue", "net_profit", "roe", "roa",
           "gross_margin", "net_margin", "debt_ratio", "current_ratio"]

# 资金流向标准格式
columns = ["date", "main_net_inflow", "main_net_inflow_pct",
           "huge_net_inflow", "large_net_inflow", "medium_net_inflow", "small_net_inflow"]
```

## 数据质量保障

1. **完整性检查**：缺失交易日检测
2. **合理性检查**：涨跌幅异常值过滤
3. **一致性检查**：多源数据交叉验证
4. **时效性检查**：数据更新时间戳标记
