"""
股票搜索服务

支持模糊输入 → 意图识别 → 股票代码解析
例: "港股智谱" → market=HK, keyword="智谱" → 搜索匹配

搜索源优先级:
1. A股: baostock (TCP协议, 不受代理影响)
2. 港股/美股: 新浪搜索 (支持中文, 覆盖全市场)
3. Fallback: Yahoo Finance (英文搜索)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from loguru import logger


@dataclass
class StockMatch:
    symbol: str       # 如 300054.SZ, 09888.HK, BABA
    name: str         # 股票名称
    market: str       # A / HK / US


# 市场关键词映射
_MARKET_KEYWORDS = {
    "A": ["a股", "A股", "沪", "深", "创业板", "科创板", "北交所"],
    "HK": ["港股", "港", "香港", "恒生", "hk"],
    "US": ["美股", "美", "纳斯达克", "纽约", "us"],
}


def parse_intent(query: str) -> tuple[Optional[str], str]:
    """从自然语言输入解析市场和关键词

    Returns:
        (market_or_none, keyword)
    """
    query = query.strip()

    # 1. 已经是标准股票代码？直接返回
    # A股: 6位数字 或 6位数字.SZ/.SH
    if re.match(r"^\d{6}(\.[A-Z]{2})?$", query):
        code = query[:6]
        if code.startswith("6"):
            return "A", code
        return "A", code

    # 港股: 5位数字 或 5位数字.HK
    if re.match(r"^\d{5}(\.HK)?$", query, re.IGNORECASE):
        return "HK", query.replace(".HK", "").replace(".hk", "")

    # 美股: 纯字母 ticker
    if re.match(r"^[A-Za-z]{1,5}$", query):
        return "US", query.upper()

    # 2. 提取市场关键词
    detected_market = None
    keyword = query
    for market, kws in _MARKET_KEYWORDS.items():
        for kw in kws:
            if kw in query:
                detected_market = market
                keyword = query.replace(kw, "").strip()
                break
        if detected_market:
            break

    return detected_market, keyword


# ==================== A股搜索（baostock TCP） ====================

_a_share_cache: Optional[pd.DataFrame] = None
_a_share_cache_time: float = 0
_CACHE_TTL = 86400  # 24小时


def _get_a_share_list() -> pd.DataFrame:
    """获取A股代码名称列表（baostock，缓存24h）"""
    global _a_share_cache, _a_share_cache_time

    if _a_share_cache is not None and (time.time() - _a_share_cache_time) < _CACHE_TTL:
        return _a_share_cache

    try:
        import baostock as bs
        bs.login()
        rs = bs.query_stock_basic()
        rows = []
        while (rs.error_code == "0") & rs.next():
            rows.append(rs.get_row_data())
        bs.logout()

        if rows:
            df = pd.DataFrame(rows, columns=rs.fields)
            df = df[df["type"] == "1"]  # 只保留股票
            _a_share_cache = df
            _a_share_cache_time = time.time()
            logger.info(f"[搜索] A股列表缓存: {len(df)} 只")
            return df
    except Exception as e:
        logger.warning(f"[搜索] baostock A股列表失败: {e}")

    return _a_share_cache if _a_share_cache is not None else pd.DataFrame()


def search_a_shares(keyword: str, limit: int = 10) -> list[StockMatch]:
    """搜索A股"""
    df = _get_a_share_list()
    if df.empty:
        return []

    # 按代码或名称模糊匹配
    mask = (
        df["code"].str.contains(keyword, case=False, na=False)
        | df["code_name"].str.contains(keyword, case=False, na=False)
    )
    matches = df[mask].head(limit)

    results = []
    for _, row in matches.iterrows():
        code = row["code"]  # sh.600000 or sz.300054
        pure_code = code.split(".")[1] if "." in code else code
        suffix = ".SH" if code.startswith("sh") else ".SZ"
        results.append(StockMatch(
            symbol=f"{pure_code}{suffix}",
            name=row["code_name"],
            market="A",
        ))
    return results


# ==================== 新浪搜索（支持中文，覆盖 A/HK/US） ====================

# 新浪 type 码 → 市场映射
_SINA_TYPE_MAP = {
    "11": "A",   # 沪A
    "12": "A",   # 深A (rarely used, 一般也是11)
    "31": "HK",  # 港股主板
    "32": "HK",  # 港股衍生品（窝轮/牛熊证）
    "41": "US",  # 美股
    "71": None,   # 外汇 → 跳过
    "81": None,   # 可转债 → 跳过
    "103": None,  # 伦敦/欧洲 → 跳过
    "21": None,   # 基金 → 跳过
    "201": None,  # 基金 → 跳过
}


def _sina_search(keyword: str, market: Optional[str] = None, limit: int = 10) -> list[StockMatch]:
    """调用新浪股票搜索 API（支持中文，返回 A/HK/US 全市场结果）"""
    import requests

    try:
        resp = requests.get(
            "https://suggest3.sinajs.cn/suggest/type=&key=" + keyword,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        if resp.status_code != 200:
            return []

        # 响应是 GBK 编码的 JS 变量
        text = resp.content.decode("gbk", errors="replace")
        # 格式: var suggestvalue="name,type,code,code2,display_name,...;name2,..."
        match = re.search(r'"(.+)"', text)
        if not match:
            return []

        raw = match.group(1)
        if not raw:
            return []

        results = []
        for item in raw.split(";"):
            parts = item.split(",")
            if len(parts) < 5:
                continue

            name = parts[0]
            sina_type = parts[1]
            code = parts[2]
            display_name = parts[4] if parts[4] else name

            # 映射市场
            stock_market = _SINA_TYPE_MAP.get(sina_type)
            if stock_market is None:
                continue  # 跳过外汇、基金、衍生品等

            # 过滤市场
            if market and stock_market != market:
                continue

            # 跳过港股衍生品（窝轮/牛熊证，type=32）
            if sina_type == "32":
                continue

            # 构造标准 symbol
            symbol = _sina_to_symbol(code, stock_market, sina_type)

            results.append(StockMatch(
                symbol=symbol,
                name=display_name,
                market=stock_market,
            ))
            if len(results) >= limit:
                break

        return results
    except Exception as e:
        logger.warning(f"[搜索] 新浪搜索失败: {e}")
        return []


def _sina_to_symbol(code: str, market: str, sina_type: str) -> str:
    """新浪代码 → 标准 symbol"""
    if market == "HK":
        # 港股: 02513 → 02513.HK
        return f"{code.zfill(5)}.HK"
    elif market == "US":
        # 美股: baba → BABA
        return code.upper()
    elif market == "A":
        # A股: sz300054 → 300054.SZ, sh600000 → 600000.SH
        if code.startswith("sh") or code.startswith("sz"):
            prefix = code[:2].upper()
            num = code[2:]
            return f"{num}.{prefix}"
        # 纯数字的情况
        if code.startswith("6"):
            return f"{code}.SH"
        return f"{code}.SZ"
    return code


# ==================== 港股/美股搜索 ====================

def search_hk_us(keyword: str, market: Optional[str] = None, limit: int = 10) -> list[StockMatch]:
    """搜索港股/美股（新浪搜索 → 支持中文 + 全市场覆盖）"""
    results = _sina_search(keyword, market=market, limit=limit)
    if results:
        return results

    logger.debug(f"[搜索] 新浪无结果，尝试 Yahoo Finance: {keyword}")
    return _yahoo_search_fallback(keyword, market, limit)


def _yahoo_search_fallback(query: str, market: Optional[str], limit: int) -> list[StockMatch]:
    """Yahoo Finance 作为 fallback（用于英文关键词搜索）"""
    import requests

    try:
        resp = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={
                "q": query,
                "quotesCount": limit * 2,
                "newsCount": 0,
                "enableFuzzyQuery": True,
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        if resp.status_code != 200:
            return []

        data = resp.json()
        results = []
        for q in data.get("quotes", []):
            sym = q.get("symbol", "")
            name = q.get("shortname") or q.get("longname") or sym
            exchange = q.get("exchange", "")
            exch_disp = q.get("exchDisp", "")

            stock_market = _classify_market(sym, exchange, exch_disp)
            if market and stock_market != market:
                continue

            results.append(StockMatch(symbol=sym, name=name, market=stock_market))
            if len(results) >= limit:
                break

        return results
    except Exception as e:
        logger.warning(f"[搜索] Yahoo search 失败: {e}")
        return []


def _classify_market(symbol: str, exchange: str, exch_disp: str) -> str:
    """根据交易所判断市场"""
    hk_exchanges = {"HKG", "HKSE", "Hong Kong"}
    us_exchanges = {"NYQ", "NMS", "NGM", "PCX", "NYSE", "NASDAQ", "AMEX", "New York", "NasdaqGS", "NasdaqGM", "NasdaqCM"}
    cn_exchanges = {"SHH", "SHZ", "Shanghai", "Shenzhen"}

    all_info = f"{exchange} {exch_disp}"
    if any(x in all_info for x in hk_exchanges) or symbol.endswith(".HK"):
        return "HK"
    if any(x in all_info for x in cn_exchanges) or symbol.endswith((".SS", ".SZ")):
        return "A"
    if any(x in all_info for x in us_exchanges):
        return "US"
    return "US"  # 默认美股


# ==================== 统一搜索入口 ====================

def search_stocks(query: str, limit: int = 10) -> list[StockMatch]:
    """统一搜索入口

    支持:
    - "300054" → 直接匹配A股代码
    - "鼎龙" → 搜索A股名称
    - "港股智谱" → 识别港股市场 + 搜索"智谱"
    - "BABA" → 识别美股 ticker
    - "智谱" (无市场前缀) → 搜索所有市场
    """
    market, keyword = parse_intent(query)

    if not keyword:
        return []

    # 新浪搜索覆盖全市场（A/HK/US），优先使用
    results = _sina_search(keyword, market=market, limit=limit)

    # 如果新浪无结果且是A股关键词，再走 baostock
    if not results and (market == "A" or market is None):
        results.extend(search_a_shares(keyword, limit))

    return results[:limit]
