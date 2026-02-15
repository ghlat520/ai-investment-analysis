"""
市场热点数据采集

AKShare API 采集市场热点数据（纯代码，无LLM，~30秒）：
- 人气榜（stock_hot_rank_em）
- 概念板块（stock_board_concept_name_em）
- 行业板块（stock_board_industry_name_em）
- 政策新闻（news_cctv）
- 概念板块成分股（stock_board_concept_cons_em）
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from loguru import logger


def collect_market_snapshot() -> dict[str, Any]:
    """聚合多个AKShare API，返回市场快照

    Returns:
        dict with keys:
          hot_stocks: 人气榜 (list[dict])
          concept_boards: 概念板块涨幅前30 (list[dict])
          industry_boards: 行业板块涨幅前20 (list[dict])
          cctv_news: 近3天政策新闻 (list[dict])
    """
    import akshare as ak

    snapshot: dict[str, Any] = {
        "hot_stocks": [],
        "concept_boards": [],
        "industry_boards": [],
        "cctv_news": [],
        "zt_pool": [],
    }

    # 1. 人气榜
    try:
        logger.info("[采集] 人气榜...")
        hot_df = ak.stock_hot_rank_em()
        if hot_df is not None and not hot_df.empty:
            snapshot["hot_stocks"] = hot_df.head(50).to_dict("records")
            logger.info(f"[采集] 人气榜: {len(snapshot['hot_stocks'])}只")
    except Exception as e:
        logger.warning(f"[采集] 人气榜失败: {e}")

    # 2. 概念板块（按涨跌幅排序取前30）
    try:
        logger.info("[采集] 概念板块...")
        concept_df = ak.stock_board_concept_name_em()
        if concept_df is not None and not concept_df.empty:
            # 按涨跌幅降序
            if "涨跌幅" in concept_df.columns:
                concept_df = concept_df.sort_values("涨跌幅", ascending=False)
            snapshot["concept_boards"] = concept_df.head(30).to_dict("records")
            logger.info(f"[采集] 概念板块: {len(snapshot['concept_boards'])}个")
    except Exception as e:
        logger.warning(f"[采集] 概念板块失败: {e}")

    # 3. 行业板块（按涨跌幅排序取前20）
    try:
        logger.info("[采集] 行业板块...")
        industry_df = ak.stock_board_industry_name_em()
        if industry_df is not None and not industry_df.empty:
            if "涨跌幅" in industry_df.columns:
                industry_df = industry_df.sort_values("涨跌幅", ascending=False)
            snapshot["industry_boards"] = industry_df.head(20).to_dict("records")
            logger.info(f"[采集] 行业板块: {len(snapshot['industry_boards'])}个")
    except Exception as e:
        logger.warning(f"[采集] 行业板块失败: {e}")

    # 4. 涨停池（今日涨停股）
    try:
        logger.info("[采集] 涨停池...")
        zt_df = ak.stock_zt_pool_em(date=date.today().strftime("%Y%m%d"))
        if zt_df is not None and not zt_df.empty:
            snapshot["zt_pool"] = zt_df.to_dict("records")
            logger.info(f"[采集] 涨停池: {len(snapshot['zt_pool'])}只")
    except Exception as e:
        logger.warning(f"[采集] 涨停池失败: {e}")

    # 5. CCTV新闻（近3天）
    try:
        logger.info("[采集] 政策新闻...")
        today = date.today()
        all_news = []
        for days_ago in range(3):
            d = today - timedelta(days=days_ago)
            try:
                news_df = ak.news_cctv(date=d.strftime("%Y%m%d"))
                if news_df is not None and not news_df.empty:
                    records = news_df.to_dict("records")
                    all_news.extend(records)
            except Exception:
                pass
        snapshot["cctv_news"] = all_news[:50]
        logger.info(f"[采集] 政策新闻: {len(snapshot['cctv_news'])}条")
    except Exception as e:
        logger.warning(f"[采集] 政策新闻失败: {e}")

    return snapshot


def rank_hot_concepts(snapshot: dict[str, Any], top_n: int = 8) -> list[dict[str, Any]]:
    """对概念板块进行综合排序

    排序公式：涨幅×0.4 + 龙头涨幅×0.3 + 人气榜重叠×0.3

    Args:
        snapshot: collect_market_snapshot() 的返回值
        top_n: 返回前N个概念

    Returns:
        排序后的概念列表，每个包含 name, code, change_pct, leader_change, hot_overlap, score
    """
    concept_boards = snapshot.get("concept_boards", [])
    hot_stocks = snapshot.get("hot_stocks", [])
    zt_pool = snapshot.get("zt_pool", [])

    if not concept_boards:
        return []

    # 提取人气榜股票代码集合
    hot_codes = set()
    for stock in hot_stocks:
        code = stock.get("股票代码", stock.get("代码", ""))
        if code:
            hot_codes.add(str(code))

    # 提取涨停股代码集合
    zt_codes = set()
    for stock in zt_pool:
        code = stock.get("代码", stock.get("股票代码", ""))
        if code:
            zt_codes.add(str(code))

    ranked = []
    for board in concept_boards:
        name = board.get("板块名称", "")
        code = board.get("板块代码", "")
        change_pct = _safe_float(board.get("涨跌幅", 0))
        leader_change = _safe_float(board.get("领涨股票-涨跌幅", 0))
        leader_name = board.get("领涨股票", "")

        # 热度重叠：概念板块的领涨股是否在人气榜/涨停池中
        leader_code = board.get("领涨股票-代码", "")
        hot_overlap = 1.0 if str(leader_code) in hot_codes else 0.0
        zt_overlap = 1.0 if str(leader_code) in zt_codes else 0.0

        # 归一化评分：涨幅×0.35 + 龙头涨幅×0.25 + 人气重叠×0.2 + 涨停重叠×0.2
        score = (change_pct * 0.35 + leader_change * 0.25 +
                 hot_overlap * 30 * 0.2 + zt_overlap * 30 * 0.2)

        ranked.append({
            "name": name,
            "code": code,
            "change_pct": round(change_pct, 2),
            "leader_name": leader_name,
            "leader_change": round(leader_change, 2),
            "hot_overlap": hot_overlap,
            "score": round(score, 2),
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    result = ranked[:top_n]
    logger.info(f"[排序] 热门概念 Top {top_n}: {[c['name'] for c in result]}")
    return result


def enrich_concepts_with_stocks(ranked_concepts: list[dict[str, Any]], stocks_per_concept: int = 10) -> list[dict[str, Any]]:
    """为每个热门概念拉取成分股

    Args:
        ranked_concepts: rank_hot_concepts() 的返回值
        stocks_per_concept: 每个概念取前N只成分股

    Returns:
        enriched concepts，每个增加 stocks 字段
    """
    import akshare as ak

    enriched = []
    for concept in ranked_concepts:
        concept_copy = dict(concept)
        concept_copy["stocks"] = []

        name = concept.get("name", "")
        try:
            cons_df = ak.stock_board_concept_cons_em(symbol=name)
            if cons_df is not None and not cons_df.empty:
                # 按总市值排序取前N
                if "总市值" in cons_df.columns:
                    cons_df = cons_df.sort_values("总市值", ascending=False)

                stocks = []
                for _, row in cons_df.head(stocks_per_concept).iterrows():
                    code = str(row.get("代码", ""))
                    stock_name = str(row.get("名称", ""))
                    # 构造标准股票代码
                    symbol = _to_standard_symbol(code)
                    stocks.append({
                        "symbol": symbol,
                        "name": stock_name,
                        "code": code,
                        "change_pct": _safe_float(row.get("涨跌幅", 0)),
                        "market_cap": _safe_float(row.get("总市值", 0)),
                    })
                concept_copy["stocks"] = stocks
                logger.info(f"[采集] 概念「{name}」成分股: {len(stocks)}只")
        except Exception as e:
            logger.warning(f"[采集] 概念「{name}」成分股失败: {e}")

        enriched.append(concept_copy)

    return enriched


def _safe_float(val: Any) -> float:
    """安全转换为float"""
    if val is None or val == "" or val == "-":
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _to_standard_symbol(code: str) -> str:
    """将纯数字代码转换为标准格式（如 000001 → 000001.SZ）"""
    code = str(code).strip()
    if "." in code:
        return code
    if code.startswith(("6",)):
        return f"{code}.SH"
    elif code.startswith(("0", "3")):
        return f"{code}.SZ"
    elif code.startswith(("8", "4")):
        return f"{code}.BJ"
    return f"{code}.SZ"
