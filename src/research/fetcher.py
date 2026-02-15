"""
研报自动采集

从东方财富（通过 akshare）自动获取最新券商研报，下载 PDF 到临时目录。

核心流程：
1. fetch_report_list()  — akshare stock_research_report_em 获取研报列表
2. download_top_reports() — 下载最新 N 份 PDF
3. auto_fetch_research()  — 一站式入口：采集 + 下载 + 返回目录路径
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import requests
from loguru import logger


# 默认下载最新 3 份研报（平衡覆盖度与 token 预算）
DEFAULT_TOP_N = 3
# 单个 PDF 最大 5MB，超过跳过
MAX_PDF_SIZE = 5 * 1024 * 1024
# 下载超时
DOWNLOAD_TIMEOUT = 30


def fetch_report_list(symbol: str) -> list[dict]:
    """获取个股研报列表

    Args:
        symbol: 股票代码（如 '300054' 或 '300054.SZ'）

    Returns:
        研报列表，每项包含 title, institution, rating, date, pdf_url, eps_forecasts
    """
    import akshare as ak

    code = symbol.split(".")[0]

    try:
        df = ak.stock_research_report_em(symbol=code)
    except Exception as e:
        logger.warning(f"[研报采集] akshare 获取研报列表失败: {e}")
        return []

    if df is None or df.empty:
        logger.info(f"[研报采集] {code} 无研报数据")
        return []

    reports = []
    for _, row in df.iterrows():
        pdf_url = str(row.get("报告PDF链接", "")).strip()
        if not pdf_url:
            continue

        # 提取 EPS 预测列
        eps = {}
        for col in df.columns:
            if "盈利预测-收益" in col:
                year = col.split("-")[0]
                val = row.get(col)
                if val is not None and str(val).strip():
                    try:
                        eps[year] = float(val)
                    except (ValueError, TypeError):
                        pass

        reports.append({
            "title": str(row.get("报告名称", "")),
            "institution": str(row.get("机构", "")),
            "rating": str(row.get("东财评级", "")),
            "date": str(row.get("日期", ""))[:10],
            "pdf_url": pdf_url,
            "eps_forecasts": eps,
        })

    logger.info(f"[研报采集] {code} 获取 {len(reports)} 份研报")
    return reports


def download_top_reports(
    reports: list[dict],
    output_dir: Path,
    top_n: int = DEFAULT_TOP_N,
) -> list[Path]:
    """下载最新 N 份研报 PDF

    Args:
        reports: fetch_report_list() 返回的列表（已按日期降序）
        output_dir: PDF 保存目录
        top_n: 下载数量

    Returns:
        成功下载的 PDF 路径列表
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for i, report in enumerate(reports[:top_n]):
        url = report["pdf_url"]
        inst = report["institution"].replace("/", "_")
        date = report["date"].replace("-", "")
        filename = f"{date}_{inst}_{i}.pdf"
        filepath = output_dir / filename

        try:
            resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True)
            resp.raise_for_status()

            # 检查大小
            content_length = int(resp.headers.get("Content-Length", 0))
            if content_length > MAX_PDF_SIZE:
                logger.warning(
                    f"[研报采集] PDF 过大({content_length/1024/1024:.1f}MB)，跳过: {report['title'][:30]}"
                )
                continue

            # 流式写入
            total = 0
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    total += len(chunk)
                    if total > MAX_PDF_SIZE:
                        logger.warning(f"[研报采集] PDF 下载超限，截断: {filename}")
                        break
                    f.write(chunk)

            downloaded.append(filepath)
            logger.info(
                f"[研报采集] 下载成功: {report['institution']} {report['date']} "
                f"({total/1024:.0f}KB) {report['title'][:30]}"
            )

        except Exception as e:
            logger.warning(f"[研报采集] PDF 下载失败: {url[:60]}... | {e}")

    return downloaded


def auto_fetch_research(
    symbol: str,
    top_n: int = DEFAULT_TOP_N,
    output_dir: Optional[Path] = None,
) -> tuple[Optional[str], dict]:
    """一站式研报自动采集

    Args:
        symbol: 股票代码
        top_n: 下载最新 N 份
        output_dir: 指定输出目录（默认使用临时目录）

    Returns:
        (research_dir, report_meta) 元组:
        - research_dir: PDF 所在目录路径（无研报时为 None）
        - report_meta: 研报元数据（评级分布、EPS 预测等），可注入 StockData.info
    """
    reports = fetch_report_list(symbol)
    if not reports:
        return None, {}

    # 构建元数据
    meta = _build_report_meta(reports)

    # 下载 PDF
    if output_dir is None:
        code = symbol.split(".")[0]
        output_dir = Path(tempfile.mkdtemp(prefix=f"research_{code}_"))

    downloaded = download_top_reports(reports, output_dir, top_n=top_n)

    if not downloaded:
        logger.warning(f"[研报采集] {symbol} PDF 全部下载失败")
        return None, meta

    logger.info(f"[研报采集] {symbol} 完成: {len(downloaded)}/{top_n} PDF → {output_dir}")
    return str(output_dir), meta


def _build_report_meta(reports: list[dict]) -> dict:
    """从研报列表构建结构化元数据

    与现有 fetch_research_reports() 返回格式兼容，可补充/覆盖。
    """
    if not reports:
        return {}

    # 评级分布
    rating_dist: dict[str, int] = {}
    institutions = set()
    for r in reports:
        rating = r.get("rating", "")
        if rating:
            rating_dist[rating] = rating_dist.get(rating, 0) + 1
        inst = r.get("institution", "")
        if inst:
            institutions.add(inst)

    # 最新 EPS 预测（取最近 5 份有 EPS 数据的）
    eps_forecasts = []
    for r in reports[:10]:
        if r.get("eps_forecasts"):
            eps_forecasts.append({
                "institution": r["institution"],
                "date": r["date"],
                **r["eps_forecasts"],
            })
        if len(eps_forecasts) >= 5:
            break

    # 近期标题
    recent_titles = [
        {
            "title": r["title"],
            "institution": r["institution"],
            "rating": r["rating"],
            "date": r["date"],
        }
        for r in reports[:8]
    ]

    # 评级变动（最近 vs 历史）
    rating_changes: dict[str, int] = {}
    recent = reports[:5]
    older = reports[5:20]
    recent_ratings = {r["institution"]: r["rating"] for r in recent if r.get("rating")}
    older_ratings = {r["institution"]: r["rating"] for r in older if r.get("rating")}
    for inst, new_rating in recent_ratings.items():
        old_rating = older_ratings.get(inst)
        if old_rating and old_rating != new_rating:
            change = f"{old_rating}→{new_rating}"
            rating_changes[change] = rating_changes.get(change, 0) + 1

    return {
        "rating_distribution": rating_dist,
        "total_reports": len(reports),
        "latest_rating": reports[0].get("rating", "") if reports else "",
        "coverage_count": len(institutions),
        "institutions": sorted(institutions)[:10],
        "eps_forecasts": eps_forecasts,
        "recent_titles": recent_titles,
        "rating_changes": rating_changes,
    }
