"""
批量分析服务

串行执行多只股票的9维深度分析，支持断点续传和汇总报告。

约束：
- Ollama N=1（Apple Silicon GPU不能并行）→ 股票必须串行
- 每股~11min → 10只≈2h
- API限流 → 股间间隔5s
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger


def _default_output_dir() -> Path:
    """默认输出目录"""
    d = Path("output/batch")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_batch_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ─── 进度管理 ──────────────────────────────────────────────


class BatchProgress:
    """断点续传进度管理"""

    def __init__(self, progress_file: Path):
        self._file = progress_file
        self.batch_id: str = ""
        self.total: int = 0
        self.symbols: list[str] = []
        self.completed: dict[str, dict] = {}   # symbol → {score, action, ...}
        self.failed: dict[str, str] = {}       # symbol → error_msg
        self.started_at: str = ""
        self.last_updated: str = ""

    def save(self) -> None:
        self.last_updated = datetime.now().isoformat()
        self._file.write_text(json.dumps({
            "batch_id": self.batch_id,
            "total": self.total,
            "symbols": self.symbols,
            "completed": self.completed,
            "failed": self.failed,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, progress_file: Path) -> "BatchProgress":
        p = cls(progress_file)
        if progress_file.exists():
            data = json.loads(progress_file.read_text(encoding="utf-8"))
            p.batch_id = data.get("batch_id", "")
            p.total = data.get("total", 0)
            p.symbols = data.get("symbols", [])
            p.completed = data.get("completed", {})
            p.failed = data.get("failed", {})
            p.started_at = data.get("started_at", "")
            p.last_updated = data.get("last_updated", "")
        return p

    @property
    def pending(self) -> list[str]:
        done = set(self.completed) | set(self.failed)
        return [s for s in self.symbols if s not in done]


# ─── 批量执行 ──────────────────────────────────────────────


def batch_analyze(
    symbols: list[str],
    output_dir: Optional[Path] = None,
    resume: bool = False,
    research_dir: Optional[str] = None,
    on_stock_done: Optional[Callable[[str, int, int, dict], None]] = None,
    cooldown: int = 5,
) -> dict[str, Any]:
    """批量分析多只股票

    Args:
        symbols: 股票代码列表
        output_dir: 输出目录
        resume: 是否断点续传
        research_dir: 研报PDF目录
        on_stock_done: 每只股票完成后的回调 (symbol, index, total, result_summary)
        cooldown: 股间冷却秒数

    Returns:
        {"batch_id": str, "completed": dict, "failed": dict, "summary": str}
    """
    from src.services.analysis_service import collect_stock_data, run_analysis

    if not output_dir:
        output_dir = _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 进度管理
    progress_file = output_dir / "progress.json"
    if resume and progress_file.exists():
        progress = BatchProgress.load(progress_file)
        # 合并新增的股票（如果有）
        for s in symbols:
            if s not in progress.symbols:
                progress.symbols.append(s)
        progress.total = len(progress.symbols)
        logger.info(
            f"[batch] 断点续传: {len(progress.completed)}已完成, "
            f"{len(progress.failed)}失败, {len(progress.pending)}待执行"
        )
    else:
        progress = BatchProgress(progress_file)
        progress.batch_id = _make_batch_id()
        progress.symbols = list(symbols)
        progress.total = len(symbols)
        progress.started_at = datetime.now().isoformat()

    pending = progress.pending
    if not pending:
        logger.info("[batch] 所有股票已完成，无需执行")
        summary = _generate_summary(progress, output_dir)
        return {
            "batch_id": progress.batch_id,
            "completed": progress.completed,
            "failed": progress.failed,
            "summary": summary,
        }

    logger.info(
        f"[batch] 开始批量分析: {len(pending)}只待执行, "
        f"预计耗时~{len(pending) * 11}min"
    )

    for i, symbol in enumerate(pending):
        idx = len(progress.completed) + len(progress.failed) + 1
        logger.info(f"\n{'='*60}")
        logger.info(f"[batch] [{idx}/{progress.total}] 开始分析: {symbol}")
        logger.info(f"{'='*60}")

        t0 = time.time()
        try:
            # 1. 数据采集
            stock_data = collect_stock_data(symbol, research_dir=research_dir)

            # 2. 9-Agent分析
            result = run_analysis(stock_data)

            elapsed = time.time() - t0
            fusion = result.get("fusion")
            report = result.get("report", "")
            signals = result.get("signals", [])

            # 3. 持久化到DB
            try:
                from src.data.storage.persist import persist_analysis
                persist_analysis(result)
            except Exception as e:
                logger.warning(f"[batch] 持久化失败（非致命）: {e}")

            # 4. 保存单股报告
            report_file = output_dir / f"{symbol.replace('.', '_')}.md"
            report_file.write_text(report, encoding="utf-8")

            # 5. 记录结果
            # 目标价：fusion层 → valuation agent metadata → 0
            target_price = 0.0
            if fusion and fusion.target_prices:
                target_price = fusion.target_prices.get(
                    "base",
                    fusion.target_prices.get("probability_weighted", 0),
                )
            if not target_price:
                for s in signals:
                    if s.agent_name == "valuation" and isinstance(s.metadata, dict):
                        raw = s.metadata.get("raw_response", {})
                        if isinstance(raw, dict) and raw.get("target_prices"):
                            target_price = raw["target_prices"].get(
                                "base",
                                raw["target_prices"].get("probability_weighted", 0),
                            )
                            break

            result_summary = {
                "score": fusion.final_score if fusion else 0,
                "action": fusion.final_action if fusion else "未知",
                "confidence": round(fusion.confidence, 2) if fusion else 0,
                "name": stock_data.name,
                "agents_ok": len(signals),
                "elapsed_s": int(elapsed),
                "target_price": round(target_price, 2) if target_price else 0,
            }
            progress.completed[symbol] = result_summary
            progress.save()

            logger.info(
                f"[batch] [{idx}/{progress.total}] {stock_data.name}({symbol}): "
                f"{fusion.final_score:+d}分 {fusion.final_action} "
                f"({elapsed:.0f}s, {len(signals)} agents)"
            )

            if on_stock_done:
                on_stock_done(symbol, idx, progress.total, result_summary)

        except Exception as e:
            elapsed = time.time() - t0
            progress.failed[symbol] = str(e)
            progress.save()
            logger.error(
                f"[batch] [{idx}/{progress.total}] {symbol} 失败 ({elapsed:.0f}s): {e}"
            )

        # 股间冷却（最后一只不需要）
        if i < len(pending) - 1:
            logger.info(f"[batch] 冷却 {cooldown}s...")
            time.sleep(cooldown)

    # 生成汇总报告
    summary = _generate_summary(progress, output_dir)

    return {
        "batch_id": progress.batch_id,
        "completed": progress.completed,
        "failed": progress.failed,
        "summary": summary,
    }


# ─── 汇总报告 ──────────────────────────────────────────────


def _generate_summary(progress: BatchProgress, output_dir: Path) -> str:
    """生成批量分析汇总报告"""
    lines = []
    lines.append(f"# 批量分析汇总报告")
    lines.append(f"")
    lines.append(f"- 批次ID: {progress.batch_id}")
    lines.append(f"- 开始时间: {progress.started_at}")
    lines.append(f"- 完成时间: {progress.last_updated}")
    lines.append(f"- 总计: {progress.total}只 | "
                 f"成功: {len(progress.completed)} | 失败: {len(progress.failed)}")
    lines.append("")

    # 排名表（按score降序）
    if progress.completed:
        sorted_results = sorted(
            progress.completed.items(),
            key=lambda x: x[1].get("score", 0),
            reverse=True,
        )

        lines.append("## 评分排名")
        lines.append("")
        lines.append(f"| 排名 | 代码 | 名称 | 评分 | 操作建议 | 置信度 | 目标价 | 耗时 |")
        lines.append(f"|------|------|------|------|---------|--------|--------|------|")

        for rank, (symbol, r) in enumerate(sorted_results, 1):
            target = f"{r.get('target_price', 0):.1f}" if r.get("target_price") else "-"
            lines.append(
                f"| {rank} | {symbol} | {r.get('name', '')} | "
                f"{r.get('score', 0):+d} | {r.get('action', '')} | "
                f"{r.get('confidence', 0):.0%} | {target} | "
                f"{r.get('elapsed_s', 0)}s |"
            )

        lines.append("")

        # 推荐清单
        recommended = [
            (s, r) for s, r in sorted_results
            if r.get("score", 0) >= 30
        ]
        if recommended:
            lines.append("## 推荐标的（score >= +30）")
            lines.append("")
            for symbol, r in recommended:
                lines.append(
                    f"- **{r.get('name', '')}({symbol})**: "
                    f"{r.get('score', 0):+d}分, {r.get('action', '')}"
                )
            lines.append("")

        # 警示清单
        warnings = [
            (s, r) for s, r in sorted_results
            if r.get("score", 0) <= -30
        ]
        if warnings:
            lines.append("## 风险标的（score <= -30）")
            lines.append("")
            for symbol, r in warnings:
                lines.append(
                    f"- **{r.get('name', '')}({symbol})**: "
                    f"{r.get('score', 0):+d}分, {r.get('action', '')}"
                )
            lines.append("")

    # 失败列表
    if progress.failed:
        lines.append("## 失败记录")
        lines.append("")
        for symbol, err in progress.failed.items():
            lines.append(f"- {symbol}: {err}")
        lines.append("")

    summary = "\n".join(lines)

    # 保存汇总文件
    summary_file = output_dir / "summary.md"
    summary_file.write_text(summary, encoding="utf-8")
    logger.info(f"[batch] 汇总报告已保存: {summary_file}")

    return summary


def parse_stock_list(stocks_str: Optional[str] = None, file_path: Optional[str] = None) -> list[str]:
    """解析股票列表

    支持：
    - 逗号分隔字符串: "300054.SZ,600150.SH"
    - 文件（每行一个代码，支持#注释）
    """
    symbols = []

    if stocks_str:
        for s in stocks_str.split(","):
            s = s.strip()
            if s:
                symbols.append(s)

    if file_path:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"股票列表文件不存在: {file_path}")
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                # 支持 "300054.SZ  鼎龙股份" 格式（只取第一列）
                symbol = line.split()[0].strip()
                if symbol:
                    symbols.append(symbol)

    # 去重保序
    seen = set()
    unique = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique
