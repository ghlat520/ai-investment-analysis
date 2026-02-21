"""
预测验证服务

实现预测验证闭环：
1. 记录预测（分歧点、关键假设）
2. 自动回测（30天后）
3. 错误分析（哪个Agent判断错了）
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Optional
from uuid import uuid4

from loguru import logger
from sqlalchemy import and_, desc, or_

from src.data.storage.database import get_database
from src.data.storage.models import (
    AnalysisResult,
    FusionDecisionRecord,
    PredictionRecord,
    StockDailyQuote,
)


class ValidationService:
    """预测验证服务"""

    # 默认验证周期（天）
    DEFAULT_VERIFICATION_DAYS = 30

    def __init__(self):
        self.db = get_database()
        self.db.create_tables()

    def record_prediction(
        self,
        run_id: str,
        symbol: str,
        fusion: Any,
        signals: list[Any],
        divergence_points: list[dict],
        current_price: Optional[float] = None,
        verification_days: int = DEFAULT_VERIFICATION_DAYS,
    ) -> PredictionRecord:
        """记录预测

        Args:
            run_id: 分析运行ID
            symbol: 股票代码
            fusion: 融合决策结果
            signals: 各Agent信号列表
            divergence_points: 分歧点列表 [{point, verify_method, verify_date}]
            current_price: 当前价格
            verification_days: 验证周期（天）

        Returns:
            PredictionRecord
        """
        prediction_date = date.today()
        verification_date = prediction_date + timedelta(days=verification_days)

        # 提取目标价
        target_prices = fusion.target_prices or {}
        agent_scores = {s.agent_name: s.signal_score for s in signals}

        with self.db.session() as session:
            record = PredictionRecord(
                run_id=run_id,
                symbol=symbol,
                prediction_date=prediction_date,
                verification_date=verification_date,
                final_score=fusion.final_score,
                final_action=fusion.final_action,
                confidence=fusion.confidence,
                target_price_conservative=target_prices.get("conservative"),
                target_price_base=target_prices.get("base"),
                target_price_optimistic=target_prices.get("optimistic"),
                current_price=current_price,
                divergence_points=divergence_points,
                agent_scores=agent_scores,
                is_verified=False,
            )
            session.add(record)
            session.flush()  # 获取id
            record_id = record.id

        logger.info(
            f"[ValidationService] 记录预测: {symbol} score={fusion.final_score:+d}, "
            f"验证日期={verification_date}, divergence_points={len(divergence_points)}"
        )

        return record

    def get_pending_verifications(self, days_overdue: int = 0) -> list[PredictionRecord]:
        """获取待验证的预测记录

        Args:
            days_overdue: 超过验证日期多少天也算（默认0，只返回当天到期的）

        Returns:
            待验证的预测记录列表
        """
        today = date.today()
        cutoff_date = today - timedelta(days=days_overdue)

        with self.db.session() as session:
            records = (
                session.query(PredictionRecord)
                .filter(
                    and_(
                        PredictionRecord.is_verified == False,
                        PredictionRecord.verification_date <= today,
                        PredictionRecord.verification_date >= cutoff_date,
                    )
                )
                .order_by(PredictionRecord.prediction_date)
                .all()
            )
            # 使对象在session外可用
            return [self._detach(r) for r in records]

    def verify_prediction(
        self,
        record_id: int,
        actual_price: float,
        error_analysis: Optional[dict] = None,
    ) -> dict:
        """验证预测

        Args:
            record_id: 预测记录ID
            actual_price: 实际价格
            error_analysis: 错误分析（可选）

        Returns:
            验证结果
        """
        with self.db.session() as session:
            record = session.query(PredictionRecord).filter_by(id=record_id).first()
            if not record:
                raise ValueError(f"PredictionRecord not found: {record_id}")

            # 计算实际收益率
            if record.current_price and record.current_price > 0:
                actual_return_pct = (actual_price - record.current_price) / record.current_price * 100
            else:
                actual_return_pct = None

            # 判断方向是否正确
            direction_correct = None
            if record.final_score > 0:
                # 看多预测
                direction_correct = actual_return_pct is not None and actual_return_pct > 0
            elif record.final_score < 0:
                # 看空预测
                direction_correct = actual_return_pct is not None and actual_return_pct < 0

            # 判断目标价是否命中
            price_target_hit = "miss"
            if record.target_price_base and actual_price:
                if actual_price >= record.target_price_optimistic:
                    price_target_hit = "optimistic"
                elif actual_price >= record.target_price_base:
                    price_target_hit = "base"
                elif actual_price >= record.target_price_conservative:
                    price_target_hit = "conservative"

            # 计算评分准确性（简单的方向准确率）
            score_accuracy = 1.0 if direction_correct else 0.0

            # 更新记录
            record.is_verified = True
            record.verified_at = datetime.now()
            record.actual_price = actual_price
            record.actual_return_pct = actual_return_pct
            record.price_target_hit = price_target_hit
            record.direction_correct = direction_correct
            record.score_accuracy = score_accuracy
            record.error_analysis = error_analysis
            record.updated_at = datetime.now()

            result = {
                "record_id": record_id,
                "symbol": record.symbol,
                "prediction_date": str(record.prediction_date),
                "final_score": record.final_score,
                "actual_return_pct": actual_return_pct,
                "direction_correct": direction_correct,
                "price_target_hit": price_target_hit,
                "verification_days": (date.today() - record.prediction_date).days,
            }

        logger.info(
            f"[ValidationService] 验证完成: {record.symbol} "
            f"score={record.final_score:+d} return={actual_return_pct:+.2f}% "
            f"direction={'✓' if direction_correct else '✗'}"
        )

        return result

    def analyze_agent_accuracy(
        self,
        record_id: int,
        signals: list[Any],
    ) -> dict:
        """分析各Agent的准确性贡献

        Args:
            record_id: 预测记录ID
            signals: 当时的Agent信号列表

        Returns:
            Agent准确性分析
        """
        with self.db.session() as session:
            record = session.query(PredictionRecord).filter_by(id=record_id).first()
            if not record or not record.is_verified:
                return {}

            agent_accuracy = {}
            actual_return = record.actual_return_pct or 0

            for signal in signals:
                agent_name = signal.agent_name
                score = signal.signal_score

                # 判断该Agent的方向是否正确
                if score > 0:
                    correct = actual_return > 0
                elif score < 0:
                    correct = actual_return < 0
                else:
                    correct = True  # 中性评分不算错

                # 计算贡献度（评分 × 方向一致性）
                contribution = abs(score) * (1 if correct else -1)

                agent_accuracy[agent_name] = {
                    "score": score,
                    "correct": correct,
                    "contribution": contribution,
                }

            # 更新记录
            record.agent_accuracy = agent_accuracy
            record.updated_at = datetime.now()

        return agent_accuracy

    def get_accuracy_statistics(
        self,
        symbol: Optional[str] = None,
        days: int = 90,
        agent_name: Optional[str] = None,
    ) -> dict:
        """获取预测准确性统计

        Args:
            symbol: 股票代码（可选，不传则统计全部）
            days: 统计最近多少天
            agent_name: Agent名称（可选，用于Agent级别统计）

        Returns:
            统计结果
        """
        start_date = date.today() - timedelta(days=days)

        with self.db.session() as session:
            query = session.query(PredictionRecord).filter(
                and_(
                    PredictionRecord.is_verified == True,
                    PredictionRecord.verified_at >= start_date,
                )
            )
            if symbol:
                query = query.filter(PredictionRecord.symbol == symbol)

            records = query.all()

            if not records:
                return {"total": 0, "message": "无已验证的预测记录"}

            # 整体统计
            total = len(records)
            direction_correct_count = sum(1 for r in records if r.direction_correct)
            avg_return = sum(r.actual_return_pct or 0 for r in records) / total

            # 按评分区间统计
            score_bins = {
                "strong_buy (>50)": {"count": 0, "correct": 0, "avg_return": 0},
                "buy (20-50)": {"count": 0, "correct": 0, "avg_return": 0},
                "neutral (-20~20)": {"count": 0, "correct": 0, "avg_return": 0},
                "sell (-50~-20)": {"count": 0, "correct": 0, "avg_return": 0},
                "strong_sell (<-50)": {"count": 0, "correct": 0, "avg_return": 0},
            }

            for r in records:
                if r.final_score > 50:
                    bin_key = "strong_buy (>50)"
                elif r.final_score > 20:
                    bin_key = "buy (20-50)"
                elif r.final_score >= -20:
                    bin_key = "neutral (-20~20)"
                elif r.final_score >= -50:
                    bin_key = "sell (-50~-20)"
                else:
                    bin_key = "strong_sell (<-50)"

                score_bins[bin_key]["count"] += 1
                if r.direction_correct:
                    score_bins[bin_key]["correct"] += 1
                score_bins[bin_key]["avg_return"] += r.actual_return_pct or 0

            # 计算平均值
            for bin_key in score_bins:
                if score_bins[bin_key]["count"] > 0:
                    score_bins[bin_key]["avg_return"] /= score_bins[bin_key]["count"]
                    score_bins[bin_key]["accuracy"] = (
                        score_bins[bin_key]["correct"] / score_bins[bin_key]["count"] * 100
                    )

            # Agent级别统计
            agent_stats = {}
            if agent_name or True:  # 始终计算Agent统计
                for r in records:
                    if not r.agent_accuracy:
                        continue
                    for an, data in r.agent_accuracy.items():
                        if agent_name and an != agent_name:
                            continue
                        if an not in agent_stats:
                            agent_stats[an] = {"count": 0, "correct": 0, "total_contribution": 0}
                        agent_stats[an]["count"] += 1
                        if data.get("correct"):
                            agent_stats[an]["correct"] += 1
                        agent_stats[an]["total_contribution"] += data.get("contribution", 0)

                for an in agent_stats:
                    if agent_stats[an]["count"] > 0:
                        agent_stats[an]["accuracy"] = (
                            agent_stats[an]["correct"] / agent_stats[an]["count"] * 100
                        )

            return {
                "total": total,
                "direction_accuracy": direction_correct_count / total * 100 if total > 0 else 0,
                "avg_return": avg_return,
                "score_bins": score_bins,
                "agent_stats": agent_stats,
                "period_days": days,
            }

    def generate_error_report(self, record_id: int) -> dict:
        """生成错误分析报告

        Args:
            record_id: 预测记录ID

        Returns:
            错误分析报告
        """
        with self.db.session() as session:
            record = session.query(PredictionRecord).filter_by(id=record_id).first()
            if not record or not record.is_verified:
                return {"error": "记录不存在或未验证"}

            if record.direction_correct:
                return {"status": "预测正确，无需错误分析"}

            # 分析错误原因
            error_report = {
                "symbol": record.symbol,
                "prediction_date": str(record.prediction_date),
                "final_score": record.final_score,
                "actual_return_pct": record.actual_return_pct,
                "error_type": self._classify_error(record),
            }

            # Agent归因
            if record.agent_accuracy:
                wrong_agents = [
                    an for an, data in record.agent_accuracy.items()
                    if not data.get("correct")
                ]
                error_report["wrong_agents"] = wrong_agents
                error_report["wrong_agent_count"] = len(wrong_agents)

            # 分歧点验证
            if record.divergence_points:
                error_report["divergence_points"] = record.divergence_points

            # 建议修复
            error_report["suggested_fixes"] = self._suggest_fixes(record, error_report)

            # 更新记录
            record.error_analysis = error_report
            record.updated_at = datetime.now()

        return error_report

    def _classify_error(self, record: PredictionRecord) -> str:
        """分类错误类型"""
        if not record.actual_return_pct:
            return "unknown"

        if record.final_score > 30 and record.actual_return_pct < -10:
            return "false_positive_bull"  # 假看多
        elif record.final_score < -30 and record.actual_return_pct > 10:
            return "false_positive_bear"  # 假看空
        elif abs(record.final_score) < 20 and abs(record.actual_return_pct) > 15:
            return "missed_move"  # 错过大行情
        else:
            return "minor_error"  # 小幅偏差

    def _suggest_fixes(self, record: PredictionRecord, error_report: dict) -> list[str]:
        """建议修复措施"""
        fixes = []

        error_type = error_report.get("error_type", "")
        wrong_agents = error_report.get("wrong_agents", [])

        if error_type == "false_positive_bull":
            fixes.append("检查是否过度乐观，fusion层辩论机制是否充分")
            fixes.append("审查估值Agent的目标价假设是否合理")
            if "sentiment" in wrong_agents:
                fixes.append("情绪面可能过度正面，考虑降低情绪权重")
            if "technical" in wrong_agents:
                fixes.append("技术面信号可能滞后，考虑增加趋势确认")

        elif error_type == "false_positive_bear":
            fixes.append("检查是否过度悲观，是否有遗漏的正面因素")
            if "moat" in wrong_agents:
                fixes.append("护城河评估可能过于保守")
            if "fundamental" in wrong_agents:
                fixes.append("基本面可能忽略了潜在改善")

        elif error_type == "missed_move":
            fixes.append("中性评分过于保守，考虑增加信号敏感度")
            fixes.append("检查是否有Agent给出了方向性信号但权重不足")

        # 通用建议
        if len(wrong_agents) >= 4:
            fixes.append("多个Agent同时出错，检查数据质量问题")
        if len(wrong_agents) <= 1:
            fixes.append(f"仅{wrong_agents[0] if wrong_agents else '某个'}Agent出错，针对性优化")

        return fixes

    def _detach(self, record: PredictionRecord) -> PredictionRecord:
        """使记录在session外可用"""
        # 创建一个新对象，复制所有属性
        detached = PredictionRecord()
        for col in record.__table__.columns:
            setattr(detached, col.name, getattr(record, col.name))
        return detached


# 单例
_validation_service: Optional[ValidationService] = None


def get_validation_service() -> ValidationService:
    """获取验证服务单例"""
    global _validation_service
    if _validation_service is None:
        _validation_service = ValidationService()
    return _validation_service
