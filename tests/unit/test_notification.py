"""通知和调度器单元测试"""

import os
from unittest.mock import MagicMock, patch

from src.notification.manager import NotificationManager, _smart_truncate
from src.scheduler.scheduler import AnalysisScheduler, _build_daily_summary
from src.agents.state import FusionDecision


class TestNotificationManager:
    """通知管理器测试"""

    def test_from_env_no_config(self):
        """无环境变量时没有通道"""
        with patch.dict(os.environ, {}, clear=True):
            mgr = NotificationManager.from_env()
        assert not mgr.has_channels
        assert mgr.channel_names == []

    def test_from_env_wechat(self):
        """配置企业微信"""
        env = {"WECHAT_WEBHOOK_URL": "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx"}
        with patch.dict(os.environ, env, clear=True):
            mgr = NotificationManager.from_env()
        assert mgr.has_channels
        assert "wechat" in mgr.channel_names

    def test_from_env_telegram(self):
        """配置Telegram"""
        env = {"TELEGRAM_BOT_TOKEN": "123:ABC", "TELEGRAM_CHAT_ID": "456"}
        with patch.dict(os.environ, env, clear=True):
            mgr = NotificationManager.from_env()
        assert "telegram" in mgr.channel_names

    def test_from_env_email(self):
        """配置邮件"""
        env = {"EMAIL_SENDER": "a@qq.com", "EMAIL_PASSWORD": "pass", "EMAIL_RECEIVERS": "b@qq.com,c@qq.com"}
        with patch.dict(os.environ, env, clear=True):
            mgr = NotificationManager.from_env()
        assert "email" in mgr.channel_names

    def test_from_env_multi_channel(self):
        """多渠道同时配置"""
        env = {
            "WECHAT_WEBHOOK_URL": "https://example.com",
            "FEISHU_WEBHOOK_URL": "https://example.com",
            "TELEGRAM_BOT_TOKEN": "tok",
            "TELEGRAM_CHAT_ID": "cid",
        }
        with patch.dict(os.environ, env, clear=True):
            mgr = NotificationManager.from_env()
        assert len(mgr.channel_names) == 3

    def test_send_returns_results(self):
        """send返回每个渠道的成功/失败状态"""
        mgr = NotificationManager(wechat_webhook_url="https://example.com")
        with patch.object(mgr, "_send_wechat", side_effect=Exception("test")):
            results = mgr.send("title", "content")
        assert results["wechat"] is False

    def test_no_channels_send_empty(self):
        """无渠道时send返回空"""
        mgr = NotificationManager()
        results = mgr.send("title", "content")
        assert results == {}


class TestSmartTruncate:
    """智能截断测试"""

    def test_short_text_unchanged(self):
        assert _smart_truncate("short text", 1000) == "short text"

    def test_long_text_truncated(self):
        text = "paragraph1\n\nparagraph2\n\nparagraph3"
        result = _smart_truncate(text, 30)
        assert len(result.encode("utf-8")) <= 30 + 10  # allow some margin

    def test_chinese_utf8_safe(self):
        text = "这是一段中文\n\n另一段中文\n\n第三段"
        result = _smart_truncate(text, 30)
        # 不应该截断在多字节字符中间
        result.encode("utf-8")  # should not raise


class TestDailySummary:
    """每日摘要测试"""

    def test_empty_results(self):
        summary = _build_daily_summary([])
        assert "无分析结果" in summary

    def test_bullish_section(self):
        results = [{
            "symbol": "000001.SZ",
            "name": "平安银行",
            "fusion": FusionDecision(
                final_score=50, final_action="建仓", confidence=0.9,
                position_pct=40, reasoning="test",
            ),
            "signals": [],
        }]
        summary = _build_daily_summary(results)
        assert "看多" in summary
        assert "平安银行" in summary

    def test_mixed_signals(self):
        results = [
            {
                "symbol": "A", "name": "看多股",
                "fusion": FusionDecision(final_score=50, final_action="建仓", confidence=0.9, reasoning=""),
                "signals": [],
            },
            {
                "symbol": "B", "name": "观望股",
                "fusion": FusionDecision(final_score=0, final_action="观望", confidence=0.5, reasoning=""),
                "signals": [],
            },
            {
                "symbol": "C", "name": "看空股",
                "fusion": FusionDecision(final_score=-50, final_action="清仓", confidence=0.8, reasoning=""),
                "signals": [],
            },
        ]
        summary = _build_daily_summary(results)
        assert "看多" in summary
        assert "中性" in summary or "观望" in summary
        assert "看空" in summary


class TestScheduler:
    """调度器测试"""

    def test_scheduler_init(self):
        sched = AnalysisScheduler(trigger_time="18:00", top_n=5)
        jobs = sched._scheduler.get_jobs()
        assert len(jobs) == 1
        assert jobs[0].name == "每日分析流水线"

    def test_scheduler_from_env(self):
        env = {"ANALYSIS_TRIGGER_TIME": "19:00", "ANALYSIS_TOP_N": "20"}
        with patch.dict(os.environ, env, clear=True):
            sched = AnalysisScheduler.from_env()
        assert sched._top_n == 20
        assert sched._trigger_time == "19:00"
