"""测试LLM增强层"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agents.llm_enhance import _parse_json_response, llm_enhance


class TestParseJsonResponse:
    """测试JSON响应解析"""

    def test_pure_json(self):
        text = '{"score_adjustment": 5, "reasoning": "test"}'
        result = _parse_json_response(text)
        assert result["score_adjustment"] == 5

    def test_json_code_block(self):
        text = 'some text\n```json\n{"score_adjustment": -3}\n```\nmore text'
        result = _parse_json_response(text)
        assert result["score_adjustment"] == -3

    def test_embedded_json(self):
        text = 'Here is my analysis: {"reasoning": "good stock", "score_adjustment": 0} end.'
        result = _parse_json_response(text)
        assert result["reasoning"] == "good stock"

    def test_invalid_returns_none_or_empty(self):
        result = _parse_json_response("no json here at all")
        # json_repair may return empty string; either None or falsy is acceptable
        assert not result or result is None

    def test_empty_string(self):
        result = _parse_json_response("")
        assert not result or result is None


class TestLlmEnhance:
    """测试llm_enhance函数"""

    def test_no_api_key_returns_fallback(self):
        """无API Key时返回code-only结果"""
        result = llm_enhance(
            agent_name="technical",
            template_name="technical.md",
            template_vars={
                "symbol": "000001.SZ",
                "name": "平安银行",
                "analysis_date": "2026-01-01",
                "indicators_summary": "test",
                "recent_quotes": "test",
            },
            code_score=50,
            code_reasoning="test reasoning",
            code_factors=["factor1"],
            code_risks=["risk1"],
        )
        assert result["enhanced"] is False
        assert result["score_adjustment"] == 0
        assert result["reasoning"] == "test reasoning"
        assert result["llm_model"] == ""

    @patch("src.agents.llm_enhance._has_llm_key", return_value=True)
    @patch("src.agents.llm_enhance._load_agent_config")
    @patch("src.agents.llm_enhance._load_prompt")
    def test_llm_success_path(self, mock_prompt, mock_config, mock_key):
        """模拟LLM成功调用"""
        mock_config.return_value = {
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "temperature": 0.2,
            "max_tokens": 2000,
            "timeout": 30,
        }
        mock_prompt.return_value = "分析{symbol}({name})，日期{analysis_date}"

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "score_adjustment": 5,
            "reasoning": "LLM深度分析：趋势向好",
            "key_patterns": ["双底形态"],
            "risks": ["量能不足"],
        })
        mock_response.model = "gpt-4o-mini"
        mock_response.total_tokens = 500
        mock_response.cost_usd = 0.001
        mock_response.latency_ms = 800

        mock_router = MagicMock()
        mock_router.invoke.return_value = mock_response

        with patch("src.llm.router.get_llm_router", return_value=mock_router):
            result = llm_enhance(
                agent_name="technical",
                template_name="technical.md",
                template_vars={
                    "symbol": "000001.SZ",
                    "name": "平安银行",
                    "analysis_date": "2026-01-01",
                    "indicators_summary": "test",
                    "recent_quotes": "test",
                },
                code_score=50,
                code_reasoning="test reasoning",
                code_factors=["factor1"],
                code_risks=["risk1"],
            )

        assert result["enhanced"] is True
        assert result["score_adjustment"] == 5
        assert result["reasoning"] == "LLM深度分析：趋势向好"
        assert "双底形态" in result["extra_factors"]
        assert "量能不足" in result["extra_risks"]
        assert result["llm_model"] == "gpt-4o-mini"
        assert result["llm_tokens"] == 500

    @patch("src.agents.llm_enhance._has_llm_key", return_value=True)
    @patch("src.agents.llm_enhance._load_agent_config")
    @patch("src.agents.llm_enhance._load_prompt")
    def test_score_adjustment_clamped(self, mock_prompt, mock_config, mock_key):
        """LLM分数调整限制在±15"""
        mock_config.return_value = {
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
        }
        mock_prompt.return_value = "分析{symbol}"

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "score_adjustment": 50,  # 超出范围
            "reasoning": "test",
        })
        mock_response.model = "gpt-4o-mini"
        mock_response.total_tokens = 100
        mock_response.cost_usd = 0.0001
        mock_response.latency_ms = 200

        mock_router = MagicMock()
        mock_router.invoke.return_value = mock_response

        with patch("src.llm.router.get_llm_router", return_value=mock_router):
            result = llm_enhance(
                agent_name="technical",
                template_name="technical.md",
                template_vars={
                    "symbol": "000001.SZ",
                    "name": "测试",
                    "analysis_date": "2026-01-01",
                    "indicators_summary": "test",
                    "recent_quotes": "test",
                },
                code_score=50,
                code_reasoning="test",
                code_factors=[],
                code_risks=[],
            )

        assert result["score_adjustment"] == 15  # clamped to max

    @patch("src.agents.llm_enhance._has_llm_key", return_value=True)
    @patch("src.agents.llm_enhance._load_agent_config")
    @patch("src.agents.llm_enhance._load_prompt")
    def test_llm_failure_graceful_fallback(self, mock_prompt, mock_config, mock_key):
        """LLM调用失败时优雅降级"""
        mock_config.return_value = {"llm_provider": "openai", "llm_model": "gpt-4o-mini"}
        mock_prompt.return_value = "分析{symbol}"

        mock_router = MagicMock()
        mock_router.invoke.side_effect = Exception("API timeout")

        with patch("src.llm.router.get_llm_router", return_value=mock_router):
            result = llm_enhance(
                agent_name="technical",
                template_name="technical.md",
                template_vars={
                    "symbol": "000001.SZ",
                    "name": "测试",
                    "analysis_date": "2026-01-01",
                    "indicators_summary": "test",
                    "recent_quotes": "test",
                },
                code_score=50,
                code_reasoning="fallback reasoning",
                code_factors=[],
                code_risks=[],
            )

        assert result["enhanced"] is False
        assert result["reasoning"] == "fallback reasoning"
        assert result["score_adjustment"] == 0
