"""筛选引擎因子评分测试"""

import numpy as np
import pandas as pd
import pytest

from src.screening.engine import ScreeningEngine


class MockSourceManager:
    """模拟数据源管理器"""
    pass


def _make_engine():
    """创建带默认配置的引擎"""
    return ScreeningEngine(MockSourceManager(), config_path=None)


def _make_df(n=100, **kwargs):
    """生成测试用DataFrame"""
    rng = np.random.default_rng(42)
    data = {
        "symbol": [f"{str(i).zfill(6)}.SZ" for i in range(n)],
        "name": [f"测试{i}" for i in range(n)],
        "eps": rng.uniform(0.1, 3.0, n),
        "bps": rng.uniform(1, 20, n),
        "roe": rng.uniform(3, 30, n),
        "gross_margin": rng.uniform(10, 80, n),
        "revenue_yoy": rng.uniform(-20, 60, n),
        "profit_yoy": rng.uniform(-30, 100, n),
        "cashflow_per_share": rng.uniform(-0.5, 3.0, n),
        "industry": ["行业A"] * n,
    }
    data.update(kwargs)
    return pd.DataFrame(data)


def test_quality_score_range():
    engine = _make_engine()
    df = _make_df()
    scores = engine._calc_quality_score(df)
    assert scores.min() >= 0
    assert scores.max() <= 100
    # 高ROE + 高毛利 应该得高分
    assert scores.std() > 1  # 有区分度


def test_growth_score_range():
    engine = _make_engine()
    df = _make_df()
    scores = engine._calc_growth_score(df)
    assert scores.min() >= 0
    assert scores.max() <= 100


def test_quality_high_roe_scores_higher():
    engine = _make_engine()
    df = _make_df(n=200)
    scores = engine._calc_quality_score(df)
    roe = pd.to_numeric(df["roe"])
    # 高ROE组的平均分应该高于低ROE组
    high_roe_mask = roe > roe.median()
    assert scores[high_roe_mask].mean() > scores[~high_roe_mask].mean()


def test_growth_high_revenue_scores_higher():
    engine = _make_engine()
    df = _make_df(n=200)
    scores = engine._calc_growth_score(df)
    rev = pd.to_numeric(df["revenue_yoy"])
    high_rev_mask = rev > rev.median()
    assert scores[high_rev_mask].mean() > scores[~high_rev_mask].mean()


def test_pre_filter_excludes_st():
    engine = _make_engine()
    df = _make_df(n=10)
    df.loc[0, "name"] = "ST测试"
    df.loc[1, "name"] = "*ST退市"
    filtered = engine._pre_filter(df, has_spot=False)
    assert len(filtered) == 8
    assert not filtered["name"].str.contains("ST").any()


def test_pre_filter_excludes_loss():
    engine = _make_engine()
    df = _make_df(n=10)
    df.loc[0, "eps"] = -0.5  # 亏损
    df.loc[1, "eps"] = 0     # 盈亏平衡
    filtered = engine._pre_filter(df, has_spot=False)
    assert len(filtered) == 8


def test_score_all_produces_composite():
    engine = _make_engine()
    df = _make_df(n=50)
    scored = engine._score_all(df, has_spot=False)
    assert "composite_score" in scored.columns
    assert "factor_scores" in scored.columns
    assert scored["composite_score"].max() > scored["composite_score"].min()
