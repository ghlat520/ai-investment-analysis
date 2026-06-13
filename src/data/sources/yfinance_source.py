"""
yfinance 数据源

覆盖全球市场（A股/港股/美股），作为最终fallback。
美股增强版：财务数据、估值数据、分红历史、分业务营收。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseDataSource


class YFinanceSource(BaseDataSource):
    name = "yfinance"
    priority = 5  # 最低优先级（A股场景）；美股场景下是唯一数据源

    @staticmethod
    def _to_yf_symbol(symbol: str) -> str:
        """转换为yfinance格式"""
        if symbol.endswith(".SH"):
            return symbol.replace(".SH", ".SS")
        if symbol.endswith(".SZ"):
            return symbol  # yfinance支持.SZ
        return symbol  # 美股直接用（NVDA, TSM, ASML等）

    def fetch_daily_quotes(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        import yfinance as yf

        yf_symbol = self._to_yf_symbol(symbol)
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(start=start_date, end=end_date)
        if df.empty:
            return df

        df = df.reset_index()
        rename_map = {
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df = df.rename(columns=rename_map)
        return df[["date", "open", "high", "low", "close", "volume"]]

    def fetch_stock_list(self, market: str = "A") -> pd.DataFrame:
        return pd.DataFrame()

    # ─── 财务数据（美股增强） ───────────────────────────────

    def fetch_financial_data(self, symbol: str) -> pd.DataFrame:
        """获取财务数据（yfinance）

        返回标准化字段，对齐 akshare_source 字段名：
        report_date, roe, roa, gross_margin, net_margin, revenue, net_profit,
        revenue_yoy, profit_yoy, debt_ratio, eps, bps, operating_cashflow
        """
        import yfinance as yf

        try:
            yf_symbol = self._to_yf_symbol(symbol)
            ticker = yf.Ticker(yf_symbol)

            income_stmt = ticker.quarterly_income_stmt
            balance_sheet = ticker.quarterly_balance_sheet
            cashflow = ticker.quarterly_cashflow

            if income_stmt.empty:
                return pd.DataFrame()

            dates = income_stmt.columns.tolist()
            rows = []

            for i, col in enumerate(dates):
                row: dict = {"report_date": col.date() if hasattr(col, "date") else col}

                # 营收 & 净利润
                rev = self._safe_get(income_stmt, col, [
                    "Total Revenue", "TotalRevenue",
                ])
                np_ = self._safe_get(income_stmt, col, [
                    "Net Income", "NetIncome", "Net Income Common Stockholders",
                ])
                gp = self._safe_get(income_stmt, col, [
                    "Gross Profit", "GrossProfit",
                ])
                opex = self._safe_get(income_stmt, col, [
                    "Total Operating Expenses", "OperatingExpense", "Operating Revenue",
                ])
                interest = self._safe_get(income_stmt, col, [
                    "Interest Expense", "InterestExpense",
                ])

                row["revenue"] = rev
                row["net_profit"] = np_
                row["gross_margin"] = (gp / rev * 100) if rev and gp and rev != 0 else None
                row["net_margin"] = (np_ / rev * 100) if rev and np_ and rev != 0 else None

                # 资产负债表
                total_assets = self._safe_get(balance_sheet, col, ["Total Assets"]) if not balance_sheet.empty else None
                total_liab = self._safe_get(balance_sheet, col, ["Total Liabilities Net Minority Interest"]) if not balance_sheet.empty else None
                total_eq = self._safe_get(balance_sheet, col, ["Stockholders Equity", "Common Stock Equity"]) if not balance_sheet.empty else None
                current_assets = self._safe_get(balance_sheet, col, ["Current Assets"]) if not balance_sheet.empty else None
                current_liab = self._safe_get(balance_sheet, col, ["Current Liabilities"]) if not balance_sheet.empty else None

                row["debt_ratio"] = (total_liab / total_assets * 100) if total_assets and total_liab and total_assets != 0 else None
                row["current_ratio"] = (current_assets / current_liab) if current_assets and current_liab and current_liab != 0 else None

                # ROE & ROA
                row["roe"] = (np_ / total_eq * 100) if np_ and total_eq and total_eq != 0 else None
                row["roa"] = (np_ / total_assets * 100) if np_ and total_assets and total_assets != 0 else None

                # EPS & BPS (TTM 近4个季度)
                shares = self._safe_get(income_stmt, col, [
                    "Basic Average Shares", "Weighted Average Shares",
                ]) or self._safe_get(balance_sheet, col, [
                    "Ordinary Shares Number", "Common Stock Shares Outstanding",
                ])
                if shares and shares != 0:
                    row["eps"] = np_ / shares if np_ else None
                    row["bps"] = total_eq / shares if total_eq else None
                else:
                    row["eps"] = None
                    row["bps"] = None

                # 现金流
                ocf = self._safe_get(cashflow, col, [
                    "Operating Cash Flow", "Cash Flow From Continuing Operating Activities",
                ]) if not cashflow.empty else None
                row["operating_cashflow"] = ocf
                row["operating_cashflow_per_share"] = (ocf / shares) if ocf and shares and shares != 0 else None

                # YoY 增速（需要同比季度）
                row["revenue_yoy"] = None
                row["profit_yoy"] = None
                if i + 4 < len(dates):
                    prev_col = dates[i + 4]  # 同一季度去年
                    prev_rev = self._safe_get(income_stmt, prev_col, ["Total Revenue", "TotalRevenue"])
                    prev_np = self._safe_get(income_stmt, prev_col, ["Net Income", "NetIncome", "Net Income Common Stockholders"])
                    if prev_rev and prev_rev != 0 and rev:
                        row["revenue_yoy"] = (rev - prev_rev) / abs(prev_rev) * 100
                    if prev_np and prev_np != 0 and np_:
                        row["profit_yoy"] = (np_ - prev_np) / abs(prev_np) * 100

                rows.append(row)

            result = pd.DataFrame(rows)
            result = result.sort_values("report_date", ascending=False).reset_index(drop=True)
            logger.debug(f"[yfinance] 财务数据: {symbol}, {len(result)}期")
            return result

        except Exception as e:
            logger.debug(f"[yfinance] 财务数据获取失败: {e}")
            return pd.DataFrame()

    # ─── 估值数据（美股增强） ───────────────────────────────

    def fetch_valuation(self, symbol: str) -> pd.DataFrame:
        """获取估值历史数据

        返回：date, pe_ttm, pb, total_market_cap
        对齐 akshare_source 字段名。
        """
        import yfinance as yf

        try:
            yf_symbol = self._to_yf_symbol(symbol)
            ticker = yf.Ticker(yf_symbol)

            # 用 history 拿 1 年收盘价 + info 拿估值
            hist = ticker.history(period="1y")
            info = ticker.info

            if hist.empty:
                return pd.DataFrame()

            pe_ttm = info.get("trailingPE") or info.get("forwardPE")
            pb = info.get("priceToBook")
            market_cap = info.get("marketCap")

            rows = []
            for idx, row in hist.iterrows():
                d = {
                    "date": idx.date() if hasattr(idx, "date") else idx,
                    "pe_ttm": pe_ttm,
                    "pb": pb,
                    "total_market_cap": market_cap,
                }
                rows.append(d)

            result = pd.DataFrame(rows)
            logger.debug(f"[yfinance] 估值数据: {symbol}, {len(result)}天")
            return result

        except Exception as e:
            logger.debug(f"[yfinance] 估值数据获取失败: {e}")
            return pd.DataFrame()

    # ─── 分红历史（美股增强） ───────────────────────────────

    def fetch_dividend_history(self, symbol: str) -> pd.DataFrame:
        """获取分红历史

        返回：report_date, div_per_share, dividend_yield, ex_date
        """
        import yfinance as yf

        try:
            yf_symbol = self._to_yf_symbol(symbol)
            ticker = yf.Ticker(yf_symbol)

            divs = ticker.dividends
            if divs.empty:
                return pd.DataFrame()

            df = divs.reset_index()
            df.columns = ["ex_date", "div_per_share"]
            df["ex_date"] = pd.to_datetime(df["ex_date"]).dt.date
            df["report_date"] = df["ex_date"]
            df["dividend_yield"] = None  # yfinance不直接提供yield per event

            result = df[["report_date", "div_per_share", "dividend_yield", "ex_date"]]
            result = result.sort_values("report_date", ascending=False).reset_index(drop=True)
            logger.debug(f"[yfinance] 分红历史: {symbol}, {len(result)}期")
            return result

        except Exception as e:
            logger.debug(f"[yfinance] 分红历史获取失败: {e}")
            return pd.DataFrame()

    # ─── 分业务营收（美股增强） ───────────────────────────────

    def fetch_business_composition(self, symbol: str) -> pd.DataFrame:
        """获取分业务营收构成

        yfinance 通过 revenue_forecasts 或 info 提供有限数据。
        返回：product, revenue, revenue_pct, report_date
        """
        import yfinance as yf

        try:
            yf_symbol = self._to_yf_symbol(symbol)
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info

            # 尝试从 analyst_price_targets 或 recommendation 获取行业信息
            # yfinance 没有直接的分业务营收 API，返回空（LLM 会自行分析）
            return pd.DataFrame()

        except Exception as e:
            logger.debug(f"[yfinance] 分业务营收获取失败: {e}")
            return pd.DataFrame()

    # ─── 辅助方法 ──────────────────────────────────────────

    @staticmethod
    def _safe_get(df: pd.DataFrame, col, keys: list[str]) -> float | None:
        """安全从 DataFrame 取值，尝试多个 key"""
        if df.empty:
            return None
        for key in keys:
            if key in df.index:
                val = df.loc[key, col]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        continue
        return None
