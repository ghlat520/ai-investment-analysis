"""
AKShare 数据源

A股数据：日线行情、财务数据、资金流向、估值。
优先级1（efinance之后）。
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from .base import BaseDataSource


class AKShareSource(BaseDataSource):
    name = "akshare"
    priority = 1

    def fetch_daily_quotes(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        import akshare as ak

        code = symbol.split(".")[0]
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust="qfq",
        )
        if df.empty:
            return df

        rename_map = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover_rate",
            "振幅": "amplitude",
            "涨跌幅": "change_pct",
            "涨跌额": "change_amount",
        }
        df = df.rename(columns=rename_map)
        return df

    def fetch_stock_list(self, market: str = "A") -> pd.DataFrame:
        import akshare as ak

        if market == "A":
            df = ak.stock_zh_a_spot_em()
            rename_map = {
                "代码": "symbol",
                "名称": "name",
            }
            df = df.rename(columns=rename_map)
            df["symbol"] = df["symbol"].apply(
                lambda x: f"{x}.SZ" if x.startswith(("0", "3")) else f"{x}.SH"
            )
            df["market"] = "A"
            return df[["symbol", "name", "market"]]
        return pd.DataFrame()

    def fetch_financial_data(self, symbol: str) -> pd.DataFrame:
        """获取财务数据（东方财富接口）

        使用 stock_financial_analysis_indicator_em，返回标准化字段：
        report_date, roe, roa, gross_margin, net_margin, revenue_yoy,
        profit_yoy, debt_ratio, current_ratio, operating_cashflow, etc.
        """
        import akshare as ak

        try:
            df = ak.stock_financial_analysis_indicator_em(symbol=symbol)
            if df.empty:
                return df

            # 东方财富字段 → 标准字段映射
            result = pd.DataFrame()
            result["report_date"] = pd.to_datetime(df["REPORT_DATE"]).dt.date
            result["report_type"] = df["REPORT_DATE_NAME"]

            # 每股指标
            result["eps"] = pd.to_numeric(df["EPSJB"], errors="coerce")
            result["bps"] = pd.to_numeric(df["BPS"], errors="coerce")

            # 盈利能力
            result["roe"] = pd.to_numeric(df["ROEJQ"], errors="coerce")
            result["roa"] = pd.to_numeric(df["ZZCJLL"], errors="coerce")
            result["gross_margin"] = pd.to_numeric(df["XSMLL"], errors="coerce")
            result["net_margin"] = pd.to_numeric(df["XSJLL"], errors="coerce")

            # 营收与利润绝对值
            result["revenue"] = pd.to_numeric(df["TOTALOPERATEREVE"], errors="coerce")
            result["net_profit"] = pd.to_numeric(df["PARENTNETPROFIT"], errors="coerce")
            result["net_profit_deducted"] = pd.to_numeric(df["KCFJCXSYJLR"], errors="coerce")

            # 成长性（同比增速）
            result["revenue_yoy"] = pd.to_numeric(df["TOTALOPERATEREVETZ"], errors="coerce")
            result["profit_yoy"] = pd.to_numeric(df["PARENTNETPROFITTZ"], errors="coerce")

            # 财务健康
            result["debt_ratio"] = pd.to_numeric(df["ZCFZL"], errors="coerce")
            result["current_ratio"] = pd.to_numeric(df.get("LD"), errors="coerce")
            result["quick_ratio"] = pd.to_numeric(df.get("SD"), errors="coerce")

            # 现金流
            result["operating_cashflow_per_share"] = pd.to_numeric(df["MGJYXJJE"], errors="coerce")
            # 经营现金流/营收
            result["cashflow_to_revenue"] = pd.to_numeric(df.get("JYXJLYYSR"), errors="coerce")

            # 按报告日期排序
            result = result.sort_values("report_date").reset_index(drop=True)

            logger.debug(f"[akshare] 财务数据: {symbol}, {len(result)}期")
            return result
        except Exception as e:
            logger.debug(f"[akshare] 财务数据获取失败: {e}")
            return pd.DataFrame()

    def fetch_valuation(self, symbol: str) -> pd.DataFrame:
        """获取估值历史数据（百度股市通接口）

        返回：date, pe_ttm, pb, total_market_cap
        """
        import akshare as ak

        code = symbol.split(".")[0]
        try:
            # 获取 PE/PB/市值 三年历史
            pe_df = ak.stock_zh_valuation_baidu(
                symbol=code, indicator="市盈率(TTM)", period="近三年"
            )
            pb_df = ak.stock_zh_valuation_baidu(
                symbol=code, indicator="市净率", period="近三年"
            )
            cap_df = ak.stock_zh_valuation_baidu(
                symbol=code, indicator="总市值", period="近一年"
            )

            result = pd.DataFrame()

            if not pe_df.empty:
                pe_df = pe_df.rename(columns={"date": "date", "value": "pe_ttm"})
                pe_df["date"] = pd.to_datetime(pe_df["date"]).dt.date
                result = pe_df[["date", "pe_ttm"]]

            if not pb_df.empty:
                pb_df = pb_df.rename(columns={"value": "pb"})
                pb_df["date"] = pd.to_datetime(pb_df["date"]).dt.date
                if result.empty:
                    result = pb_df[["date", "pb"]]
                else:
                    result = result.merge(pb_df[["date", "pb"]], on="date", how="outer")

            if not cap_df.empty:
                cap_df = cap_df.rename(columns={"value": "total_market_cap"})
                cap_df["date"] = pd.to_datetime(cap_df["date"]).dt.date
                if result.empty:
                    result = cap_df[["date", "total_market_cap"]]
                else:
                    result = result.merge(
                        cap_df[["date", "total_market_cap"]], on="date", how="outer"
                    )

            if not result.empty:
                result = result.sort_values("date").reset_index(drop=True)
                logger.debug(f"[akshare] 估值数据: {code}, {len(result)}天")

            return result
        except Exception as e:
            logger.debug(f"[akshare] 估值数据获取失败: {e}")
            return pd.DataFrame()

    def fetch_batch_financial(self, report_date: str = "") -> pd.DataFrame:
        """获取全市场业绩报表（东方财富datacenter接口）

        使用 stock_yjbb_em，一次返回 ~5000+ 只股票的核心财务指标。
        """
        import akshare as ak

        if not report_date:
            # 默认取最近一个季报：上年三季报 or 上年年报
            from datetime import date as d
            today = d.today()
            if today.month >= 11:
                report_date = f"{today.year}0930"
            elif today.month >= 5:
                report_date = f"{today.year - 1}1231"
            else:
                report_date = f"{today.year - 1}0930"

        try:
            df = ak.stock_yjbb_em(date=report_date)
            if df.empty:
                return df

            result = pd.DataFrame()
            result["symbol"] = df["股票代码"].apply(
                lambda x: f"{x}.SZ" if x.startswith(("0", "3")) else f"{x}.SH"
            )
            result["name"] = df["股票简称"]
            result["eps"] = pd.to_numeric(df["每股收益"], errors="coerce")
            result["bps"] = pd.to_numeric(df["每股净资产"], errors="coerce")
            result["roe"] = pd.to_numeric(df["净资产收益率"], errors="coerce")
            result["gross_margin"] = pd.to_numeric(df["销售毛利率"], errors="coerce")
            result["revenue_yoy"] = pd.to_numeric(df["营业总收入-同比增长"], errors="coerce")
            result["profit_yoy"] = pd.to_numeric(df["净利润-同比增长"], errors="coerce")
            result["cashflow_per_share"] = pd.to_numeric(df["每股经营现金流量"], errors="coerce")
            result["industry"] = df["所处行业"]
            result["report_date"] = report_date

            logger.debug(f"[akshare] 批量财务数据: {len(result)}只, 报告期={report_date}")
            return result
        except Exception as e:
            logger.debug(f"[akshare] 批量财务数据失败: {e}")
            return pd.DataFrame()

    def fetch_spot_data(self, market: str = "A") -> pd.DataFrame:
        """获取全市场实时行情快照"""
        import akshare as ak

        if market != "A":
            return pd.DataFrame()

        try:
            df = ak.stock_zh_a_spot_em()
            if df.empty:
                return df

            result = pd.DataFrame()
            result["symbol"] = df["代码"].apply(
                lambda x: f"{x}.SZ" if x.startswith(("0", "3")) else f"{x}.SH"
            )
            result["name"] = df["名称"]
            result["price"] = pd.to_numeric(df["最新价"], errors="coerce")
            result["change_pct"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
            result["volume"] = pd.to_numeric(df["成交量"], errors="coerce")
            result["amount"] = pd.to_numeric(df["成交额"], errors="coerce")
            result["turnover_rate"] = pd.to_numeric(df["换手率"], errors="coerce")
            result["amplitude"] = pd.to_numeric(df["振幅"], errors="coerce")
            result["pe"] = pd.to_numeric(df.get("市盈率-动态"), errors="coerce")
            result["pb"] = pd.to_numeric(df.get("市净率"), errors="coerce")
            result["total_market_cap"] = pd.to_numeric(df.get("总市值"), errors="coerce")
            result["float_market_cap"] = pd.to_numeric(df.get("流通市值"), errors="coerce")
            result["change_pct_60d"] = pd.to_numeric(df.get("60日涨跌幅"), errors="coerce")
            result["change_pct_ytd"] = pd.to_numeric(df.get("年初至今涨跌幅"), errors="coerce")
            result["volume_ratio"] = pd.to_numeric(df.get("量比"), errors="coerce")

            logger.debug(f"[akshare] 实时行情: {len(result)}只")
            return result
        except Exception as e:
            logger.debug(f"[akshare] 实时行情获取失败: {e}")
            return pd.DataFrame()

    def fetch_stock_news(self, symbol: str, limit: int = 20) -> list[dict]:
        """获取个股新闻（东方财富接口）"""
        import akshare as ak

        code = symbol.split(".")[0]
        try:
            df = ak.stock_news_em(symbol=code)
            if df.empty:
                return []

            results = []
            for _, row in df.head(limit).iterrows():
                results.append({
                    "title": str(row.get("新闻标题", "")),
                    "content": str(row.get("新闻内容", "")),
                    "datetime": str(row.get("发布时间", "")),
                    "source": str(row.get("文章来源", "")),
                    "url": str(row.get("新闻链接", "")),
                })
            logger.debug(f"[akshare] 个股新闻: {code}, {len(results)}条")
            return results
        except Exception as e:
            logger.debug(f"[akshare] 个股新闻获取失败: {e}")
            return []

    def fetch_money_flow(self, symbol: str, days: int = 20) -> pd.DataFrame:
        import akshare as ak

        code = symbol.split(".")[0]
        try:
            df = ak.stock_individual_fund_flow(
                stock=code,
                market="sh" if symbol.endswith(".SH") else "sz",
            )
            if df.empty:
                return df
            return df.tail(days)
        except Exception as e:
            logger.debug(f"[akshare] 资金流向获取失败: {e}")
            return pd.DataFrame()
