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
            # API 需要带后缀格式，如 300054.SZ
            code = symbol.split(".")[0]
            suffix = ".SH" if code.startswith("6") else ".SZ"
            df = ak.stock_financial_analysis_indicator_em(symbol=f"{code}{suffix}")
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
            result["current_ratio"] = pd.to_numeric(df["LD"], errors="coerce")
            result["quick_ratio"] = pd.to_numeric(df["SD"], errors="coerce")

            # 现金流
            result["operating_cashflow_per_share"] = pd.to_numeric(df["MGJYXJJE"], errors="coerce")
            # 经营现金流/营收比（注意：此值是小数形式如 0.28，非百分比）
            result["cashflow_to_revenue"] = pd.to_numeric(df["JYXJLYYSR"], errors="coerce")
            # 经营现金流总额 = cashflow_to_revenue(小数) * revenue
            result["operating_cashflow"] = result["cashflow_to_revenue"] * result["revenue"]

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

    def fetch_business_composition(self, symbol: str) -> pd.DataFrame:
        """获取分业务/分产品营收构成（东方财富主营业务分析）"""
        import akshare as ak

        code = symbol.split(".")[0]
        # stock_zygc_em 需要 SH/SZ 前缀格式
        prefix = "SH" if code.startswith("6") else "SZ"
        em_symbol = f"{prefix}{code}"
        try:
            df = ak.stock_zygc_em(symbol=em_symbol)
            if df.empty:
                return df

            result = pd.DataFrame()
            result["product"] = df.get("主营构成", df.columns[0])
            result["revenue"] = pd.to_numeric(df.get("主营收入", 0), errors="coerce")
            result["revenue_pct"] = pd.to_numeric(df.get("收入比例", 0), errors="coerce")
            result["gross_margin"] = pd.to_numeric(df.get("毛利率", 0), errors="coerce")
            result["cost"] = pd.to_numeric(df.get("主营成本", 0), errors="coerce")
            result["report_date"] = df.get("报告日期", "").astype(str)
            result["classification"] = df.get("分类类型", "").astype(str)

            logger.debug(f"[akshare] 分业务构成: {code}, {len(result)}项")
            return result
        except Exception as e:
            logger.debug(f"[akshare] 分业务构成获取失败: {e}")
            return pd.DataFrame()

    def fetch_profit_forecast(self, symbol: str) -> pd.DataFrame:
        """获取券商盈利预测/一致预期EPS（同花顺接口）

        返回标准化列：year, num_institutions, eps_min, eps_mean, eps_max, industry_avg
        """
        import akshare as ak

        code = symbol.split(".")[0]
        try:
            df = ak.stock_profit_forecast_ths(symbol=code)
            if df.empty:
                return df

            result = pd.DataFrame()
            result["year"] = df.get("年度", "")
            result["num_institutions"] = pd.to_numeric(df.get("预测机构数", 0), errors="coerce")
            result["eps_min"] = pd.to_numeric(df.get("最小值", 0), errors="coerce")
            result["eps_mean"] = pd.to_numeric(df.get("均值", 0), errors="coerce")
            result["eps_max"] = pd.to_numeric(df.get("最大值", 0), errors="coerce")
            result["industry_avg"] = pd.to_numeric(df.get("行业平均数", 0), errors="coerce")

            logger.debug(f"[akshare] 券商盈利预测: {code}, {len(result)}年")
            return result
        except Exception as e:
            logger.debug(f"[akshare] 券商盈利预测获取失败: {e}")
            return pd.DataFrame()

    def fetch_shareholder_count(self, symbol: str) -> pd.DataFrame:
        """获取股东人数变化历史（东方财富接口）"""
        import akshare as ak

        code = symbol.split(".")[0]
        try:
            df = ak.stock_zh_a_gdhs_detail_em(symbol=code)
            if df.empty:
                return df

            result = pd.DataFrame()
            result["date"] = pd.to_datetime(df["股东户数统计截止日"]).dt.date
            result["count"] = pd.to_numeric(df["股东户数-本次"], errors="coerce")
            result["change_pct"] = pd.to_numeric(df["股东户数-增减比例"], errors="coerce")
            result["avg_market_value"] = pd.to_numeric(df["户均持股市值"], errors="coerce")
            result["total_market_cap"] = pd.to_numeric(df["总市值"], errors="coerce")

            result = result.sort_values("date").reset_index(drop=True)
            # 取最近8期（约2年）
            result = result.tail(8).reset_index(drop=True)
            logger.debug(f"[akshare] 股东人数: {code}, {len(result)}期")
            return result
        except Exception as e:
            logger.debug(f"[akshare] 股东人数获取失败: {e}")
            return pd.DataFrame()

    def fetch_northbound_holdings(self, symbol: str) -> pd.DataFrame:
        """获取北向资金个股持股历史（东方财富接口）"""
        import akshare as ak

        code = symbol.split(".")[0]
        try:
            df = ak.stock_hsgt_individual_em(symbol=code)
            if df.empty:
                return df

            result = pd.DataFrame()
            result["date"] = pd.to_datetime(df["持股日期"]).dt.date
            result["close"] = pd.to_numeric(df["当日收盘价"], errors="coerce")
            result["hold_shares"] = pd.to_numeric(df["持股数量"], errors="coerce")
            result["hold_market_value"] = pd.to_numeric(df["持股市值"], errors="coerce")
            result["hold_ratio_float"] = pd.to_numeric(df["持股数量占A股百分比"], errors="coerce")
            result["change_shares"] = pd.to_numeric(df["今日增持股数"], errors="coerce")
            result["change_market_value"] = pd.to_numeric(df["今日增持资金"], errors="coerce")

            result = result.sort_values("date").reset_index(drop=True)
            # 取最近20条
            result = result.tail(20).reset_index(drop=True)
            logger.debug(f"[akshare] 北向持股: {code}, {len(result)}条")
            return result
        except Exception as e:
            logger.debug(f"[akshare] 北向持股获取失败: {e}")
            return pd.DataFrame()

    def fetch_dividend_history(self, symbol: str) -> pd.DataFrame:
        """获取分红送配历史（东方财富接口）"""
        import akshare as ak

        code = symbol.split(".")[0]
        try:
            df = ak.stock_fhps_detail_em(symbol=code)
            if df.empty:
                return df

            result = pd.DataFrame()
            result["report_date"] = pd.to_datetime(df["报告期"]).dt.date
            result["div_per_share"] = pd.to_numeric(df["现金分红-现金分红比例"], errors="coerce") / 10  # 10派X → 每股X/10
            result["dividend_yield"] = pd.to_numeric(df["现金分红-股息率"], errors="coerce") * 100  # 小数→百分比
            result["eps"] = pd.to_numeric(df["每股收益"], errors="coerce")
            result["ex_date"] = df.get("除权除息日", "").astype(str)

            # 计算派息率
            result["payout_ratio"] = result.apply(
                lambda r: round(r["div_per_share"] / r["eps"] * 100, 1)
                if pd.notna(r["eps"]) and r["eps"] > 0 and pd.notna(r["div_per_share"])
                else None,
                axis=1,
            )

            result = result.sort_values("report_date").reset_index(drop=True)
            logger.debug(f"[akshare] 分红历史: {code}, {len(result)}期")
            return result
        except Exception as e:
            logger.debug(f"[akshare] 分红历史获取失败: {e}")
            return pd.DataFrame()

    def fetch_dragon_tiger(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """获取龙虎榜+大宗交易（合并，东方财富接口）"""
        import akshare as ak
        from datetime import date, timedelta

        code = symbol.split(".")[0]
        end_date = date.today().strftime("%Y%m%d")
        start_date = (date.today() - timedelta(days=days)).strftime("%Y%m%d")

        results = []

        # 1. 龙虎榜（全市场，按代码过滤）
        try:
            df = ak.stock_lhb_detail_em(start_date=start_date, end_date=end_date)
            if not df.empty:
                filtered = df[df["代码"] == code]
                for _, row in filtered.iterrows():
                    results.append({
                        "date": str(row.get("上榜日", ""))[:10],
                        "source": "lhb",
                        "net_buy_amount": float(row.get("龙虎榜净买额", 0) or 0),
                        "buy_amount": float(row.get("龙虎榜买入额", 0) or 0),
                        "sell_amount": float(row.get("龙虎榜卖出额", 0) or 0),
                        "premium_rate": None,
                        "reason": str(row.get("上榜原因", "")),
                    })
        except Exception as e:
            logger.debug(f"[akshare] 龙虎榜获取失败: {e}")

        # 2. 大宗交易（全市场A股，按代码过滤）
        try:
            df = ak.stock_dzjy_mrmx(symbol="A股", start_date=start_date, end_date=end_date)
            if not df.empty:
                filtered = df[df["证券代码"] == code]
                for _, row in filtered.iterrows():
                    results.append({
                        "date": str(row.get("交易日期", ""))[:10],
                        "source": "dzjy",
                        "net_buy_amount": float(row.get("成交额", 0) or 0),
                        "buy_amount": float(row.get("成交额", 0) or 0),
                        "sell_amount": 0,
                        "premium_rate": float(row.get("折溢率", 0) or 0) * 100,  # 小数→百分比
                        "reason": f"买方:{row.get('买方营业部', '')}" if "机构专用" in str(row.get("买方营业部", "")) else "",
                    })
        except Exception as e:
            logger.debug(f"[akshare] 大宗交易获取失败: {e}")

        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("date").reset_index(drop=True)
        logger.debug(f"[akshare] 龙虎榜/大宗: {code}, {len(result_df)}条")
        return result_df

    def fetch_research_reports(self, symbol: str) -> dict:
        """获取个股研报数据（东方财富接口）

        返回结构化摘要：一致评级、EPS预测、覆盖机构数、评级变动、机构参与度。
        """
        import akshare as ak

        code = symbol.split(".")[0]
        result: dict = {}

        # 1. 个股研报列表（评级+EPS预测+机构）
        try:
            df = ak.stock_research_report_em(symbol=code)
            if not df.empty:
                recent = df.head(30)

                rating_col = next((c for c in df.columns if "评级" in str(c)), None)
                inst_col = next((c for c in df.columns if "机构" in str(c)), None)
                date_col = next((c for c in df.columns if "日期" in str(c)), None)
                title_col = next((c for c in df.columns if "报告名称" in str(c) or "标题" in str(c)), None)

                if rating_col:
                    ratings = recent[rating_col].dropna()
                    result["rating_distribution"] = ratings.value_counts().to_dict()
                    result["total_reports"] = len(recent)
                    result["latest_rating"] = str(ratings.iloc[0]) if len(ratings) > 0 else "未知"

                if inst_col:
                    institutions = recent[inst_col].dropna().unique().tolist()
                    result["coverage_count"] = len(institutions)
                    result["institutions"] = institutions[:10]

                # EPS预测（最近5条研报）
                # 列名格式如 "2025-盈利预测-收益", "2026-盈利预测-收益" 等
                eps_cols = [c for c in df.columns if "盈利预测-收益" in str(c) or "EPS" in str(c).upper()]
                if eps_cols:
                    eps_forecasts = []
                    for _, row in recent.head(5).iterrows():
                        forecast = {}
                        for col in eps_cols:
                            val = row.get(col)
                            if pd.notna(val):
                                forecast[str(col)] = float(val)
                        if forecast:
                            forecast["institution"] = str(row.get(inst_col, "")) if inst_col else ""
                            forecast["date"] = str(row.get(date_col, ""))[:10] if date_col else ""
                            eps_forecasts.append(forecast)
                    result["eps_forecasts"] = eps_forecasts

                # 研报标题（最近10条）
                if title_col:
                    titles = []
                    for _, row in recent.head(10).iterrows():
                        entry = {"title": str(row.get(title_col, ""))}
                        if inst_col:
                            entry["institution"] = str(row.get(inst_col, ""))
                        if date_col:
                            entry["date"] = str(row.get(date_col, ""))[:10]
                        if rating_col:
                            entry["rating"] = str(row.get(rating_col, ""))
                        titles.append(entry)
                    result["recent_titles"] = titles

                logger.debug(f"[akshare] 个股研报: {code}, {len(recent)}条, 覆盖{result.get('coverage_count', 0)}家机构")
        except Exception as e:
            logger.debug(f"[akshare] 个股研报获取失败: {e}")

        # 2. 机构参与度（日频时间序列）
        try:
            df2 = ak.stock_comment_detail_zlkp_jgcyd_em(symbol=code)
            if not df2.empty:
                recent_participation = []
                date_col2 = df2.columns[0]
                val_col2 = df2.columns[-1]
                for _, row in df2.tail(10).iterrows():
                    recent_participation.append({
                        "date": str(row.get(date_col2, ""))[:10],
                        "participation": float(row.get(val_col2, 0)),
                    })
                result["institutional_participation"] = recent_participation
                if recent_participation:
                    result["latest_participation"] = recent_participation[-1]["participation"]
                logger.debug(f"[akshare] 机构参与度: {code}, {len(recent_participation)}天")
        except Exception as e:
            logger.debug(f"[akshare] 机构参与度获取失败: {e}")

        return result

    def fetch_margin_data(self, symbol: str) -> pd.DataFrame:
        """获取个股融资融券数据（东方财富/交易所接口）"""
        import akshare as ak
        from datetime import date, timedelta

        code = symbol.split(".")[0]
        is_sh = code.startswith("6")
        target_date = date.today()

        # 尝试最近5个交易日（节假日可能无数据）
        for offset in range(5):
            d = (target_date - timedelta(days=offset)).strftime("%Y%m%d")
            try:
                if is_sh:
                    df = ak.stock_margin_detail_sse(date=d)
                    if df.empty:
                        continue
                    code_col = "标的证券代码"
                else:
                    df = ak.stock_margin_detail_szse(date=d)
                    if df.empty:
                        continue
                    code_col = "证券代码"

                filtered = df[df[code_col].astype(str) == code]
                if filtered.empty:
                    continue

                row = filtered.iloc[0]
                result = pd.DataFrame([{
                    "date": d,
                    "margin_buy": pd.to_numeric(row.get("融资买入额", 0), errors="coerce"),
                    "margin_balance": pd.to_numeric(row.get("融资余额", 0), errors="coerce"),
                    "short_sell": pd.to_numeric(row.get("融券卖出量", row.get("融券卖出量", 0)), errors="coerce"),
                    "short_balance": pd.to_numeric(row.get("融券余量", 0), errors="coerce"),
                    "total_balance": pd.to_numeric(row.get("融资融券余额", 0), errors="coerce"),
                }])
                logger.debug(f"[akshare] 融资融券: {code}, 日期={d}")
                return result
            except Exception:
                continue

        logger.debug(f"[akshare] 融资融券: {code}, 近5日均无数据")
        return pd.DataFrame()

    def fetch_performance_forecast(self, report_date: str = "") -> pd.DataFrame:
        """获取全市场业绩预告（东方财富接口）"""
        import akshare as ak

        if not report_date:
            from datetime import date as d
            today = d.today()
            if today.month >= 11:
                report_date = f"{today.year}0930"
            elif today.month >= 5:
                report_date = f"{today.year - 1}1231"
            else:
                report_date = f"{today.year - 1}0930"

        try:
            df = ak.stock_yjyg_em(date=report_date)
            if df.empty:
                return df

            result = pd.DataFrame()
            result["symbol"] = df["股票代码"].apply(
                lambda x: f"{x}.SZ" if x.startswith(("0", "3")) else f"{x}.SH"
            )
            result["name"] = df["股票简称"]
            result["forecast_type"] = df["预告类型"]
            result["change_pct"] = pd.to_numeric(df["业绩变动幅度"], errors="coerce")
            result["forecast_content"] = df["业绩变动"]
            result["forecast_profit"] = pd.to_numeric(df["预测数值"], errors="coerce")
            result["prev_profit"] = pd.to_numeric(df["上年同期值"], errors="coerce")
            result["announce_date"] = df["公告日期"]
            result["report_date"] = report_date

            logger.debug(f"[akshare] 业绩预告: {len(result)}条, 报告期={report_date}")
            return result
        except Exception as e:
            logger.debug(f"[akshare] 业绩预告获取失败: {e}")
            return pd.DataFrame()

    def fetch_hot_rank(self) -> pd.DataFrame:
        """获取东方财富人气排名Top100"""
        import akshare as ak

        try:
            df = ak.stock_hot_rank_em()
            if df.empty:
                return df

            result = pd.DataFrame()
            result["rank"] = df["当前排名"]
            # 代码格式：SH600410 → 600410.SH
            result["symbol"] = df["代码"].apply(
                lambda x: f"{x[2:]}.{x[:2]}" if len(x) > 2 else x
            )
            result["name"] = df["股票名称"]
            result["price"] = pd.to_numeric(df["最新价"], errors="coerce")
            result["change_pct"] = pd.to_numeric(df["涨跌幅"], errors="coerce")

            logger.debug(f"[akshare] 人气排名: {len(result)}只")
            return result
        except Exception as e:
            logger.debug(f"[akshare] 人气排名获取失败: {e}")
            return pd.DataFrame()

    def fetch_zt_pool(self, date: str = "") -> pd.DataFrame:
        """获取涨停池（东方财富接口）"""
        import akshare as ak
        from datetime import date as d

        if not date:
            # 取最近交易日
            date = d.today().strftime("%Y%m%d")

        try:
            df = ak.stock_zt_pool_em(date=date)
            if df.empty:
                return df

            result = pd.DataFrame()
            result["symbol"] = df["代码"].apply(
                lambda x: f"{x}.SZ" if x.startswith(("0", "3")) else f"{x}.SH"
            )
            result["name"] = df["名称"]
            result["change_pct"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
            result["price"] = pd.to_numeric(df["最新价"], errors="coerce")
            result["amount"] = pd.to_numeric(df["成交额"], errors="coerce")
            result["seal_amount"] = pd.to_numeric(df["封板资金"], errors="coerce")
            result["first_time"] = df["首次封板时间"]
            result["last_time"] = df["最后封板时间"]
            result["break_count"] = pd.to_numeric(df["炸板次数"], errors="coerce")
            result["continuous"] = pd.to_numeric(df["连板数"], errors="coerce")
            result["industry"] = df["所属行业"]

            logger.debug(f"[akshare] 涨停池: {len(result)}只, 日期={date}")
            return result
        except Exception as e:
            logger.debug(f"[akshare] 涨停池获取失败: {e}")
            return pd.DataFrame()

    def fetch_industry_peers(self, symbol: str, industry: str = "") -> pd.DataFrame:
        """获取同行业个股（需传入行业板块名称）"""
        import akshare as ak

        if not industry:
            logger.debug(f"[akshare] 行业成分股: 无行业名称，跳过")
            return pd.DataFrame()

        try:
            cons = ak.stock_board_industry_cons_em(symbol=industry)
            if cons.empty:
                return pd.DataFrame()

            result = pd.DataFrame()
            result["symbol"] = cons["代码"].apply(
                lambda x: f"{x}.SZ" if x.startswith(("0", "3")) else f"{x}.SH"
            )
            result["name"] = cons["名称"]
            result["price"] = pd.to_numeric(cons["最新价"], errors="coerce")
            result["change_pct"] = pd.to_numeric(cons["涨跌幅"], errors="coerce")
            result["pe"] = pd.to_numeric(cons.get("市盈率-动态"), errors="coerce")
            result["pb"] = pd.to_numeric(cons.get("市净率"), errors="coerce")
            result["turnover_rate"] = pd.to_numeric(cons.get("换手率"), errors="coerce")
            result["industry"] = industry

            logger.debug(f"[akshare] 行业成分股: {industry}, {len(result)}只")
            return result
        except Exception as e:
            logger.debug(f"[akshare] 行业成分股获取失败: {e}")
            return pd.DataFrame()

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
