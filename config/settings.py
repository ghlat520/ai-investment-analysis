"""
Pydantic Settings — 类型安全的配置管理

所有配置从环境变量/.env文件加载，支持默认值。
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE_CONF = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", **_ENV_FILE_CONF)

    database_url: str = "postgresql+asyncpg://localhost:5432/ai_investment"
    database_url_sync: str = "postgresql+psycopg2://localhost:5432/ai_investment"
    redis_url: str = "redis://localhost:6379/0"


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", **_ENV_FILE_CONF)

    # 全局默认 Provider（ollama/openai/anthropic）
    llm_default_provider: str = "ollama"

    # OpenAI（兼容智谱GLM等OpenAI格式API，通过.env配置base_url）
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_default_model: str = "gpt-4o-mini"

    # Anthropic（支持智谱代理等自定义端点）
    anthropic_api_key: Optional[str] = None
    anthropic_base_url: Optional[str] = None  # 自定义API端点，如智谱代理
    anthropic_default_model: str = "claude-sonnet-4-20250514"

    # Ollama（本地部署）
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_default_model: str = "qwen2.5:14b"
    ollama_max_concurrent: int = 1       # 匹配 OLLAMA_NUM_PARALLEL，默认串行
    ollama_request_timeout: int = 300    # 单次推理超时（秒），不含排队等待

    # 通用
    llm_temperature: float = 0.3
    llm_max_retries: int = 3
    llm_request_timeout: int = 60
    llm_cache_ttl: int = 3600  # 秒

    def get_default_model(self, provider: str) -> str:
        """获取指定 provider 的默认模型"""
        mapping = {
            "openai": self.openai_default_model,
            "anthropic": self.anthropic_default_model,
            "ollama": self.ollama_default_model,
        }
        return mapping.get(provider, "gpt-4o-mini")


class DataSourceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", **_ENV_FILE_CONF)

    tushare_token: Optional[str] = None

    # 数据源冷却时间（秒）
    source_cooldown_seconds: int = 300
    # 数据源请求超时（秒）
    source_request_timeout: int = 30


class SearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", **_ENV_FILE_CONF)

    bocha_api_keys: str = ""  # 逗号分隔
    tavily_api_keys: str = ""
    brave_api_keys: str = ""
    serpapi_keys: str = ""

    def get_key_list(self, provider: str) -> list[str]:
        raw = getattr(self, f"{provider}_api_keys", "") or getattr(self, f"{provider}_keys", "")
        return [k.strip() for k in raw.split(",") if k.strip()]


class NotificationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", **_ENV_FILE_CONF)

    wechat_webhook_url: Optional[str] = None
    feishu_webhook_url: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    email_sender: Optional[str] = None
    email_password: Optional[str] = None
    email_receivers: str = ""  # 逗号分隔


class SchedulerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", **_ENV_FILE_CONF)

    analysis_trigger_time: str = "17:00"
    timezone: str = "Asia/Shanghai"


class ScreeningSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCREENING_", **_ENV_FILE_CONF)

    top_n: int = 50
    min_market_cap: float = 20e8  # 最小市值 20亿
    min_volume_20d: float = 1e6  # 最小20日均成交量
    exclude_st: bool = True
    exclude_new_stock_days: int = 60  # 排除上市不满60天的新股


class Settings(BaseSettings):
    """聚合所有子配置的根配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 子配置
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    data_source: DataSourceSettings = Field(default_factory=DataSourceSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    notification: NotificationSettings = Field(default_factory=NotificationSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    screening: ScreeningSettings = Field(default_factory=ScreeningSettings)

    # 全局
    log_level: str = "INFO"
    debug: bool = False


# 全局单例
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
