"""
PostgreSQL 数据库连接管理

SQLAlchemy 2.0 异步引擎 + 会话管理。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from loguru import logger


class Database:
    """数据库连接管理器"""

    def __init__(self, url: str, echo: bool = False) -> None:
        self._engine: AsyncEngine = create_async_engine(
            url,
            echo=echo,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info(f"数据库引擎创建完成: {url.split('@')[-1]}")  # 不打印密码

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """获取数据库会话"""
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def close(self) -> None:
        """关闭数据库连接"""
        await self._engine.dispose()
        logger.info("数据库连接已关闭")


# 全局实例
_db: Optional[Database] = None


def get_database() -> Database:
    global _db
    if _db is None:
        from config.settings import get_settings
        settings = get_settings()
        _db = Database(settings.db.database_url)
    return _db
