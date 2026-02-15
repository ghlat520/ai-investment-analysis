"""
数据库连接管理

Phase 1：同步 SQLite（本地开发）
Phase 2+：异步 PostgreSQL（生产环境）
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from loguru import logger

from .models import Base


_DEFAULT_SQLITE_PATH = Path(__file__).parent.parent.parent.parent / "data" / "ai_invest.db"


class Database:
    """同步数据库连接管理器"""

    def __init__(self, url: Optional[str] = None, echo: bool = False) -> None:
        if url is None:
            _DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{_DEFAULT_SQLITE_PATH}"

        self._url = url
        connect_args = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        self._engine = create_engine(url, echo=echo, connect_args=connect_args)
        self._session_factory = sessionmaker(
            self._engine,
            expire_on_commit=False,
        )
        # 隐藏密码
        display_url = url.split("@")[-1] if "@" in url else url
        logger.debug(f"数据库引擎: {display_url}")

    def create_tables(self) -> None:
        """创建所有表"""
        Base.metadata.create_all(self._engine)
        logger.info("数据库表创建完成")

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """获取数据库会话"""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self) -> None:
        """关闭数据库连接"""
        self._engine.dispose()


# 全局实例
_db: Optional[Database] = None


def get_database() -> Database:
    global _db
    if _db is None:
        _db = Database()
        atexit.register(_db.close)
    return _db
