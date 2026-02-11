"""
通知管理器

多通道通知推送：企业微信、飞书、Telegram、邮件。
复用 daily_stock_analysis/src/notification.py 的模式。
"""

from __future__ import annotations

from typing import Optional

import requests
from loguru import logger


class NotificationManager:
    """通知管理器 — 多通道推送"""

    def __init__(
        self,
        wechat_webhook_url: Optional[str] = None,
        feishu_webhook_url: Optional[str] = None,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
    ) -> None:
        self._channels: list[dict] = []

        if wechat_webhook_url:
            self._channels.append({"type": "wechat", "url": wechat_webhook_url})
        if feishu_webhook_url:
            self._channels.append({"type": "feishu", "url": feishu_webhook_url})
        if telegram_bot_token and telegram_chat_id:
            self._channels.append({
                "type": "telegram",
                "token": telegram_bot_token,
                "chat_id": telegram_chat_id,
            })

        logger.info(f"通知渠道: {[c['type'] for c in self._channels]}")

    def send(self, title: str, content: str) -> None:
        """向所有渠道发送通知"""
        for channel in self._channels:
            try:
                if channel["type"] == "wechat":
                    self._send_wechat(channel["url"], title, content)
                elif channel["type"] == "feishu":
                    self._send_feishu(channel["url"], title, content)
                elif channel["type"] == "telegram":
                    self._send_telegram(channel["token"], channel["chat_id"], title, content)
            except Exception as e:
                logger.error(f"[{channel['type']}] 发送失败: {e}")

    @staticmethod
    def _send_wechat(url: str, title: str, content: str) -> None:
        """企业微信 Webhook"""
        # 企业微信限制4096字节
        text = f"## {title}\n\n{content}"
        if len(text.encode("utf-8")) > 4000:
            text = text[:1500] + "\n\n...(内容过长已截断)"

        resp = requests.post(url, json={
            "msgtype": "markdown",
            "markdown": {"content": text},
        }, timeout=10)
        resp.raise_for_status()
        logger.debug(f"[wechat] 发送成功")

    @staticmethod
    def _send_feishu(url: str, title: str, content: str) -> None:
        """飞书 Webhook"""
        text = f"{title}\n\n{content}"
        if len(text.encode("utf-8")) > 20000:
            text = text[:8000] + "\n\n...(内容过长已截断)"

        resp = requests.post(url, json={
            "msg_type": "text",
            "content": {"text": text},
        }, timeout=10)
        resp.raise_for_status()
        logger.debug(f"[feishu] 发送成功")

    @staticmethod
    def _send_telegram(token: str, chat_id: str, title: str, content: str) -> None:
        """Telegram Bot"""
        text = f"*{title}*\n\n{content}"
        # Telegram 4096字符限制
        if len(text) > 4000:
            text = text[:3900] + "\n\n...(内容过长已截断)"

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }, timeout=10)
        resp.raise_for_status()
        logger.debug(f"[telegram] 发送成功")
