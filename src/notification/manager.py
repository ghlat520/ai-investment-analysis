"""
通知管理器

多通道通知推送：企业微信、飞书、Telegram、邮件。
自动检测已配置的渠道（从环境变量），向所有可用渠道推送。
"""

from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import httpx
from loguru import logger


# SMTP服务器自动检测
_SMTP_SERVERS: dict[str, tuple[str, int, bool]] = {
    "qq.com": ("smtp.qq.com", 465, True),
    "163.com": ("smtp.163.com", 465, True),
    "126.com": ("smtp.126.com", 465, True),
    "gmail.com": ("smtp.gmail.com", 587, False),
    "outlook.com": ("smtp-mail.outlook.com", 587, False),
    "hotmail.com": ("smtp-mail.outlook.com", 587, False),
    "aliyun.com": ("smtp.aliyun.com", 465, True),
}


class NotificationManager:
    """通知管理器 — 多通道推送

    用法:
        mgr = NotificationManager.from_env()  # 从环境变量自动配置
        mgr.send("分析报告", "报告内容...")
    """

    def __init__(
        self,
        wechat_webhook_url: Optional[str] = None,
        feishu_webhook_url: Optional[str] = None,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        email_sender: Optional[str] = None,
        email_password: Optional[str] = None,
        email_receivers: Optional[list[str]] = None,
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
        if email_sender and email_password and email_receivers:
            self._channels.append({
                "type": "email",
                "sender": email_sender,
                "password": email_password,
                "receivers": email_receivers,
            })

        if self._channels:
            logger.info(f"通知渠道: {[c['type'] for c in self._channels]}")
        else:
            logger.debug("未配置任何通知渠道")

    @classmethod
    def from_env(cls) -> "NotificationManager":
        """从环境变量自动检测并配置通知渠道"""
        receivers_str = os.environ.get("EMAIL_RECEIVERS", "")
        receivers = [r.strip() for r in receivers_str.split(",") if r.strip()] if receivers_str else None

        return cls(
            wechat_webhook_url=os.environ.get("WECHAT_WEBHOOK_URL") or None,
            feishu_webhook_url=os.environ.get("FEISHU_WEBHOOK_URL") or None,
            telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN") or None,
            telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID") or None,
            email_sender=os.environ.get("EMAIL_SENDER") or None,
            email_password=os.environ.get("EMAIL_PASSWORD") or None,
            email_receivers=receivers,
        )

    @property
    def channel_names(self) -> list[str]:
        return [c["type"] for c in self._channels]

    @property
    def has_channels(self) -> bool:
        return len(self._channels) > 0

    def send(self, title: str, content: str) -> dict[str, bool]:
        """向所有渠道发送通知

        Returns: {channel_type: success_bool}
        """
        results = {}
        for channel in self._channels:
            ch_type = channel["type"]
            try:
                if ch_type == "wechat":
                    self._send_wechat(channel["url"], title, content)
                elif ch_type == "feishu":
                    self._send_feishu(channel["url"], title, content)
                elif ch_type == "telegram":
                    self._send_telegram(channel["token"], channel["chat_id"], title, content)
                elif ch_type == "email":
                    self._send_email(channel["sender"], channel["password"], channel["receivers"], title, content)
                results[ch_type] = True
                logger.info(f"[{ch_type}] 推送成功")
            except Exception as e:
                results[ch_type] = False
                logger.error(f"[{ch_type}] 推送失败: {e}")
        return results

    @staticmethod
    def _send_wechat(url: str, title: str, content: str) -> None:
        """企业微信 Webhook（markdown, 4096字节限制）"""
        text = f"## {title}\n\n{content}"
        text_bytes = text.encode("utf-8")
        if len(text_bytes) > 4000:
            # 按段落智能截断
            text = _smart_truncate(text, 3800) + "\n\n...(内容过长已截断)"

        with httpx.Client(timeout=10) as client:
            resp = client.post(url, json={
                "msgtype": "markdown",
                "markdown": {"content": text},
            })
            resp.raise_for_status()

    @staticmethod
    def _send_feishu(url: str, title: str, content: str) -> None:
        """飞书 Webhook（富文本卡片, 20KB限制）"""
        text = f"{title}\n\n{content}"
        if len(text.encode("utf-8")) > 20000:
            text = _smart_truncate(text, 18000) + "\n\n...(内容过长已截断)"

        with httpx.Client(timeout=10) as client:
            resp = client.post(url, json={
                "msg_type": "interactive",
                "card": {
                    "header": {"title": {"tag": "plain_text", "content": title}},
                    "elements": [{"tag": "markdown", "content": text}],
                },
            })
            resp.raise_for_status()

    @staticmethod
    def _send_telegram(token: str, chat_id: str, title: str, content: str) -> None:
        """Telegram Bot（Markdown, 4096字符限制）"""
        # Telegram Markdown: **bold** → *bold*, ## header → 去掉
        text = f"*{title}*\n\n{content}"
        text = text.replace("**", "*").replace("## ", "").replace("### ", "")
        if len(text) > 4000:
            text = text[:3900] + "\n\n...(内容过长已截断)"

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        with httpx.Client(timeout=15) as client:
            resp = client.post(url, json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown",
            })
            resp.raise_for_status()

    @staticmethod
    def _send_email(sender: str, password: str, receivers: list[str], title: str, content: str) -> None:
        """Email（SMTP, 自动检测服务器）"""
        domain = sender.split("@")[-1].lower()
        smtp_cfg = _SMTP_SERVERS.get(domain)
        if smtp_cfg is None:
            # 默认尝试TLS
            smtp_cfg = (f"smtp.{domain}", 587, False)

        host, port, use_ssl = smtp_cfg

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[AI投研] {title}"
        msg["From"] = sender
        msg["To"] = ", ".join(receivers)

        # 纯文本 + 简单HTML
        msg.attach(MIMEText(content, "plain", "utf-8"))
        html = _markdown_to_simple_html(title, content)
        msg.attach(MIMEText(html, "html", "utf-8"))

        if use_ssl:
            with smtplib.SMTP_SSL(host, port, timeout=15) as smtp:
                smtp.login(sender, password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=15) as smtp:
                smtp.starttls()
                smtp.login(sender, password)
                smtp.send_message(msg)


def _smart_truncate(text: str, max_bytes: int) -> str:
    """按段落智能截断，不破坏UTF-8"""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text

    # 按段落分割，逐段累加
    paragraphs = text.split("\n\n")
    result = []
    current_size = 0

    for p in paragraphs:
        p_size = len(p.encode("utf-8")) + 2  # +2 for \n\n
        if current_size + p_size > max_bytes:
            break
        result.append(p)
        current_size += p_size

    return "\n\n".join(result) if result else text[:max_bytes // 3]


def _markdown_to_simple_html(title: str, content: str) -> str:
    """简单的Markdown→HTML转换（邮件用）"""
    html_content = content
    # 标题
    html_content = html_content.replace("### ", "<h4>").replace("\n", "</h4>\n", 1) if "### " in html_content else html_content
    # 加粗
    import re
    html_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_content)
    # 列表
    lines = []
    for line in html_content.split("\n"):
        if line.startswith("- "):
            lines.append(f"<li>{line[2:]}</li>")
        else:
            lines.append(f"<p>{line}</p>" if line.strip() else "")
    html_content = "\n".join(lines)

    return f"""<html><body style="font-family:sans-serif;max-width:800px;margin:0 auto;padding:20px;">
<h2>{title}</h2>
{html_content}
<hr><p style="color:#999;font-size:12px;">AI投研助手系统自动发送</p>
</body></html>"""
