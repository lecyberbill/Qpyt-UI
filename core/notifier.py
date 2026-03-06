import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import logging
from core.config import config

logger = logging.getLogger("qpyt-notifier")

class Notifier:
    @staticmethod
    def send_email(subject: str, body: str, to_email: str = None, attachment_path: str = None):
        """Send an email notification if enabled in config. Returns (success, message)"""
        try:
            # Access settings via config singleton
            notifications = getattr(config, "NOTIFICATIONS", {})
            
            if not notifications.get("EMAIL_ENABLED", False):
                return True, "Email disabled"

            to_email = to_email or notifications.get("EMAIL_TO")
            smtp_server = notifications.get("SMTP_SERVER")
            smtp_port = notifications.get("SMTP_PORT", 587)
            smtp_user = notifications.get("SMTP_USER")
            smtp_pass = notifications.get("SMTP_PASSWORD")

            if not all([to_email, smtp_server, smtp_user, smtp_pass]):
                msg = "[Notifier] Email settings incomplete. Check config.json."
                logger.warning(msg)
                return False, msg

            msg = MIMEMultipart()
            msg['From'] = smtp_user
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            # Handle attachment
            if attachment_path and os.path.exists(attachment_path):
                try:
                    with open(attachment_path, "rb") as f:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename={os.path.basename(attachment_path)}",
                    )
                    msg.attach(part)
                except Exception as att_err:
                    logger.error(f"[Notifier] Failed to attach file: {att_err}")
                    # We continue sending even if attachment fails

            # Standard SMTP with STARTTLS
            with smtplib.SMTP(smtp_server, smtp_port, timeout=10) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
            
            logger.info(f"[Notifier] Notification email sent to {to_email}")
            return True, f"Email sent to {to_email}"
        except Exception as e:
            error_msg = f"[Notifier] Failed to send email: {e}"
            logger.error(error_msg)
            return False, error_msg
