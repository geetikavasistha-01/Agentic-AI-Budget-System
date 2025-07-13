import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv
from langchain.tools import tool
import logging

# Load environment variables from .env
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@tool("send_notification_email", return_direct=True)
def send_notification_email(message: str, subject: str = "\u26a0\ufe0f Financial Validation Alert") -> str:
    """
    Sends an email notification with the given message and subject using SMTP credentials from .env.
    Args:
        message (str): The alert or error message to send.
        subject (str): The email subject (default: Financial Validation Alert).
    Returns:
        str: Confirmation of success or failure.
    """
    sender = os.getenv("ALERT_SENDER_EMAIL")
    receiver = os.getenv("ALERT_RECEIVER_EMAIL")
    password = os.getenv("ALERT_EMAIL_PASSWORD")

    if not all([sender, receiver, password]):
        error_msg = "❌ Missing email configuration in .env file."
        logging.error(error_msg)
        return error_msg

    # Compose the email
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    body = f"{message}\n\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    msg.attach(MIMEText(body, "plain"))

    # Print alert summary
    print("\n==== EMAIL ALERT SUMMARY ====")
    print(f"To: {receiver}")
    print(f"Subject: {subject}")
    print(f"Message:\n{message}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("============================\n")

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        logging.info("✅ Email sent successfully.")
        return f"✅ Alert email sent to {receiver}."
    except Exception as e:
        error_msg = f"❌ Failed to send email: {e}"
        logging.error(error_msg)
        return error_msg
