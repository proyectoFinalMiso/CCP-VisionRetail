import smtplib
from collections import defaultdict
from os import getenv

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class SendEmail:

    def __init__(self, body):
        self.body = body
    
    def build_email_body(self, grouped_data):
        img_path = 'https://storage.googleapis.com/ccp-recommendations-videos/buffered-keyframes/'
        html = "<h2>Recomendación de productos y tiendas</h2>"
        for name, messages in grouped_data.items():
            image_url = img_path + name
            html += f"""
            <div style="margin-bottom: 20px;">
                <img src="{image_url}" alt="{name}" style="max-width: 500px;"><br>
                <ul>
            """
            for msg in messages:
                html += f"<li>{msg}</li>"
            html += "</ul></div>"
        return html
    
    def send_email(self, subject, html_body):
        EMAIL_SENDER = getenv("EMAIL_SENDER")
        EMAIL_RECEIVER = [self.body['customer'], [self.body['seller']]]
        SMTP_PASSWORD = getenv("SMTP_PASSWORD")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = getenv("EMAIL_SENDER")
        msg["To"] = getenv("EMAIL_RECEIVER")

        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, SMTP_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
    
    def execute(self):
        grouped = defaultdict(list)
        for rec in self.body["message"]:
            grouped[rec["name"]].append(rec["message"])
        html_body = self.build_email_body(grouped)
        self.send_email("Recomendación de productos y tiendas", html_body)
        return {'response': 'Se ha enviado correctamente el correo', 'status_code': 200}