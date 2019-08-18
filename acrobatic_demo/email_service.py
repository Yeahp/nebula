import smtplib
from email.mime.text import MIMEText
from email.header import Header

"""
We send email via SMTP(Simple Mail Transfer Protocol) and MIME(Multipurpose Internet Mail Extensions), 
which require that both the sender and receiver should open his SMTP service.
Otherwise, the request for sending email will fail.
"""

if __name__ == "__main__":
    sender = "yeerxi@163.com"
    receivers = ['yeerxi@163.com']

    # three parameters: text content, text format and encoding
    message = MIMEText("test for sending email ", "plain", "utf-8")
    message["From"] = Header("yeerxi <yeerxi@163.com>", "utf-8")
    message["To"] = "qierpeng <qierpeng@163.com>"
    subject = "Python SMTP email test"
    message["Subject"] = Header(subject, "utf-8").encode()

    try:
        smtp = smtplib.SMTP()
        smtp.connect('smtp.163.com', 25)
        smtp.set_debuglevel(1)
        smtp.login('yeerxi', 'qierpeng')
        smtp.sendmail(sender, receivers, message.as_string())
        print("Success: finish sending email!")
        smtp.quit()
    except smtplib.SMTPException:
        print("Error: cannot send email!")
