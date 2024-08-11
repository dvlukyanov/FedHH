import requests
from architectures.shared.config import Config

__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


def notify_slack(message):
    headers = {
        'Content-Type': "application/json",
    }
    response = requests.post(
        url=Config()['notification']['slack'],
        json={"text": message},
        headers=headers
    )
    if response.status_code == 200:
        print(f"Notification sent: {message}")
    else:
        print(f"Notification failed: {response.content}")