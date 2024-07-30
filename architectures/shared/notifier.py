import requests

def notify_slack(url, message):
    headers = {
        'Content-Type': "application/json",
    }
    response = requests.post(
        url=url,
        json={"text": message},
        headers=headers
    )
    if response.status_code == 200:
        print(f"Notification sent: {message}")
    else:
        print(f"Notification failed: {response.content}")