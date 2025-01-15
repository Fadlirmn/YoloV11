import requests

# Replace 'YOUR_BOT_TOKEN' with your actual bot token
TOKEN = "7029178812:AAF3JlXBlNsVKcG34Dr0G4PDDb3jD0MqD9g"
BASE_URL = f"https://api.telegram.org/bot{TOKEN}"

def send_message(chat_id, text):
    # Send a message to a specific chat ID
    url = f"{BASE_URL}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text
    }
    response = requests.post(url, data=payload)
    
    if response.status_code == 200:
        print("Message sent successfully!")
    else:
        print(f"Failed to send message. Status code: {response.status_code}")
        print(response.json())

if __name__ == "__main__":
    # Replace 'YOUR_CHAT_ID' with the actual chat ID you want to send a message to
    chat_id = "@trafficitera"
    message = "Hello, this is a test message!"
    
    send_message(chat_id, message)
