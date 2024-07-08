def send_notification(message):
    pass

def llog(message):
    pass

'''
def send_notification(message):
    try:
        url = "https://ntfy.sh/Kiranet_carts_test_notice2"
        headers = {
            "Content-Type": "text/plain"
        }
        response = requests.post(url, data=message, headers=headers)
        print("Notifica inviata con successo!")
    except:
        print(f"Errore nell'invio della notifica: {response.status_code} - {response.text}")
'''