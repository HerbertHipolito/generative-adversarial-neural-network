import telebot
from telegram_bot.variables import telegram_variabels
import time

def send_msg_telegram(msg, max_retries=4):
    
    retries = 0
    while retries < max_retries:
        try:
            bot = telebot.TeleBot(telegram_variabels['key'])
            bot.send_message(telegram_variabels['chat_id'],msg)
            return True
        except Exception as e:
            print(f"An error occurred: {e}")
            retries += 1
            print(f"attempt to reconnect {retries}")
            time.sleep(2)  # Introduce a delay before retrying
            
    return False
        
def send_image(path_img):
    bot = telebot.TeleBot(telegram_variabels['key'])
    try:
        with open(path_img, 'rb') as photo:
            bot.send_photo(telegram_variabels['chat_id'], photo)
    except FileNotFoundError:
        bot.send_message(telegram_variabels['chat_id'], 'Image not found!')
    except Exception as e:
        print(f"Something went wrong during the image saving: {e}")
