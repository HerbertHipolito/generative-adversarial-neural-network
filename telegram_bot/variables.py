from dotenv import load_dotenv
import os

load_dotenv()

telegram_variabels = {
    'key':os.getenv('key'),
    'chat_id':os.getenv('chat_id')
}