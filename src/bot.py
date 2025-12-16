
import telebot
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from src.model import KoreanChatbotModel


load_dotenv()


BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or config.TELEGRAM_CONFIG.get("token")

if not BOT_TOKEN or BOT_TOKEN == "ТВОЙ_ТОКЕН_ЕСЛИ_НЕТ_ENV":
    print("ОШИБКА: Токен не найден!")
    print("Создай файл .env с содержимым: TELEGRAM_BOT_TOKEN=твой_токен")
    sys.exit(1)

bot = telebot.TeleBot(BOT_TOKEN)

print("Инициализация нейросети...")
try:
    chatbot = KoreanChatbotModel()
    print("Модель успешно загружена!")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    chatbot = None

@bot.message_handler(commands=['start'])
def send_welcome(message):
    welcome_text = "안녕하세요! 👋\nЯ бот для практики корейского языка. Напиши мне что-нибудь!"
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['help'])
def send_help(message):
    bot.reply_to(message, "Просто пиши мне на корейском, а я постараюсь ответить.")


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    if not chatbot:
        bot.reply_to(message, "Извините, модель сейчас недоступна (технические работы).")
        return

    bot.send_chat_action(message.chat.id, 'typing')

    try:
        user_input = message.text
        response = chatbot.generate_response(user_input, user_id=message.chat.id)

        if response is None:
            print("ВНИМАНИЕ: generate_response вернул None.")
            response = "죄송합니다, 잠시 문제가 생겼어요."

        if not response.strip():
            response = "죄송합니다, 이해하지 못했습니다. (Извините, я не понял)"

        bot.reply_to(message, response)
    except Exception as e:
        print(f"Error generating response: {e}")
        bot.reply_to(message, "Произошла ошибка при обработке сообщения.")

if __name__ == "__main__":
    print("Бот запущен и готов к работе...")
    bot.infinity_polling()
