import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
client = OpenAI()
MODEL = "gpt-4o-mini"  # любая доступная тебе модель

# --- ШАГ 1. Инструменты-заглушки (вместо настоящей базы клиники) ---
SLOTS = {"2026-10-27": ["10:00", "12:00", "15:30"]}
BOOKINGS = []

def get_free_slots(date):
    return {"date": date, "slots": SLOTS.get(date, [])}

def book_appointment(name, phone, slot):
    BOOKINGS.append({"name": name, "phone": phone, "slot": slot})
    return {"status": "ok", "booking_id": len(BOOKINGS)}

FUNCTIONS = {"get_free_slots": get_free_slots,
             "book_appointment": book_appointment}

# Описание инструментов для модели (JSON-схемы)
TOOLS = [
    {"type": "function", "function": {
        "name": "get_free_slots",
        "description": "Показать свободные слоты на дату",
        "parameters": {"type": "object",
            "properties": {"date": {"type": "string", "description": "Дата YYYY-MM-DD"}},
            "required": ["date"]}}},
    {"type": "function", "function": {
        "name": "book_appointment",
        "description": "Записать пациента на слот",
        "parameters": {"type": "object",
            "properties": {
                "name": {"type": "string"},
                "phone": {"type": "string"},
                "slot": {"type": "string", "description": "YYYY-MM-DD HH:MM"}},
            "required": ["name", "phone", "slot"]}}},
]

messages = [{"role": "system", "content":
    "Ты администратор клиники. Сегодня 2026-10-26. "
    "Перед записью узнай имя, телефон и услугу."}]

# --- ШАГ 2. Цикл агента ---
def run_agent(user_text):
    messages.append({"role": "user", "content": user_text})
    while True:
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:          # модель ничего не просит, значит это финальный ответ
            return msg.content

        for call in msg.tool_calls:     # модель просит вызвать инструмент(ы)
            args = json.loads(call.function.arguments)
            print(f"  [инструмент] {call.function.name}({args})")
            result = FUNCTIONS[call.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, ensure_ascii=False)})

# --- ШАГ 3. Structured output: итоговая заявка по Pydantic-схеме ---
class Booking(BaseModel):
    name: str
    phone: str
    slot: str
    service: str

def extract_booking():
    resp = client.chat.completions.parse(
        model=MODEL,
        messages=messages + [{"role": "user", "content": "Собери итоговую заявку из диалога."}],
        response_format=Booking)
    return resp.choices[0].message.parsed

if __name__ == "__main__":
    while True:
        text = input("Вы: ")
        if text == "выход":
            break
        print("Бот:", run_agent(text))
    print("Заявка:", extract_booking())