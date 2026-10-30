from dotenv import load_dotenv
load_dotenv()  # берёт ключ из .env

from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

CONSENT = {"да", "ок", "подтверждаю"}


def _last_user_text(messages) -> str:
    """Текст последнего сообщения клиента."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
        elif isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content", "")
        else:
            continue
        return content if isinstance(content, str) else str(content)
    return ""


def _is_consent(text: str) -> bool:
    """Согласие — всё сообщение целиком: да, ок или подтверждаю."""
    return text.strip().lower().rstrip(".!?") in CONSENT


# --- Инструменты: декоратор @tool сам делает JSON-схему из функции ---
@tool
def get_free_slots(date: str) -> list[str]:
    """Свободные окна на дату в формате YYYY-MM-DD."""
    return ["10:00", "12:00", "15:30"]

@tool
def book_appointment(
    name: str,
    date: str,
    time: str,
    state: Annotated[dict, InjectedState],
) -> str:
    """Записать пациента на дату и время. Только после явного согласия клиента."""
    if not _is_consent(_last_user_text(state["messages"])):
        return "Сначала спроси подтверждение у клиента"
    return f"Готово: {name} записан на {date} в {time}"

@tool
def handoff_to_human(reason: str) -> str:
    """Передать вопрос живому администратору.
    Клиент просит человека, жалуется, описывает срочный или опасный случай,
    или спрашивает то, на что нет инструментов.
    """
    print(f"🔔 АДМИНИСТРАТОРУ: {reason}")
    return "Передал ваш вопрос администратору, он скоро ответит"

tools = [get_free_slots, book_appointment, handoff_to_human]

# Модель, которой показали инструменты
llm = init_chat_model("groq:openai/gpt-oss-20b").bind_tools(tools)

SYSTEM = {"role": "system",
          "content": (
              "Ты администратор клиники. Сегодня среда, 2026-10-28. "
              "Перед вызовом book_appointment повтори клиенту имя, дату и время "
              "и спроси «Подтверждаете?». "
              "Вызывай book_appointment только после явного согласия клиента: "
              "«да», «ок» или «подтверждаю». "
              "Вызывай handoff_to_human, если клиент просит человека, жалуется, "
              "описывает срочный или опасный случай, или спрашивает то, "
              "на что у тебя нет инструментов."
          )}

# --- Узел model: зовёт модель с текущей перепиской ---
def call_model(state: MessagesState):
    answer = llm.invoke([SYSTEM] + state["messages"])
    return {"messages": [answer]}  # ответ добавится в State

# --- Собираем граф ---
builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_node("tools", ToolNode(tools))       # готовый узел, выполняет инструменты
builder.add_edge(START, "model")                 # начинаем с модели
builder.add_conditional_edges("model", tools_condition)  # просит инструмент -> tools, иначе конец
builder.add_edge("tools", "model")               # после инструмента снова к модели

graph = builder.compile(checkpointer=InMemorySaver())  # память включена

def ask(text, thread):
    config = {"configurable": {"thread_id": thread}}
    result = graph.invoke({"messages": [{"role": "user", "content": text}]}, config)
    print(f"[разговор {thread}] Бот:", result["messages"][-1].content)
    return config

if __name__ == "__main__":
    print("\n=== разговор yes: запись после «да» ===")
    ask("Какие свободные окна на 2026-10-30?", "yes")
    ask("Запишите меня, Дильмурад, на 10:00", "yes")
    ask("да", "yes")

    print("\n=== разговор change: отказ и другое время ===")
    ask("Какие свободные окна на 2026-10-30?", "change")
    ask("Запишите меня, Дильмурад, на 10:00", "change")
    ask("нет, давайте в 11", "change")

    print("\n=== разговор human: передача администратору ===")
    ask("Позовите администратора", "human")
