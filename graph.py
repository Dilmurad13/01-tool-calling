from dotenv import load_dotenv
load_dotenv()  # берёт ключ из .env

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

# --- Инструменты: декоратор @tool сам делает JSON-схему из функции ---
@tool
def get_free_slots(date: str) -> list[str]:
    """Свободные окна на дату в формате YYYY-MM-DD."""
    return ["10:00", "12:00", "15:30"]

@tool
def book_appointment(name: str, date: str, time: str) -> str:
    """Записать пациента на дату и время."""
    return f"Готово: {name} записан на {date} в {time}"

tools = [get_free_slots, book_appointment]

# Модель, которой показали инструменты
llm = init_chat_model("groq:openai/gpt-oss-20b").bind_tools(tools)

SYSTEM = {"role": "system",
          "content": "Ты администратор клиники. Сегодня среда, 2026-10-28."}

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

# --- Шаги 3 и 4: проверяем память ---
cfg = ask("Есть окна на пятницу?", "1")
ask("Запиши меня на первое окно, я Дильмурад", "1")   # помнит пятницу
ask("Запиши меня на первое окно, я Дильмурад", "2")   # новый разговор, пятницу не знает

# Что лежит в памяти разговора 1
for m in graph.get_state(cfg).values["messages"]:
    print(type(m).__name__, ":", m.content or m.tool_calls)