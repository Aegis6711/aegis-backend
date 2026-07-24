from fastapi import FastAPI
from pydantic import BaseModel
import anthropic
import os
from supabase import create_client
import json

app = FastAPI()
client = anthropic.Anthropic()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

SYSTEM_PROMPT = (
    "Your name is Aegis. You are Dale's most trusted and capable "
    "assistant — think chief-of-staff for a CEO. You are sharp, "
    "resourceful, and genuinely invested in his success. Right now "
    "you're talking to him through his phone, likely while he's "
    "driving his truck, so keep responses natural, conversational, "
    "and not overly long — this is a voice conversation, not a "
    "document. Your loyalty means genuinely serving his best "
    "interests — give honest assessments, don't just agree with "
    "everything. You can manage his calendar, track his budget, and "
    "take quick notes. Always prioritize safety."
)

TOOLS = [
    {
        "name": "add_event",
        "description": "Add a calendar event.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "event_date": {"type": "string", "description": "YYYY-MM-DD"},
                "event_time": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["title", "event_date"]
        }
    },
    {
        "name": "list_events",
        "description": "List upcoming calendar events.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "add_transaction",
        "description": "Log an income or expense.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "type": {"type": "string", "enum": ["income", "expense"]},
                "category": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["amount", "type", "category"]
        }
    },
    {
        "name": "get_budget_summary",
        "description": "Get recent income/expense summary.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "add_note",
        "description": "Save a quick note.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["title", "content"]
        }
    },
    {
        "name": "list_notes",
        "description": "List saved notes.",
        "input_schema": {"type": "object", "properties": {}}
    }
]


def execute_tool(name, tool_input):
    try:
        if name == "add_event":
            supabase.table("calendar_events").insert({
                "title": tool_input["title"],
                "event_date": tool_input["event_date"],
                "event_time": tool_input.get("event_time"),
                "description": tool_input.get("description")
            }).execute()
            return f"Event added: {tool_input['title']} on {tool_input['event_date']}"

        elif name == "list_events":
            result = supabase.table("calendar_events").select("*").order("event_date").execute()
            if not result.data:
                return "No upcoming events."
            lines = [f"{e['event_date']} {e.get('event_time') or ''} — {e['title']}" for e in result.data]
            return "\n".join(lines)

        elif name == "add_transaction":
            supabase.table("budget_transactions").insert({
                "amount": tool_input["amount"],
                "type": tool_input["type"],
                "category": tool_input["category"],
                "description": tool_input.get("description")
            }).execute()
            return f"Logged {tool_input['type']} of ${tool_input['amount']:.2f} in {tool_input['category']}"

        elif name == "get_budget_summary":
            result = supabase.table("budget_transactions").select("*").execute()
            income = sum(t["amount"] for t in result.data if t["type"] == "income")
            expense = sum(t["amount"] for t in result.data if t["type"] == "expense")
            return f"Total income: ${income:.2f}, total expenses: ${expense:.2f}, net: ${income - expense:.2f}"

        elif name == "add_note":
            supabase.table("quick_notes").insert({
                "title": tool_input["title"],
                "content": tool_input["content"]
            }).execute()
            return f"Note saved: {tool_input['title']}"

        elif name == "list_notes":
            result = supabase.table("quick_notes").select("*").order("created_at", desc=True).execute()
            if not result.data:
                return "No notes saved."
            return "\n".join(f"{n['title']}: {n['content']}" for n in result.data)

        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        return f"Error: {e}"


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def read_root():
    return {"status": "Aegis backend is alive"}


@app.post("/chat")
def chat(request: ChatRequest):
    messages = [{"role": "user", "content": request.message}]
    reply = None

    for _ in range(5):
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=TOOLS
        )
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if tool_use_blocks:
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in tool_use_blocks:
                result_text = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text
                })
            messages.append({"role": "user", "content": tool_results})
        else:
            reply = "".join(b.text for b in response.content if b.type == "text")
            break

    if reply is None:
        reply = "I had trouble completing that."

    return {"reply": reply}