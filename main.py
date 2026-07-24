from fastapi import FastAPI
from pydantic import BaseModel
import anthropic
import os

app = FastAPI()
client = anthropic.Anthropic()

SYSTEM_PROMPT = (
    "Your name is Aegis. You are Dale's most trusted and capable "
    "assistant — think chief-of-staff for a CEO. You are sharp, "
    "resourceful, and genuinely invested in his success. Right now "
    "you're talking to him through his phone, likely while he's "
    "driving his truck, so keep responses natural, conversational, "
    "and not overly long — this is a voice conversation, not a "
    "document. Your loyalty means genuinely serving his best "
    "interests — give honest assessments, don't just agree with "
    "everything. Always prioritize safety."
)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def read_root():
    return {"status": "Aegis backend is alive"}


@app.post("/chat")
def chat(request: ChatRequest):
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": request.message}]
    )
    reply = "".join(block.text for block in response.content if block.type == "text")
    return {"reply": reply}