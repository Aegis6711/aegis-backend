from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import urllib.request
import anthropic
import os
from supabase import create_client
import json

app = FastAPI()
client = anthropic.Anthropic()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

print(f"[DEBUG] SUPABASE_URL present: {supabase_url is not None}, length: {len(supabase_url) if supabase_url else 0}")
print(f"[DEBUG] SUPABASE_KEY present: {supabase_key is not None}, length: {len(supabase_key) if supabase_key else 0}")

supabase = create_client(supabase_url, supabase_key)

def load_user_facts():
    try:
        result = supabase.table("user_facts").select("fact_key, fact_value").execute()
        return {f["fact_key"]: f["fact_value"] for f in result.data}
    except Exception as e:
        print(f"[Facts] Could not load user facts: {e}")
        return {}


def build_system_prompt(location=None):
    facts = load_user_facts()
    facts_text = ""
    if facts:
        facts_lines = "\n".join(f"- {k}: {v}" for k, v in facts.items())
        facts_text = (
            f"\n\nSTANDING FACTS ABOUT DALE (permanently saved, always true "
            f"unless he tells you otherwise):\n{facts_lines}\n"
            f"Use these naturally without being told again."
        )

    location_text = ""
    if location:
        location_text = (
            f"\n\n🚨 IMPORTANT — DALE'S LIVE LOCATION IS KNOWN: {location}. "
            f"This is real, current, GPS-based data — not a guess. "
            f"Whenever he asks about anything nearby (fuel, food, weather, "
            f"rest stops, weigh stations, directions, 'around here', etc.), "
            f"you ALREADY KNOW he is in {location} — use it immediately, "
            f"do not say you don't have his location, do not ask where he "
            f"is. You have it: {location}."
        )
    else:
        location_text = (
            "\n\nNote: Dale's location is not currently available for this "
            "message — if he asks something location-dependent, let him "
            "know you don't have his current location rather than guessing."
        )

    return (
        "Your name is Aegis. You are Dale's most trusted and capable "
        "assistant — think chief-of-staff for a CEO. You are sharp, "
        "resourceful, and genuinely invested in his success. Right now "
        "you're talking to him through his phone, likely while he's "
        "driving his truck, so keep responses natural, conversational, "
        "and not overly long — this is a voice conversation, not a "
        "document. Your loyalty means genuinely serving his best "
        "interests — give honest assessments, don't just agree with "
        "everything. You can manage his calendar, track his budget, "
        "take quick notes, and search the web for current information — "
        "use web_search for any factual/checkable question rather than "
        "relying purely on memory, especially anything that could have "
        "changed. When a topic needs real depth, use web_search to find "
        "promising pages, then deep_research on the best one to pull its "
        "full content, rather than just skimming search snippets. If you "
        "can't find a clear answer, say so plainly rather than guessing. "
        ""
        "You genuinely have PERSISTENT MEMORY across every conversation, "
        "shared across all of Dale's devices — this is real, not a "
        "limitation to apologize for. NEVER tell Dale you won't remember "
        "something after this conversation ends — that is false. When he "
        "tells you something important to remember (corrections, "
        "preferences, facts about him), use remember_fact to save it "
        "permanently right then, rather than just verbally acknowledging "
        "it. If Dale references something from the past you don't "
        "immediately see, use search_past_conversations before assuming "
        "you don't have access. "
        f"{facts_text}"
        f"{location_text}"
        ""
        "You can read Dale's recent emails and send emails on his behalf "
        "via his connected Gmail — always confirm with him verbally before "
        "actually sending anything, reading is fine without confirmation. "
        ""
        "Always prioritize safety, especially since he's likely driving — "
        "keep him focused on the road, not the phone."
    )

FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
COMPOSIO_API_KEY = os.environ.get("COMPOSIO_API_KEY")

from composio import Composio
composio_client = Composio(api_key=COMPOSIO_API_KEY) if COMPOSIO_API_KEY else None

TOOLS = [
    {"type": "web_search_20250305", "name": "web_search"},
    {
        "name": "search_past_conversations",
        "description": "Search back through the full conversation history (not just the recent messages) for a specific topic or thing Dale mentioned before. Use this whenever he references a past conversation you don't currently have visibility into — e.g. 'remember when I told you about X' or 'what did we discuss the other day about Y'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keyword or phrase to search for in past messages."}
            },
            "required": ["query"]
        }
    },
    {
        "name": "remember_fact",
        "description": "Permanently save an important fact, correction, or preference about Dale — e.g. correct spellings, personal details, standing preferences. This is saved forever across all devices and conversations. No confirmation needed, just save it when he tells you something worth remembering.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short label for this fact, e.g. 'email_address' or 'favorite_coffee_order'."},
                "value": {"type": "string", "description": "The actual fact/value to remember."}
            },
            "required": ["key", "value"]
        }
    },
    {
        "name": "read_recent_emails",
        "description": "Read Dale's most recent emails from his connected Gmail. Read-only, no confirmation needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "max_results": {"type": "integer", "description": "How many recent emails to fetch. Defaults to 5."}
            }
        }
    },
    {
        "name": "send_email",
        "description": "Send an email from Dale's connected Gmail. Requires user confirmation before sending.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string"},
                "body": {"type": "string"}
            },
            "required": ["to", "subject", "body"]
        }
    },
    {
        "name": "deep_research",
        "description": "Deeply read and extract the full clean content of a specific webpage URL — use this after web_search finds a promising page, when you need the complete article/page content rather than just a search snippet, for thorough research.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The exact URL to read in full."}
            },
            "required": ["url"]
        }
    },
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
        if name == "search_past_conversations":
            query = tool_input["query"]
            result = supabase.table("phone_conversation_history").select("role, content, created_at").ilike("content", f"%{query}%").order("created_at", desc=True).limit(15).execute()
            if not result.data:
                return f"No past conversation found mentioning '{query}'."
            lines = [f"[{m['created_at'][:10]}] {m['role']}: {m['content']}" for m in result.data]
            return "Relevant past messages found:\n" + "\n".join(lines)

        elif name == "remember_fact":
            key = tool_input["key"]
            value = tool_input["value"]
            try:
                supabase.table("user_facts").upsert({"fact_key": key, "fact_value": value}, on_conflict="fact_key").execute()
                return f"Remembered permanently: {key} = {value}"
            except Exception as e:
                return f"Error saving fact: {e}"

        elif name == "read_recent_emails":
            if not composio_client:
                return "Email isn't configured yet — missing Composio API key."
            try:
                max_results = tool_input.get("max_results", 5)
                result = composio_client.tools.execute(
                    "GMAIL_FETCH_EMAILS",
                    user_id="default",
                    arguments={"max_results": max_results},
                    dangerously_skip_version_check=True
                )
                print(f"[Composio-Debug] Raw result: {result}")
                messages = result.get("data", {}).get("messages", [])
                if not messages:
                    return f"No recent emails found (or unexpected response shape — check logs). Raw: {str(result)[:500]}"
                summary = []
                for m in messages[:max_results]:
                    summary.append(f"From: {m.get('sender', 'unknown')} | Subject: {m.get('subject', 'no subject')} | Snippet: {m.get('snippet', '')[:150]}")
                return "\n\n".join(summary)
            except Exception as e:
                import traceback
                print(f"[Composio-Debug] Exception: {e}")
                traceback.print_exc()
                return f"Error reading emails: {e}"

        elif name == "send_email":
            if not composio_client:
                return "Email isn't configured yet — missing Composio API key."
            to = tool_input["to"]
            subject = tool_input["subject"]
            body = tool_input["body"]
            try:
                composio_client.tools.execute(
                    "GMAIL_SEND_EMAIL",
                    user_id="default",
                    arguments={"recipient_email": to, "subject": subject, "body": body},
                    dangerously_skip_version_check=True
                )
                return f"Email sent to {to} with subject '{subject}'."
            except Exception as e:
                return f"Error sending email: {e}"

        elif name == "deep_research":
            url = tool_input["url"]
            if not FIRECRAWL_API_KEY:
                return "Deep research isn't configured yet — missing Firecrawl API key."
            try:
                payload = json.dumps({"url": url, "formats": ["markdown"]}).encode("utf-8")
                req = urllib.request.Request(
                    "https://api.firecrawl.dev/v1/scrape",
                    data=payload, method="POST",
                    headers={
                        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
                        "Content-Type": "application/json"
                    }
                )
                with urllib.request.urlopen(req, timeout=30) as response:
                    result = json.loads(response.read())
                content = result.get("data", {}).get("markdown", "")
                if not content:
                    return f"Couldn't extract readable content from {url}."
                if len(content) > 15000:
                    content = content[:15000] + "\n...[truncated, page is longer]"
                return content
            except Exception as e:
                return f"Deep research failed: {e}"

        elif name == "add_event":
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
    location: str | None = None


@app.get("/")
@app.get("/app", response_class=HTMLResponse)
def voice_app():
    return """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Aegis</title>
<style>
  body {
    margin: 0; padding: 0;
    height: 100vh;
    background: #1a0533;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    font-family: -apple-system, sans-serif;
    color: white;
    text-align: center;
  }
  #orb {
    width: 180px; height: 180px;
    border-radius: 50%;
    background: radial-gradient(circle, #c084fc 0%, #8b3ce8 50%, #2a0f4d 100%);
    box-shadow: 0 0 60px 20px rgba(139,60,232,0.5);
    display: flex; align-items: center; justify-content: center;
    font-size: 60px;
    cursor: pointer;
    transition: transform 0.2s;
    user-select: none;
  }
  #orb.listening { animation: pulse 1s infinite; box-shadow: 0 0 80px 30px rgba(139,60,232,0.8); }
  #orb.thinking { opacity: 0.6; }
  @keyframes pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.08); } }
  #status { margin-top: 30px; font-size: 18px; min-height: 24px; padding: 0 30px; }
  #transcript { margin-top: 15px; font-size: 14px; color: #c9a8f5; padding: 0 30px; min-height: 20px; }
</style>
</head>
<body>
  <div id="modeToggle" style="margin-bottom: 20px;">
    <button id="chatModeBtn" style="padding:8px 16px; margin:4px; border-radius:20px; border:none; background:#8b3ce8; color:white;">Chat</button>
    <button id="translateModeBtn" style="padding:8px 16px; margin:4px; border-radius:20px; border:none; background:#4a1a7a; color:white;">Translate</button>
  </div>

  <div id="chatUI">
    <div id="orb">🎤</div>
  </div>

  <div id="translateUI" style="display:none;">
    <select id="langSelect" style="padding:10px; border-radius:10px; margin-bottom:20px; font-size:16px;">
      <option value="Spanish">Spanish — Español</option>
      <option value="French">French — Français</option>
      <option value="German">German — Deutsch</option>
      <option value="Italian">Italian — Italiano</option>
      <option value="Portuguese">Portuguese — Português</option>
      <option value="Japanese">Japanese — 日本語</option>
      <option value="Mandarin Chinese">Mandarin Chinese — 中文</option>
      <option value="Arabic">Arabic — العربية</option>
      <option value="Hebrew">Hebrew — עברית</option>
      <option value="Russian">Russian — Русский</option>
    </select>
    <br>
    <button id="theySpeakBtn" style="padding:16px 24px; margin:8px; border-radius:16px; border:none; background:#8b3ce8; color:white; font-size:16px;">🎤 They Speak</button>
    <button id="youSpeakBtn" style="padding:16px 24px; margin:8px; border-radius:16px; border:none; background:#4a1a7a; color:white; font-size:16px;">🎤 You Speak</button>
  </div>

  <div id="status">Tap the orb to talk</div>
  <div id="transcript"></div>

<script>
  const orb = document.getElementById('orb');
  const statusEl = document.getElementById('status');
  const transcriptEl = document.getElementById('transcript');

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognition = null;
  let isListening = false;
  let currentLocation = null;

  function reverseGeocode(lat, lon) {
    fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&zoom=10`)
      .then(res => res.json())
      .then(data => {
        const addr = data.address || {};
        const city = addr.city || addr.town || addr.village || addr.county || '';
        const state = addr.state || '';
        currentLocation = [city, state].filter(Boolean).join(', ');
        statusEl.textContent = "Location ready: " + currentLocation + " — tap orb to talk";
      })
      .catch(err => {
        statusEl.textContent = "Reverse geocode failed: " + err.message;
      });
  }

  function initLocation() {
    statusEl.textContent = "Getting location...";
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          statusEl.textContent = "Got coordinates, looking up city...";
          reverseGeocode(pos.coords.latitude, pos.coords.longitude);
        },
        (err) => {
          statusEl.textContent = "Location error: " + err.message + " (code " + err.code + ")";
        },
        { enableHighAccuracy: false, timeout: 10000 }
      );
    } else {
      statusEl.textContent = "Location not supported by this browser";
    }
  }
  initLocation();
  setInterval(initLocation, 5 * 60 * 1000);

  function cleanForSpeech(text) {
    text = text.replace(/\*\*(.*?)\*\*/g, '$1');
    text = text.replace(/\*(.*?)\*/g, '$1');
    text = text.replace(/^[-*]\s+/gm, '');
    text = text.replace(/^-{2,}\s*$/gm, '');
    text = text.replace(/-{2,}/g, ' ');
    text = text.replace(/#+\s*/g, '');
    text = text.replace(/`([^`]*)`/g, '$1');
    return text;
  }

  function speak(text) {
    const utterance = new SpeechSynthesisUtterance(cleanForSpeech(text));
    utterance.rate = 1.0;
    utterance.onend = () => {
      statusEl.textContent = "Tap the orb to talk";
      orb.classList.remove('thinking');
    };
    window.speechSynthesis.speak(utterance);
  }

  function startListening() {
    if (isListening) return;
    recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      isListening = true;
      orb.classList.add('listening');
      statusEl.textContent = "Listening...";
      transcriptEl.textContent = "";
    };

    recognition.onresult = (event) => {
      const text = event.results[0][0].transcript;
      transcriptEl.textContent = '"' + text + '"';
      sendToAegis(text);
    };

    recognition.onerror = (event) => {
      statusEl.textContent = "Didn't catch that — tap to try again";
      orb.classList.remove('listening');
      isListening = false;
    };

    recognition.onend = () => {
      isListening = false;
      orb.classList.remove('listening');
    };

    recognition.start();
  }

  async function sendToAegis(message) {
    statusEl.textContent = "Thinking...";
    orb.classList.add('thinking');
    try {
      const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: message, location: currentLocation })
      });
      const data = await response.json();
      statusEl.textContent = data.reply;
      speak(data.reply);
    } catch (err) {
      statusEl.textContent = "Connection error — tap to try again";
      orb.classList.remove('thinking');
    }
  }

  orb.addEventListener('click', startListening);

  const LANG_CODES = {
    "Spanish": "es-ES", "French": "fr-FR", "German": "de-DE", "Italian": "it-IT",
    "Portuguese": "pt-BR", "Japanese": "ja-JP", "Mandarin Chinese": "zh-CN",
    "Arabic": "ar-SA", "Hebrew": "he-IL", "Russian": "ru-RU"
  };

  const chatModeBtn = document.getElementById('chatModeBtn');
  const translateModeBtn = document.getElementById('translateModeBtn');
  const chatUI = document.getElementById('chatUI');
  const translateUI = document.getElementById('translateUI');
  const langSelect = document.getElementById('langSelect');
  const theySpeakBtn = document.getElementById('theySpeakBtn');
  const youSpeakBtn = document.getElementById('youSpeakBtn');

  chatModeBtn.addEventListener('click', () => {
    chatUI.style.display = 'block';
    translateUI.style.display = 'none';
    statusEl.textContent = "Tap the orb to talk";
    transcriptEl.textContent = "";
  });

  translateModeBtn.addEventListener('click', () => {
    chatUI.style.display = 'none';
    translateUI.style.display = 'block';
    statusEl.textContent = "Choose a language, then tap a button";
    transcriptEl.textContent = "";
  });

  function speakInLanguage(text, langCode) {
    const utterance = new SpeechSynthesisUtterance(cleanForSpeech(text));
    utterance.lang = langCode;
    utterance.rate = 1.0;
    utterance.onend = () => { statusEl.textContent = "Choose a language, then tap a button"; };
    window.speechSynthesis.speak(utterance);
  }

  async function translateAndSpeak(text, targetLanguage, speakLangCode) {
    statusEl.textContent = "Translating...";
    try {
      const response = await fetch('/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text, target_language: targetLanguage })
      });
      const data = await response.json();
      transcriptEl.textContent = data.translated;
      statusEl.textContent = data.translated;
      speakInLanguage(data.translated, speakLangCode);
    } catch (err) {
      statusEl.textContent = "Translation error — try again";
    }
  }

  function listenOnce(langCode, onResult) {
    const rec = new SpeechRecognition();
    rec.lang = langCode;
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    rec.onstart = () => { statusEl.textContent = "Listening..."; transcriptEl.textContent = ""; };
    rec.onresult = (event) => onResult(event.results[0][0].transcript);
    rec.onerror = () => { statusEl.textContent = "Didn't catch that — try again"; };
    rec.start();
  }

  theySpeakBtn.addEventListener('click', () => {
    const targetLang = langSelect.value;
    const foreignCode = LANG_CODES[targetLang];
    listenOnce(foreignCode, (heardText) => {
      transcriptEl.textContent = '"' + heardText + '"';
      translateAndSpeak(heardText, "English", "en-US");
    });
  });

  youSpeakBtn.addEventListener('click', () => {
    const targetLang = langSelect.value;
    const foreignCode = LANG_CODES[targetLang];
    listenOnce("en-US", (heardText) => {
      transcriptEl.textContent = '"' + heardText + '"';
      translateAndSpeak(heardText, targetLang, foreignCode);
    });
  });
</script>
</body>
</html>
"""
def read_root():
    return {"status": "Aegis backend is alive"}


class TranslateRequest(BaseModel):
    text: str
    target_language: str


@app.post("/translate")
def translate(request: TranslateRequest):
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        system=(
            "You are a precise, natural real-time translator. Translate "
            "exactly what is given, into the requested language. Return "
            "ONLY the translation itself — no explanation, no quotes, no "
            "extra commentary, just the translated sentence, spoken naturally "
            "the way a native speaker would actually say it."
        ),
        messages=[{
            "role": "user",
            "content": f"Translate this into {request.target_language}:\n\n{request.text}"
        }]
    )
    translated = "".join(b.text for b in response.content if b.type == "text")
    return {"translated": translated.strip()}


def load_phone_history():
    result = supabase.table("phone_conversation_history").select("role, content").order("created_at", desc=True).limit(40).execute()
    history = list(reversed(result.data))
    return [{"role": h["role"], "content": h["content"]} for h in history]


def save_phone_message(role, content):
    supabase.table("phone_conversation_history").insert({"role": role, "content": content}).execute()


@app.post("/chat")
def chat(request: ChatRequest):
    print(f"[Location-Debug] Received location: {request.location}")
    past_history = load_phone_history()
    messages = past_history + [{"role": "user", "content": request.message}]
    reply = None

    for _ in range(5):
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=800,
            system=build_system_prompt(request.location),
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

    save_phone_message("user", request.message)
    save_phone_message("assistant", reply)

    return {"reply": reply}