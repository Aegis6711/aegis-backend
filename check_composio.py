import os
import json
from composio import Composio

api_key = os.environ.get("COMPOSIO_API_KEY") or input("Paste your Composio API key: ").strip()

composio = Composio(api_key=api_key)
accounts = composio.connected_accounts.list()

for acc in accounts.items:
    try:
        print(json.dumps(acc.model_dump(), indent=2, default=str))
    except Exception:
        print(acc)
    print("---")