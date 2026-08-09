import base64
import json
import os
import re
import time
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google import genai
from notion_client import Client as NotionClient

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


def _normalize_notion_id(s: str) -> str:
    """Normalize a Notion ID to UUID-with-dashes form."""
    if not s:
        return s
    raw = s.strip()
    if "-" in raw:
        return raw
    if re.fullmatch(r"[0-9a-fA-F]{32}", raw):
        return f"{raw[0:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:32]}"
    return raw


def load_gmail_service():
    creds = None
    token_path = os.getenv("GMAIL_TOKEN_PATH", "token.json")
    creds_path = os.getenv("GMAIL_CREDENTIALS_PATH", "credentials.json")

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def extract_text_from_payload(payload: dict) -> str:
    """Extract readable message body from Gmail API payload.

    Preference: text/plain > text/html (stripped).
    """
    if not payload:
        return ""

    mime_type = payload.get("mimeType", "")
    body = payload.get("body", {})
    parts = payload.get("parts", [])

    if parts:
        for part in parts:
            if part.get("mimeType") == "text/plain":
                return extract_text_from_payload(part)
        for part in parts:
            if part.get("mimeType") == "text/html":
                return extract_text_from_payload(part)
        for part in parts:
            text = extract_text_from_payload(part)
            if text:
                return text

    data = body.get("data")
    if data:
        decoded = base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", errors="ignore")
        if mime_type == "text/html":
            decoded = re.sub(r"<style[\s\S]*?</style>", " ", decoded, flags=re.IGNORECASE)
            decoded = re.sub(r"<script[\s\S]*?</script>", " ", decoded, flags=re.IGNORECASE)
            decoded = re.sub(r"<[^>]+>", " ", decoded)
            decoded = re.sub(r"\s+", " ", decoded).strip()
        return decoded

    return ""


def _should_skip_email(body_text: str) -> bool:
    """Skip upcoming mandate notifications or other non-transaction alerts."""
    if not body_text:
        return False
    text = body_text.casefold()
    if "there is an upcoming e-mandate (auto payment)" in text:
        return True
    return False


def _normalize_date(
    date_str: str, *, fallback_date: Optional[datetime.date] = None
) -> tuple[str, bool]:
    """Normalize common date formats to ISO (YYYY-MM-DD)."""
    if not date_str:
        return "", False
    raw = str(date_str).strip()
    if not raw:
        return "", False

    # Fast-path ISO
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        parsed = datetime.strptime(raw, "%Y-%m-%d").date()
    else:
        parsed = None
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y", "%Y/%m/%d", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(raw, fmt).date()
                break
            except ValueError:
                continue

    if parsed is None:
        return "", False

    if parsed.year < 100:
        parsed = parsed.replace(year=parsed.year + 2000)

    if fallback_date and abs((parsed - fallback_date).days) >= 2:
        return fallback_date.isoformat(), True

    return parsed.isoformat(), False


def build_gmail_query() -> str:
    """Build Gmail search query for transaction alert emails."""
    # Keep sender filters strict to avoid pulling marketing campaigns.
    # Configured via BANK_ALERT_SENDERS (comma-separated) so this works for any bank.
    senders = [
        s.strip()
        for s in os.getenv("BANK_ALERT_SENDERS", "").split(",")
        if s.strip()
    ]
    if not senders:
        raise RuntimeError(
            "BANK_ALERT_SENDERS is empty. Set it in .env to a comma-separated list "
            "of your bank's alert addresses, e.g. alerts@yourbank.com"
        )
    sender_query = " OR ".join(f"from:{s}" for s in senders)

    # Different banks use different wording in transaction alerts.
    tx_phrases = [
        "debited",
        "credited",
        "deposited",
        "spent",
        "received",
        "\"added to your account\"",
        "\"has been debited\"",
        "\"has been credited\"",
        "\"has been deposited\"",
    ]
    tx_query = " OR ".join(tx_phrases)

    exclude_from = ["aclmails.in", "offers", "newsletter"]
    exclude_subject = ["voucher", "offer", "newsletter", "easyemi", "mailer"]
    exclude_query = " ".join([f"-from:{s}" for s in exclude_from] + [f"-subject:{s}" for s in exclude_subject])

    return f"is:unread ({sender_query}) ({tx_query}) {exclude_query}"


def call_gemini_extract(
    client,
    body_text: str,
    message_id: str,
    *,
    allowed_categories: Optional[list[str]] = None,
) -> dict:
    """Parse transaction details from email body using Gemini."""
    categories_rule = (
        "- category: choose ONE from this list (match spelling exactly): "
        + ", ".join(allowed_categories)
        + ". "
        if allowed_categories
        else "- category: pick the best category, default to transport. "
    )

    prompt = (
        "You are a transaction parser. Extract transaction info from the email body. "
        "Return STRICT JSON only with these keys: "
        "name, amount, type, category, payment_method, date. "
        "Rules: "
        "- name: merchant/payer name. "
        "- amount: numeric INR only, no currency symbols. "
        "- type: Expense if debited/spent, Income if credited. "
        + categories_rule
        + "- payment_method: CC XXXX, Debit XXXX, or UPI if present. "
        "- date: ISO format (YYYY-MM-DD) if present in the email. "
        "If a field is missing, still return it with an empty string. "
        f"Email Body:\n{body_text}"
    )

    for model in ("gemini-2.5-flash", "gemini-2.0-flash"):
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={"thinking_config": {"thinking_budget": 0}},
                )
                break
            except Exception as e:
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    wait = 2 ** attempt
                    print(f"{model} 503, retrying in {wait}s (attempt {attempt + 1}/3)...")
                    time.sleep(wait)
                else:
                    raise
        else:
            print(f"{model} unavailable after 3 retries, trying next model...")
            continue
        break
    else:
        raise RuntimeError("All Gemini models unavailable")
    raw = response.text.strip()

    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*", "", raw).strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    data = json.loads(raw)
    data["link"] = f"https://mail.google.com/mail/u/0/#inbox/{message_id}"
    return data


def _notion_db_query(
    notion: NotionClient,
    database_id: str,
    *,
    filter: Optional[dict] = None,
    page_size: int = 100,
):
    """Query a Notion database (handles API version differences)."""
    body: dict = {"page_size": page_size}
    if filter:
        body["filter"] = filter

    if hasattr(getattr(notion, "databases", None), "query"):
        return notion.databases.query(database_id=database_id, **body)

    def _req_post(path: str):
        try:
            return notion.request(method="POST", path=path, body=body)
        except TypeError:
            return notion.request(method="POST", path=path, json=body)

    last_err: Optional[Exception] = None

    for path in (f"databases/{database_id}/query", f"/databases/{database_id}/query"):
        try:
            return _req_post(path)
        except Exception as e:
            last_err = e

    # Fallback: try newer data_sources endpoint
    try:
        db = notion.databases.retrieve(database_id=database_id)
        data_sources = db.get("data_sources") or []
        if data_sources and isinstance(data_sources, list):
            ds_id = (data_sources[0] or {}).get("id", "")
            if ds_id:
                for path in (f"data_sources/{ds_id}/query", f"/data_sources/{ds_id}/query"):
                    try:
                        return _req_post(path)
                    except Exception as e:
                        last_err = e
    except Exception as e:
        last_err = e

    raise last_err or RuntimeError("Failed to query Notion database")


def _find_category_page_id(
    notion: NotionClient,
    *,
    category_database_id: str,
    category_name: str,
    category_title_prop: str = "Name",
    create_missing: bool = False,
    _cache: Optional[dict] = None,
) -> str:
    """Resolve a category name to its Notion page ID."""
    if not category_database_id or not category_name:
        return ""

    name_key = category_name.strip()
    if not name_key:
        return ""

    if _cache is not None and name_key in _cache:
        return _cache[name_key]

    try:
        res = _notion_db_query(
            notion,
            category_database_id,
            filter={"property": category_title_prop, "title": {"contains": name_key}},
            page_size=25,
        )
        target = name_key.casefold()
        for page in res.get("results", []):
            props = page.get("properties", {})
            title_obj = (props.get(category_title_prop) or {}).get("title", [])
            title_text = "".join([(t.get("plain_text") or "") for t in title_obj]).strip()
            if title_text.casefold() == target:
                page_id = page.get("id", "")
                if _cache is not None:
                    _cache[name_key] = page_id
                return page_id
    except Exception:
        pass

    if not create_missing:
        return ""

    created = notion.pages.create(
        parent={"database_id": category_database_id},
        properties={category_title_prop: {"title": [{"text": {"content": name_key}}]}},
    )
    page_id = created.get("id", "")
    if _cache is not None:
        _cache[name_key] = page_id
    return page_id


def _list_category_names(
    notion: NotionClient,
    *,
    category_database_id: str,
    category_title_prop: str = "Name",
    page_size: int = 100,
) -> list[str]:
    """Return existing category names from the Category DB."""
    if not category_database_id:
        return []

    try:
        res = _notion_db_query(notion, category_database_id, page_size=page_size)
        names: list[str] = []
        for page in res.get("results", []):
            props = page.get("properties", {})
            title_obj = (props.get(category_title_prop) or {}).get("title", [])
            title_text = "".join([(t.get("plain_text") or "") for t in title_obj]).strip()
            if title_text:
                names.append(title_text)
        return sorted(set(names), key=str.casefold)
    except Exception as e:
        print(f"ERROR listing categories: {e}")
        return []


def create_notion_page(
    notion: NotionClient,
    database_id: str,
    data: dict,
    *,
    category_relation_prop: str = "Category",
    category_database_id: str = "",
    category_title_prop: str = "Name",
    create_missing_categories: bool = False,
    _category_cache: Optional[dict] = None,
):
    """Create a transaction row in the Notion database."""
    props = {
        "Heading": {"title": [{"text": {"content": data.get("name", "")}}]},
        "Amount": {"number": float(data.get("amount") or 0)},
        "Transaction Type": {"select": {"name": data.get("type", "")}},
        "Date": {"date": {"start": data.get("date", "")}},
        "Link": {"url": data.get("link", "")},
    }

    payment_method = (data.get("payment_method") or "").strip()
    if payment_method:
        props["Payment Method"] = {"select": {"name": payment_method}}

    category_name = (data.get("category") or "").strip()
    if category_database_id and category_name:
        cat_page_id = _find_category_page_id(
            notion,
            category_database_id=category_database_id,
            category_name=category_name,
            category_title_prop=category_title_prop,
            create_missing=create_missing_categories,
            _cache=_category_cache,
        )
        if cat_page_id:
            props[category_relation_prop] = {"relation": [{"id": cat_page_id}]}

    notion.pages.create(parent={"database_id": database_id}, properties=props)


def mark_as_read(service, msg_id: str):
    service.users().messages().modify(
        userId="me",
        id=msg_id,
        body={"removeLabelIds": ["UNREAD"]},
    ).execute()


def main():
    load_dotenv()

    gemini_key = os.getenv("LOGGER_GEMINI_KEY")
    notion_token = os.getenv("NOTION_TOKEN")
    database_id = _normalize_notion_id(os.getenv("NOTION_DATABASE_ID") or "")

    if not gemini_key:
        raise RuntimeError("Missing LOGGER_GEMINI_KEY in environment")
    if not notion_token or not database_id:
        raise RuntimeError("Missing NOTION_TOKEN or NOTION_DATABASE_ID in environment")

    # Category relation support (optional)
    category_database_id = _normalize_notion_id(os.getenv("NOTION_CATEGORY_DATABASE_ID", ""))
    category_relation_prop = os.getenv("NOTION_CATEGORY_RELATION_PROP", "Category")
    category_title_prop = os.getenv("NOTION_CATEGORY_TITLE_PROP", "Name")
    create_missing_categories = os.getenv("NOTION_CREATE_MISSING_CATEGORIES") == "1"
    category_cache: dict = {}

    gemini = genai.Client(api_key=gemini_key)
    gmail = load_gmail_service()
    notion = NotionClient(auth=notion_token)

    # Fetch allowed categories for Gemini prompt
    allowed_categories = _list_category_names(
        notion,
        category_database_id=category_database_id,
        category_title_prop=category_title_prop,
    )

    # Fetch unread transaction emails
    query = build_gmail_query()
    results = gmail.users().messages().list(userId="me", q=query, maxResults=50).execute()
    messages = results.get("messages", [])

    if not messages:
        print("No unread transaction emails found.")
        return

    print(f"Found {len(messages)} unread transaction email(s).\n")

    for msg in messages:
        msg_id = msg.get("id")
        if not msg_id:
            continue

        try:
            full = gmail.users().messages().get(userId="me", id=msg_id, format="full").execute()
            body_text = extract_text_from_payload(full.get("payload", {}))
            internal_ms = int(full.get("internalDate", "0") or 0)
            internal_date = datetime.fromtimestamp(internal_ms / 1000).date() if internal_ms else None

            if not body_text.strip():
                print(f"⚠ Skipped {msg_id}: empty email body")
                continue
            if _should_skip_email(body_text):
                print(f"⚠ Skipped {msg_id}: upcoming/mandate notice")
                continue

            # Parse with Gemini
            data = call_gemini_extract(gemini, body_text, msg_id, allowed_categories=allowed_categories)

            # Normalize/validate date; fallback to Gmail internal date
            raw_date = data.get("date", "")
            data["date"], used_fallback = _normalize_date(raw_date, fallback_date=internal_date)
            if used_fallback:
                print(
                    "⚠ Date fallback used: "
                    f"msg_id={msg_id} raw='{raw_date}' "
                    f"normalized='{data['date']}' internal_date='{internal_date}'"
                )
            if not data.get("date") and internal_date:
                data["date"] = internal_date.isoformat()

            if not data.get("date"):
                print(f"⚠ Skipped {msg_id}: could not determine date")
                continue

            # Write to Notion
            create_notion_page(
                notion,
                database_id,
                data,
                category_relation_prop=category_relation_prop,
                category_database_id=category_database_id,
                category_title_prop=category_title_prop,
                create_missing_categories=create_missing_categories,
                _category_cache=category_cache,
            )
            mark_as_read(gmail, msg_id)

            # Confirmation
            name = data.get("name", "?")
            amount = data.get("amount", "?")
            tx_type = data.get("type", "?")
            category = data.get("category", "")
            payment = data.get("payment_method", "")
            parts = [name, f"₹{amount}", tx_type]
            if category:
                parts.append(category)
            if payment:
                parts.append(payment)
            print(f"✓ Added to Notion: {' | '.join(parts)}")

        except Exception as exc:
            print(f"✗ Failed {msg_id}: {exc}")


if __name__ == "__main__":
    main()
