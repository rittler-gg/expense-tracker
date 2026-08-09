# Expense Logger

Reads bank transaction alerts out of Gmail, uses an LLM to pull the structured details out of each one, and writes them to a Notion database. Runs unattended; marks emails read so it never double-logs.

## Why a model instead of rules

Every bank formats its alert emails differently, and they change the layout without telling anyone. A regex or template approach needs one template per sender and breaks silently the first time a bank tweaks its HTML. Extraction from heterogeneous, unstable text is what models are actually good at, so that's the part the model does — and only that part.

## How it works

```
Gmail  ──►  filter to bank alerts  ──►  Gemini extracts JSON  ──►  Notion row
                                              │
                                    categories constrained to
                                   those already in the database
```

1. **Query Gmail** for unread mail from configured bank alert addresses containing transaction wording (`debited`, `credited`, `spent`…), excluding marketing and EMI offers.
2. **Extract with Gemini** into a strict JSON contract: `name`, `amount`, `type`, `category`, `payment_method`, `date`.
3. **Write to Notion**, linking back to the source email.
4. **Mark the email read**, so a re-run doesn't duplicate it.

## Keeping the model on rails

The interesting problem here isn't extraction, it's stopping the model inventing things. Three constraints do most of the work:

- **Categories are read from the Notion database at runtime** and passed into the prompt as a closed list, so the model picks an existing category instead of inventing "Miscellaneous Expenses (Travel)".
- **Strict JSON contract** — every field is returned even when empty, so a missing value never shifts the shape of the response.
- **Model fallback and retry** — `gemini-2.5-flash`, falling back to `gemini-1.5-flash`, with exponential backoff on 503s, so a transient outage doesn't lose a run.

## Known gap

There's no correction loop yet. If a transaction is miscategorised, the fix is manual in Notion, and nothing feeds that back. Extraction was the easy half; knowing when the output is wrong and making correction cheap is the half that decides whether a tool like this is actually usable long-term. That's the next thing to build.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env    # then fill it in
```

You'll need:

- **A Gemini API key** from Google AI Studio.
- **A Notion integration token**, and the target database shared with that integration.
- **Gmail API credentials** (`credentials.json`) from a Google Cloud project with the Gmail API enabled. First run opens a browser to authorise and writes `token.json`.

Your Notion database needs properties matching the extracted fields: name, amount, type, category, payment method, date, and a link.

## Run

```bash
python expense_logger.py
```

Schedule it with cron or launchd to run unattended.

## Note

`.env`, `credentials.json` and `token.json` are gitignored and should never be committed. Bank sender addresses are configured via `BANK_ALERT_SENDERS` rather than hardcoded.
