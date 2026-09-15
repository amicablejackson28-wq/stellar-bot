"""
Thin Supabase (PostgREST) client for the bot's persistent bookkeeping.

IMPORTANT: Stellar remains the source of truth for actual holdings and
executed trades. This module is the bot's *bookkeeping/history* layer —
it never overrides what the chain says, it only remembers what the bot
believes so a restart can compare the two instead of guessing.

Talks directly to Supabase's built-in REST API (PostgREST) over HTTPS —
no supabase-py dependency, just `requests`, to keep this bot's dependency
footprint small and easy to audit.

Reads two env vars:
    SUPABASE_URL          e.g. https://xxxxx.supabase.co
    SUPABASE_SERVICE_KEY  service_role key (server-side use only —
                           never expose this in a frontend)

If either is missing, every function here becomes a safe no-op that
returns None — see bot_trend_rsi.py's startup check, which refuses to
run with DRY_RUN=False if persistence isn't configured.
"""
import os
import logging
from datetime import date

import requests

log = logging.getLogger("stellar_bot")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Content-Type": "application/json",
}
_TIMEOUT = 15


def is_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _url(table: str) -> str:
    return f"{SUPABASE_URL}/rest/v1/{table}"


def get_pair(asset_code: str, asset_issuer: str) -> dict | None:
    """Fetch the persisted bookkeeping row for a pair, or None if unseen before."""
    if not is_enabled():
        return None
    try:
        resp = requests.get(
            _url("bot_pairs"), headers=_HEADERS, timeout=_TIMEOUT,
            params={"asset_code": f"eq.{asset_code}", "asset_issuer": f"eq.{asset_issuer}"},
        )
        resp.raise_for_status()
        rows = resp.json()
        return rows[0] if rows else None
    except requests.RequestException as e:
        log.error(f"Supabase get_pair failed for {asset_code}: {e}")
        return None


def upsert_pair(row: dict) -> bool:
    """Insert or update a pair's persisted bookkeeping row. Returns success."""
    if not is_enabled():
        return False
    try:
        resp = requests.post(
            _url("bot_pairs"), headers={**_HEADERS, "Prefer": "resolution=merge-duplicates"},
            params={"on_conflict": "asset_code,asset_issuer"}, json=row, timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        log.error(f"Supabase upsert_pair failed: {e}")
        return False


def get_daily_risk(trade_date: str) -> dict | None:
    if not is_enabled():
        return None
    try:
        resp = requests.get(
            _url("bot_daily_risk"), headers=_HEADERS, timeout=_TIMEOUT,
            params={"trade_date": f"eq.{trade_date}"},
        )
        resp.raise_for_status()
        rows = resp.json()
        return rows[0] if rows else None
    except requests.RequestException as e:
        log.error(f"Supabase get_daily_risk failed: {e}")
        return None


def upsert_daily_risk(row: dict) -> bool:
    if not is_enabled():
        return False
    try:
        resp = requests.post(
            _url("bot_daily_risk"), headers={**_HEADERS, "Prefer": "resolution=merge-duplicates"},
            params={"on_conflict": "trade_date"}, json=row, timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        log.error(f"Supabase upsert_daily_risk failed: {e}")
        return False


def record_order(row: dict) -> int | None:
    """Insert an order/execution record. Returns its new row id, or None on failure."""
    if not is_enabled():
        return None
    try:
        resp = requests.post(
            _url("bot_orders"), headers={**_HEADERS, "Prefer": "return=representation"},
            json=row, timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        rows = resp.json()
        return rows[0]["id"] if rows else None
    except requests.RequestException as e:
        log.error(f"Supabase record_order failed: {e}")
        return None


def record_trade(row: dict) -> bool:
    if not is_enabled():
        return False
    try:
        resp = requests.post(_url("bot_trades"), headers=_HEADERS, json=row, timeout=_TIMEOUT)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        log.error(f"Supabase record_trade failed: {e}")
        return False


def get_known_offer_ids(asset_code: str, asset_issuer: str) -> set:
    """
    Offer IDs this bot has itself created for this pair, per our own order
    history. Used on startup to tell 'our own still-resting partial fill'
    apart from 'a foreign offer we've never seen' — the latter halts.
    """
    if not is_enabled():
        return set()
    try:
        resp = requests.get(
            _url("bot_orders"), headers=_HEADERS, timeout=_TIMEOUT,
            params={
                "asset_code": f"eq.{asset_code}",
                "asset_issuer": f"eq.{asset_issuer}",
                "offer_id": "not.is.null",
                "select": "offer_id",
            },
        )
        resp.raise_for_status()
        return {row["offer_id"] for row in resp.json() if row.get("offer_id")}
    except requests.RequestException as e:
        log.error(f"Supabase get_known_offer_ids failed: {e}")
        return set()


def today() -> str:
    return date.today().isoformat()
