"""
Stellar DEX (SDEX) Multi-Pair Trading Bot — Trend + Buy-Low/Sell-High (RSI)
=============================================================================

STRATEGY (combines two ideas you asked for)
- TREND: a moving-average crossover tells us the broader direction
  (short MA above long MA = uptrend; below = downtrend).
- BUY LOW / SELL HIGH: an RSI (Relative Strength Index) indicator flags
  when a price looks "oversold" (cheap, RSI < 30) or "overbought"
  (expensive, RSI > 70).

  BUY only when: RSI is oversold (cheap) AND the broader trend is still
                 up (so we're buying a dip within an uptrend, not
                 catching a falling knife in a downtrend).
  SELL when:     RSI is overbought (take profit) OR the trend flips down
                 (protect against giving back gains).

HONESTY CHECK — please actually read this part
- Combining indicators does not make a bot "very good" at trading. It
  makes assumptions more explicit, which can reduce (not eliminate) bad
  trades in some conditions and does nothing to prevent bad trades in
  others. Nobody — including me — can promise this makes money.
- Still start with DRY_RUN = True, then testnet, then small amounts on
  mainnet if you go live at all.

PERSISTENCE / RECONCILIATION ARCHITECTURE
- Stellar (the blockchain) is the source of truth for actual holdings
  and executed trades — always.
- Supabase (see persistence.py) is the bot's own BOOKKEEPING/MEMORY —
  what it believes it did. It is never trusted over the chain for
  quantities, only for context (entry prices, halt reasons, history)
  the chain itself doesn't record.
- On every startup, the bot reconciles the two. Anything it can't
  explain — an unrecognized balance, an unrecognized resting offer, an
  unverifiable order — halts that pair for a human to check. The bot
  never invents an entry price and never auto-cancels an offer it can't
  prove is its own.
- A "daily" halt (risk-limit breach) auto-clears at the next trade date.
  A "reconciliation" halt (something doesn't add up) NEVER auto-clears —
  only a human editing the bot_pairs table in Supabase can clear it.

SETUP
    pip install stellar-sdk requests

    Environment variables:
    STELLAR_SECRET_KEY     - your account's secret seed (starts with S...)
    STELLAR_NETWORK         - "testnet" or "mainnet" (default: testnet)
    SUPABASE_URL            - required before DRY_RUN can be turned off
    SUPABASE_SERVICE_KEY    - required before DRY_RUN can be turned off
"""

import os
import time
import socket
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from stellar_sdk import Server, Keypair, Asset, TransactionBuilder, Network
from stellar_sdk.exceptions import BaseHorizonError

import persistence

# Hard safety net: caps EVERY raw socket operation in this process (Stellar
# SDK calls, Supabase requests calls, everything) at 30s, regardless of
# whether any individual library's own timeout parameter is correctly wired
# through to the underlying connection. Without this, a single stalled
# network call can hang the bot's single-threaded loop indefinitely with no
# error and no way to recover except a manual restart — which is exactly
# what was observed happening in production.
socket.setdefaulttimeout(30)

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

DRY_RUN = True  # <<< keep True until tested

NETWORK = os.environ.get("STELLAR_NETWORK", "testnet")
SECRET_KEY = os.environ.get("STELLAR_SECRET_KEY")

BASE_ASSET = Asset.native()  # XLM — every pair trades against XLM

SHORT_WINDOW = 10
LONG_WINDOW = 30
RSI_PERIOD = 14
RSI_OVERSOLD = 30    # below this = "buy low" candidate
RSI_OVERBOUGHT = 70  # above this = "sell high" candidate

POLL_INTERVAL_SECONDS = 60
REFRESH_ASSETS_EVERY_N_CYCLES = 30

MAX_POSITION_SIZE = 50.0
STOP_LOSS_PCT = 0.05
MAX_DAILY_LOSS_PER_PAIR = 20.0
MAX_TOTAL_DAILY_LOSS = 100.0
ORDER_SIZE = 10.0
MAX_PAIRS_TO_TRADE = 10

# Watchdog timing (see _run_with_timeout): PER_CALL_TIMEOUT_SECONDS bounds any
# single network call; MAX_CYCLE_SECONDS bounds the whole per-pair processing
# loop so a bot with many pairs can't quietly stack up delays even if each
# individual call succeeds within its own timeout. Kept internally consistent:
# the cycle budget comfortably fits several calls, not just one or two.
PER_CALL_TIMEOUT_SECONDS = 30
MAX_CYCLE_SECONDS = 120

HORIZON_URLS = {
    "testnet": "https://horizon-testnet.stellar.org",
    "mainnet": "https://horizon.stellar.org",
}
NETWORK_PASSPHRASES = {
    "testnet": Network.TESTNET_NETWORK_PASSPHRASE,
    "mainnet": Network.PUBLIC_NETWORK_PASSPHRASE,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stellar_bot")


# --------------------------------------------------------------------------
# STATE
# --------------------------------------------------------------------------

@dataclass
class PairState:
    asset: Asset
    prices: deque = field(default_factory=lambda: deque(maxlen=max(LONG_WINDOW, RSI_PERIOD + 1)))

    position_open: bool = False
    position_size: float = 0.0
    entry_price: float = 0.0
    entry_price_verified: bool = False  # False = approximated, never a proven purchase price

    trade_date: str = field(default_factory=lambda: datetime.now(timezone.utc).date().isoformat())
    daily_pnl: float = 0.0
    halted_daily: bool = False           # auto-clears at the next trade_date

    halted_reconciliation: bool = False  # NEVER auto-cleared — human must clear in Supabase
    halt_reason: str = ""

    @property
    def halted(self) -> bool:
        return self.halted_daily or self.halted_reconciliation


@dataclass
class BotState:
    pairs: dict = field(default_factory=dict)
    total_daily_pnl: float = 0.0
    daily_pnl_date: str = field(default_factory=lambda: datetime.now(timezone.utc).date().isoformat())
    globally_halted: bool = False
    # Incremented once per main-loop cycle. Used to let a process_pair() call
    # that got abandoned by the watchdog (see _run_with_timeout) detect, if
    # it eventually finishes late in its zombie thread, that the main loop
    # already moved on — so it can discard its own result instead of writing
    # stale mutations into shared state from a thread nobody is tracking
    # anymore.
    cycle_generation: int = 0


def asset_key(asset: Asset) -> str:
    return "XLM" if asset.is_native() else f"{asset.code}:{asset.issuer}"


def _run_with_timeout(func, timeout, *args, **kwargs):
    """
    Runs func in a separate daemon thread and waits up to `timeout` seconds.

    WHY THIS EXISTS: in production, a Stellar/Supabase network call was
    observed hanging far longer than its documented timeout (multiple
    confirmed 5+ minute freezes with zero errors, zero log output). Both
    the library's own timeout parameters and socket.setdefaulttimeout()
    failed to bound it — likely because urllib3 (used by both stellar-sdk
    and requests) manages connection timeouts internally and doesn't
    reliably honor the OS-level default. This makes the bot's main loop
    unconditionally responsive regardless of that underlying cause: if the
    call doesn't finish in time, we stop WAITING for it (the thread itself
    is abandoned running in the background, harmless since it's a daemon
    thread) and move on, rather than freezing the whole bot indefinitely.

    Returns (True, result) on success, or (False, None) on timeout. Raises
    whatever exception func itself raised, if it completed but failed.
    """
    box = {}

    def wrapper():
        try:
            box["value"] = func(*args, **kwargs)
            box["ok"] = True
        except Exception as e:
            box["error"] = e
            box["ok"] = False

    t = threading.Thread(target=wrapper, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return False, None
    if not box.get("ok", False):
        raise box.get("error", RuntimeError("_run_with_timeout: unknown failure"))
    return True, box.get("value")


# --------------------------------------------------------------------------
# PERSISTENCE HELPERS (bookkeeping only — never overrides the chain)
# --------------------------------------------------------------------------

def _persist_pair(pair_state: PairState):
    persistence.upsert_pair({
        "asset_code": pair_state.asset.code,
        "asset_issuer": pair_state.asset.issuer,
        "position_open": pair_state.position_open,
        "position_size": pair_state.position_size,
        "entry_price": pair_state.entry_price if pair_state.position_open else None,
        "entry_price_verified": pair_state.entry_price_verified,
        "trade_date": pair_state.trade_date,
        "daily_pnl": pair_state.daily_pnl,
        "halted_daily": pair_state.halted_daily,
        "halted_reconciliation": pair_state.halted_reconciliation,
        "halt_reason": pair_state.halt_reason,
    })


def _persist_daily_risk(state: BotState):
    persistence.upsert_daily_risk({
        "trade_date": state.daily_pnl_date,
        "total_daily_pnl": state.total_daily_pnl,
        "globally_halted": state.globally_halted,
    })


def _persist_order_record(record: "ExecutionRecord", counter_asset: Asset):
    """Logs every order ATTEMPT — including UNCERTAIN ones — as audit history."""
    return persistence.record_order({
        "transaction_hash": record.transaction_hash,
        "asset_code": counter_asset.code,
        "asset_issuer": counter_asset.issuer,
        "pair": record.pair,
        "side": record.side,
        "requested_amount": record.requested_amount,
        "limit_price": record.limit_price,
        "offer_id": record.offer_id,
        "actual_filled_amount": record.actual_filled_amount,
        "actual_execution_price": record.actual_execution_price,
        "actual_fees": record.actual_fees,
        "remaining_amount": record.remaining_amount,
        "verification_status": record.verification_status,
    })


def _persist_trade_record(record: "ExecutionRecord", counter_asset: Asset, order_row_id, pnl):
    persistence.record_trade({
        "order_id": order_row_id,
        "asset_code": counter_asset.code,
        "asset_issuer": counter_asset.issuer,
        "side": record.side,
        "filled_amount": record.actual_filled_amount,
        "execution_price": record.actual_execution_price,
        "realized_pnl": pnl,
    })


# --------------------------------------------------------------------------
# INDICATORS
# --------------------------------------------------------------------------

def moving_average(prices: deque, window: int):
    if len(prices) < window:
        return None
    return sum(list(prices)[-window:]) / window


def compute_rsi(prices: deque, period: int):
    """Standard RSI calculation. Returns None until enough data exists."""
    if len(prices) < period + 1:
        return None

    price_list = list(prices)[-(period + 1):]
    gains, losses = [], []

    for i in range(1, len(price_list)):
        change = price_list[i] - price_list[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0  # no losses at all = maximally overbought

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# --------------------------------------------------------------------------
# ACCOUNT / ASSET DISCOVERY
# --------------------------------------------------------------------------

def discover_tradable_assets(server: Server, public_key: str) -> list:
    """Returns a list of (Asset, balance) tuples for non-XLM assets held."""
    account = server.accounts().account_id(public_key).call()
    assets = []

    for balance in account.get("balances", []):
        asset_type = balance.get("asset_type")
        if asset_type in ("native", "liquidity_pool_shares"):
            continue
        code = balance.get("asset_code")
        issuer = balance.get("asset_issuer")
        if code and issuer:
            assets.append((Asset(code=code, issuer=issuer), float(balance.get("balance", 0))))

    if len(assets) > MAX_PAIRS_TO_TRADE:
        log.warning(f"Account holds {len(assets)} assets; capping to {MAX_PAIRS_TO_TRADE}.")
        assets = assets[:MAX_PAIRS_TO_TRADE]

    return assets


def reconcile_pair_on_startup(server: Server, keypair, asset: Asset, balance: float) -> PairState:
    """
    Rebuild a pair's state from BOTH the blockchain and persisted bookkeeping
    (Supabase) — never from a blank slate, and never by inventing history.

      - Stellar tells us what's actually held right now (the quantity).
      - Supabase tells us what the bot itself believes happened (entry
        price, halt state, history).
      - Any disagreement between the two, or an on-chain fact Supabase has
        never recorded (an unrecognized balance or resting offer), halts
        the pair for a human to look at. Nothing here auto-cancels an
        offer or auto-invents a price to make the numbers line up.
    """
    today_str = persistence.today()
    persisted = persistence.get_pair(asset.code, asset.issuer)
    pair_state = PairState(asset=asset, trade_date=today_str)

    # ---- 1. Resting offers: only "known" if WE persisted their offer_id ----
    known_offer_ids = persistence.get_known_offer_ids(asset.code, asset.issuer)
    try:
        offers = server.offers().for_seller(keypair.public_key).call()
        for rec in offers.get("_embedded", {}).get("records", []):
            selling, buying = rec.get("selling", {}), rec.get("buying", {})
            is_this_pair = (
                (selling.get("asset_type") != "native"
                 and selling.get("asset_code") == asset.code
                 and selling.get("asset_issuer") == asset.issuer
                 and buying.get("asset_type") == "native")
                or
                (buying.get("asset_type") != "native"
                 and buying.get("asset_code") == asset.code
                 and buying.get("asset_issuer") == asset.issuer
                 and selling.get("asset_type") == "native")
            )
            if is_this_pair and str(rec.get("id")) not in known_offer_ids:
                pair_state.halted_reconciliation = True
                pair_state.halt_reason = (
                    f"Unrecognized resting offer {rec.get('id')} found on startup — not in "
                    f"this bot's own order history. Clear manually in Supabase (bot_pairs) "
                    f"after checking where it came from."
                )
                log.error(f"XLM/{asset.code} {pair_state.halt_reason}")
    except BaseHorizonError as e:
        pair_state.halted_reconciliation = True
        pair_state.halt_reason = f"Could not verify open offers on startup: {e}"
        log.error(f"XLM/{asset.code} {pair_state.halt_reason}")

    if pair_state.halted_reconciliation:
        _persist_pair(pair_state)
        return pair_state

    # ---- 2. Balance reconciliation against persisted bookkeeping ----
    if persisted is None:
        if balance > 1e-7:
            pair_state.halted_reconciliation = True
            pair_state.halt_reason = (
                f"Found an on-chain balance of {balance:.4f} with no persisted record of "
                f"ever buying it. Refusing to invent an entry price — clear manually in "
                f"Supabase after confirming where this balance came from."
            )
            log.error(f"XLM/{asset.code} {pair_state.halt_reason}")
        # else: brand-new asset, zero balance — clean start, proceed normally.
    else:
        pair_state.halted_reconciliation = bool(persisted.get("halted_reconciliation"))
        pair_state.halt_reason = persisted.get("halt_reason") or ""

        persisted_date = persisted.get("trade_date")
        same_day = persisted_date == today_str
        pair_state.trade_date = persisted_date if same_day else today_str
        pair_state.daily_pnl = float(persisted.get("daily_pnl", 0)) if same_day else 0.0
        pair_state.halted_daily = bool(persisted.get("halted_daily")) if same_day else False

        if not pair_state.halted_reconciliation:
            persisted_size = float(persisted.get("position_size", 0))
            persisted_open = bool(persisted.get("position_open"))
            persisted_entry_price = float(persisted.get("entry_price") or 0.0)
            mismatch = (
                (persisted_open and abs(persisted_size - balance) > max(1e-4, persisted_size * 0.01))
                or (not persisted_open and balance > 1e-7)
                or (persisted_open and balance <= 1e-7)
                or (persisted_open and persisted_entry_price <= 0)
            )
            if mismatch:
                pair_state.halted_reconciliation = True
                if persisted_open and persisted_entry_price <= 0:
                    pair_state.halt_reason = (
                        f"Bookkeeping says this position is open but has no valid entry "
                        f"price ({persisted_entry_price}) — refusing to trade on it, since "
                        f"that would divide by zero in the stop-loss check. Fix the "
                        f"entry_price in Supabase (bot_pairs) and clear this halt manually."
                    )
                else:
                    pair_state.halt_reason = (
                        f"Balance mismatch on restart: bookkeeping expected "
                        f"{'an open position of ' + str(persisted_size) if persisted_open else 'no position'}, "
                        f"chain shows {balance:.4f}. Clear manually after investigating."
                    )
                log.error(f"XLM/{asset.code} {pair_state.halt_reason}")
            else:
                pair_state.position_open = persisted_open
                pair_state.position_size = balance  # trust the chain for quantity
                pair_state.entry_price = persisted_entry_price
                pair_state.entry_price_verified = bool(persisted.get("entry_price_verified"))
                log.info(f"XLM/{asset.code} restored from bookkeeping: "
                         f"position_open={pair_state.position_open} "
                         f"size={pair_state.position_size:.4f} "
                         f"entry_price={pair_state.entry_price:.6f} "
                         f"(verified={pair_state.entry_price_verified})")

    _persist_pair(pair_state)
    return pair_state


# --------------------------------------------------------------------------
# CORE LOGIC
# --------------------------------------------------------------------------

def get_mid_price(server: Server, counter_asset: Asset) -> float:
    order_book = server.orderbook(selling=BASE_ASSET, buying=counter_asset).limit(1).call()
    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])
    if not bids or not asks:
        raise ValueError(f"No liquidity right now for XLM/{counter_asset.code}")
    return (float(bids[0]["price"]) + float(asks[0]["price"])) / 2.0


def reset_daily_pnl_if_new_day(state: BotState):
    today_str = persistence.today()
    if today_str != state.daily_pnl_date:
        log.info(f"New day — resetting daily PnL (total was {state.total_daily_pnl:.2f})")
        state.daily_pnl_date = today_str
        state.total_daily_pnl = 0.0
        state.globally_halted = False
        _persist_daily_risk(state)
        for p in state.pairs.values():
            p.trade_date = today_str
            p.daily_pnl = 0.0
            p.halted_daily = False
            # halted_reconciliation is deliberately NOT touched here.
            _persist_pair(p)


def _cancel_offer(server, keypair, counter_asset, side, offer_id, price):
    """
