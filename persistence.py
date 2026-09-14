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
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from stellar_sdk import Server, Keypair, Asset, TransactionBuilder, Network
from stellar_sdk.exceptions import BaseHorizonError

import persistence

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


def asset_key(asset: Asset) -> str:
    return "XLM" if asset.is_native() else f"{asset.code}:{asset.issuer}"


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
        offers = server.offers().account(keypair.public_key).call()
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
            mismatch = (
                (persisted_open and abs(persisted_size - balance) > max(1e-4, persisted_size * 0.01))
                or (not persisted_open and balance > 1e-7)
                or (persisted_open and balance <= 1e-7)
            )
            if mismatch:
                pair_state.halted_reconciliation = True
                pair_state.halt_reason = (
                    f"Balance mismatch on restart: bookkeeping expected "
                    f"{'an open position of ' + str(persisted_size) if persisted_open else 'no position'}, "
                    f"chain shows {balance:.4f}. Clear manually after investigating."
                )
                log.error(f"XLM/{asset.code} {pair_state.halt_reason}")
            else:
                pair_state.position_open = persisted_open
                pair_state.position_size = balance  # trust the chain for quantity
                pair_state.entry_price = float(persisted.get("entry_price") or 0.0)
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
    """Cancel a resting (unfilled or partially filled) offer by setting amount to 0."""
    account = server.load_account(keypair.public_key)
    tx_builder = TransactionBuilder(
        source_account=account,
        network_passphrase=NETWORK_PASSPHRASES[NETWORK],
        base_fee=server.fetch_base_fee(),
    )
    if side == "buy":
        tx_builder.append_manage_buy_offer_op(
            selling=counter_asset, buying=BASE_ASSET,
            amount="0", price=str(round(price, 7)), offer_id=int(offer_id),
        )
    else:
        tx_builder.append_manage_sell_offer_op(
            selling=BASE_ASSET, buying=counter_asset,
            amount="0", price=str(round(price, 7)), offer_id=int(offer_id),
        )
    tx = tx_builder.set_timeout(30).build()
    tx.sign(keypair)
    try:
        server.submit_transaction(tx)
        log.info(f"Cancelled unfilled remainder of offer {offer_id}.")
    except BaseHorizonError as e:
        log.error(f"Failed to cancel offer {offer_id}: {e}")


class FillStatus:
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class ExecutionRecord:
    """
    Full audit record for one submitted order. Anything we can't actually
    verify is left as None rather than backfilled with an approximation —
    a None here is a known gap, not a silent guess.
    """
    transaction_hash: str | None
    pair: str
    side: str
    requested_amount: float
    limit_price: float
    offer_id: str | None = None
    actual_filled_amount: float = 0.0
    actual_execution_price: float | None = None  # not yet computed — see note below
    actual_fees: float | None = None              # not yet computed — see note below
    remaining_amount: float = 0.0
    verification_status: str = FillStatus.UNCERTAIN
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def submit_order(server, keypair, counter_asset, side, amount, price) -> ExecutionRecord:
    """
    Submits an order and returns a full ExecutionRecord.

    IMPORTANT: append_manage_*_offer_op places a resting limit order on the
    SDEX orderbook. Successful submission does NOT mean it filled — it may
    partially fill, or not fill at all, and sit on the books. This function
    checks what actually happened and cancels any unfilled remainder so we
    never leave a stray order unmanaged between polling cycles.

    KNOWN GAP (not yet closed): fill detection below infers "fully filled"
    from the absence of a resting offer on this exact selling/buying pair.
    Stellar's protocol guarantees a nonzero manage-offer operation either
    deletes (fully matched) or rests with a remainder — there's no third
    case — so that part is sound. What's NOT independently proven is that
    the absence we observe corresponds to *this* transaction specifically,
    rather than some other resting offer on the same pair. Today that gap
    is closed by a design invariant (this bot never has more than one
    resting order per pair at a time), not by direct transaction-level
    correlation. Closing it properly means tying the check to this
    transaction's own offer_id via operation/effect records, which needs
    an SDK call this codebase hasn't verified yet — so it's left as a
    documented gap rather than a guessed implementation.

    actual_execution_price and actual_fees are similarly left unpopulated
    (None) for the same reason. Callers fall back to the submitted
    mid-price for PnL, which is an approximation, not a verified price.
    """
    pair_label = f"XLM/{counter_asset.code}"
    tolerance = 1.002 if side == "buy" else 0.998
    limit_price = price * tolerance

    if DRY_RUN:
        log.info(f"[DRY RUN] {side.upper()} {amount:.4f} XLM @ {price:.6f} {counter_asset.code}")
        return ExecutionRecord(
            transaction_hash=None, pair=pair_label, side=side,
            requested_amount=amount, limit_price=limit_price,
            actual_filled_amount=amount, remaining_amount=0.0,
            verification_status=FillStatus.FILLED,
        )

    account = server.load_account(keypair.public_key)
    tx_builder = TransactionBuilder(
        source_account=account,
        network_passphrase=NETWORK_PASSPHRASES[NETWORK],
        base_fee=server.fetch_base_fee(),
    )

    if side == "buy":
        tx_builder.append_manage_buy_offer_op(
            selling=counter_asset, buying=BASE_ASSET,
            amount=str(round(amount, 7)), price=str(round(limit_price, 7)),
        )
    else:
        tx_builder.append_manage_sell_offer_op(
            selling=BASE_ASSET, buying=counter_asset,
            amount=str(round(amount, 7)), price=str(round(limit_price, 7)),
        )

    tx = tx_builder.set_timeout(30).build()
    tx.sign(keypair)
    try:
        response = server.submit_transaction(tx)
        tx_hash = response["hash"]
        log.info(f"Order submitted for {pair_label}: {tx_hash}")
    except BaseHorizonError as e:
        log.error(f"Order failed for {pair_label}: {e}")
        return ExecutionRecord(
            transaction_hash=None, pair=pair_label, side=side,
            requested_amount=amount, limit_price=limit_price,
            verification_status=FillStatus.UNCERTAIN,
        )

    record = ExecutionRecord(
        transaction_hash=tx_hash, pair=pair_label, side=side,
        requested_amount=amount, limit_price=limit_price,
    )

    sell_asset = counter_asset if side == "buy" else BASE_ASSET
    buy_asset = BASE_ASSET if side == "buy" else counter_asset
    try:
        offers = (server.offers().account(keypair.public_key)
                  .for_selling(sell_asset).for_buying(buy_asset).call())
        offer_records = offers.get("_embedded", {}).get("records", [])
        if offer_records:
            offer = offer_records[0]
            remaining = float(offer.get("amount", 0))
            filled = max(0.0, amount - remaining)
            record.offer_id = offer.get("id")
            record.actual_filled_amount = filled
            record.remaining_amount = remaining
            if remaining > 0:
                record.verification_status = FillStatus.PARTIALLY_FILLED
                log.warning(f"{pair_label} order partially filled "
                            f"({filled:.4f}/{amount:.4f}); cancelling remainder.")
                _cancel_offer(server, keypair, counter_asset, side, offer.get("id"), limit_price)
            else:
                record.verification_status = FillStatus.FILLED
        else:
            record.actual_filled_amount = amount
            record.remaining_amount = 0.0
            record.verification_status = FillStatus.FILLED
    except BaseHorizonError as e:
        log.error(f"{pair_label} could not verify fill status — "
                  f"halting this pair rather than assuming a fill happened: {e}")
        record.verification_status = FillStatus.UNCERTAIN
        record.actual_filled_amount = 0.0

    log.info(f"EXECUTION RECORD: {record}")
    return record


def process_pair(server, keypair, state: BotState, pair_state: PairState):
    counter_asset = pair_state.asset
    label = f"XLM/{counter_asset.code}"

    if pair_state.halted:
        return

    try:
        price = get_mid_price(server, counter_asset)
    except ValueError as e:
        log.warning(f"[{label}] {e}")
        return

    pair_state.prices.append(price)

    # Stop-loss check
    if pair_state.position_open:
        loss_pct = (pair_state.entry_price - price) / pair_state.entry_price
        if loss_pct >= STOP_LOSS_PCT:
            record = submit_order(server, keypair, counter_asset, "sell", pair_state.position_size, price)
            order_row_id = _persist_order_record(record, counter_asset)
            if record.verification_status == FillStatus.UNCERTAIN:
                pair_state.halted_reconciliation = True
                pair_state.halt_reason = f"Stop-loss order status unverifiable (tx {record.transaction_hash})."
                _persist_pair(pair_state)
                log.error(f"[{label}] Stop-loss order status could not be verified. "
                          f"Halting this pair for manual review rather than guessing.")
                return
            filled = record.actual_filled_amount
            if filled > 0:
                pnl = (price - pair_state.entry_price) * filled
                pair_state.daily_pnl += pnl
                state.total_daily_pnl += pnl
                pair_state.position_size -= filled
                if pair_state.position_size <= 1e-7:
                    pair_state.position_open = False
                    pair_state.entry_price = 0.0
                    pair_state.entry_price_verified = False
                _persist_trade_record(record, counter_asset, order_row_id, pnl)
                _persist_pair(pair_state)
                _persist_daily_risk(state)
                log.warning(f"[{label}] Stop-loss hit ({loss_pct:.2%}). "
                            f"Sold {filled:.4f}/position. PnL: {pnl:.2f}")
            else:
                log.warning(f"[{label}] Stop-loss triggered but order did not fill. Still holding position.")

    if pair_state.daily_pnl <= -MAX_DAILY_LOSS_PER_PAIR:
        log.warning(f"[{label}] Daily loss limit hit. Halting {label} for today.")
        pair_state.halted_daily = True
        _persist_pair(pair_state)
        return

    if state.total_daily_pnl <= -MAX_TOTAL_DAILY_LOSS:
        log.warning("Total daily loss limit hit. Halting entire bot for today.")
        state.globally_halted = True
        _persist_daily_risk(state)
        return

    short_ma = moving_average(pair_state.prices, SHORT_WINDOW)
    long_ma = moving_average(pair_state.prices, LONG_WINDOW)
    rsi = compute_rsi(pair_state.prices, RSI_PERIOD)

    if short_ma is None or long_ma is None or rsi is None:
        log.info(f"[{label}] Warming up ({len(pair_state.prices)} samples) price={price:.6f}")
        return

    uptrend = short_ma > long_ma
    oversold = rsi < RSI_OVERSOLD
    overbought = rsi > RSI_OVERBOUGHT

    log.info(f"[{label}] price={price:.6f} trend={'UP' if uptrend else 'DOWN'} "
              f"rsi={rsi:.1f} position_open={pair_state.position_open}")

    # BUY LOW: oversold dip, but only within an uptrend
    if not pair_state.position_open and oversold and uptrend:
        size = min(ORDER_SIZE, MAX_POSITION_SIZE / price)
        record = submit_order(server, keypair, counter_asset, "buy", size, price)
        order_row_id = _persist_order_record(record, counter_asset)
        if record.verification_status == FillStatus.UNCERTAIN:
            pair_state.halted_reconciliation = True
            pair_state.halt_reason = f"BUY order status unverifiable (tx {record.transaction_hash})."
            _persist_pair(pair_state)
            log.error(f"[{label}] BUY order status could not be verified. "
                      f"Halting this pair for manual review rather than guessing.")
            return
        filled = record.actual_filled_amount
        if filled > 0:
            pair_state.position_open = True
            pair_state.entry_price = price
            pair_state.entry_price_verified = False  # approximation — see known execution-price gap
            pair_state.position_size = filled
            _persist_trade_record(record, counter_asset, order_row_id, None)
            _persist_pair(pair_state)
            log.info(f"[{label}] BUY (dip in uptrend, RSI={rsi:.1f}) — {filled:.4f} XLM at {price:.6f}")
        else:
            log.info(f"[{label}] BUY signal fired but order did not fill.")

    # SELL HIGH: overbought, OR trend has flipped down (protect gains)
    elif pair_state.position_open and (overbought or not uptrend):
        reason = "overbought" if overbought else "trend flipped down"
        record = submit_order(server, keypair, counter_asset, "sell", pair_state.position_size, price)
        order_row_id = _persist_order_record(record, counter_asset)
        if record.verification_status == FillStatus.UNCERTAIN:
            pair_state.halted_reconciliation = True
            pair_state.halt_reason = f"SELL order status unverifiable (tx {record.transaction_hash})."
            _persist_pair(pair_state)
            log.error(f"[{label}] SELL order status could not be verified. "
                      f"Halting this pair for manual review rather than guessing.")
            return
        filled = record.actual_filled_amount
        if filled > 0:
            pnl = (price - pair_state.entry_price) * filled
            pair_state.daily_pnl += pnl
            state.total_daily_pnl += pnl
            pair_state.position_size -= filled
            if pair_state.position_size <= 1e-7:
                pair_state.position_open = False
                pair_state.entry_price = 0.0
                pair_state.entry_price_verified = False
            _persist_trade_record(record, counter_asset, order_row_id, pnl)
            _persist_pair(pair_state)
            _persist_daily_risk(state)
            log.info(f"[{label}] SELL ({reason}, RSI={rsi:.1f}) — sold {filled:.4f}. PnL: {pnl:.2f}")
        else:
            log.info(f"[{label}] SELL signal fired but order did not fill.")


def run_bot():
    if not DRY_RUN and not SECRET_KEY:
        raise SystemExit("STELLAR_SECRET_KEY is required for live trading.")

    if not DRY_RUN and not persistence.is_enabled():
        raise SystemExit(
            "SUPABASE_URL / SUPABASE_SERVICE_KEY are required before DRY_RUN can be turned "
            "off. This bot refuses to trade live without a persistent bookkeeping layer to "
            "reconcile against after a restart — see persistence.py."
        )

    server = Server(HORIZON_URLS[NETWORK])
    keypair = Keypair.from_secret(SECRET_KEY) if SECRET_KEY else None
    if not keypair:
        raise SystemExit("STELLAR_SECRET_KEY is required even for dry runs "
                          "(the bot reads pairs from your account).")

    if not persistence.is_enabled():
        log.warning("Supabase persistence is NOT configured (SUPABASE_URL / "
                     "SUPABASE_SERVICE_KEY missing) — running with in-memory state only. "
                     "Fine for DRY_RUN testing; required before going live.")
    else:
        log.info("Supabase persistence is configured — bookkeeping will be written every cycle.")

    today_str = persistence.today()
    persisted_risk = persistence.get_daily_risk(today_str)
    state = BotState(
        daily_pnl_date=today_str,
        total_daily_pnl=float(persisted_risk.get("total_daily_pnl", 0)) if persisted_risk else 0.0,
        globally_halted=bool(persisted_risk.get("globally_halted")) if persisted_risk else False,
    )
    _persist_daily_risk(state)

    cycle_count = 0
    mode = "DRY RUN" if DRY_RUN else f"LIVE ({NETWORK})"
    log.info(f"Starting Stellar multi-pair bot (trend + RSI) — mode: {mode}")

    while True:
        try:
            reset_daily_pnl_if_new_day(state)

            if state.globally_halted:
                log.warning("Globally halted for today. Sleeping...")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            if cycle_count % REFRESH_ASSETS_EVERY_N_CYCLES == 0:
                for asset, balance in discover_tradable_assets(server, keypair.public_key):
                    key = asset_key(asset)
                    if key not in state.pairs:
                        log.info(f"Discovered new asset: {asset.code} — adding to bot")
                        state.pairs[key] = reconcile_pair_on_startup(server, keypair, asset, balance)

                if not state.pairs:
                    log.warning("No tradable (non-XLM) trustlines found yet in this account.")

            for pair_state in list(state.pairs.values()):
                try:
                    process_pair(server, keypair, state, pair_state)
                except BaseHorizonError as e:
                    log.error(f"[{pair_state.asset.code}] Horizon API error: {e}")
                except Exception as e:
                    log.exception(f"[{pair_state.asset.code}] Unexpected error: {e}")

            cycle_count += 1

        except BaseHorizonError as e:
            log.error(f"Horizon API error: {e}")
        except Exception as e:
            log.exception(f"Unexpected error: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_bot()
