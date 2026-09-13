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

SETUP
    pip install stellar-sdk

    Environment variables:
    STELLAR_SECRET_KEY   - your account's secret seed (starts with S...)
    STELLAR_NETWORK       - "testnet" or "mainnet" (default: testnet)
"""

import os
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from stellar_sdk import Server, Keypair, Asset, TransactionBuilder, Network
from stellar_sdk.exceptions import BaseHorizonError

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
    entry_price: float = 0.0
    position_size: float = 0.0
    daily_pnl: float = 0.0
    halted: bool = False


@dataclass
class BotState:
    pairs: dict = field(default_factory=dict)
    daily_pnl_date: str = field(default_factory=lambda: datetime.now(timezone.utc).date().isoformat())
    total_daily_pnl: float = 0.0
    globally_halted: bool = False


def asset_key(asset: Asset) -> str:
    return "XLM" if asset.is_native() else f"{asset.code}:{asset.issuer}"


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
    account = server.accounts().account_id(public_key).call()
    assets = []

    for balance in account.get("balances", []):
        asset_type = balance.get("asset_type")
        if asset_type in ("native", "liquidity_pool_shares"):
            continue
        code = balance.get("asset_code")
        issuer = balance.get("asset_issuer")
        if code and issuer:
            assets.append(Asset(code=code, issuer=issuer))

    if len(assets) > MAX_PAIRS_TO_TRADE:
        log.warning(f"Account holds {len(assets)} assets; capping to {MAX_PAIRS_TO_TRADE}.")
        assets = assets[:MAX_PAIRS_TO_TRADE]

    return assets


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
    today = datetime.now(timezone.utc).date().isoformat()
    if today != state.daily_pnl_date:
        log.info(f"New day — resetting daily PnL (total was {state.total_daily_pnl:.2f})")
        state.daily_pnl_date = today
        state.total_daily_pnl = 0.0
        state.globally_halted = False
        for p in state.pairs.values():
            p.daily_pnl = 0.0
            p.halted = False


def submit_order(server, keypair, counter_asset, side, amount, price):
    if DRY_RUN:
        log.info(f"[DRY RUN] {side.upper()} {amount:.4f} XLM @ {price:.6f} {counter_asset.code}")
        return {"dry_run": True}

    account = server.load_account(keypair.public_key)
    tx_builder = TransactionBuilder(
        source_account=account,
        network_passphrase=NETWORK_PASSPHRASES[NETWORK],
        base_fee=server.fetch_base_fee(),
    )
    tolerance = 1.002 if side == "buy" else 0.998
    limit_price = price * tolerance

    if side == "buy":
        tx_builder.append_manage_buy_offer_op(
            selling=counter_asset, buying=BASE_ASSET,
            amount=str(round(amount, 7)), price=str(round(1 / limit_price, 7)),
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
        log.info(f"Order submitted for XLM/{counter_asset.code}: {response['hash']}")
        return response
    except BaseHorizonError as e:
        log.error(f"Order failed for XLM/{counter_asset.code}: {e}")
        return None


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
            submit_order(server, keypair, counter_asset, "sell", pair_state.position_size, price)
            pnl = (price - pair_state.entry_price) * pair_state.position_size
            pair_state.daily_pnl += pnl
            state.total_daily_pnl += pnl
            pair_state.position_open = False
            log.warning(f"[{label}] Stop-loss hit ({loss_pct:.2%}). Closed. PnL: {pnl:.2f}")

    if pair_state.daily_pnl <= -MAX_DAILY_LOSS_PER_PAIR:
        log.warning(f"[{label}] Daily loss limit hit. Halting {label} for today.")
        pair_state.halted = True
        return

    if state.total_daily_pnl <= -MAX_TOTAL_DAILY_LOSS:
        log.warning("Total daily loss limit hit. Halting entire bot for today.")
        state.globally_halted = True
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
        submit_order(server, keypair, counter_asset, "buy", size, price)
        pair_state.position_open = True
        pair_state.entry_price = price
        pair_state.position_size = size
        log.info(f"[{label}] BUY (dip in uptrend, RSI={rsi:.1f}) — {size:.4f} XLM at {price:.6f}")

    # SELL HIGH: overbought, OR trend has flipped down (protect gains)
    elif pair_state.position_open and (overbought or not uptrend):
        reason = "overbought" if overbought else "trend flipped down"
        submit_order(server, keypair, counter_asset, "sell", pair_state.position_size, price)
        pnl = (price - pair_state.entry_price) * pair_state.position_size
        pair_state.daily_pnl += pnl
        state.total_daily_pnl += pnl
        log.info(f"[{label}] SELL ({reason}, RSI={rsi:.1f}) — closed. PnL: {pnl:.2f}")
        pair_state.position_open = False


def run_bot():
    if not DRY_RUN and not SECRET_KEY:
        raise SystemExit("STELLAR_SECRET_KEY is required for live trading.")

    server = Server(HORIZON_URLS[NETWORK])
    keypair = Keypair.from_secret(SECRET_KEY) if SECRET_KEY else None
    if not keypair:
        raise SystemExit("STELLAR_SECRET_KEY is required even for dry runs "
                          "(the bot reads pairs from your account).")

    state = BotState()
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
                for asset in discover_tradable_assets(server, keypair.public_key):
                    key = asset_key(asset)
                    if key not in state.pairs:
                        log.info(f"Discovered new asset: {asset.code} — adding to bot")
                        state.pairs[key] = PairState(asset=asset)

                if not state.pairs:
                    log.warning("No tradable (non-XLM) trustlines found yet in this account.")

            for pair_state in list(state.pairs.values()):
                process_pair(server, keypair, state, pair_state)

            cycle_count += 1

        except BaseHorizonError as e:
            log.error(f"Horizon API error: {e}")
        except Exception as e:
            log.exception(f"Unexpected error: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_bot()
