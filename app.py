"""Bybit SMA Bot 2.3 - candle-close scheduler and single worker."""
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pybit.unified_trading import HTTP

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
app = FastAPI(title="Bybit SMA Bot", version="2.3.0")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

TIMEFRAMES = {"1": 60, "3": 180, "5": 300, "15": 900, "30": 1800,
              "45": 2700, "60": 3600, "120": 7200, "240": 14400, "D": 86400}
TF_LABELS = {"1":"1 минута", "3":"3 минуты", "5":"5 минут", "15":"15 минут",
             "30":"30 минут", "45":"45 минут", "60":"1 час", "120":"2 часа",
             "240":"4 часа", "D":"1 день"}
STATE = {
    "session": None, "demo": True, "symbol": "BTCUSDT", "category": "linear",
    "timeframe": "3", "min_move": 0.3, "max_move": 0.8, "balance_pct": 15.0,
    "leverage": 50, "running": False, "last_signal": "Нет сигнала",
    "last_action": "Бот остановлен", "last_candle": None, "last_candle_ms": None,
    "last_price": None, "last_zone": None, "position_side": None, "position_size": None,
    "logs": [], "last_error": None, "initialized": False, "next_check": None,
}
STATE_LOCK = threading.RLock()
WORKER = None
STOP_EVENT = threading.Event()

class ConnectRequest(BaseModel):
    api_key: str = Field(min_length=1, max_length=200)
    api_secret: str = Field(min_length=1, max_length=200)
    demo: bool = True

class SettingsRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str = "3"
    min_move: float = 0.3
    max_move: float = 0.8
    balance_pct: float = 15.0
    leverage: int = 50


def log(message):
    with STATE_LOCK:
        stamp = datetime.now().strftime("%H:%M:%S")
        STATE["logs"] = ([f"{stamp} — {message}"] + STATE["logs"])[:100]
        STATE["last_action"] = message
        if message.startswith("Ошибка:"):
            STATE["last_error"] = message


def client():
    with STATE_LOCK:
        sess = STATE["session"]
    if sess is None:
        raise HTTPException(400, "Сначала подключите API-ключи.")
    return sess

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")

@app.get("/api/status")
def status():
    # This endpoint is deliberately local-only: it never calls Bybit.
    with STATE_LOCK:
        safe = {k: v for k, v in STATE.items() if k != "session"}
        safe["connected"] = STATE["session"] is not None
        safe["api_mode"] = "Демо" if STATE["demo"] else "Реальный аккаунт"
        safe["timeframe_label"] = TF_LABELS[STATE["timeframe"]]
        return safe

@app.post("/api/connect")
def connect(req: ConnectRequest):
    try:
        # pybit's demo=True uses Bybit's demo trading environment.
        sess = HTTP(testnet=False, demo=req.demo, api_key=req.api_key.strip(), api_secret=req.api_secret.strip())
        resp = sess.get_wallet_balance(accountType="UNIFIED")
        if resp.get("retCode") != 0:
            raise RuntimeError(resp.get("retMsg", "Не удалось проверить API"))
        with STATE_LOCK:
            if STATE["running"]:
                raise HTTPException(409, "Сначала остановите бота перед сменой подключения.")
            STATE["session"] = sess
            STATE["demo"] = req.demo
        log("Подключение успешно: " + ("деморежим" if req.demo else "РЕАЛЬНЫЙ аккаунт"))
        return {"ok": True, "message": "Подключено"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, "Не удалось подключиться к Bybit. Проверьте ключи, режим и права API. Детали: " + str(e)[:200])

@app.post("/api/settings")
def settings(req: SettingsRequest):
    symbol = req.symbol.upper().strip()
    if not symbol.isalnum():
        raise HTTPException(400, "Некорректный символ.")
    if req.timeframe not in TIMEFRAMES:
        raise HTTPException(400, "Неподдерживаемый таймфрейм.")
    if not (0 < req.min_move <= req.max_move <= 20):
        raise HTTPException(400, "Проверьте диапазон фильтра.")
    if not (0.1 <= req.balance_pct <= 100):
        raise HTTPException(400, "Доля баланса должна быть от 0,1 до 100%.")
    if not (1 <= req.leverage <= 100):
        raise HTTPException(400, "Плечо должно быть от 1 до 100.")
    with STATE_LOCK:
        if STATE["running"]:
            raise HTTPException(409, "Остановите бота перед изменением настроек.")
        STATE.update(symbol=symbol, timeframe=req.timeframe, min_move=req.min_move,
                     max_move=req.max_move, balance_pct=req.balance_pct, leverage=req.leverage,
                     initialized=False, last_candle_ms=None)
    log(f"Настройки сохранены: {symbol}, {TF_LABELS[req.timeframe]}, фильтр {req.min_move}-{req.max_move}%, маржа {req.balance_pct}%, плечо {req.leverage}x")
    return {"ok": True}


def _raw_bars(sess, interval, limit):
    resp = sess.get_kline(category=STATE["category"], symbol=STATE["symbol"], interval=interval, limit=limit)
    if resp.get("retCode") != 0:
        raise RuntimeError(f"Bybit kline: {resp.get('retCode')} {resp.get('retMsg')}")
    return sorted(resp["result"]["list"], key=lambda row: int(row[0]))


def get_bars(sess):
    """Fetch once per scheduled candle close. 45m candles are built from 15m bars."""
    tf = STATE["timeframe"]
    if tf == "45":
        raw = _raw_bars(sess, "15", 600)
        bucket_ms = 45 * 60_000
        grouped = {}
        for r in raw:
            start = int(r[0]); bucket = (start // bucket_ms) * bucket_ms
            g = grouped.setdefault(bucket, [])
            g.append(r)
        out = []
        for start, group in sorted(grouped.items()):
            group = sorted(group, key=lambda x: int(x[0]))
            if len(group) != 3 or [int(x[0]) for x in group] != [start, start+900_000, start+1_800_000]:
                continue
            out.append([str(start), group[0][1], max(float(x[2]) for x in group), min(float(x[3]) for x in group), group[-1][4], str(sum(float(x[5]) for x in group)), str(sum(float(x[6]) for x in group))])
        return out
    limit = 200 if tf != "D" else 200
    return _raw_bars(sess, tf, limit)


def closed_bars(rows, timeframe, now_ms=None):
    now_ms = now_ms or int(time.time() * 1000)
    duration = TIMEFRAMES[timeframe] * 1000
    return [r for r in rows if int(r[0]) + duration <= now_ms]


def sma(values, period, end):
    return sum(values[end-period+1:end+1]) / period


def zone(close, s20, s80):
    lo, hi = min(s20, s80), max(s20, s80)
    if close > hi: return "G"
    if close < lo: return "R"
    return "N"


def candle_zones(closed):
    closes = [float(r[4]) for r in closed]
    result = [None] * len(closes)
    for i, price in enumerate(closes):
        if i >= 79:
            result[i] = zone(price, sma(closes, 20, i), sma(closes, 80, i))
    return closes, result


def qty_for(sess, price):
    wallet = sess.get_wallet_balance(accountType="UNIFIED")
    if wallet.get("retCode") != 0:
        raise RuntimeError("Не удалось получить баланс: " + str(wallet.get("retMsg")))
    accounts = wallet.get("result", {}).get("list", [])
    coins = accounts[0].get("coin", []) if accounts else []
    usdt = next((float(c.get("walletBalance", 0)) for c in coins if c.get("coin") == "USDT"), 0.0)
    margin = usdt * STATE["balance_pct"] / 100.0
    notional = margin * STATE["leverage"]
    inst_resp = sess.get_instruments_info(category="linear", symbol=STATE["symbol"])
    if inst_resp.get("retCode") != 0 or not inst_resp.get("result", {}).get("list"):
        raise RuntimeError("Не удалось получить параметры инструмента.")
    inst = inst_resp["result"]["list"][0]
    lot = inst["lotSizeFilter"]
    step, min_qty, max_qty = float(lot["qtyStep"]), float(lot["minOrderQty"]), float(lot["maxMktOrderQty"])
    raw = min(max_qty, notional / price)
    qty = math.floor(raw / step) * step
    decimals = max(0, len(str(lot["qtyStep"]).split(".")[-1].rstrip("0")))
    qty = round(qty, decimals)
    if qty < min_qty:
        raise ValueError(f"Размер позиции меньше минимального количества {min_qty}.")
    return qty, usdt, margin, inst


def round_to_tick(price, tick, rounding):
    from decimal import Decimal, ROUND_DOWN, ROUND_UP
    p, t = Decimal(str(price)), Decimal(str(tick))
    mode = ROUND_DOWN if rounding == "down" else ROUND_UP
    return float((p / t).to_integral_value(rounding=mode) * t)


def get_position(sess):
    resp = sess.get_positions(category="linear", symbol=STATE["symbol"])
    if resp.get("retCode") != 0:
        raise RuntimeError("Не удалось проверить позицию: " + str(resp.get("retMsg")))
    return next((p for p in resp.get("result", {}).get("list", []) if float(p.get("size", 0)) > 0), None)


def maybe_trade(sess, closed, closes, zones):
    idx = len(closed) - 1
    if idx < 80 or zones[idx] is None:
        return
    current = zones[idx]
    previous = zones[idx-1] if idx else None
    price = closes[idx]
    STATE["last_price"] = price
    STATE["last_candle_ms"] = int(closed[idx][0])
    STATE["last_candle"] = datetime.fromtimestamp(int(closed[idx][0])/1000, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    STATE["last_zone"] = current
    STATE["last_signal"] = f"Зона {current} | закрытие {price:.8f}"

    # Bootstrap only records the latest closed candle; it never trades a stale signal.
    if not STATE["initialized"]:
        STATE["initialized"] = True
        log(f"Инициализация: последняя закрытая свеча {STATE['last_candle']}, цена {price}")
        return

    # A signal can only occur on the first candle of the destination zone.
    if current == previous or current not in ("G", "R"):
        return
    compact = []
    compact_indices = []
    for i in range(max(79, idx-200), idx+1):
        z = zones[i]
        if z is None: continue
        if not compact or compact[-1] != z:
            compact.append(z); compact_indices.append(i)
    signal = None
    if len(compact) >= 3:
        a, b, c = compact[-3:]
        if (a, b, c) == ("G", "N", "R") and current == "R": signal = "Sell"
        elif (a, b, c) == ("R", "N", "G") and current == "G": signal = "Buy"
    if not signal:
        return

    # SL reference is the last close of the directional zone before the grey zone.
    prior_direction = "G" if signal == "Sell" else "R"
    neutral_idx = compact_indices[-2]
    ref_idx = None
    for j in range(neutral_idx-1, max(78, idx-200)-1, -1):
        if zones[j] == prior_direction:
            ref_idx = j
            break
    if ref_idx is None:
        log("Сигнал найден, но пропущен: не найден уровень SL.")
        return
    ref = closes[ref_idx]
    move_pct = abs(price-ref)/price*100
    if not (STATE["min_move"] <= move_pct <= STATE["max_move"]):
        log(f"Сигнал {signal} пропущен: расстояние до SL {move_pct:.3f}% вне фильтра.")
        return
    # Validate no live position at the exchange before opening.
    pos = get_position(sess)
    if pos:
        STATE["position_side"], STATE["position_size"] = pos["side"], pos["size"]
        log("Сигнал пропущен: на бирже уже есть открытая позиция.")
        return
    sl = ref
    risk = abs(price-ref)
    tp = price-risk if signal == "Sell" else price+risk
    qty, usdt, margin, inst = qty_for(sess, price)
    tick = inst["priceFilter"]["tickSize"]
    sl = round_to_tick(sl, tick, "up" if signal == "Sell" else "down")
    tp = round_to_tick(tp, tick, "down" if signal == "Sell" else "up")
    if (signal == "Buy" and not (sl < price < tp)) or (signal == "Sell" and not (tp < price < sl)):
        log("Сигнал пропущен: после округления SL/TP уровни некорректны.")
        return
    lev = str(STATE["leverage"])
    lev_resp = sess.set_leverage(category="linear", symbol=STATE["symbol"], buyLeverage=lev, sellLeverage=lev)
    if lev_resp.get("retCode") not in (0, 110043):
        raise RuntimeError("Не удалось установить плечо: " + str(lev_resp.get("retMsg")))
    order = sess.place_order(category="linear", symbol=STATE["symbol"], side=signal,
                             orderType="Market", qty=str(qty), positionIdx=0)
    if order.get("retCode") != 0:
        raise RuntimeError("Ордер не принят Bybit: " + str(order.get("retMsg")))
    # If protection fails, immediately attempt to close the just-opened position.
    try:
        protect = sess.set_trading_stop(category="linear", symbol=STATE["symbol"], tpslMode="Full",
                positionIdx=0, stopLoss=str(sl), takeProfit=str(tp), slTriggerBy="MarkPrice", tpTriggerBy="MarkPrice")
        if protect.get("retCode") != 0:
            raise RuntimeError(protect.get("retMsg", "SL/TP не приняты"))
    except Exception as e:
        log("КРИТИЧЕСКИ: SL/TP не установлены, пытаюсь закрыть позицию: " + str(e)[:140])
        try:
            sess.place_order(category="linear", symbol=STATE["symbol"], side="Buy" if signal == "Sell" else "Sell",
                             orderType="Market", qty=str(qty), positionIdx=0, reduceOnly=True)
            log("Аварийная команда закрытия отправлена. Проверьте позицию в Bybit немедленно.")
        except Exception as close_e:
            log("КРИТИЧЕСКИ: не удалось отправить аварийное закрытие. Срочно проверьте Bybit: " + str(close_e)[:140])
        return
    STATE["position_side"], STATE["position_size"] = signal, qty
    log(f"Открыт {'лонг' if signal == 'Buy' else 'шорт'} {qty} {STATE['symbol']}; закрытие свечи≈{price}, SL={sl}, TP={tp}, маржа≈{margin:.2f} USDT.")


def process_once(bootstrap=False):
    sess = client()
    rows = get_bars(sess)
    tf = STATE["timeframe"]
    closed = closed_bars(rows, tf)
    if len(closed) < 81:
        log(f"Недостаточно закрытых свечей для SMA80: получено {len(closed)}.")
        return
    latest_ms = int(closed[-1][0])
    if STATE["last_candle_ms"] == latest_ms and STATE["initialized"]:
        return
    closes, zones = candle_zones(closed)
    if bootstrap:
        STATE["initialized"] = False
        maybe_trade(sess, closed, closes, zones)
        return
    # If the process woke late, analyze only the newest fully closed candle; never replay old entries.
    maybe_trade(sess, closed, closes, zones)


def next_boundary_delay(tf):
    seconds = TIMEFRAMES[tf]
    now = time.time()
    boundary = (math.floor(now / seconds) + 1) * seconds
    return max(1.0, boundary + 1.0 - now)


def loop():
    # One initial history request; subsequent kline requests happen only after candle closes.
    try:
        process_once(bootstrap=True)
    except Exception as e:
        log("Ошибка начальной загрузки: " + str(e)[:220])
    while not STOP_EVENT.is_set():
        with STATE_LOCK:
            tf = STATE["timeframe"]
        delay = next_boundary_delay(tf)
        STATE["next_check"] = datetime.fromtimestamp(time.time()+delay, timezone.utc).strftime("%H:%M:%S UTC")
        if STOP_EVENT.wait(delay):
            break
        if STOP_EVENT.is_set(): break
        try:
            process_once()
        except Exception as e:
            # Do not hammer Bybit after 10006 or any network error. Try only at the next boundary.
            log("Ошибка: " + str(e)[:220] + ". Повторная попытка — после следующего закрытия свечи.")
    with STATE_LOCK:
        STATE["running"] = False
        STATE["next_check"] = None

@app.post("/api/start")
def start():
    global WORKER
    client()
    with STATE_LOCK:
        if STATE["running"]:
            return {"ok": True, "message": "Уже запущен"}
        STATE["running"] = True
        STATE["initialized"] = False
        STATE["last_candle_ms"] = None
        STOP_EVENT.clear()
        WORKER = threading.Thread(target=loop, name="bybit-sma-worker", daemon=True)
        WORKER.start()
    log(f"Бот запущен. Первичная загрузка, затем запрос свечей по закрытию: {TF_LABELS[STATE['timeframe']]}.")
    return {"ok": True}

@app.post("/api/stop")
def stop():
    with STATE_LOCK:
        STATE["running"] = False
        STOP_EVENT.set()
    log("Бот остановлен. Открытая позиция остаётся на бирже под установленными SL/TP.")
    return {"ok": True}

@app.post("/api/position/refresh")
def refresh_position():
    sess = client()
    pos = get_position(sess)
    with STATE_LOCK:
        if pos:
            STATE["position_side"], STATE["position_size"] = pos["side"], pos["size"]
        else:
            STATE["position_side"], STATE["position_size"] = None, None
    return {"position": pos}

@app.get("/api/health")
def health():
    return {"ok": True, "version": "2.3.0"}
