# Betting Boyz — Multi-market picks (H2H + Totals + Spreads)
# WhatsApp-only version (FULL)
#
# ✅ Morning push = picks for ALL matches happening today (regardless of kickoff time)
# ✅ Afternoon push = different picks later the same day (auto-detected or via --slot afternoon)
# ✅ All available soccer leagues (auto-refresh from Odds API sports list)
#
# Run (auto slot):
#   python betting_boyz_real_final_verified_FULL_PATCHED.py
# Run explicit:
#   python betting_boyz_real_final_verified_FULL_PATCHED.py --slot morning
#   python betting_boyz_real_final_verified_FULL_PATCHED.py --slot afternoon
# Refresh sports cache:
#   python betting_boyz_real_final_verified_FULL_PATCHED.py --refresh-sports
# Quick self-test (no API calls):
#   python betting_boyz_real_final_verified_FULL_PATCHED.py --self-test

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from twilio.rest import Client

print("🔹 Betting Boyz automation starting...")

# ------------------------------
# CONFIG (env overrides supported)
# ------------------------------
# NOTE: Prefer environment variables for secrets.
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "ACbe06b36ff54b6ae49a596094496cb4fd")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "ee6ede0285c51592a59d29fbcd17eca2")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
TO_WHATSAPP_NUMBER = os.getenv("TO_WHATSAPP_NUMBER", "whatsapp:+27733647400")
WHATSAPP_CHANNEL_LINK = os.getenv(
    "WHATSAPP_CHANNEL_LINK", "https://whatsapp.com/channel/0029Vb7f24O4Y9lpZ4aVrP16"
)

BETS_LOG_FILE = os.getenv("BETS_LOG_FILE", "bets_log.json")
USED_BETS_FILE = os.getenv("USED_BETS_FILE", "used_bets.json")
DEBUG_JSON_FILE = os.getenv("DEBUG_JSON_FILE", "bets_debug.json")
DEBUG_LOG_FILE = os.getenv("DEBUG_LOG_FILE", "bets_debug.log")
SPORTS_FILE = os.getenv("SPORTS_FILE", "bets_sports.json")

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "78571d4dc644b92bc268b1b80b8dbd24")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports"

# ✅ Empty means: auto-load ALL available soccer leagues from sports list file / API
SPORTS: List[str] = []

# When True, script will refresh sports list at startup (safe: falls back if it fails)
AUTO_REFRESH_SPORTS = True

# Determines auto slot
MORNING_CUTOFF_HOUR = 12

# Market preference weights (tiny bias)
MARKET_WEIGHT = {"h2h": 1.00, "totals": 0.95, "spreads": 0.90}

# Odds guardrail
MAX_DECIMAL_ODDS = 10.0


# ------------------------------
# JSON helpers (datetime-safe)
# ------------------------------
class DateTimeEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return super().default(o)


def make_datetimes_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: make_datetimes_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_datetimes_serializable(v) for v in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return obj


def load_json_file(fn: str, default: Any = None) -> Any:
    if os.path.exists(fn):
        with open(fn, "r", encoding="utf-8") as f:
            return json.load(f)
    return default if default is not None else {}


def save_json_file(fn: str, data: Any) -> None:
    safe = make_datetimes_serializable(data)
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)


def append_debug_log(text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
        # ✅ FIXED: properly closed f-string with newline
        f.write(f"[{ts}] {text}\n")


# ------------------------------
# Live sports list refresh
# ------------------------------
def fetch_sports_list(api_key: str) -> List[Dict[str, Any]]:
    url = f"{ODDS_API_URL}?apiKey={api_key}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()


def refresh_sports_file(api_key: str, out_file: str = SPORTS_FILE) -> List[Dict[str, Any]]:
    data = fetch_sports_list(api_key)
    save_json_file(out_file, data)
    append_debug_log(f"Fetched live sports list and saved to {out_file}")
    return data


def load_all_soccer_sports_from_file() -> List[str]:
    """
    Loads sports list from SPORTS_FILE and returns ALL soccer keys.
    """
    data = load_json_file(SPORTS_FILE, default=[])
    keys: List[str] = []
    if isinstance(data, list):
        for item in data:
            k = (item or {}).get("key")
            if isinstance(k, str) and k.startswith("soccer"):
                keys.append(k)
    return keys


# ------------------------------
# Time parsing
# ------------------------------
def _parse_commence_time(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone()
    except Exception:
        return None


# ------------------------------
# Fetch matches + bookmakers/markets
# ------------------------------
def fetch_real_matches(sports_list: List[str]) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []

    for sport in sports_list:
        url = (
            f"{ODDS_API_URL}/{sport}/odds/"
            f"?apiKey={ODDS_API_KEY}&regions=uk,eu&markets=h2h,spreads,totals&oddsFormat=decimal"
        )
        try:
            response = requests.get(url, timeout=20)
        except Exception as e:
            append_debug_log(f"Network error fetching {sport}: {e}")
            continue

        if response.status_code != 200:
            append_debug_log(f"Failed to fetch {sport}: {response.status_code} {response.text[:200]}")
            continue

        try:
            data = response.json()
        except Exception as e:
            append_debug_log(f"Failed to parse JSON for {sport}: {e}")
            continue

        for game in data:
            try:
                commence_dt = _parse_commence_time(game.get("commence_time"))
                matches.append(
                    {
                        "kickoff_time": commence_dt.strftime("%H:%M") if commence_dt else None,
                        "kickoff_dt": commence_dt,
                        "kickoff_date": commence_dt.date().isoformat() if commence_dt else None,
                        "league": sport,
                        "home": game.get("home_team"),
                        "away": game.get("away_team"),
                        "bookmakers": game.get("bookmakers", []) or [],
                    }
                )
            except Exception as e:
                append_debug_log(f"Skipping match due to error: {e}")

    append_debug_log(f"Fetched {len(matches)} matches")
    return matches


# ------------------------------
# Odds extraction (multi-market)
# ------------------------------
Outcome = Dict[str, Any]


def iter_market_outcomes(match: Dict[str, Any]) -> Iterable[Outcome]:
    """Yield outcomes across h2h/totals/spreads: {market_key,name,price,point}."""
    for bm in match.get("bookmakers", []) or []:
        for market in bm.get("markets", []) or []:
            mkey = market.get("key")
            if mkey not in ("h2h", "totals", "spreads"):
                continue
            for o in market.get("outcomes", []) or []:
                price = o.get("price")
                name = o.get("name")
                point = o.get("point")
                if name and price and float(price) > 1:
                    yield {"market_key": mkey, "name": name, "price": float(price), "point": point}


def best_prices_by_outcome(match: Dict[str, Any]) -> List[Outcome]:
    """Best prices across bookmakers for each (market_key,name,point)."""
    best: Dict[Tuple[str, str, Any], Outcome] = {}
    for o in iter_market_outcomes(match):
        k = (o["market_key"], o["name"], o.get("point"))
        if k not in best or o["price"] > best[k]["price"]:
            best[k] = o
    return list(best.values())


def consensus_probs(outcomes: List[Outcome]) -> Dict[Tuple[str, str, Any], float]:
    """Median-price implied probs, normalized per market."""
    grouped: Dict[Tuple[str, str, Any], List[float]] = {}
    for o in outcomes:
        key = (o["market_key"], o["name"], o.get("point"))
        grouped.setdefault(key, []).append(float(o["price"]))

    per_market_raw: Dict[str, Dict[Tuple[str, str, Any], float]] = {}
    for (mkey, name, point), prices in grouped.items():
        prices = sorted(prices)
        n = len(prices)
        median = prices[n // 2] if n % 2 == 1 else (prices[n // 2 - 1] + prices[n // 2]) / 2
        per_market_raw.setdefault(mkey, {})[(mkey, name, point)] = (1.0 / median) if median else 0.0

    probs: Dict[Tuple[str, str, Any], float] = {}
    for mkey, raw in per_market_raw.items():
        s = sum(raw.values()) or 1.0
        for k, v in raw.items():
            probs[k] = v / s
    return probs


def display_market_label(mkey: str) -> str:
    return {"h2h": "Match Result", "totals": "Totals", "spreads": "Handicap"}.get(mkey, mkey)


def display_selection(mkey: str, name: str, point: Any) -> str:
    if mkey == "totals":
        return f"{name} {point} Goals" if point is not None else f"{name} Goals"
    if mkey == "spreads":
        if point is None:
            return f"{name} (Handicap)"
        try:
            p = float(point)
            sign = "+" if p > 0 else ""
            return f"{name} {sign}{point}"
        except Exception:
            return f"{name} {point}"
    return name


def confidence_tier(p: Optional[float], ev: float, odds: float) -> Tuple[str, str]:
    if p is None:
        return ("Low", "⭐")
    if ev >= 0.05 and p >= 0.60 and odds <= 2.2:
        return ("High", "⭐⭐⭐⭐")
    if ev >= 0.02 and p >= 0.52:
        return ("Medium", "⭐⭐⭐")
    if ev >= 0 and p >= 0.45:
        return ("Medium-Low", "⭐⭐")
    return ("Low", "⭐")


# ------------------------------
# Candidate building + SAFE/VALUE selection
# ------------------------------
Candidate = Dict[str, Any]


def build_candidates(matches: List[Dict[str, Any]], max_odds: float = MAX_DECIMAL_ODDS) -> List[Candidate]:
    candidates: List[Candidate] = []
    for m in matches:
        best_outcomes = best_prices_by_outcome(m)
        if not best_outcomes:
            continue

        all_outcomes = list(iter_market_outcomes(m))
        if not all_outcomes:
            all_outcomes = best_outcomes

        probs = consensus_probs(all_outcomes)

        for o in best_outcomes:
            mkey = o["market_key"]
            odds = float(o["price"])
            if odds <= 1.01 or odds > max_odds:
                continue

            key = (mkey, o["name"], o.get("point"))
            p = probs.get(key)
            if p is None:
                market_best = [x for x in best_outcomes if x["market_key"] == mkey]
                inv = sum(1.0 / float(x["price"]) for x in market_best if float(x["price"]) > 0) or 1.0
                p = (1.0 / odds) / inv

            ev = (p * odds) - 1.0

            w = MARKET_WEIGHT.get(mkey, 0.9)
            risk_penalty = 0.02 * max(0.0, odds - 2.5)
            score = (ev * w) - risk_penalty

            conf, stars = confidence_tier(p, ev, odds)

            candidates.append(
                {
                    "match": m,
                    "market_key": mkey,
                    "market": display_market_label(mkey),
                    "outcome_name": o["name"],
                    "point": o.get("point"),
                    "selection": display_selection(mkey, o["name"], o.get("point")),
                    "odds": odds,
                    "p_consensus": p,
                    "ev": ev,
                    "score": score,
                    "confidence": conf,
                    "stars": stars,
                }
            )

    candidates.sort(key=lambda x: (x["score"], x["ev"], -x["odds"]), reverse=True)
    return candidates


def pick_two_distinct(candidates: List[Candidate]) -> Tuple[Optional[Candidate], Optional[Candidate]]:
    if not candidates:
        return None, None

    value = candidates[0]

    safe_pool = [c for c in candidates if c["odds"] <= 1.75 and c["ev"] >= -0.02]
    safe_pool.sort(key=lambda x: (x["odds"], -(x.get("p_consensus") or 0.0), -x["score"]))

    safe: Optional[Candidate] = None
    value_key = f"{value['match'].get('home')} vs {value['match'].get('away')}"
    for c in safe_pool + candidates:
        ck = f"{c['match'].get('home')} vs {c['match'].get('away')}"
        if ck != value_key:
            safe = c
            break

    return safe, value


def build_bet_obj(candidate: Candidate) -> Dict[str, Any]:
    m = candidate["match"]
    odds = float(candidate.get("odds") or 0.0)
    p = candidate.get("p_consensus")
    ev = float(candidate.get("ev") or 0.0)

    implied = (1.0 / odds) if odds else 0.0
    no_stake = ev <= 0

    return {
        "home": m.get("home"),
        "away": m.get("away"),
        "league": m.get("league"),
        "kickoff_time": m.get("kickoff_time"),
        "kickoff_dt": m.get("kickoff_dt"),
        "kickoff_date": m.get("kickoff_date"),
        "type": candidate.get("market"),
        "pick": {
            "market_key": candidate.get("market_key"),
            "market": candidate.get("market"),
            "selection": candidate.get("selection"),
            "name": candidate.get("outcome_name"),
            "point": candidate.get("point"),
        },
        "odds": odds,
        "p_consensus": p,
        "ev": ev,
        "confidence": candidate.get("confidence"),
        "stars": candidate.get("stars"),
        "no_stake_recommended": bool(no_stake),
        "reason": (
            f"Implied {implied:.1%} vs model p={(p or 0.0):.1%}. "
            f"EV={ev:.3f}. Market={candidate.get('market')}. Conf={candidate.get('confidence')}."
        ),
    }


# ------------------------------
# Parlay (3 legs, no duplicate matches)
# ------------------------------
def build_parlay(candidates: List[Candidate], size: int = 3) -> Optional[Dict[str, Any]]:
    legs: List[Dict[str, Any]] = []
    used_matches: set[str] = set()

    for c in candidates:
        m = c["match"]
        mk = f"{m.get('home')} vs {m.get('away')}"
        if mk in used_matches:
            continue
        if c.get("odds") is None or float(c["odds"]) > 4.5:
            continue
        legs.append(
            {
                "match": mk,
                "market": c["market"],
                "pick": c["selection"],
                "odds": float(c["odds"]),
                "p_consensus": float(c.get("p_consensus") or 0.0),
            }
        )
        used_matches.add(mk)
        if len(legs) >= size:
            break

    if len(legs) < size:
        return None

    odds_parlay = 1.0
    p_parlay = 1.0
    for leg in legs:
        odds_parlay *= leg["odds"]
        p_parlay *= leg["p_consensus"]

    ev = (p_parlay * odds_parlay) - 1.0
    implied = (1.0 / odds_parlay) if odds_parlay else 0.0

    return {
        "legs": legs,
        "odds": round(odds_parlay, 3),
        "p_consensus": p_parlay,
        "ev": round(ev, 4),
        "no_stake_recommended": bool(ev <= 0),
        "reason": f"Combined {odds_parlay:.3f} imply {implied:.1%}; model p={p_parlay:.2%}. EV={ev:.3f}.",
    }


# ------------------------------
# Message formatting (WhatsApp)
# ------------------------------
def _countdown(dt: Optional[datetime]) -> str:
    if not dt:
        return "TBC"
    try:
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        delta = dt - now
        if delta.total_seconds() <= 0:
            return "Live / Started"
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m = rem // 60
        return f"Starts in {h}h {m}m"
    except Exception:
        return "TBC"


def _bet_lines(bet: Dict[str, Any]) -> str:
    pick = bet.get("pick") or {}
    selection = pick.get("selection") or str(bet.get("pick"))
    market = pick.get("market") or bet.get("type")
    label = " — NO STAKE RECOMMENDED" if bet.get("no_stake_recommended") else ""

    return (
        f"{bet.get('stars','')} ({market})\n"
        f"{bet.get('home','')} vs {bet.get('away','')} ({bet.get('league','')})\n"
        f"⏰ Kick-off: {bet.get('kickoff_time') or 'TBC'} — {_countdown(bet.get('kickoff_dt'))}\n"
        f"Pick: {selection}\n"
        f"Odds: {bet.get('odds')}\n"
        f"Confidence: {bet.get('confidence','')}\n"
        f"Reason: {bet.get('reason','')}{label}\n"
    )


def format_message_whatsapp(
    safe_bet: Dict[str, Any],
    value_bet: Dict[str, Any],
    slot_label: str,
    parlay: Optional[Dict[str, Any]] = None,
) -> str:
    today = datetime.now().strftime("%A, %d %B %Y")
    msg = f"⚽ BETTING BOYZ — REAL MATCH PICKS\n📅 {today}\n🕒 {slot_label.upper()} PUSH\n\n"

    msg += "🔒 SAFE BET\n" + _bet_lines(safe_bet) + "\n"
    msg += "🎯 VALUE BET\n" + _bet_lines(value_bet) + "\n"

    if parlay:
        label = " — NO STAKE RECOMMENDED" if parlay.get("no_stake_recommended") else ""
        msg += "🔗 3-LEG PARLAY\n"
        msg += f"Combined Odds: {parlay.get('odds')}\n"
        msg += "Legs:\n"
        for leg in parlay.get("legs", []):
            msg += f"- {leg['match']}: {leg['pick']} ({leg['market']}) @ {leg['odds']}\n"
        msg += f"Reason: {parlay.get('reason','')}{label}\n\n"

    msg += f"👉 Join the Boyz: {WHATSAPP_CHANNEL_LINK}\n— Betting Boyz"
    return msg


# ------------------------------
# WhatsApp sender
# ------------------------------
def send_whatsapp(message: str) -> None:
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    msg = client.messages.create(body=message, from_=TWILIO_WHATSAPP_FROM, to=TO_WHATSAPP_NUMBER)
    append_debug_log(f"WhatsApp message sent! SID: {msg.sid}")


# ------------------------------
# Self-tests (no API calls)
# ------------------------------
def _self_test() -> None:
    append_debug_log("SELFTEST: debug log write")

    fake_match = {
        "home": "Team A",
        "away": "Team B",
        "league": "soccer_test",
        "kickoff_dt": datetime.now().astimezone() + timedelta(hours=3),
        "kickoff_time": (datetime.now().astimezone() + timedelta(hours=3)).strftime("%H:%M"),
        "kickoff_date": datetime.now().astimezone().date().isoformat(),
        "bookmakers": [],
    }
    cand = {
        "match": fake_match,
        "market_key": "totals",
        "market": "Totals",
        "outcome_name": "Over",
        "point": 2.5,
        "selection": display_selection("totals", "Over", 2.5),
        "odds": 1.95,
        "p_consensus": 0.55,
        "ev": 0.0725,
        "score": 0.07,
        "confidence": "Medium",
        "stars": "⭐⭐⭐",
    }
    bet = build_bet_obj(cand)
    msg = format_message_whatsapp(bet, bet, slot_label="self-test")
    assert "Over 2.5 Goals" in msg, "Expected totals selection to include the line (e.g., Over 2.5 Goals)"
    print("✅ Self-test passed")


# ------------------------------
# MAIN
# ------------------------------
def main(args: argparse.Namespace) -> None:
    global SPORTS

    # Determine slot
    now = datetime.now().astimezone()
    slot = args.slot
    if slot is None:
        slot = "morning" if now.hour < MORNING_CUTOFF_HOUR else "afternoon"

    # Refresh sports file if requested or auto
    if args.refresh_sports or AUTO_REFRESH_SPORTS:
        try:
            refresh_sports_file(ODDS_API_KEY)
        except Exception as e:
            append_debug_log(f"Sports refresh failed (continuing): {e}")

    # Build sports list (all soccer keys)
    sports_from_file = load_all_soccer_sports_from_file()
    if sports_from_file:
        SPORTS = sports_from_file
        append_debug_log(f"Loaded {len(SPORTS)} soccer leagues from {SPORTS_FILE}")
    else:
        # Fallback if file missing/empty
        if not SPORTS:
            SPORTS = ["soccer_epl", "soccer_spain_la_liga", "soccer_italy_serie_a"]
        append_debug_log(f"Using fallback sports list: {len(SPORTS)}")

    if args.self_test:
        _self_test()
        return

    matches = fetch_real_matches(SPORTS)
    today_str = now.strftime("%Y-%m-%d")

    # Normalize kickoff_dt for sorting; keep real kickoff_date for filtering
    for m in matches:
        if m.get("kickoff_dt") is None:
            # Push unknown times to end; treat as today for selection visibility
            m["kickoff_dt"] = now + timedelta(days=365)
            m["kickoff_date"] = today_str
            m["kickoff_time"] = "TBC"
        else:
            if m["kickoff_dt"].tzinfo is None:
                m["kickoff_dt"] = m["kickoff_dt"].replace(tzinfo=timezone.utc).astimezone()
            else:
                m["kickoff_dt"] = m["kickoff_dt"].astimezone()
            m["kickoff_time"] = m["kickoff_dt"].strftime("%H:%M")

    # ✅ Morning uses ALL matches happening today (regardless of kickoff time)
    todays_matches = [m for m in matches if m.get("kickoff_date") == today_str]
    if not todays_matches:
        todays_matches = matches
        append_debug_log("No kickoff_date matches for today; using all matches as fallback.")

    used: List[str] = load_json_file(USED_BETS_FILE, default=[])

    # Exclude used matches so afternoon gets different ones automatically
    available = sorted(
        [m for m in todays_matches if f"{m.get('home')} vs {m.get('away')}" not in used],
        key=lambda x: x["kickoff_dt"],
    )
    if not available:
        append_debug_log("No available matches after filtering used bets. Resetting used list.")
        used = []
        available = sorted(todays_matches, key=lambda x: x["kickoff_dt"])

    candidates = build_candidates(available)
    if not candidates:
        append_debug_log("No candidates built (no odds/markets). Exiting.")
        print("No candidates found.")
        return

    safe_c, value_c = pick_two_distinct(candidates)
    if not safe_c or not value_c:
        append_debug_log("Unable to pick SAFE/VALUE (insufficient candidates). Exiting.")
        print("Insufficient candidates for SAFE/VALUE.")
        return

    safe_bet = build_bet_obj(safe_c)
    value_bet = build_bet_obj(value_c)

    # Mark used immediately so next run gets different matches
    used.append(f"{safe_bet['home']} vs {safe_bet['away']}")
    used.append(f"{value_bet['home']} vs {value_bet['away']}")

    parlay = build_parlay(candidates, size=3)
    if parlay:
        used.append(f"PARLAY|{today_str}_{slot}|odds:{parlay['odds']}")

    # Save debug snapshot
    save_json_file(
        DEBUG_JSON_FILE,
        {"timestamp": datetime.now(), "slot": slot, "candidates": candidates[:250]},
    )

    # Store bets by slot for history
    log: Dict[str, Any] = load_json_file(BETS_LOG_FILE, default={})
    key = today_str if slot == "morning" else f"{today_str}_aft"
    log[key] = {"safe": safe_bet, "value": value_bet, "parlay": parlay}
    save_json_file(BETS_LOG_FILE, log)
    save_json_file(USED_BETS_FILE, used)

    msg = format_message_whatsapp(safe_bet, value_bet, slot_label=slot, parlay=parlay)
    send_whatsapp(msg)

    print(f"✅ Sent {slot} picks. Available matches: {len(available)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-sports", action="store_true", help="Fetch live sports list and save to file")
    parser.add_argument("--self-test", action="store_true", help="Run quick self-tests (no API calls)")
    parser.add_argument("--slot", choices=["morning", "afternoon"], default=None, help="Force morning/afternoon push")
    args = parser.parse_args()
    main(args)
