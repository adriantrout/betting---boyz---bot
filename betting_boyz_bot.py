# Betting Boyz — Multi-market picks (H2H + Totals + Spreads)
# WhatsApp-only version (FULL) — SAST timezone lock + results/performance tracking
#
# ✅ Timezone lock: Africa/Johannesburg (SAST) for slot + scheduling decisions
# ✅ Results: auto-score yesterday's bets using The Odds API scores endpoint (best-effort)
# ✅ Monthly stats: wins/losses + simple ROI (1 unit per bet)
#
# Run (auto slot, SAST):
#   python betting_boyz_bot.py
# Run explicit:
#   python betting_boyz_bot.py --slot morning
#   python betting_boyz_bot.py --slot afternoon
# Refresh sports cache:
#   python betting_boyz_bot.py --refresh-sports
# Self-test (no API calls, no WhatsApp send):
#   python betting_boyz_bot.py --self-test
#
# Environment variables (set via GitHub Secrets / Actions env):
# - TWILIO_ACCOUNT_SID
# - TWILIO_AUTH_TOKEN
# - TWILIO_WHATSAPP_FROM
# - TO_WHATSAPP_NUMBER
# - ODDS_API_KEY
# - WHATSAPP_CHANNEL_LINK
# Optional:
# - TZ_NAME (default Africa/Johannesburg)
# - SEND_ENABLED (default "1"; set "0" for dry-run)

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from twilio.rest import Client

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

print("🔹 Betting Boyz automation starting...")

# ------------------------------
# CONFIG (env overrides supported)
# ------------------------------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
TO_WHATSAPP_NUMBER = os.getenv("TO_WHATSAPP_NUMBER", "")
WHATSAPP_CHANNEL_LINK = os.getenv("WHATSAPP_CHANNEL_LINK", "")

BETS_LOG_FILE = os.getenv("BETS_LOG_FILE", "bets_log.json")
USED_BETS_FILE = os.getenv("USED_BETS_FILE", "used_bets.json")
DEBUG_JSON_FILE = os.getenv("DEBUG_JSON_FILE", "bets_debug.json")
DEBUG_LOG_FILE = os.getenv("DEBUG_LOG_FILE", "bets_debug.log")
SPORTS_FILE = os.getenv("SPORTS_FILE", "bets_sports.json")

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports"

# ✅ All soccer leagues
SPORTS: List[str] = []
AUTO_REFRESH_SPORTS = True

# Slot decision cutoff (SAST)
MORNING_CUTOFF_HOUR = 12

# Market preference weights (tiny bias)
MARKET_WEIGHT = {"h2h": 1.00, "totals": 0.95, "spreads": 0.90}

# Odds guardrail
MAX_DECIMAL_ODDS = 10.0

# Timezone lock
TZ_NAME = os.getenv("TZ_NAME", "Africa/Johannesburg")

# Sending toggle (useful for testing in Actions)
SEND_ENABLED = os.getenv("SEND_ENABLED", "1") != "0"


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
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {text}\n")


# ------------------------------
# Time helpers (SAST)
# ------------------------------
def now_local() -> datetime:
    if ZoneInfo is None:
        # Fallback: assume system tz
        return datetime.now().astimezone()
    try:
        return datetime.now(ZoneInfo(TZ_NAME))
    except Exception:
        return datetime.now().astimezone()


def to_local(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if ZoneInfo is None:
        return dt.astimezone()
    try:
        return dt.astimezone(ZoneInfo(TZ_NAME))
    except Exception:
        return dt.astimezone()


# ------------------------------
# Live sports list refresh
# ------------------------------
def fetch_sports_list(api_key: str) -> List[Dict[str, Any]]:
    url = f"{ODDS_API_URL}?apiKey={api_key}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()


def refresh_sports_file(api_key: str, out_file: str = SPORTS_FILE) -> List[Dict[str, Any]]:
    data = fetch_sports_list(api_key)
    save_json_file(out_file, data)
    append_debug_log(f"Fetched sports list and saved to {out_file}")
    return data


def load_all_soccer_sports_from_file() -> List[str]:
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
        return dt
    except Exception:
        return None


# ------------------------------
# Fetch matches + bookmakers/markets
# ------------------------------

def _fetch_odds_for_sport(sport: str, markets: str) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch odds payload for a sport. Returns list of games or None on hard failure.
    Uses wide regions for better bookmaker coverage.
    """
    url = (
        f"{ODDS_API_URL}/{sport}/odds/"
        f"?apiKey={ODDS_API_KEY}&regions=eu,uk,us,au&markets={markets}&oddsFormat=decimal"
    )
    try:
        response = requests.get(url, timeout=25)
    except Exception as e:
        append_debug_log(f"Network error fetching {sport} markets={markets}: {e}")
        return None
    if response.status_code != 200:
        append_debug_log(f"Failed to fetch {sport} markets={markets}: {response.status_code} {response.text[:200]}")
        return None
    try:
        return response.json()
    except Exception as e:
        append_debug_log(f"Failed to parse JSON for {sport} markets={markets}: {e}")
        return None


def fetch_real_matches(sports_list: List[str]) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []

    for sport in sports_list:
        # First attempt: full markets (h2h, totals, spreads)
        data = _fetch_odds_for_sport(sport, markets="h2h,spreads,totals")
        if data is None:
            continue

        # Fallback: if bookmaker coverage is very low, try h2h-only (often has better coverage)
        with_bm = sum(1 for g in data if (g.get("bookmakers") or []))
        if len(data) > 0 and with_bm == 0:
            append_debug_log(f"No bookmakers for {sport} with full markets; retrying h2h-only.")
            data2 = _fetch_odds_for_sport(sport, markets="h2h")
            if data2 is not None:
                data = data2


        for game in data:
            try:
                commence_dt_utc = _parse_commence_time(game.get("commence_time"))
                commence_local = to_local(commence_dt_utc) if commence_dt_utc else None
                matches.append(
                    {
                        "event_id": game.get("id"),
                        "kickoff_time": commence_local.strftime("%H:%M") if commence_local else None,
                        "kickoff_dt": commence_local,
                        "kickoff_dt_utc": commence_dt_utc,
                        "kickoff_date": commence_local.date().isoformat() if commence_local else None,
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
    best: Dict[Tuple[str, str, Any], Outcome] = {}
    for o in iter_market_outcomes(match):
        k = (o["market_key"], o["name"], o.get("point"))
        if k not in best or o["price"] > best[k]["price"]:
            best[k] = o
    return list(best.values())


def consensus_probs(outcomes: List[Outcome]) -> Dict[Tuple[str, str, Any], float]:
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

        all_outcomes = list(iter_market_outcomes(m)) or best_outcomes
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
        "event_id": m.get("event_id"),
        "home": m.get("home"),
        "away": m.get("away"),
        "league": m.get("league"),
        "kickoff_time": m.get("kickoff_time"),
        "kickoff_dt": m.get("kickoff_dt"),
        "kickoff_dt_utc": m.get("kickoff_dt_utc"),
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
        "result": "",  # will be populated by scoring
        "reason": (
            f"Implied {implied:.1%} vs model p={(p or 0.0):.1%}. "
            f"EV={ev:.3f}. Market={candidate.get('market')}. Conf={candidate.get('confidence')}."
        ),
    }


# ------------------------------
# Results scoring (best-effort via The Odds API scores)
# ------------------------------
def fetch_scores_for_sport(sport_key: str, days_from: int = 1) -> List[Dict[str, Any]]:
    # Odds API v4 scores endpoint (best-effort)
    url = f"{ODDS_API_URL}/{sport_key}/scores/?apiKey={ODDS_API_KEY}&daysFrom={days_from}"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        append_debug_log(f"Scores fetch failed for {sport_key}: {r.status_code} {r.text[:120]}")
        return []
    try:
        return r.json()
    except Exception as e:
        append_debug_log(f"Scores JSON parse failed for {sport_key}: {e}")
        return []


def _extract_final_scores(score_obj: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """
    Attempts to extract (home_score, away_score) from Odds API score object.
    Returns None if not available/final.
    """
    # Must be completed
    completed = score_obj.get("completed")
    if completed is False:
        return None

    scores = score_obj.get("scores")
    if not isinstance(scores, list) or len(scores) < 2:
        return None

    # scores list elements often look like {"name": "...", "score": "2"}
    def _to_int(x):
        try:
            return int(float(x))
        except Exception:
            return None

    s0 = _to_int(scores[0].get("score"))
    s1 = _to_int(scores[1].get("score"))
    if s0 is None or s1 is None:
        return None

    # Need to map which entry is home/away by name
    home = score_obj.get("home_team")
    away = score_obj.get("away_team")
    if isinstance(home, str) and isinstance(away, str):
        name0 = scores[0].get("name")
        name1 = scores[1].get("name")
        if name0 == home and name1 == away:
            return (s0, s1)
        if name0 == away and name1 == home:
            return (s1, s0)

    # Fallback: assume order matches
    return (s0, s1)


def score_bet_with_final(bet: Dict[str, Any], home_score: int, away_score: int) -> str:
    pick = bet.get("pick", {}) or {}
    mkey = pick.get("market_key")
    selection = (pick.get("selection") or "").lower()
    point = pick.get("point")

    if mkey == "h2h":
        # selection is team name
        sel = (pick.get("name") or "").lower()
        home = (bet.get("home") or "").lower()
        away = (bet.get("away") or "").lower()
        if sel == home:
            return "✅ Won" if home_score > away_score else "❌ Lost"
        if sel == away:
            return "✅ Won" if away_score > home_score else "❌ Lost"
        # draw case
        if sel == "draw":
            return "✅ Won" if home_score == away_score else "❌ Lost"
        # unknown
        return ""

    if mkey == "totals":
        try:
            line = float(point)
        except Exception:
            return ""
        total = home_score + away_score
        if "over" in selection:
            return "✅ Won" if total > line else "❌ Lost"
        if "under" in selection:
            return "✅ Won" if total < line else "❌ Lost"
        return ""

    if mkey == "spreads":
        # selection is team name + handicap point
        try:
            line = float(point)
        except Exception:
            return ""
        sel = (pick.get("name") or "").lower()
        home = (bet.get("home") or "").lower()
        away = (bet.get("away") or "").lower()
        if sel == home:
            return "✅ Won" if (home_score + line) > away_score else "❌ Lost"
        if sel == away:
            return "✅ Won" if (away_score + line) > home_score else "❌ Lost"
        return ""

    return ""


def update_yesterday_results(log: Dict[str, Any]) -> None:
    """
    Updates results for yesterday's entries (morning + afternoon) if they don't have results yet.
    """
    local_now = now_local()
    yday = (local_now.date() - timedelta(days=1)).isoformat()

    # collect keys for yesterday
    y_keys = [yday, f"{yday}_aft"]
    # gather sports involved
    sports = set()
    for k in y_keys:
        entry = log.get(k)
        if not isinstance(entry, dict):
            continue
        for role in ("safe", "value"):
            bet = entry.get(role) or {}
            if bet and (bet.get("result") or "") == "":
                if bet.get("league"):
                    sports.add(bet["league"])

    if not sports:
        return

    # fetch score data per sport (daysFrom=1 is usually enough for yesterday)
    score_map: Dict[str, Dict[str, Tuple[int, int]]] = {}
    for sp in sports:
        games = fetch_scores_for_sport(sp, days_from=1)
        by_id: Dict[str, Tuple[int, int]] = {}
        for g in games or []:
            gid = g.get("id")
            final = _extract_final_scores(g)
            if gid and final:
                by_id[gid] = final
        score_map[sp] = by_id

    updated_any = False
    for k in y_keys:
        entry = log.get(k)
        if not isinstance(entry, dict):
            continue
        for role in ("safe", "value"):
            bet = entry.get(role)
            if not isinstance(bet, dict):
                continue
            if (bet.get("result") or "") != "":
                continue
            sp = bet.get("league")
            gid = bet.get("event_id")
            if not (sp and gid):
                continue
            final = score_map.get(sp, {}).get(gid)
            if not final:
                continue
            hs, aw = final
            res = score_bet_with_final(bet, hs, aw)
            if res:
                bet["result"] = res
                bet["final_score"] = f"{hs}-{aw}"
                updated_any = True

    if updated_any:
        append_debug_log("Updated yesterday results.")
    else:
        append_debug_log("No yesterday results available yet.")


# ------------------------------
# Simple performance stats
# ------------------------------
def compute_month_stats(log: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes wins/losses and simple ROI assuming 1 unit stake per bet.
    ROI = profit / stake, profit uses decimal odds:
      win profit = odds-1, loss = -1
    """
    local_now = now_local()
    ym = (local_now.year, local_now.month)

    def profit_for(bet: Dict[str, Any]) -> Optional[float]:
        r = (bet.get("result") or "").strip()
        odds = bet.get("odds")
        try:
            odds = float(odds)
        except Exception:
            return None
        if r == "✅ Won":
            return odds - 1.0
        if r == "❌ Lost":
            return -1.0
        return None

    out = {
        "safe_wins": 0, "safe_losses": 0, "safe_total": 0,
        "value_wins": 0, "value_losses": 0, "value_total": 0,
        "stake_units": 0.0, "profit_units": 0.0, "roi": 0.0,
    }

    for k, entry in (log or {}).items():
        if not isinstance(entry, dict):
            continue
        # key starts with YYYY-MM-DD
        try:
            ds = k.split("_")[0]
            dt = datetime.strptime(ds, "%Y-%m-%d")
        except Exception:
            continue
        if (dt.year, dt.month) != ym:
            continue

        for role in ("safe", "value"):
            bet = entry.get(role)
            if not isinstance(bet, dict):
                continue
            out[f"{role}_total"] += 1
            r = (bet.get("result") or "").strip()
            if r == "✅ Won":
                out[f"{role}_wins"] += 1
            elif r == "❌ Lost":
                out[f"{role}_losses"] += 1

            p = profit_for(bet)
            if p is not None:
                out["stake_units"] += 1.0
                out["profit_units"] += p

    out["roi"] = (out["profit_units"] / out["stake_units"]) if out["stake_units"] > 0 else 0.0
    return out


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
        now = now_local()
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
    score = f" | FT {bet.get('final_score')}" if bet.get("final_score") else ""

    return (
        f"{bet.get('stars','')} ({market})\n"
        f"{bet.get('home','')} vs {bet.get('away','')} ({bet.get('league','')}){score}\n"
        f"⏰ Kick-off: {bet.get('kickoff_time') or 'TBC'} — {_countdown(bet.get('kickoff_dt'))}\n"
        f"Pick: {selection}\n"
        f"Odds: {bet.get('odds')}\n"
        f"Confidence: {bet.get('confidence','')}\n"
        f"Result: {bet.get('result','')}\n"
        f"Reason: {bet.get('reason','')}{label}\n"
    )


def format_message_whatsapp(
    safe_bet: Dict[str, Any],
    value_bet: Dict[str, Any],
    slot_label: str,
    stats: Optional[Dict[str, Any]] = None,
    yesterday_summary: Optional[str] = None,
    parlay: Optional[Dict[str, Any]] = None,
) -> str:
    today = now_local().strftime("%A, %d %B %Y")
    msg = f"⚽ BETTING BOYZ — REAL MATCH PICKS\n📅 {today}\n🕒 {slot_label.upper()} PUSH (SAST)\n\n"

    if stats:
        msg += (
            "📊 Monthly Performance (1u per bet)\n"
            f"SAFE: {stats['safe_wins']}-{stats['safe_losses']} (Total {stats['safe_total']})\n"
            f"VALUE: {stats['value_wins']}-{stats['value_losses']} (Total {stats['value_total']})\n"
            f"Profit: {stats['profit_units']:.2f}u | ROI: {stats['roi']:.1%}\n\n"
        )

    if yesterday_summary:
        msg += f"📅 Yesterday Update\n{yesterday_summary}\n\n"

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
    if not SEND_ENABLED:
        print("🧪 SEND_ENABLED=0, skipping WhatsApp send (dry-run).")
        append_debug_log("Dry-run: message prepared, not sent.")
        return

    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TO_WHATSAPP_NUMBER and TWILIO_WHATSAPP_FROM):
        raise RuntimeError("Missing Twilio configuration environment variables.")

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    msg = client.messages.create(body=message, from_=TWILIO_WHATSAPP_FROM, to=TO_WHATSAPP_NUMBER)
    append_debug_log(f"WhatsApp message sent! SID: {msg.sid}")


# ------------------------------
# Self-tests (no API calls)
# ------------------------------
def _self_test() -> None:
    # Log writer
    append_debug_log("SELFTEST: debug log write")

    fake_match = {
        "event_id": "evt_test_1",
        "home": "Team A",
        "away": "Team B",
        "league": "soccer_test",
        "kickoff_dt": now_local() + timedelta(hours=3),
        "kickoff_dt_utc": datetime.now(timezone.utc) + timedelta(hours=1),
        "kickoff_time": (now_local() + timedelta(hours=3)).strftime("%H:%M"),
        "kickoff_date": now_local().date().isoformat(),
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
    assert "Over 2.5 Goals" in msg, "Totals selection should include goal line"
    assert "(SAST)" in msg, "Should mention SAST"
    print("✅ Self-test passed")


# ------------------------------
# MAIN
# ------------------------------
def main(args: argparse.Namespace) -> None:
    global SPORTS

    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY env var is missing. Add it to GitHub Secrets.")

    # Slot decision in SAST
    local_now = now_local()
    slot = args.slot
    if slot is None:
        slot = "morning" if local_now.hour < MORNING_CUTOFF_HOUR else "afternoon"

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
        SPORTS = ["soccer_epl", "soccer_spain_la_liga", "soccer_italy_serie_a"]
        append_debug_log(f"Using fallback sports list: {len(SPORTS)}")

    if args.self_test:
        _self_test()
        return

    # Load logs and attempt to update yesterday results first
    log: Dict[str, Any] = load_json_file(BETS_LOG_FILE, default={})
    update_yesterday_results(log)
    save_json_file(BETS_LOG_FILE, log)

    stats = compute_month_stats(log)

    matches = fetch_real_matches(SPORTS)
    with_bm = sum(1 for m in matches if (m.get('bookmakers') or []))
    print(f"Matches fetched: {len(matches)} | With bookmakers: {with_bm}")
    append_debug_log(f"Matches fetched: {len(matches)} | With bookmakers: {with_bm}")
    today_str = local_now.date().isoformat()

    # Filter: today's matches only (local date)
    todays_matches = [m for m in matches if m.get("kickoff_date") == today_str]
    if not todays_matches:
        todays_matches = matches
        append_debug_log("No kickoff_date matches for today; using all matches as fallback.")

    used: List[str] = load_json_file(USED_BETS_FILE, default=[])

    # Exclude used matches (so afternoon is different)
    available = sorted(
        [m for m in todays_matches if f"{m.get('home')} vs {m.get('away')}" not in used],
        key=lambda x: x.get("kickoff_dt") or (local_now + timedelta(days=365)),
    )
    if not available:
        append_debug_log("No available matches after filtering used bets. Resetting used list.")
        used = []
        available = sorted(todays_matches, key=lambda x: x.get("kickoff_dt") or (local_now + timedelta(days=365)))

    candidates = build_candidates(available)
    print(f"Candidates built: {len(candidates)}")
    append_debug_log(f"Candidates built: {len(candidates)}")
    if not candidates:
        # Fallback: consider next 48 hours of matches (local time) to avoid blank days
        next_48h_cutoff = local_now + timedelta(hours=48)
        next_48h = [m for m in matches if m.get('kickoff_dt') and m['kickoff_dt'] <= next_48h_cutoff]
        append_debug_log(f"No candidates today; trying next 48h window: matches={len(next_48h)}")
        candidates = build_candidates(next_48h)
        print(f"Candidates built (next 48h): {len(candidates)}")
        append_debug_log(f"Candidates built (next 48h): {len(candidates)}")
    if not candidates:
        append_debug_log("No candidates built even after 48h fallback. Exiting.")
        print("No candidates found.")
        return

    safe_c, value_c = pick_two_distinct(candidates)
    if not safe_c or not value_c:
        append_debug_log("Unable to pick SAFE/VALUE (insufficient candidates). Exiting.")
        print("Insufficient candidates for SAFE/VALUE.")
        return

    safe_bet = build_bet_obj(safe_c)
    value_bet = build_bet_obj(value_c)

    # Mark used immediately
    used.append(f"{safe_bet['home']} vs {safe_bet['away']}")
    used.append(f"{value_bet['home']} vs {value_bet['away']}")

    parlay = build_parlay(candidates, size=3)
    if parlay:
        used.append(f"PARLAY|{today_str}_{slot}|odds:{parlay['odds']}")

    # Save debug snapshot
    save_json_file(DEBUG_JSON_FILE, {"timestamp": datetime.now(timezone.utc), "slot": slot, "candidates": candidates[:250]})

    # Store bets by slot for history
    key = today_str if slot == "morning" else f"{today_str}_aft"
    log[key] = {"safe": safe_bet, "value": value_bet, "parlay": parlay}
    save_json_file(BETS_LOG_FILE, log)
    save_json_file(USED_BETS_FILE, used)

    # Yesterday summary lines (if available)
    yday = (local_now.date() - timedelta(days=1)).isoformat()
    y_summary_parts = []
    for k in [yday, f"{yday}_aft"]:
        e = log.get(k) or {}
        if isinstance(e, dict) and (e.get("safe") or e.get("value")):
            s = e.get("safe", {})
            v = e.get("value", {})
            y_summary_parts.append(
                f"{k}: SAFE {s.get('result','')} ({s.get('home','')} vs {s.get('away','')}) | "
                f"VALUE {v.get('result','')} ({v.get('home','')} vs {v.get('away','')})"
            )
    yesterday_summary = "\n".join(y_summary_parts) if y_summary_parts else None

    msg = format_message_whatsapp(safe_bet, value_bet, slot_label=slot, stats=stats, yesterday_summary=yesterday_summary, parlay=parlay)
    send_whatsapp(msg)

    print(f"✅ Sent {slot} picks (SAST). Candidates: {len(candidates)} | Available matches: {len(available)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-sports", action="store_true", help="Fetch live sports list and save to file")
    parser.add_argument("--self-test", action="store_true", help="Run quick self-tests (no API calls)")
    parser.add_argument("--slot", choices=["morning", "afternoon"], default=None, help="Force morning/afternoon push")
    args = parser.parse_args()
    main(args)
