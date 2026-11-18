#!/usr/bin/env python3
"""FACEIT CS2 loss-correlation analyzer.

Pulls a player's recent CS2 matches, joins per-match stats with match metadata,
and surfaces which variables separate wins from losses, including a cohort
analysis of frequent teammates. Run:

    python analyze.py --nickname jsn789 --limit 200
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("Europe/Sofia")
except Exception:  # pragma: no cover - fallback if tz data unavailable
    LOCAL_TZ = timezone.utc

BASE_URL = "https://open.faceit.com/data/v4"
CACHE_DIR = "./cache"
TARGETS = ["kpace", "nuki_tbg", "Kovkomeister", "eradicate-", "dobaka_"]
SESSION_GAP_S = 2 * 3600  # new session when gap to previous match > 2h


# --------------------------------------------------------------------------- #
# API client: cache-first, paginated, 429-backoff.
# --------------------------------------------------------------------------- #
def load_api_key():
    if os.path.exists(".env"):
        for line in open(".env"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    key = os.environ.get("FACEIT_API_KEY", "").strip()
    if not key:
        sys.exit("ERROR: FACEIT_API_KEY missing. Add it to .env (see .env.example).")
    return key


def _cache_path(path, params):
    raw = path + "?" + "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
    h = hashlib.md5(raw.encode()).hexdigest()[:10]
    safe = path.strip("/").replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe}_{h}.json")


def api_get(session, path, params=None):
    """GET with on-disk cache. Reruns hit cache, never the API."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = _cache_path(path, params)
    if os.path.exists(cp):
        with open(cp) as f:
            return json.load(f)
    url = BASE_URL + path
    for attempt in range(6):
        r = session.get(url, params=params, timeout=30)
        if r.status_code == 429:
            wait = 2 ** attempt
            print(f"  429 rate-limited; backing off {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        with open(cp, "w") as f:
            json.dump(data, f)
        return data
    sys.exit("ERROR: repeated 429s; aborting.")


def paginate(session, path, base_params, limit):
    """Walk offset-based pages (limit<=100 per call) up to `limit` items."""
    items, offset = [], 0
    while len(items) < limit:
        page = min(100, limit - len(items))
        params = dict(base_params, offset=offset, limit=page)
        data = api_get(session, path, params)
        batch = (data or {}).get("items", [])
        if not batch:
            break
        items.extend(batch)
        offset += page
        if len(batch) < page:
            break
    return items[:limit]


# --------------------------------------------------------------------------- #
# Defensive stat-key mapping (key names vary by API version).
# --------------------------------------------------------------------------- #
def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def map_stat_keys(sample):
    """Return {canonical: actual_key} by fuzzy-matching one stats item's keys."""
    keys = list(sample.keys())
    low = {k.lower(): k for k in keys}

    def find(*needles):
        for n in needles:
            if n in low:
                return low[n]
        for k in keys:
            kl = k.lower()
            if all(part in kl for part in needles[0].split()):
                return k
        return None

    mapping = {
        "kd": find("k/d ratio", "k/d"),
        "kr": find("k/r ratio", "k/r"),
        "hs": find("headshots %", "headshots%", "hs %"),
        "adr": find("adr", "average damage per round", "damage per round"),
        "kills": find("kills"),
        "rounds": find("rounds"),
        "result": find("result"),
        "match_id": find("match id", "matchid"),
        "map": find("map"),
        "date": find("match finished at", "created at", "date"),
    }
    print("\n[key-map] sample stats keys ->", keys)
    print("[key-map] resolved:", {k: v for k, v in mapping.items() if v})
    missing = [k for k, v in mapping.items() if not v]
    if missing:
        print("[key-map] WARNING unresolved:", missing)
    return mapping


# --------------------------------------------------------------------------- #
# Data pull.
# --------------------------------------------------------------------------- #
def pull(nickname, limit):
    key = load_api_key()
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {key}", "Accept": "application/json"})

    player = api_get(s, "/players", {"nickname": nickname, "game": "cs2"})
    if not player:
        sys.exit(f"ERROR: player '{nickname}' not found for cs2.")
    pid = player["player_id"]
    cs2 = (player.get("games") or {}).get("cs2", {})
    print(f"\nPlayer {nickname} | id={pid} | elo={cs2.get('faceit_elo')} "
          f"| skill_level={cs2.get('skill_level')}")

    stats_items = paginate(s, f"/players/{pid}/games/cs2/stats", {}, limit)
    history = paginate(s, f"/players/{pid}/history", {"game": "cs2"}, limit)
    print(f"Pulled {len(stats_items)} stats rows, {len(history)} history rows.")
    return s, pid, player, stats_items, history


def build_frames(pid, stats_items, history):
    """Join stats (2) and history (3) on match_id; compute derived fields."""
    if not stats_items:
        sys.exit("ERROR: no stats returned.")
    km = map_stat_keys(stats_items[0]["stats"])

    rows = []
    for it in stats_items:
        st = it["stats"]
        mid = st.get(km["match_id"]) or it.get("match_id")
        rows.append({
            "match_id": mid,
            "map": st.get(km["map"]),
            "result": int(num(st.get(km["result"]))) if km["result"] else None,
            "kd": num(st.get(km["kd"])),
            "kr": num(st.get(km["kr"])),
            "hs": num(st.get(km["hs"])),
            "adr": num(st.get(km["adr"])),
            "rounds": num(st.get(km["rounds"])),
        })
    stats_df = pd.DataFrame(rows).dropna(subset=["match_id"])

    # History: finished_at, winner faction, rosters with skill levels.
    hrows, rosters = [], {}
    for h in history:
        mid = h.get("match_id")
        teams = h.get("teams", {})
        my_faction, mate_levels, opp_levels = None, [], []
        team_rosters = {}
        for fac, team in teams.items():
            players = team.get("roster", []) or team.get("players", [])
            team_rosters[fac] = players
            ids = [p.get("player_id") for p in players]
            if pid in ids:
                my_faction = fac
        for fac, players in team_rosters.items():
            for p in players:
                lvl = num(p.get("game_skill_level") or p.get("skill_level"))
                if fac == my_faction:
                    if p.get("player_id") != pid:
                        mate_levels.append(lvl)
                else:
                    opp_levels.append(lvl)
        rosters[mid] = {"my_faction": my_faction, "teams": team_rosters}
        winner = (h.get("results") or {}).get("winner")
        hrows.append({
            "match_id": mid,
            "finished_at": h.get("finished_at"),
            "winner": winner,
            "my_faction": my_faction,
            "won_hist": int(winner == my_faction) if winner and my_faction else None,
            "teammate_avg_level": (sum(mate_levels) / len(mate_levels)) if mate_levels else float("nan"),
            "opp_avg_level": (sum(opp_levels) / len(opp_levels)) if opp_levels else float("nan"),
        })
    hist_df = pd.DataFrame(hrows).dropna(subset=["match_id"])

    df = stats_df.merge(hist_df, on="match_id", how="inner")
    if df.empty:
        sys.exit("ERROR: no overlap between stats and history on match_id.")

    # result fallback from history if stats result missing.
    df["result"] = df["result"].fillna(df["won_hist"])

    # Derived fields. Sort chronologically (oldest first) for sessions/rolling.
    df["finished_dt"] = pd.to_datetime(df["finished_at"], unit="s", utc=True)
    df = df.sort_values("finished_dt").reset_index(drop=True)
    df["hour_local"] = df["finished_dt"].dt.tz_convert(LOCAL_TZ).dt.hour

    my_level = None  # resolved by caller; fill team_skill_gap there.
    # perf_z: z-score of ADR vs my own expanding rolling mean (S8 rating fallback).
    roll_mean = df["adr"].expanding(min_periods=1).mean()
    roll_std = df["adr"].expanding(min_periods=2).std().fillna(df["adr"].std())
    roll_std = roll_std.replace(0, float("nan")).fillna(df["adr"].std() or 1.0)
    df["perf_z"] = (df["adr"] - roll_mean) / roll_std
    df["below_baseline"] = df["perf_z"] < 0

    # Sessions: new session when gap to previous match > 2h.
    gap = df["finished_dt"].diff().dt.total_seconds()
    df["session_id"] = (gap.isna() | (gap > SESSION_GAP_S)).cumsum()
    df["match_in_session"] = df.groupby("session_id").cumcount() + 1
    df["mis_bucket"] = df["match_in_session"].apply(
        lambda n: f"{n}st" if n == 1 else ("2nd" if n == 2 else ("3rd" if n == 3 else "4th+")))
    return df, rosters, km


# --------------------------------------------------------------------------- #
# Solo analyses.
# --------------------------------------------------------------------------- #
def grouped_winrate(df, col, min_n=3):
    g = df.groupby(col).agg(
        N=("result", "size"),
        win_rate=("result", "mean"),
        avg_adr=("adr", "mean"),
        avg_kd=("kd", "mean"),
    )
    g = g[g["N"] >= min_n].copy()
    g["win_rate"] = (g["win_rate"] * 100).round(1)
    g["avg_adr"] = g["avg_adr"].round(1)
    g["avg_kd"] = g["avg_kd"].round(2)
    return g.sort_values("win_rate", ascending=False)


def teammate_quality_bucket(df, my_level):
    df = df.copy()
    df["team_skill_gap"] = my_level - df["teammate_avg_level"]

    def b(gap):
        if pd.isna(gap):
            return None
        if gap > 0.5:
            return "below my level"
        if gap < -0.5:
            return "above my level"
        return "equal to my level"
    df["mate_quality"] = df["team_skill_gap"].apply(b)
    return df, grouped_winrate(df.dropna(subset=["mate_quality"]), "mate_quality")


def hour_bucket(h):
    if 6 <= h < 12:
        return "morning 06-12"
    if 12 <= h < 18:
        return "afternoon 12-18"
    if 18 <= h < 24:
        return "evening 18-24"
    return "late 00-06"


def delta(g):
    if g.empty:
        return None
    return round(g["win_rate"].max() - g["win_rate"].min(), 1)


# --------------------------------------------------------------------------- #
# Cohort analysis.
# --------------------------------------------------------------------------- #
def pearson(a, b):
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(s) < 3 or s["a"].std() == 0 or s["b"].std() == 0:
        return float("nan")
    return round(s["a"].corr(s["b"]), 3)


def resolve_targets(rosters, pid):
    """Build {nickname_lower: player_id} from MY teammates across all lobbies."""
    found, want = {}, {t.lower(): t for t in TARGETS}
    for mid, r in rosters.items():
        myf = r["my_faction"]
        if not myf:
            continue
        for p in r["teams"].get(myf, []):
            nl = (p.get("nickname") or "").lower()
            if nl in want and nl not in found:
                found[nl] = {"player_id": p.get("player_id"), "nickname": p.get("nickname")}
    missing = [want[k] for k in want if k not in found]
    if missing:
        print("[cohort] target(s) not found in window:", missing)
    return found


def fetch_match_stats(session, df, rosters, targets, pid):
    """Pull /matches/{id}/stats only for matches with >=1 target on my faction."""
    tid = {t["player_id"] for t in targets.values()}
    shared = {}
    for mid in df["match_id"]:
        r = rosters.get(mid)
        if not r or not r["my_faction"]:
            continue
        roster_ids = {p.get("player_id") for p in r["teams"].get(r["my_faction"], [])}
        if tid & roster_ids:
            shared[mid] = roster_ids
    print(f"[cohort] fetching match stats for {len(shared)} shared matches.")
    per_player = {}  # match_id -> {player_id: {adr,kd}}
    for mid in shared:
        data = api_get(session, f"/matches/{mid}/stats")
        if not data:
            continue
        for rnd in data.get("rounds", []):
            for team in rnd.get("teams", []):
                for p in team.get("players", []):
                    st = p.get("player_stats", {})
                    pp = per_player.setdefault(mid, {})
                    pp[p.get("player_id")] = {
                        "adr": num(st.get("ADR") or st.get("Average Damage per Round")),
                        "kd": num(st.get("K/D Ratio")),
                    }
    return shared, per_player


def collect_role_stats(session, df, rosters, targets, shared, pid):
    """Aggregate per-player role signals across shared games (target on my faction).

    Returns a DataFrame indexed by nickname with role metrics, plus win/loss ADR.
    """
    idname = {pid: "jsn789"}
    for t in targets.values():
        idname[t["player_id"]] = t["nickname"]
    res_by_mid = df.set_index("match_id")["result"].to_dict()

    raw = {pi: {} for pi in idname}          # cumulative sums/lists
    adr_w = {pi: [] for pi in idname}
    adr_l = {pi: [] for pi in idname}
    fields = ["ADR", "K/D Ratio", "K/R Ratio", "Headshots %", "Entry Count",
              "Entry Wins", "First Kills", "Clutch Kills", "Sniper Kills",
              "Flash Count", "Flash Successes", "Utility Damage", "Assists"]
    for pi in idname:
        raw[pi] = {f: [] for f in fields}

    for mid in shared:
        r = rosters.get(mid)
        myf = r["my_faction"]
        myfac_ids = {p.get("player_id") for p in r["teams"].get(myf, [])}
        data = api_get(session, f"/matches/{mid}/stats")
        if not data:
            continue
        res = res_by_mid.get(mid)
        for rnd in data.get("rounds", []):
            for team in rnd.get("teams", []):
                for p in team.get("players", []):
                    pi = p.get("player_id")
                    if pi not in idname or pi not in myfac_ids:
                        continue
                    st = p.get("player_stats", {})
                    for f in fields:
                        v = num(st.get(f))
                        if v == v:
                            raw[pi][f].append(v)
                    a = num(st.get("ADR"))
                    if a == a and res is not None:
                        (adr_w if res == 1 else adr_l)[pi].append(a)

    rows = []
    for pi, name in idname.items():
        adrs = raw[pi]["ADR"]
        n = len(adrs)
        if n == 0:
            continue
        entry = sum(raw[pi]["Entry Count"])
        ewin = sum(raw[pi]["Entry Wins"])
        fc = sum(raw[pi]["Flash Count"])
        fs = sum(raw[pi]["Flash Successes"])
        mean = lambda f: (sum(raw[pi][f]) / len(raw[pi][f])) if raw[pi][f] else float("nan")
        aw = (sum(adr_w[pi]) / len(adr_w[pi])) if adr_w[pi] else float("nan")
        al = (sum(adr_l[pi]) / len(adr_l[pi])) if adr_l[pi] else float("nan")
        rows.append({
            "player": name, "N": n,
            "ADR": round(mean("ADR"), 1), "KD": round(mean("K/D Ratio"), 2),
            "KR": round(mean("K/R Ratio"), 2), "HS%": round(mean("Headshots %"), 1),
            "entry_pg": round(mean("Entry Count"), 2),
            "entry_win%": round(100 * ewin / entry, 1) if entry else float("nan"),
            "first_kills_pg": round(mean("First Kills"), 2),
            "clutch_pg": round(mean("Clutch Kills"), 2),
            "sniper_pg": round(mean("Sniper Kills"), 2),
            "flash_succ%": round(100 * fs / fc, 1) if fc else float("nan"),
            "util_dmg_pg": round(mean("Utility Damage"), 1),
            "assists_pg": round(mean("Assists"), 2),
            "adr_win": round(aw, 1), "adr_loss": round(al, 1),
            "adr_swing": round(aw - al, 1) if aw == aw and al == al else float("nan"),
        })
    out = pd.DataFrame(rows).set_index("player")
    order = ["jsn789"] + [t["nickname"] for t in targets.values() if t["nickname"] in out.index]
    return out.loc[[p for p in order if p in out.index]]


def collect_player_detail(session, df, rosters, targets, shared, pid, min_n=3):
    """Per-player deep data: per-map WR/ADR and per-match series (for trend + splits).

    Returns {nickname: {"per_map": DataFrame, "series": DataFrame}} where series has
    one row per shared match the player was in: finished_dt, map, result, adr, kd, kr.
    """
    idname = {pid: "jsn789"}
    for t in targets.values():
        idname[t["player_id"]] = t["nickname"]
    meta = df.set_index("match_id")[["map", "result", "finished_dt"]].to_dict("index")

    series = {pi: [] for pi in idname}
    for mid in shared:
        r = rosters.get(mid)
        myf = r["my_faction"]
        myfac_ids = {p.get("player_id") for p in r["teams"].get(myf, [])}
        data = api_get(session, f"/matches/{mid}/stats")
        if not data or mid not in meta:
            continue
        m = meta[mid]
        for rnd in data.get("rounds", []):
            for team in rnd.get("teams", []):
                for p in team.get("players", []):
                    pi = p.get("player_id")
                    if pi not in idname or pi not in myfac_ids:
                        continue
                    st = p.get("player_stats", {})
                    series[pi].append({
                        "finished_dt": m["finished_dt"], "map": m["map"],
                        "result": m["result"], "adr": num(st.get("ADR")),
                        "kd": num(st.get("K/D Ratio")), "kr": num(st.get("K/R Ratio")),
                        "hs": num(st.get("Headshots %")),
                    })

    detail = {}
    for pi, name in idname.items():
        sdf = pd.DataFrame(series[pi])
        if sdf.empty:
            continue
        sdf = sdf.sort_values("finished_dt").reset_index(drop=True)
        pm = sdf.groupby("map").agg(
            N=("result", "size"), win_rate=("result", "mean"),
            avg_adr=("adr", "mean"), avg_kd=("kd", "mean")).reset_index()
        pm = pm[pm["N"] >= min_n].copy()
        pm["win_rate"] = (pm["win_rate"] * 100).round(1)
        pm["avg_adr"] = pm["avg_adr"].round(1)
        pm["avg_kd"] = pm["avg_kd"].round(2)
        pm = pm.sort_values("win_rate", ascending=False)
        detail[name] = {"per_map": pm, "series": sdf}
    return detail


def collect_map_team_stats(df, detail, min_n=3):
    """Per-map team summary + per-player splits.

    Returns {map: {"team": {"N","wins","wr"}, "players": DataFrame}} for maps with
    at least `min_n` team games, ordered by games played desc.
    """
    team = df.groupby("map").agg(N=("result", "size"), wins=("result", "sum"),
                                 wr=("result", "mean")).reset_index()
    team = team[team["N"] >= min_n].sort_values("N", ascending=False)

    out = {}
    for _, t in team.iterrows():
        m = t["map"]
        rows = []
        for name, d in detail.items():
            ser = d["series"]
            sub = ser[ser["map"] == m]
            if len(sub) < min_n:
                continue
            rows.append({
                "player": name, "N": len(sub),
                "win_rate": round(100 * sub["result"].mean(), 1),
                "ADR": round(sub["adr"].mean(), 1),
                "KD": round(sub["kd"].mean(), 2),
                "KR": round(sub["kr"].mean(), 2),
                "HS%": round(sub["hs"].mean(), 1),
            })
        pdf = pd.DataFrame(rows).sort_values("ADR", ascending=False) if rows else pd.DataFrame()
        out[m] = {
            "team": {"N": int(t["N"]), "wins": int(t["wins"]),
                     "wr": round(100 * t["wr"], 1)},
            "players": pdf,
        }
    return out


def collect_map_deep_stats(session, df, rosters, targets, shared, pid, min_n=3):
    """Per-map deep dive from raw match-stats: half splits, close/OT record, and
    per-player opening-duel / clutch / utility / multi-kill profiles.

    The FACEIT API exposes only match-aggregate stats (no per-round data) and does
    NOT label which half was CT vs T, so half splits are reported as
    first-half vs second-half round win rate (an honest proxy for "start strong vs
    fade"). Returns {map: {...}} for maps with >= min_n team games.
    """
    idname = {pid: "jsn789"}
    for t in targets.values():
        idname[t["player_id"]] = t["nickname"]
    meta = df.set_index("match_id")[["map", "result"]].to_dict("index")

    # team-level per-map accumulators
    tagg = {}            # map -> dict of running totals
    # player-level per-map accumulators: map -> player -> field lists/sums
    pagg = {}
    pfields = ["ADR", "Entry Count", "Entry Wins", "First Kills", "Clutch Kills",
               "Utility Damage", "K/R Ratio", "Headshots %", "Double Kills",
               "Triple Kills", "Quadro Kills", "1v1Count", "1v1Wins",
               "1v2Count", "1v2Wins"]

    for mid in shared:
        if mid not in meta:
            continue
        m = meta[mid]["map"]
        res = meta[mid]["result"]
        r = rosters.get(mid)
        myf = r["my_faction"]
        myfac_ids = {p.get("player_id") for p in r["teams"].get(myf, [])}
        data = api_get(session, f"/matches/{mid}/stats")
        if not data:
            continue
        for rnd in data.get("rounds", []):
            # locate our team's team_stats for half/OT/final
            myteam = None
            for team in rnd.get("teams", []):
                tids = {p.get("player_id") for p in team.get("players", [])}
                if tids & myfac_ids:
                    myteam = team
                    break
            t = tagg.setdefault(m, {"N": 0, "wins": 0, "fh_rounds": 0, "sh_rounds": 0,
                                    "fh_total": 0, "sh_total": 0, "close": 0, "close_w": 0,
                                    "ot": 0, "ot_w": 0})
            t["N"] += 1
            t["wins"] += 1 if res == 1 else 0
            if myteam is not None:
                ts = myteam.get("team_stats", {})
                opp = next((x for x in rnd.get("teams", []) if x is not myteam), None)
                ots = opp.get("team_stats", {}) if opp else {}
                my_fh, my_sh = num(ts.get("First Half Score")), num(ts.get("Second Half Score"))
                op_fh, op_sh = num(ots.get("First Half Score")), num(ots.get("Second Half Score"))
                if my_fh == my_fh and op_fh == op_fh:
                    t["fh_rounds"] += my_fh
                    t["fh_total"] += my_fh + op_fh
                if my_sh == my_sh and op_sh == op_sh:
                    t["sh_rounds"] += my_sh
                    t["sh_total"] += my_sh + op_sh
                my_fin, op_fin = num(ts.get("Final Score")), num(ots.get("Final Score"))
                if my_fin == my_fin and op_fin == op_fin and abs(my_fin - op_fin) <= 2:
                    t["close"] += 1
                    t["close_w"] += 1 if res == 1 else 0
                ot = num(ts.get("Overtime score"))
                if ot == ot and ot > 0:
                    t["ot"] += 1
                    t["ot_w"] += 1 if res == 1 else 0

            # player deep fields
            for team in rnd.get("teams", []):
                for p in team.get("players", []):
                    pi = p.get("player_id")
                    if pi not in idname or pi not in myfac_ids:
                        continue
                    name = idname[pi]
                    pm = pagg.setdefault(m, {})
                    acc = pm.setdefault(name, {f: [] for f in pfields})
                    acc.setdefault("_n", [0])
                    acc["_n"][0] += 1
                    st = p.get("player_stats", {})
                    for f in pfields:
                        v = num(st.get(f))
                        if v == v:
                            acc[f].append(v)

    out = {}
    for m, t in tagg.items():
        if t["N"] < min_n:
            continue
        fh_wr = round(100 * t["fh_rounds"] / t["fh_total"], 1) if t["fh_total"] else float("nan")
        sh_wr = round(100 * t["sh_rounds"] / t["sh_total"], 1) if t["sh_total"] else float("nan")
        prows = []
        for name, acc in pagg.get(m, {}).items():
            n = acc["_n"][0]
            if n < min_n:
                continue
            s = lambda f: sum(acc[f])
            mean = lambda f: (sum(acc[f]) / len(acc[f])) if acc[f] else float("nan")
            entry = s("Entry Count")
            ewin = s("Entry Wins")
            c1, c1w = s("1v1Count"), s("1v1Wins")
            c2, c2w = s("1v2Count"), s("1v2Wins")
            clutch_n = c1 + c2
            clutch_w = c1w + c2w
            multi = (s("Double Kills") + s("Triple Kills") + s("Quadro Kills"))
            prows.append({
                "player": name, "N": n,
                "entry_pg": round(entry / n, 2),
                "entry_win%": round(100 * ewin / entry, 1) if entry else float("nan"),
                "first_kills_pg": round(s("First Kills") / n, 2),
                "clutch_won_pg": round(clutch_w / n, 2),
                "clutch_win%": round(100 * clutch_w / clutch_n, 1) if clutch_n else float("nan"),
                "util_dmg_pg": round(mean("Utility Damage"), 1),
                "multikill_pg": round(multi / n, 2),
                "HS%": round(mean("Headshots %"), 1),
            })
        pdf = (pd.DataFrame(prows).sort_values("entry_pg", ascending=False)
               if prows else pd.DataFrame())
        out[m] = {
            "fh_wr": fh_wr, "sh_wr": sh_wr,
            "close": {"n": t["close"], "w": t["close_w"]},
            "ot": {"n": t["ot"], "w": t["ot_w"]},
            "players": pdf,
        }
    return out


def map_recommendations(m, team, deep, base_wr):
    """Rule-based, data-grounded recommendations for one map."""
    recs = []
    wr = team["wr"]
    fh, sh = deep["fh_wr"], deep["sh_wr"]
    # half-phase read
    if fh == fh and sh == sh:
        gap = fh - sh
        if gap >= 8:
            recs.append("Strong starts, weak closes: the team wins the first half "
                        f"({fh}% of rounds) but fades after the swap ({sh}%). Review "
                        "second-half setups/economy and protect leads.")
        elif gap <= -8:
            recs.append("Slow starts: the team is sharper in the second half "
                        f"({sh}%) than the first ({fh}%). Tighten opening-half defaults "
                        "and pistol/anti-eco execution.")
        else:
            recs.append(f"Balanced halves ({fh}% / {sh}%): no obvious side/phase "
                        "weakness — gains will come from individual duels and utility.")
    # close-game resilience
    c = deep["close"]
    if c["n"] >= 2:
        crate = round(100 * c["w"] / c["n"], 1)
        if crate < 45:
            recs.append(f"Loses the close ones: {c['w']}\u2013{c['n']-c['w']} in games "
                        f"decided by \u22642 rounds ({crate}%). Practice timeout discipline, "
                        "save rounds and forced-buy decisions.")
        elif crate >= 60:
            recs.append(f"Clutch under pressure: {c['w']}\u2013{c['n']-c['w']} in close "
                        f"games ({crate}%) — a real strength to lean on.")
    # entry / opening duels
    pdf = deep["players"]
    if not pdf.empty:
        ent = pdf[pdf["entry_pg"] >= 3].sort_values("entry_win%")
        if not ent.empty:
            worst = ent.iloc[0]
            best = ent.iloc[-1]
            if worst["entry_win%"] < 45:
                recs.append(f"Opening duels: {worst['player']} takes "
                            f"{worst['entry_pg']} entries/game on this map but wins only "
                            f"{worst['entry_win%']}% — consider handing first contact to "
                            f"{best['player']} ({best['entry_win%']}%) or trading tighter.")
        # utility
        util = pdf["util_dmg_pg"].mean()
        if m in ("de_mirage", "de_inferno", "de_nuke") and util < 80:
            recs.append(f"Under-utilitied for {_pretty_map(m)} (avg {round(util,1)} util "
                        "dmg/game): this map rewards smokes/molly setups — invest in "
                        "structured utility on executes and retakes.")
    # overall vs baseline
    if wr < base_wr - 5:
        recs.append(f"Below baseline ({wr}% vs {base_wr}%): consider lowering this map "
                    "in the veto until the issues above are addressed.")
    elif wr >= base_wr + 8:
        recs.append(f"Above baseline ({wr}% vs {base_wr}%): a comfort pick — prioritize "
                    "keeping it in the active pool.")
    return recs


IGL = "jsn789"  # in-game leader; not stat-derived, so it's called out alongside the role.


def auto_role(r, name=None):
    """Heuristic role label from role-stat row. Entry-star wins over AWP when the
    opening-duel profile is dominant (high entries + high entry win%)."""
    if r["entry_pg"] >= 5.0 and r["entry_win%"] >= 55:
        role = "Entry / star"
    elif r["sniper_pg"] >= 2.0:
        role = "AWPer"
    elif r["ADR"] >= 80:
        role = "Secondary fragger"
    elif r["util_dmg_pg"] >= 100 and r["KR"] < 0.6:
        role = "Support / utility"
    else:
        role = "Rifler (flex)"
    return f"{role} / IGL" if name == IGL else role


NARRATIVE = {"tips": {}, "stack_html": "", "stack_text": ""}


def build_narrative(role_df, df, cohort_df, map_g, base_wr):
    """Generate per-player tips and the stack summary from computed stats."""
    n, wins = len(df), int(df["result"].sum())
    wr_with = {r["teammate"]: r for _, r in cohort_df.iterrows()} if not cohort_df.empty else {}
    metrics = [("ADR", "ADR"), ("entry_pg", "entry volume"), ("util_dmg_pg", "utility damage"),
               ("KD", "K/D"), ("clutch_pg", "clutches"), ("HS%", "headshot rate")]
    best = {c: role_df[c].idxmax() for c, _ in metrics}
    worst = {c: role_df[c].idxmin() for c, _ in metrics}

    tips = {}
    for p in role_df.index:
        r = role_df.loc[p]
        s = [f"{r['ADR']} ADR, {r['entry_pg']} entries/game at "
             f"{r['entry_win%']}% win"
             + (f", {r['sniper_pg']} sniper kills/game" if r["sniper_pg"] >= 2 else "") + "."]
        if p == IGL:
            s.append("Also the in-game leader, so calling sits on top of the fragging load.")
        tops = [lab for c, lab in metrics if best[c] == p][:3]
        lows = [lab for c, lab in metrics if worst[c] == p][:2]
        if tops:
            s.append("Highest in the group for " + ", ".join(tops) + ".")
        if lows:
            s.append("Lowest in the group for " + ", ".join(lows) + ".")
        sw = r["adr_swing"]
        if sw >= 15:
            s.append(f"ADR is {sw} higher in wins than losses, so results follow their form.")
        elif sw <= 0:
            s.append(f"ADR is {'flat' if sw > -5 else 'lower'} in wins ({sw:+}), so output "
                     "doesn't track results.")
        w = wr_with.get(p)
        if w is not None and w["wr_baseline_without"] == w["wr_baseline_without"]:
            s.append(f"Team wins {w['wr_with']}% with them in the lineup ({int(w['N'])} games) "
                     f"vs {w['wr_baseline_without']}% without.")
        if r["N"] < 20:
            s.append(f"Small sample (N={int(r['N'])}), so treat this as directional.")
        tips[p] = " ".join(s)

    facts = [f"{n} games, {wins}W\u2013{n - wins}L ({base_wr}% win rate). "
             f"{IGL} is the in-game leader and played all of them."]
    freq = ", ".join(f"{p} {int(role_df.loc[p, 'N'])}" for p in
                     role_df["N"].sort_values(ascending=False).index if p != IGL)
    facts.append(f"The lineup rotated: games alongside {freq}.")
    big = map_g[map_g["N"] >= 5].sort_values("win_rate")
    if len(big) >= 2:
        lo, hi = big.index[0], big.index[-1]
        facts.append(f"Strongest map: {_pretty_map(hi)} ({big.loc[hi, 'win_rate']}% over "
                     f"{int(big.loc[hi, 'N'])}); weakest: {_pretty_map(lo)} "
                     f"({big.loc[lo, 'win_rate']}% over {int(big.loc[lo, 'N'])}).")
    early, late = df[df["match_in_session"] <= 2], df[df["match_in_session"] >= 3]
    if len(late) >= 5 and len(early) >= 5:
        e, l = round(100 * early["result"].mean(), 1), round(100 * late["result"].mean(), 1)
        if abs(e - l) >= 10:
            facts.append(f"Session length matters: {e}% in the first two matches of a session "
                         f"({len(early)} games) vs {l}% from the third on ({len(late)} games).")
    NARRATIVE["tips"] = tips
    NARRATIVE["stack_text"] = " ".join(facts)
    NARRATIVE["stack_html"] = " ".join(facts)


PLAYER_CLIPS = {
    "kpace": "https://allstar.gg/clip/6aa5d13c0ff9d4e066939f5d?ref=695ed7e0db3c83c310df630a",
    "jsn789": "https://allstar.gg/clip/6aa463c19dc43cf2c99b6e6e?ref=695ed7e0db3c83c310df630a",
    "eradicate-": "https://allstar.gg/clip/6aa5c5482124e563fc54a14a?ref=695ed7e0db3c83c310df630a",
    "Kovkomeister": "https://allstar.gg/clip/6aa7c9013896a9880f025c31?ref=695ed7e0db3c83c310df630a",
    "nuki_tbg": "https://allstar.gg/clip/6aa7cb255bc68a7a57c0acf8?ref=695ed7e0db3c83c310df630a",
}


def build_dashboard(role_df, df, base_wr):
    """Multi-panel PNG: role profiles, win/loss ADR, stack overview, tips."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    players = list(role_df.index)
    colors = plt.cm.tab10(np.linspace(0, 1, len(players)))
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f"Gaming weekend — role & impact overview  (baseline WR {base_wr}%)",
                 fontsize=16, fontweight="bold")
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.1, 1.0], hspace=0.45, wspace=0.3)

    # Panel 1: ADR + KD per player.
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(players))
    ax.bar(x, role_df["ADR"], color=colors)
    ax.set_title("Firepower: avg ADR"); ax.set_xticks(x)
    ax.set_xticklabels(players, rotation=40, ha="right")
    for i, v in enumerate(role_df["ADR"]):
        ax.text(i, v + 1, str(v), ha="center", fontsize=8)
    ax.axhline(80, ls="--", c="grey", lw=0.8)

    # Panel 2: Entry role (entries/g vs entry win%).
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(role_df["entry_pg"], role_df["entry_win%"], s=120, color=colors)
    for i, p in enumerate(players):
        ax.annotate(p, (role_df["entry_pg"].iloc[i], role_df["entry_win%"].iloc[i]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("entries / game"); ax.set_ylabel("entry win %")
    ax.set_title("Entry role"); ax.axhline(50, ls="--", c="grey", lw=0.8)

    # Panel 3: Support/AWP signature (util dmg vs sniper kills).
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(role_df["util_dmg_pg"], role_df["sniper_pg"], s=120, color=colors)
    for i, p in enumerate(players):
        ax.annotate(p, (role_df["util_dmg_pg"].iloc[i], role_df["sniper_pg"].iloc[i]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("utility dmg / game"); ax.set_ylabel("sniper kills / game")
    ax.set_title("Support (util) vs AWP")

    # Panel 4: ADR in wins vs losses (clutch factor).
    ax = fig.add_subplot(gs[1, 0])
    w = 0.38
    ax.bar(x - w / 2, role_df["adr_win"], w, label="win", color="seagreen")
    ax.bar(x + w / 2, role_df["adr_loss"], w, label="loss", color="indianred")
    ax.set_title("ADR in wins vs losses"); ax.set_xticks(x)
    ax.set_xticklabels(players, rotation=40, ha="right"); ax.legend(fontsize=8)

    # Panel 5: normalized role radar-ish bars (KR, HS%, clutch).
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(x - w / 2, role_df["KR"], w, label="K/R", color="steelblue")
    ax.bar(x + w / 2, role_df["clutch_pg"] / role_df["clutch_pg"].max(), w,
           label="clutch (norm)", color="goldenrod")
    ax.set_title("Impact: K/R & clutch"); ax.set_xticks(x)
    ax.set_xticklabels(players, rotation=40, ha="right"); ax.legend(fontsize=8)

    # Panel 6: sample size + role label table.
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    cells = [[p, int(role_df.loc[p, "N"]), auto_role(role_df.loc[p], p)] for p in players]
    tbl = ax.table(cellText=cells, colLabels=["player", "games", "role"],
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    ax.set_title("Roles & sample size", pad=20)

    # Panel 7-9: tips text spanning bottom row.
    ax = fig.add_subplot(gs[2, :]); ax.axis("off")
    ax.set_title("Per-player tips & weekend read", loc="left", fontweight="bold", pad=10)
    y = 0.92
    stack_line = NARRATIVE["stack_text"]
    ax.text(0.01, y, stack_line, fontsize=10, wrap=True, va="top",
            bbox=dict(boxstyle="round", fc="#eef", ec="grey"))
    y -= 0.20
    for p in players:
        tip = NARRATIVE["tips"].get(p, "")
        ax.text(0.01, y, f"• {p}: {tip}", fontsize=9, va="top", wrap=True)
        y -= 0.135
    fig.savefig("stack_dashboard.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("Wrote stack_dashboard.png")


HTML_CSS = """
:root { --bg:#0f1117; --card:#1a1d27; --ink:#e6e8ef; --muted:#9aa3b2; --accent:#ff7a2f;
  --good:#3fb950; --bad:#f85149; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
a { color:var(--accent); text-decoration:none; }
a:hover { text-decoration:underline; }
nav { display:flex; flex-wrap:wrap; gap:6px; align-items:center; padding:10px 28px;
  background:#13151d; border-bottom:1px solid #262a36; position:sticky; top:0; z-index:20; }
nav a { color:var(--ink); font-size:13px; padding:6px 10px; border-radius:8px;
  border:1px solid transparent; }
nav a:hover { background:#20242f; text-decoration:none; }
nav a.active { background:var(--accent); color:#10121a; font-weight:700; }
nav .grp { color:var(--muted); font-size:12px; margin:0 4px 0 10px; }
.dd { position:relative; }
.dd > .ddbtn { color:var(--ink); font-size:13px; padding:6px 10px; border-radius:8px;
  border:1px solid transparent; cursor:pointer; background:none; font:inherit;
  display:inline-flex; align-items:center; gap:5px; }
.dd > .ddbtn:hover, .dd:hover > .ddbtn { background:#20242f; }
.dd > .ddbtn.active { background:var(--accent); color:#10121a; font-weight:700; }
.dd > .ddbtn .caret { font-size:9px; opacity:.7; }
.dd .menu { position:absolute; left:0; top:100%; min-width:180px;
  background:#13151d; border:1px solid #262a36; border-radius:10px; padding:6px;
  margin-top:6px; flex-direction:column; gap:2px; z-index:30;
  box-shadow:0 12px 30px rgba(0,0,0,.45);
  opacity:0; visibility:hidden; transform:translateY(-4px);
  transition:opacity .12s ease, transform .12s ease, visibility 0s linear .25s;
  display:flex; pointer-events:none; }
/* invisible bridge so moving the cursor down to the menu doesn't drop the hover */
.dd .menu::before { content:''; position:absolute; left:0; right:0; top:-8px; height:8px; }
.dd:hover .menu, .dd:focus-within .menu {
  opacity:1; visibility:visible; transform:translateY(0);
  transition:opacity .12s ease, transform .12s ease; pointer-events:auto; }
.dd .menu a { white-space:nowrap; }
.dd .menu a.active { background:var(--accent); color:#10121a; font-weight:700; }
.stacked { display:flex; flex-direction:column; gap:16px; }
.stacked .chartbox canvas { min-height:340px; }
header { padding:24px 28px; border-bottom:1px solid #262a36; }
h1 { margin:0; font-size:22px; } h1 span { color:var(--accent); }
.sub { color:var(--muted); margin-top:6px; }
.crumb { color:var(--muted); font-size:13px; margin-bottom:4px; }
.wrap { padding:22px 28px; max-width:1180px; margin:0 auto; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:16px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
.stat { background:var(--card); border:1px solid #262a36; border-radius:12px; padding:14px; }
.stat .v { font-size:24px; font-weight:700; } .stat .k { color:var(--muted); font-size:12px; }
.card { background:var(--card); border:1px solid #262a36; border-radius:12px; padding:12px;
  display:flex; align-items:center; justify-content:center; }
.card img { width:100%; height:auto; border-radius:6px; }
.card.chartbox { display:block; padding:14px 16px; }
.chartbox canvas { width:100% !important; }
h2 { margin:30px 0 12px; font-size:17px; border-left:3px solid var(--accent); padding-left:10px; }
table { width:100%; border-collapse:collapse; background:var(--card); border-radius:12px;
  overflow:hidden; font-size:13px; }
th,td { padding:8px 10px; text-align:right; border-bottom:1px solid #262a36; }
th { background:#222633; color:var(--muted); font-weight:600; }
td.pl { text-align:left; font-weight:700; }
td.pl a { color:var(--accent); }
td.role,th:nth-child(2) { text-align:left; }
tbody tr:hover { background:#20242f; }
.stack { background:#181c28; border:1px solid #2c3142; border-left:4px solid var(--accent);
  border-radius:10px; padding:16px 18px; margin-bottom:8px; }
ul.tips { list-style:none; padding:0; }
ul.tips li { background:var(--card); border:1px solid #262a36; border-radius:10px;
  padding:12px 14px; margin-bottom:10px; }
.rl { color:var(--accent); font-weight:600; }
.cols { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.good { color:var(--good); } .bad { color:var(--bad); }
ul.sw { margin:0; padding-left:18px; } ul.sw li { margin-bottom:6px; }
footer { color:var(--muted); font-size:12px; padding:18px 28px; border-top:1px solid #262a36; }
.clipwrap { position:relative; width:100%; padding-top:56.25%; background:var(--card);
  border:1px solid #262a36; border-radius:12px; overflow:hidden; }
.clipwrap iframe { position:absolute; inset:0; width:100%; height:100%; border:0; }
@media (max-width:760px) { .cols { grid-template-columns:1fr; } }
"""


def _slug(name):
    import re
    return "player_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + ".html"


def _map_slug(mapname):
    import re
    return "map_" + re.sub(r"[^a-z0-9]+", "_", mapname.lower()).strip("_") + ".html"


def _pretty_map(mapname):
    return mapname.replace("de_", "").capitalize()


def _fmt(v):
    """Render a possibly-NaN numeric value for HTML display."""
    return "–" if v != v else v


def nav(players, maps, active=""):
    """Shared top navigation bar with Players/Maps dropdown menus.

    `active` is the current page's href; the matching link (or its parent
    dropdown trigger) is highlighted.
    """
    import html as _html

    def link(href, label):
        cls = " class='active'" if href == active else ""
        return f"<a href='{href}'{cls}>{_html.escape(label)}</a>"

    def dropdown(label, items):
        """items: list of (href, label). Trigger is active if `active` is inside."""
        hrefs = {h for h, _ in items}
        btn_cls = " active" if active in hrefs else ""
        menu = "".join(link(h, lab) for h, lab in items)
        return (f"<div class='dd'><button class='ddbtn{btn_cls}'>{label}"
                f"<span class='caret'>\u25be</span></button>"
                f"<div class='menu'>{menu}</div></div>")

    parts = [link("dashboard.html", "Overview"), link("maps.html", "Maps overview")]
    parts.append(dropdown("Players", [(_slug(p), p) for p in players]))
    parts.append(dropdown("Map pages", [(_map_slug(m), _pretty_map(m)) for m in maps]))
    return "<nav>" + "".join(parts) + "</nav>"


def _chart_b64(fig):
    """Render a matplotlib figure to a base64 PNG data URI (used for PNG exports)."""
    import base64
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# Chart.js (CDN) + a shared theme so every chart blends with the dark palette and
# shows hover tooltips. Palette mirrors the CSS :root variables.
CHARTJS_HEAD = (
    "<script src='https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js'>"
    "</script>"
    "<script>"
    "window.PAL=['#ff7a2f','#3fb950','#58a6ff','#d2a8ff','#f0b429','#ff6b81',"
    "'#56d4dd','#a5d6a7'];"
    "window.chartReady=function(){"
    "if(!window.Chart)return;"
    "const C=Chart.defaults;"
    "C.color='#9aa3b2';C.borderColor='#262a36';"
    "C.font.family=\"-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif\";"
    "C.plugins.legend.labels.color='#e6e8ef';"
    "C.plugins.tooltip.backgroundColor='#0f1117';"
    "C.plugins.tooltip.titleColor='#ff7a2f';"
    "C.plugins.tooltip.bodyColor='#e6e8ef';"
    "C.plugins.tooltip.borderColor='#2c3142';"
    "C.plugins.tooltip.borderWidth=1;"
    "C.plugins.tooltip.padding=10;"
    "C.scales={category:{grid:{color:'#222633'}},linear:{grid:{color:'#222633'}}};"
    "};"
    "</script>"
)


def _canvas(cid, config, height=300):
    """Return (html, script) for a themed Chart.js canvas.

    `config` is a Python dict serialized to JSON for the Chart constructor.
    The init runs after `chartReady()` applies the shared dark theme.
    """
    import json
    html = (f"<div class='card chartbox'>"
            f"<canvas id='{cid}' height='{height}'></canvas></div>")
    script = (f"chartReady();new Chart(document.getElementById('{cid}'),"
              f"{json.dumps(config)});")
    return html, script


def build_html_dashboard(role_df, df, base_wr, map_g, mis_g, maps=None):
    """Self-contained HTML dashboard with interactive Chart.js charts."""
    import html as _html

    players = list(role_df.index)
    maps = maps or []
    chart_html, chart_js = {}, []

    def bar(cid, title, labels, values, ylabel, ref=None, color="#ff7a2f"):
        ds = [{"label": ylabel, "data": values, "backgroundColor": color,
               "borderRadius": 4}]
        ann = {}
        cfg = {"type": "bar",
               "data": {"labels": labels, "datasets": ds},
               "options": {"responsive": True, "maintainAspectRatio": False,
                           "plugins": {"legend": {"display": False},
                                       "title": {"display": True, "text": title,
                                                 "color": "#e6e8ef"}},
                           "scales": {"y": {"title": {"display": True, "text": ylabel}}}}}
        if ref is not None:
            # reference line drawn as a flat dataset overlay
            cfg["data"]["datasets"].append({
                "type": "line", "label": f"baseline {ref}",
                "data": [ref] * len(labels), "borderColor": "#f85149",
                "borderDash": [6, 4], "pointRadius": 0, "borderWidth": 1.5})
            cfg["options"]["plugins"]["legend"] = {"display": True}
        h, s = _canvas(cid, cfg)
        chart_html[cid] = h
        chart_js.append(s)

    pal = ["#ff7a2f", "#3fb950", "#58a6ff", "#d2a8ff", "#f0b429", "#ff6b81",
           "#56d4dd", "#a5d6a7"]

    def scatter(cid, title, xlabel, ylabel, pts):
        # one dataset per player so the legend/tooltip name the point; an inline
        # plugin also draws each player's name beside its marker on the chart.
        import json as _json
        datasets = [{"label": p, "data": [{"x": xv, "y": yv}],
                     "backgroundColor": pal[i % len(pal)], "pointRadius": 7,
                     "pointHoverRadius": 9}
                    for i, (p, xv, yv) in enumerate(pts)]
        cfg = {"type": "scatter",
               "data": {"datasets": datasets},
               "options": {"responsive": True, "maintainAspectRatio": False,
                           "plugins": {"title": {"display": True, "text": title,
                                                 "color": "#e6e8ef"}},
                           "scales": {"x": {"title": {"display": True, "text": xlabel}},
                                      "y": {"title": {"display": True, "text": ylabel}}}}}
        # build init manually so we can attach a point-label plugin
        labels_plugin = (
            "{id:'ptlabels',afterDatasetsDraw:function(c){"
            "var x=c.ctx;x.save();x.font='600 11px -apple-system,Segoe UI,Roboto,sans-serif';"
            "x.fillStyle='#e6e8ef';"
            "c.data.datasets.forEach(function(ds,di){"
            "var m=c.getDatasetMeta(di);if(!m.data[0])return;"
            "var pt=m.data[0];x.fillText(ds.label,pt.x+9,pt.y+4);});"
            "x.restore();}}")
        script = (f"chartReady();new Chart(document.getElementById('{cid}'),"
                  f"Object.assign({_json.dumps(cfg)},{{plugins:[{labels_plugin}]}}));")
        h = (f"<div class='card chartbox'>"
             f"<canvas id='{cid}' height='300'></canvas></div>")
        chart_html[cid] = h
        chart_js.append(script)

    pl = lambda c: [round(float(v), 2) for v in role_df[c]]
    bar("c_adr", "Firepower: avg ADR", players, pl("ADR"), "ADR", ref=80)
    scatter("c_entry", "Entry role", "entries / game", "entry win %",
            list(zip(players, pl("entry_pg"), pl("entry_win%"))))
    scatter("c_support", "Support (util) vs AWP", "utility dmg / game",
            "sniper kills / game",
            list(zip(players, pl("util_dmg_pg"), pl("sniper_pg"))))

    # win vs loss grouped bars
    cfg = {"type": "bar",
           "data": {"labels": players,
                    "datasets": [
                        {"label": "ADR in wins", "data": pl("adr_win"),
                         "backgroundColor": "#3fb950", "borderRadius": 4},
                        {"label": "ADR in losses", "data": pl("adr_loss"),
                         "backgroundColor": "#f85149", "borderRadius": 4}]},
           "options": {"responsive": True, "maintainAspectRatio": False,
                       "plugins": {"title": {"display": True,
                                             "text": "ADR in wins vs losses",
                                             "color": "#e6e8ef"}}}}
    h, s = _canvas("c_winloss", cfg)
    chart_html["c_winloss"] = h
    chart_js.append(s)

    if not map_g.empty:
        bar("c_map", "Win rate by map",
            [_pretty_map(str(i)) for i in map_g.index],
            [round(float(v), 1) for v in map_g["win_rate"]], "win rate %",
            ref=base_wr, color="#58a6ff")
    if not mis_g.empty:
        bar("c_session", "Win rate by match-in-session",
            [str(i) for i in mis_g.index],
            [round(float(v), 1) for v in mis_g["win_rate"]], "win rate %",
            ref=base_wr, color="#f0b429")

    # Role table rows.
    role_cols = ["N", "ADR", "KD", "KR", "HS%", "entry_pg", "entry_win%",
                 "sniper_pg", "util_dmg_pg", "clutch_pg", "adr_swing"]
    head = "".join(f"<th>{c}</th>" for c in ["player", "role"] + role_cols)
    trs = []
    for p in players:
        r = role_df.loc[p]
        cells = "".join(f"<td>{int(r[c]) if c == 'N' else r[c]}</td>" for c in role_cols)
        link = f"<a href='{_slug(p)}'>{_html.escape(p)}</a>"
        trs.append(f"<tr><td class='pl'>{link}</td>"
                   f"<td class='role'>{auto_role(r, p)}</td>{cells}</tr>")
    table_html = f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(trs)}</tbody></table>"

    tips_html = "".join(
        f"<li><b><a href='{_slug(p)}'>{_html.escape(p)}</a></b> — "
        f"<span class='rl'>{auto_role(role_df.loc[p], p)}</span>: "
        f"{_html.escape(NARRATIVE['tips'].get(p, ''))}</li>" for p in players)

    stack_line = _html.escape(NARRATIVE["stack_html"])

    def img(key, title):
        return chart_html.get(key, "")

    d0, d1 = df["finished_dt"].min(), df["finished_dt"].max()
    span = f"{d0:%d %b} \u2013 {d1:%d %b %Y}"

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gaming Weekend Analysis</title>
{CHARTJS_HEAD}
<style>{HTML_CSS}</style></head><body>
{nav(players, maps, active="dashboard.html")}
<header>
  <h1><span>Gaming Weekend</span> — role &amp; impact overview</h1>
  <div class="sub">{len(df)} matches ({span}) · baseline win rate {base_wr}% · cohort: {', '.join(players[1:])}</div>
</header>
<div class="wrap">
  <h2>How this weekend played out</h2>
  <div class="stack">{stack_line}</div>

  <h2>Role &amp; performance table (shared games)</h2>
  <div class="sub" style="margin-bottom:10px">Click a player name for their detailed page.</div>
  {table_html}

  <h2>Charts</h2>
  <div class="stacked">
    {img('c_adr','ADR')}{img('c_entry','Entry role')}{img('c_support','Support vs AWP')}
    {img('c_winloss','ADR win vs loss')}{img('c_map','Win rate by map')}{img('c_session','Win rate by session')}
  </div>

  <h2>Per-player tips</h2>
  <ul class="tips">{tips_html}</ul>

  <h2>Map analysis</h2>
  <div class="stack">See the <a href="maps.html">Maps overview</a> for team win rate by map
    and per-player breakdowns on each map.</div>
</div>
<footer>Generated from cached FACEIT Data API responses · win-rate deltas in a tight premade are noisy — pooled correlation is the reliable read.</footer>
<script>{''.join(chart_js)}</script>
</body></html>"""
    with open("dashboard.html", "w") as fh:
        fh.write(doc)
    print("Wrote dashboard.html")


def _strengths_weaknesses(name, role_df):
    """Derive strengths/weaknesses by comparing a player to cohort medians."""
    r = role_df.loc[name]
    med = role_df.median(numeric_only=True)
    rules = [
        ("ADR", "raw damage", True), ("KD", "K/D", True), ("KR", "K/R impact", True),
        ("HS%", "aim (HS%)", True), ("entry_win%", "opening duels", True),
        ("util_dmg_pg", "utility damage", True), ("clutch_pg", "clutching", True),
        ("sniper_pg", "AWP presence", True), ("adr_swing", "shows up in wins", True),
    ]
    strengths, weaknesses = [], []
    for col, label, higher_better in rules:
        if col not in role_df.columns or pd.isna(r[col]) or pd.isna(med[col]):
            continue
        ref = med[col]
        if ref == 0:
            continue
        diff = (r[col] - ref) / abs(ref)
        if diff >= 0.15:
            strengths.append(f"{label} ({r[col]} vs cohort median {round(ref,1)})")
        elif diff <= -0.15:
            weaknesses.append(f"{label} ({r[col]} vs cohort median {round(ref,1)})")
    return strengths, weaknesses


def build_player_pages(role_df, detail, base_wr, maps=None):
    """One detailed HTML page per player, linked from the dashboard."""
    import html as _html

    players = list(role_df.index)
    maps = maps or []
    for name in players:
        if name not in detail:
            continue
        r = role_df.loc[name]
        sdf = detail[name]["series"]
        pm = detail[name]["per_map"]
        role = auto_role(r, name)

        # Stat cards.
        cards = [("Games", int(r["N"])), ("ADR", r["ADR"]), ("K/D", r["KD"]),
                 ("K/R", r["KR"]), ("HS%", f"{r['HS%']}%"),
                 ("Entry win%", f"{r['entry_win%']}%"), ("Entries/g", r["entry_pg"]),
                 ("Clutch/g", r["clutch_pg"]), ("Util dmg/g", r["util_dmg_pg"]),
                 ("Sniper/g", r["sniper_pg"]), ("ADR win", r["adr_win"]),
                 ("ADR loss", r["adr_loss"])]
        cards_html = "".join(
            f"<div class='stat'><div class='v'>{v}</div><div class='k'>{k}</div></div>"
            for k, v in cards)

        # Form trend chart (Chart.js): ADR per shared match colored by win/loss,
        # with a 5-game rolling-average line overlaid.
        adrs = [round(float(v), 1) if v == v else None for v in sdf["adr"]]
        results = list(sdf["result"])
        maps_seq = list(sdf["map"])
        labels = [str(i + 1) for i in range(len(adrs))]
        bar_colors = ["#3fb950" if x == 1 else "#f85149" for x in results]
        roll = [round(float(v), 1) for v in sdf["adr"].rolling(5, min_periods=1).mean()]
        point_labels = [f"#{i+1} · {_pretty_map(maps_seq[i])} · "
                        f"{'W' if results[i] == 1 else 'L'}"
                        for i in range(len(adrs))]
        cid = "c_form"
        cfg = {"type": "bar",
               "data": {"labels": labels,
                        "datasets": [
                            {"label": "ADR (green=win, red=loss)", "data": adrs,
                             "backgroundColor": bar_colors, "borderRadius": 3,
                             "order": 2},
                            {"type": "line", "label": "5-game avg ADR", "data": roll,
                             "borderColor": "#ff7a2f", "borderWidth": 2,
                             "pointRadius": 0, "tension": 0.3, "order": 1}]},
               "options": {"responsive": True, "maintainAspectRatio": False,
                           "plugins": {"title": {"display": True,
                                                 "text": f"{name}: ADR over shared games",
                                                 "color": "#e6e8ef"},
                                       "tooltip": {"callbacks": {}}},
                           "scales": {"x": {"title": {"display": True,
                                                      "text": "shared match (oldest \u2192 newest)"}},
                                      "y": {"title": {"display": True, "text": "ADR"}}}}}
        trend_html, trend_js = _canvas(cid, cfg, height=260)
        # attach descriptive labels for the tooltip title via a side array
        import json as _json
        trend_js = (f"window._pl_{cid}={_json.dumps(point_labels)};" + trend_js)
        trend_js = trend_js.replace(
            '"callbacks": {}',
            f'"callbacks": {{"title": function(it){{return window._pl_{cid}[it[0].dataIndex];}}}}')
        trend_js = trend_js.replace(
            '"callbacks":{}',
            f'"callbacks":{{"title":function(it){{return window._pl_{cid}[it[0].dataIndex];}}}}')

        # Per-map table.
        if not pm.empty:
            rows = "".join(
                f"<tr><td class='pl'>{_html.escape(row['map'])}</td><td>{row['N']}</td>"
                f"<td>{row['win_rate']}</td><td>{row['avg_adr']}</td><td>{row['avg_kd']}</td></tr>"
                for _, row in pm.iterrows())
            map_table = (f"<table><thead><tr><th>map</th><th>N</th><th>win%</th>"
                         f"<th>avg ADR</th><th>avg KD</th></tr></thead><tbody>{rows}</tbody></table>")
        else:
            map_table = "<p class='sub'>No shared games recorded on any map.</p>"

        strengths, weaknesses = _strengths_weaknesses(name, role_df)
        sw = (f"<div class='cols'><div><h2>Strengths</h2><ul class='sw'>"
              + ("".join(f"<li class='good'>{_html.escape(s)}</li>" for s in strengths)
                 or "<li class='sub'>No standout strengths vs cohort.</li>")
              + "</ul></div><div><h2>Weaknesses</h2><ul class='sw'>"
              + ("".join(f"<li class='bad'>{_html.escape(w)}</li>" for w in weaknesses)
                 or "<li class='sub'>No standout weaknesses vs cohort.</li>")
              + "</ul></div></div>")

        wins = int(sdf["result"].sum())
        n = len(sdf)
        wr = round(100 * wins / n, 1) if n else 0
        tip = _html.escape(NARRATIVE["tips"].get(name, ""))

        clip_url = PLAYER_CLIPS.get(name)
        clip_section = ""
        if clip_url:
            clip_section = (
                "<h2>Highlight clip</h2>"
                "<div class='clipwrap'><iframe src='" + _html.escape(clip_url) + "' "
                "loading='lazy' allow='fullscreen' allowfullscreen "
                "referrerpolicy='strict-origin-when-cross-origin'></iframe></div>")

        doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(name)} — detail</title>
{CHARTJS_HEAD}
<style>{HTML_CSS}</style></head><body>
{nav(players, maps, active=_slug(name))}
<header>
  <div class="crumb"><a href="dashboard.html">&larr; back to dashboard</a></div>
  <h1><span>{_html.escape(name)}</span> &middot; {role}</h1>
  <div class="sub">{n} shared games · {wins}W&ndash;{n-wins}L · {wr}% win rate
    (stack baseline {base_wr}%)</div>
</header>
<div class="wrap">
  <h2>Summary</h2>
  <div class="stack">{tip}</div>

  <h2>Key stats (shared games)</h2>
  <div class="cards">{cards_html}</div>

  {sw}

  <h2>Form &amp; consistency</h2>
  {trend_html}

  <h2>By map</h2>
  {map_table}

  {clip_section}
</div>
<footer>Stats from cached FACEIT match data, scoped to games played on jsn789's team.
  Small per-map samples are directional. <a href="dashboard.html">Back to dashboard</a></footer>
<script>{trend_js}</script>
</body></html>"""
        with open(_slug(name), "w") as fh:
            fh.write(doc)
    print(f"Wrote {len([p for p in players if p in detail])} player pages.")


def build_map_pages(map_stats, players, base_wr, deep_stats=None):
    """Maps overview (maps.html) + one detail page per map, with navigation."""
    deep_stats = deep_stats or {}
    import html as _html

    maps = list(map_stats.keys())

    # ---- Overview page: team win rate per map (Chart.js) + summary table. ----
    wrs = [round(float(map_stats[m]["team"]["wr"]), 1) for m in maps]
    ov_cfg = {"type": "bar",
              "data": {"labels": [_pretty_map(m) for m in maps],
                       "datasets": [
                           {"label": "win rate %", "data": wrs,
                            "backgroundColor": ["#3fb950" if w >= base_wr else "#f85149"
                                                for w in wrs],
                            "borderRadius": 4, "order": 2},
                           {"type": "line", "label": f"team baseline {base_wr}%",
                            "data": [base_wr] * len(maps), "borderColor": "#ff7a2f",
                            "borderDash": [6, 4], "pointRadius": 0, "borderWidth": 1.5,
                            "order": 1}]},
              "options": {"responsive": True, "maintainAspectRatio": False,
                          "plugins": {"title": {"display": True,
                                                "text": "Team win rate by map",
                                                "color": "#e6e8ef"}},
                          "scales": {"y": {"title": {"display": True,
                                                     "text": "win rate %"}}}}}
    ov_html, ov_js = _canvas("c_ovmap", ov_cfg, height=320)

    rows = ""
    for m in maps:
        t = map_stats[m]["team"]
        cls = "good" if t["wr"] >= base_wr else "bad"
        rows += (f"<tr><td class='pl'><a href='{_map_slug(m)}'>{_pretty_map(m)}</a></td>"
                 f"<td>{t['N']}</td><td>{t['wins']}\u2013{t['N']-t['wins']}</td>"
                 f"<td class='{cls}'>{t['wr']}</td></tr>")
    ov_table = (f"<table><thead><tr><th>map</th><th>games</th><th>W\u2013L</th>"
                f"<th>win%</th></tr></thead><tbody>{rows}</tbody></table>")

    nav_html = nav(players, maps, active="maps.html")
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Maps overview</title>
{CHARTJS_HEAD}
<style>{HTML_CSS}</style></head><body>
{nav_html}
<header>
  <h1><span>Maps</span> — team overview</h1>
  <div class="sub">Team win rate by map across the analyzed window · baseline {base_wr}%.
    Click a map for the per-player breakdown.</div>
</header>
<div class="wrap">
  {ov_html}
  <h2>Map summary</h2>
  {ov_table}
</div>
<footer>Generated from cached FACEIT match data. Every map played this window is included; small samples are directional.</footer>
<script>{ov_js}</script>
</body></html>"""
    with open("maps.html", "w") as fh:
        fh.write(doc)

    # ---- Per-map detail pages. ----
    for m in maps:
        t = map_stats[m]["team"]
        pdf = map_stats[m]["players"]
        delta = round(t["wr"] - base_wr, 1)
        dcls = "good" if delta >= 0 else "bad"
        page_js = []

        if not pdf.empty:
            prows = ""
            for _, row in pdf.iterrows():
                wcls = "good" if row["win_rate"] >= base_wr else "bad"
                prows += (f"<tr><td class='pl'><a href='{_slug(row['player'])}'>"
                          f"{_html.escape(row['player'])}</a></td><td>{row['N']}</td>"
                          f"<td class='{wcls}'>{row['win_rate']}</td><td>{row['ADR']}</td>"
                          f"<td>{row['KD']}</td><td>{row['KR']}</td><td>{row['HS%']}</td></tr>")
            ptable = (f"<table><thead><tr><th>player</th><th>N</th><th>win%</th>"
                      f"<th>ADR</th><th>KD</th><th>KR</th><th>HS%</th></tr></thead>"
                      f"<tbody>{prows}</tbody></table>")
            # Per-player ADR bar chart on this map (Chart.js).
            adr_cfg = {"type": "bar",
                       "data": {"labels": list(pdf["player"]),
                                "datasets": [{"label": "avg ADR",
                                              "data": [round(float(v), 1) for v in pdf["ADR"]],
                                              "backgroundColor": "#58a6ff",
                                              "borderRadius": 4}]},
                       "options": {"responsive": True, "maintainAspectRatio": False,
                                   "plugins": {"legend": {"display": False},
                                               "title": {"display": True,
                                                         "text": f"{_pretty_map(m)}: avg ADR per player",
                                                         "color": "#e6e8ef"}},
                                   "scales": {"y": {"title": {"display": True,
                                                              "text": "ADR"}}}}}
            mchart, mjs = _canvas(f"c_adr_{_map_slug(m)[4:-5]}", adr_cfg, height=280)
            page_js.append(mjs)
            top = pdf.iloc[0]["player"]
            note = (f"Most-played: <b>{t['N']} games</b>. Team win rate "
                    f"<span class='{dcls}'>{t['wr']}%</span> ({delta:+} vs baseline). "
                    f"Top ADR on this map: <b>{_html.escape(top)}</b>.")
        else:
            ptable = "<p class='sub'>No player has 3+ games on this map.</p>"
            mchart = ""
            note = (f"<b>{t['N']} games</b>, team win rate "
                    f"<span class='{dcls}'>{t['wr']}%</span> ({delta:+} vs baseline).")

        # ---- Deep sections (half splits, close/OT, opening duels, recs). ----
        d = deep_stats.get(m)
        deep_html = ""
        if d:
            fh, sh = d["fh_wr"], d["sh_wr"]
            def _phcls(v):
                return "good" if v == v and v >= 50 else "bad"
            half_cards = (
                f"<div class='stat'><div class='v {_phcls(fh)}'>{fh}%</div>"
                f"<div class='k'>1st-half round win%</div></div>"
                f"<div class='stat'><div class='v {_phcls(sh)}'>{sh}%</div>"
                f"<div class='k'>2nd-half round win%</div></div>")
            c, ot = d["close"], d["ot"]
            crate = round(100 * c["w"] / c["n"], 1) if c["n"] else None
            otrate = round(100 * ot["w"] / ot["n"], 1) if ot["n"] else None
            close_cards = (
                f"<div class='stat'><div class='v'>{c['w']}\u2013{c['n']-c['w']}"
                f"</div><div class='k'>close games (\u22642 rds)"
                + (f" · {crate}%" if crate is not None else "") + "</div></div>"
                f"<div class='stat'><div class='v'>{ot['w']}\u2013{ot['n']-ot['w']}"
                f"</div><div class='k'>overtime games"
                + (f" · {otrate}%" if otrate is not None else "") + "</div></div>")
            deep_html += ("<h2>Phase &amp; resilience</h2>"
                          "<div class='sub' style='margin-bottom:10px'>Round win% by half "
                          "(API doesn't label CT/T, so this is start-strong vs finish-strong), "
                          "plus record in close and overtime games.</div>"
                          f"<div class='cards'>{half_cards}{close_cards}</div>")

            dpdf = d["players"]
            if not dpdf.empty:
                drows = ""
                for _, row in dpdf.iterrows():
                    ecls = ("good" if row["entry_win%"] == row["entry_win%"]
                            and row["entry_win%"] >= 50 else "bad")
                    drows += (f"<tr><td class='pl'><a href='{_slug(row['player'])}'>"
                              f"{_html.escape(row['player'])}</a></td>"
                              f"<td>{row['N']}</td>"
                              f"<td>{row['entry_pg']}</td>"
                              f"<td class='{ecls}'>{_fmt(row['entry_win%'])}</td>"
                              f"<td>{row['first_kills_pg']}</td>"
                              f"<td>{row['clutch_won_pg']}</td>"
                              f"<td>{_fmt(row['clutch_win%'])}</td>"
                              f"<td>{row['util_dmg_pg']}</td>"
                              f"<td>{row['multikill_pg']}</td>"
                              f"<td>{row['HS%']}</td></tr>")
                deep_html += (
                    "<h2>Opening duels, clutches &amp; utility (per game on this map)</h2>"
                    "<table><thead><tr><th>player</th><th>N</th><th>entries/g</th>"
                    "<th>entry win%</th><th>1st kills/g</th><th>clutches won/g</th>"
                    "<th>clutch win%</th><th>util dmg/g</th><th>multi-K/g</th>"
                    f"<th>HS%</th></tr></thead><tbody>{drows}</tbody></table>")

            recs = map_recommendations(m, t, d, base_wr)
            if recs:
                deep_html += ("<h2>Recommendations</h2><ul class='tips'>"
                              + "".join(f"<li>{_html.escape(rc)}</li>" for rc in recs)
                              + "</ul>")

        nav_html = nav(players, maps, active=_map_slug(m))
        doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_pretty_map(m)} — team analysis</title>
{CHARTJS_HEAD}
<style>{HTML_CSS}</style></head><body>
{nav_html}
<header>
  <div class="crumb"><a href="maps.html">&larr; all maps</a></div>
  <h1><span>{_pretty_map(m)}</span> — team analysis</h1>
  <div class="sub">{t['wins']}W&ndash;{t['N']-t['wins']}L · {t['wr']}% win rate
    (baseline {base_wr}%)</div>
</header>
<div class="wrap">
  <h2>Summary</h2>
  <div class="stack">{note}</div>
  {deep_html}
  <h2>Per-player on {_pretty_map(m)}</h2>
  {ptable}
  {('<h2>Firepower</h2>' + mchart) if mchart else ''}
</div>
<footer>Scoped to jsn789's team games on this map. Stats are match aggregates (no
  per-round data available from the API); small samples are directional.
  <a href="maps.html">Back to maps</a></footer>
<script>{''.join(page_js)}</script>
</body></html>"""
        with open(_map_slug(m), "w") as fh:
            fh.write(doc)
    print(f"Wrote maps.html + {len(maps)} map pages.")


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #
def table_md(g):
    return g.to_markdown() if not g.empty else "_(no rows meet min N)_"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nickname", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--png", action="store_true", help="write optional PNG charts")
    ap.add_argument("--dashboard", action="store_true",
                    help="write stack_dashboard.png (role overview + tips)")
    ap.add_argument("--html", action="store_true",
                    help="write dashboard.html (self-contained, viewable in browser)")
    args = ap.parse_args()

    session, pid, player, stats_items, history = pull(args.nickname, args.limit)
    df, rosters, km = build_frames(pid, stats_items, history)
    my_level = num((player.get("games") or {}).get("cs2", {}).get("skill_level"))

    df, mate_g = teammate_quality_bucket(df, my_level)
    df["hour_bucket"] = df["hour_local"].apply(hour_bucket)

    base_wr = round(df["result"].mean() * 100, 1)
    map_g = grouped_winrate(df, "map", min_n=1)
    hour_g = grouped_winrate(df, "hour_bucket")
    mis_g = grouped_winrate(df, "mis_bucket")
    base_g = grouped_winrate(df.assign(zone=df["below_baseline"].map(
        {True: "below baseline", False: "above baseline"})), "zone")

    deltas = {"map": delta(map_g), "teammate quality": delta(mate_g),
              "hour": delta(hour_g), "match_in_session": delta(mis_g)}
    deltas = {k: v for k, v in deltas.items() if v is not None}
    headline = max(deltas, key=deltas.get) if deltas else None

    lines = [f"# FACEIT loss-correlation report — {args.nickname}", "",
             f"Matches analyzed: **{len(df)}** | baseline win rate: **{base_wr}%** "
             f"| skill_level: **{int(my_level) if my_level==my_level else '?'}**", ""]
    print("\n" + "=" * 60 + "\nSOLO BREAKDOWNS\n" + "=" * 60)
    for title, g in [("1. By map", map_g), ("2. By teammate quality", mate_g),
                     ("3a. By hour bucket", hour_g), ("3b. By match_in_session", mis_g),
                     ("4. Above vs below my baseline (perf_z)", base_g)]:
        print(f"\n{title}\n{g}")
        lines += [f"## {title}", "", table_md(g), ""]
    if headline:
        msg = (f"5. Headline: **{headline}** shows the largest win-rate swing "
               f"(delta {deltas[headline]} pts between best and worst bucket).")
        print("\n" + msg)
        lines += ["## 5. Headline", "", msg, ""]

    # ---- Cohort ----
    print("\n" + "=" * 60 + "\nCOHORT\n" + "=" * 60)
    lines += ["# Cohort — frequent teammates", ""]
    targets = resolve_targets(rosters, pid)
    shared, per_player = fetch_match_stats(session, df, rosters, targets, pid)
    lines += [f"> Note: this is a tight premade — targets appear in {len(shared)}/{len(df)} "
              "matches. The \"without them\" baseline is computed from few, mutually "
              "overlapping games, so per-teammate win-rate deltas are noisy. Treat the "
              "**pooled correlation** below as the reliable read.", ""]

    cohort_rows, corr_rows = [], []
    for nl, t in targets.items():
        tpid = t["player_id"]
        mids = [m for m, ids in shared.items() if tpid in ids]
        with_df = df[df["match_id"].isin(mids)]
        without_df = df[~df["match_id"].isin(mids)]
        n = len(with_df)
        if n == 0:
            continue
        wr_with = round(with_df["result"].mean() * 100, 1)
        wr_without = round(without_df["result"].mean() * 100, 1) if len(without_df) else float("nan")
        # teammate's own perf_z across shared games (vs their own mean).
        t_adr = pd.Series({m: per_player.get(m, {}).get(tpid, {}).get("adr") for m in mids}, dtype="float64")
        t_z = (t_adr - t_adr.mean()) / (t_adr.std() or 1.0)
        my_z = with_df.set_index("match_id")["perf_z"].reindex(mids)
        my_res = with_df.set_index("match_id")["result"].reindex(mids)
        cohort_rows.append({
            "teammate": t["nickname"], "N": n, "low_conf": n < 4,
            "W": int(with_df["result"].sum()), "L": int(n - with_df["result"].sum()),
            "wr_with": wr_with, "wr_baseline_without": wr_without,
            "wr_delta": round(wr_with - wr_without, 1) if wr_without == wr_without else float("nan"),
            "my_adr_with": round(with_df["adr"].mean(), 1),
            "my_adr_without": round(without_df["adr"].mean(), 1) if len(without_df) else float("nan"),
            "my_perfz_with": round(with_df["perf_z"].mean(), 2),
            "their_avg_adr": round(t_adr.mean(), 1) if len(t_adr.dropna()) else float("nan"),
        })
        corr_rows.append({
            "teammate": t["nickname"], "N": n,
            "r_myz_vs_theirz": pearson(my_z.values, t_z.values),
            "r_theirz_vs_result": pearson(t_z.values, my_res.values),
        })

    cohort_df = pd.DataFrame(cohort_rows)
    if not cohort_df.empty:
        cohort_df = cohort_df.sort_values("wr_delta", ascending=False)
        print("\nPer-teammate (sorted by win-rate delta):\n", cohort_df.to_string(index=False))
        lines += ["## Per-teammate breakdown", "",
                  "Win rate **with** vs my baseline **without**; N<4 flagged low-confidence.",
                  "", cohort_df.set_index("teammate").to_markdown(), ""]
        corr_df = pd.DataFrame(corr_rows)
        print("\nImpact correlation (per-teammate, directional at small N):\n",
              corr_df.to_string(index=False))
        # Pooled correlation across all shared matches.
        pool_my, pool_their, pool_res = [], [], []
        for nl, t in targets.items():
            tpid = t["player_id"]
            for m in [mm for mm, ids in shared.items() if tpid in ids]:
                z = per_player.get(m, {}).get(tpid, {}).get("adr")
                row = df[df["match_id"] == m]
                if z is not None and z == z and not row.empty:
                    pool_my.append(row["perf_z"].iloc[0])
                    pool_their.append(z)
                    pool_res.append(row["result"].iloc[0])
        pooled_their_z = pd.Series(pool_their)
        pooled_their_z = (pooled_their_z - pooled_their_z.mean()) / (pooled_their_z.std() or 1.0)
        r_pool_myz = pearson(pool_my, pooled_their_z.values)
        r_pool_res = pearson(pooled_their_z.values, pool_res)
        pooled_line = (f"Pooled across {len(pool_my)} shared matches: "
                       f"r(my perf_z, teammate perf_z) = {r_pool_myz}; "
                       f"r(teammate perf_z, result) = {r_pool_res}.")
        print("\n" + pooled_line)
        lines += ["## Impact correlation", "",
                  "Per-teammate r is directional given small samples; pooled is the reliable read.",
                  "", corr_df.set_index("teammate").to_markdown(), "", pooled_line, ""]

        # Synthesis: carry vs prop-up.
        synth = []
        for _, r in cohort_df.iterrows():
            if r["my_perfz_with"] < 0 and r["wr_delta"] == r["wr_delta"] and r["wr_delta"] > 0:
                synth.append(f"- **{r['teammate']}**: win rate up (+{r['wr_delta']}) while my perf_z drops "
                             f"({r['my_perfz_with']}) — they likely carry.")
            elif r["my_perfz_with"] > 0 and r["wr_delta"] == r["wr_delta"] and r["wr_delta"] < 0:
                synth.append(f"- **{r['teammate']}**: my perf_z up ({r['my_perfz_with']}) but win rate down "
                             f"({r['wr_delta']}) — I likely prop them up.")
        rank = " > ".join(cohort_df["teammate"].tolist())
        lines += ["## Synthesis", "", f"Win-rate-delta rank: {rank}", ""] + (synth or ["- No clear carry/prop-up pattern."]) + [""]
        print("\nSynthesis rank:", rank)
        for sline in synth:
            print(sline)
    else:
        lines += ["_No target teammates found in this window._", ""]

    with open("report.md", "w") as f:
        f.write("\n".join(lines))
    print("\nWrote report.md")

    if (args.dashboard or args.html) and targets:
        role_df = collect_role_stats(session, df, rosters, targets, shared, pid)
        if not role_df.empty:
            build_narrative(role_df, df, cohort_df, map_g, base_wr)
            if args.dashboard:
                build_dashboard(role_df, df, base_wr)
            if args.html:
                detail = collect_player_detail(session, df, rosters, targets, shared, pid, min_n=1)
                map_stats = collect_map_team_stats(df, detail, min_n=1)
                deep_stats = collect_map_deep_stats(session, df, rosters, targets, shared, pid, min_n=1)
                maps = list(map_stats.keys())
                build_html_dashboard(role_df, df, base_wr, map_g, mis_g, maps)
                build_player_pages(role_df, detail, base_wr, maps)
                build_map_pages(map_stats, list(role_df.index), base_wr, deep_stats)

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not map_g.empty:
            ax = map_g["win_rate"].plot(kind="bar", title="Win rate by map")
            ax.set_ylabel("win rate %"); plt.tight_layout(); plt.savefig("winrate_by_map.png"); plt.clf()
        if not mis_g.empty:
            ax = mis_g["win_rate"].plot(kind="bar", title="Win rate by match_in_session")
            ax.set_ylabel("win rate %"); plt.tight_layout(); plt.savefig("winrate_by_session.png"); plt.clf()
        print("Wrote PNG charts.")


if __name__ == "__main__":
    main()
