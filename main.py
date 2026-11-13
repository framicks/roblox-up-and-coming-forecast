#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (make.fun roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson + /input/task.json, writes /output/forecast.json.
Pure Python 3.13 stdlib, deterministic (no randomness, no wall-clock).

Model: the chart sorts new games (<=28 days old) by user growth, and it churns
fast (median 6 of the top 10 change between reads a few hours apart). So instead
of persistence (the house baseline), each game gets a projected rank mu and a
strength w from:
  - last rank, rank velocity (climbing games keep climbing),
  - playing-count growth over the last ~24h of reads (the chart sorts by growth),
  - age: games approaching 28 days old leave the chart during the window, so
    their strength is scaled by the fraction of the window they stay eligible.
For place k, P(game at k) is proportional to w * exp(-(k-mu)^2 / (2*sigma^2))
with sigma=4 (ranks move a lot over 72h), normalised per place, leaving a
small unnamed share that spreads evenly over games we did not name.
"""
import json
import math
import os
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SIGMA = float(os.environ.get("FC_SIGMA", "5.0"))          # rank spread: ranks move a lot over 72h
UNNAMED = float(os.environ.get("FC_UNNAMED", "0.20"))       # share of each place left unnamed (spreads over un-named games)
MAX_AGE_DAYS = 28.0 # chart only lists games published within the last 28 days
WINDOW_HOURS = 72.0


def parse_time(s):
    if s is None:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    output_name = task.get("output", "forecast.json")
    window_from = parse_time(task.get("window", {}).get("from"))
    if window_from is None:
        window_from = parse_time(task.get("history_until"))

    reads = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                reads.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    reads.sort(key=lambda r: r.get("at") or "")

    if not reads:
        raise SystemExit("no reads in history")

    last = reads[-1]
    last_at = parse_time(last.get("at")) or window_from
    window_start = window_from or last_at

    # --- per-game series -------------------------------------------------
    # gid -> {ranks: [(t, rank)], playings: [(t, playing)], created: datetime|None,
    #         votes: (up, down)|None}
    games = {}
    for r in reads:
        t = parse_time(r.get("at"))
        chart = r.get("up_and_coming") or []
        details = r.get("details") or {}
        for g in chart:
            gid = g.get("universe_id")
            if gid is None:
                continue
            e = games.setdefault(gid, {"ranks": [], "playings": [], "created": None,
                                       "up": None, "down": None})
            if g.get("rank") is not None:
                e["ranks"].append((t, float(g["rank"])))
            if g.get("playing") is not None:
                e["playings"].append((t, float(g["playing"])))
            d = details.get(str(gid)) or details.get(gid) or {}
            if d:
                c = parse_time(d.get("created"))
                if c is not None and e["created"] is None:
                    e["created"] = c
                if e["up"] is None and d.get("up_votes") is not None:
                    e["up"] = d.get("up_votes")
                    e["down"] = d.get("down_votes")

    # --- features per game ------------------------------------------------
    cands = []  # (gid, mu, w)
    for gid, e in games.items():
        ranks = e["ranks"]
        if not ranks:
            continue
        last_rank = ranks[-1][1]
        # rank velocity: rank ~24h ago minus rank now (positive = climbing).
        # Use the earliest rank within the last 24h of reads, else the oldest.
        t_now = ranks[-1][0]
        ref = None
        for t, rk in ranks:
            if t is not None and t_now is not None and (t_now - t).total_seconds() <= 26 * 3600:
                ref = rk
                break
        if ref is None:
            ref = ranks[0][1]
        vel = ref - last_rank  # positive when climbing

        # playing growth over the last ~24h (log ratio).
        play = e["playings"]
        growth = 0.0
        if len(play) >= 2 and t_now is not None:
            p_now = play[-1][1]
            p_ref = None
            for t, p in play:
                if t is not None and (t_now - t).total_seconds() <= 26 * 3600:
                    p_ref = p
                    break
            if p_ref is None:
                p_ref = play[0][1]
            if p_ref and p_ref > 0 and p_now > 0:
                growth = math.log(p_now / p_ref)
        growth = max(-1.5, min(1.5, growth))

        # age / eligibility over the 72h window.
        elig = 1.0
        if e["created"] is not None and window_start is not None:
            age_days = (window_start - e["created"]).total_seconds() / 86400.0
            # fraction of the window (in days) the game stays <=28 days old
            elig = max(0.0, min(1.0, (MAX_AGE_DAYS - age_days) / (WINDOW_HOURS / 24.0)))
            if elig <= 0.0:
                continue  # already too old to be on the chart in the window

        # projected rank: last rank, pulled by velocity and growth.
        mu = last_rank - 0.6 * vel - 1.5 * growth
        mu = max(1.0, min(45.0, mu))

        # strength: rank base (decays with rank) boosted by growth/velocity,
        # scaled by eligibility.
        w = math.exp(-(last_rank - 1.0) / 14.0)
        w *= (1.0 + 0.35 * max(0.0, growth))
        w *= (1.0 + 0.10 * max(0.0, vel))
        w *= (0.25 + 0.75 * elig)
        if w <= 0:
            continue
        cands.append((gid, mu, w))

    # Also include games that were in the chart recently but not in the last
    # read (they may re-enter): they are already in `games` with their last
    # known rank; mu/w computed above covers them (last_rank is stale but the
    # Gaussian spread + unnamed share handles the uncertainty).

    # --- probabilities per place ------------------------------------------
    probabilities = []
    for k in range(1, places + 1):
        raw = []
        for gid, mu, w in cands:
            v = w * math.exp(-((k - mu) ** 2) / (2.0 * SIGMA * SIGMA))
            if v > 1e-6:
                raw.append((gid, v))
        if not raw:
            continue
        total = sum(v for _, v in raw)
        # cap rows per place at 100 (rules) and keep only meaningful mass
        raw.sort(key=lambda x: -x[1])
        raw = raw[:100]
        total = sum(v for _, v in raw)
        if total <= 0:
            continue
        for gid, v in raw:
            p = (1.0 - UNNAMED) * v / total
            if p >= 0.0005:
                probabilities.append({"universe_id": int(gid), "place": k,
                                      "p": round(p, 6)})

    # --- top10: games with the highest total mass across places -----------
    mass = {}
    for row in probabilities:
        mass[row["universe_id"]] = mass.get(row["universe_id"], 0.0) + row["p"]
    # tie-break by projected rank (lower mu first)
    mu_of = {gid: mu for gid, mu, _ in cands}
    top10 = sorted(mass, key=lambda g: (-mass[g], mu_of.get(g, 99.0)))[:places]
    # pad with best-mu games if fewer than `places`
    if len(top10) < places:
        rest = sorted((gid for gid, _, _ in cands if gid not in top10),
                      key=lambda g: mu_of.get(g, 99.0))
        top10 = top10 + [int(g) for g in rest[: places - len(top10)]]
    top10 = [int(g) for g in top10[:places]]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as f:
        f.write(json.dumps({"probabilities": probabilities, "top10": top10}) + "\n")


if __name__ == "__main__":
    main()
