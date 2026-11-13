# Roblox Up-and-Coming Forecast

Forecast Roblox's Up-and-Coming chart top 10 for the 72 hours after entries
close, for the make.fun `roblox-up-and-coming-forecast-01` bounty.

## Entry

`main.py` (Python 3.13, stdlib only, deterministic) reads
`/input/history.ndjson` + `/input/task.json` and writes
`/output/forecast.json` (`probabilities` + `top10`).

## Model

The chart sorts new games (<=28 days old) by user growth and churns fast
(median 6 of the top 10 change between reads a few hours apart), so
persistence (the house baseline) is a weak forecast. Each game gets a
projected rank `mu` and a strength `w` from:

- last rank, rank velocity (climbing games keep climbing),
- playing-count growth over the last ~24h of reads (the chart sorts by growth),
- age: games approaching 28 days old leave the chart during the window, so
  their strength is scaled by the fraction of the window they stay eligible.

For place k, P(game at k) is proportional to
`w * exp(-(k-mu)^2 / (2*sigma^2))` with sigma=5, normalised per place,
leaving 0.20 of each place unnamed (spreads evenly over un-named games).

## Local backtest (on the kit's practice history)

Forecast from a truncated history, scored against the reads in the next 72h:

| cutoff | this model | house baseline | delta |
|---|---|---|---|
| 2026-11-02T23:00Z | -38.32 | -49.13 | +10.80 |
| 2026-11-04T23:00Z | -38.77 | -51.77 | +13.01 |

Log score (higher is better); the baseline scores -43.51 on the practice
window. Two runs of `main.py` on the same input are byte-identical.

## License

MIT
