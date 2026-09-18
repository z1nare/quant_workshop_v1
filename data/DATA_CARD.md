# Data card: synthetic workshop market

**Files**
* `workshop_market.csv`: 5 fictional assets × 39 sessions × 78 five-minute bars = 15,210 rows (2025-03-03 → 2025-04-24, fictional dates).
* `stress_market.csv`: same assets, 10 sessions (2025-04-28 → 2025-05-09), volatility doubled and the momentum effect reversed.

**Columns** `timestamp` (ISO-8601 with New York offset, bar START), `session` (YYYY-MM-DD), `symbol`, `open`, `high`, `low`, `close` (USD, 6 dp), `volume` (shares).

**Assets** AURA (Aura Robotics), BOLT (Bolt Energy), CRUX (Crux Semiconductors), DUNE (Dune Minerals), ECHO (Echo Media), all fictional.
Per-bar volatility σ = 9, 12, 15, 20, 28 bps respectively.

**Mechanism** (see `quantsoc/market.py` docstring and the notebook's final section)
`r[i,t] = β·clip(m[i,t−1]) + σ_i·(ρ·F[t] + √(1−ρ²)·ε[i,t])` with β = 0.07, ρ = 0.55, m = sum of the previous three
close-to-close log returns in the same session, F a shared factor; opens are the previous close times a small gap
(15 % of σ intraday, 3σ overnight); high/low wrap open/close with half-σ wicks; U-shaped volume.

**Generation** `python generate_data.py` (deterministic, seeds 20250917 / 20250918); `python generate_data.py --check` verifies the files.

**Calibration** (`tools/calibrate_market.py`, seeds 1–8 plus the classroom seed, reference configuration, 2 bps costs):
median validation correlation ≈ 0.12, MSE improvement ≈ 1.5 %, net validation profit in 7/9 seeds, net test profit in 8/9.
The classroom seed was fixed *before* looking at outcomes and was not selected for its test result.

**Intended use** teaching only. The pattern exists by construction; nothing here describes real markets.
