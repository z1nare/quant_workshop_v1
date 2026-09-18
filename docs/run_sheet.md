# Run sheet: 120 minutes

| Clock | Section | Cells | What happens | Presenter action |
|---|---|---|---|---|
| 0:00 | Your first fund | title, setup, imports | participants run Setup (20–60 s) | disclosure: synthetic market, planted pattern, not investment advice |
| 0:04 | | demo, fund name | 30-second animation; name the fund; initial hypothesis | ask the Predict question (momentum continues? reverses?) |
| 0:08 | Understand the market | load, explorer | inspect bars, sessions, normalised prices | define bar, return, bps, session |
| 0:12 | | **Ex 1** | per-asset returns within session | cut-off 0:18 → `ex.use_reference(1)` |
| 0:16 | | checkpoint 1 | histograms; volatility differences | ask which asset is widest |
| 0:18 | Find a candidate signal | worked example | lagged feature; research table; timing contract | draw fill at *next open* on the board |
| 0:24 | | **Ex 2** | 3-bar momentum | cut-off 0:33 |
| 0:28 | | signal explorer | bucket means, training rows only; record hypothesis | "why only training rows?" |
| 0:33 | Train a predictor | splits, baseline | chronological 60/20/20; zero-forecast MSE | why not shuffle |
| 0:40 | | **Ex 3** | fit linear model on train only | cut-off 0:50 |
| 0:45 | | checkpoint 3 | MSE vs baseline, correlation, decile plot, coefficients | interpret uncertainty |
| 0:50 | Build a portfolio | equal weight, manager | weights, cash, exposure; controls | "with threshold 15 bps why all cash?" |
| 0:56 | | **Ex 4** | allocation rule | cut-off 0:68 |
| 1:04 | | checkpoint 4 | edge cases table; manager with own function | |
| 1:08 | Evaluate | backtest | equity / drawdown / turnover vs benchmark | fill and cost assumptions |
| 1:12 | | cost explorer | gross vs net; break-even cost | Predict before Apply |
| 1:15 | | **Ex 5** | choose + justify config on validation | cut-off 1:20 |
| 1:18 | | freeze + test | final-test reveal (once) | "one draw; look at drawdown & benchmark" |
| 1:20 | Develop the application | worked example | explicit inputs/outputs | runner loads model once |
| 1:24 | | **Ex 6a**, **Ex 6b** | `%%writefile strategy.py` | cut-off 1:38 |
| 1:36 | | checkpoint 6 | parity table research vs exported | |
| 1:40 | | export | ZIP download checkpoint | **everyone downloads now** |
| 1:43 | | mocked cycle | rejection + partial fill in the log | projector if short on time |
| 1:45 | Connect and run | replay monitor | play/pause/step; replay == backtest | replay is simulated |
| 1:49 | | Alpaca (optional) | connectivity, preview, one supervised submit | projector only |
| 1:53 | | app / monitor | runner path + inline state monitor; local commands | |
| 1:55 | Reveal and extend | mechanism | equation, parameters | ask "what happens on stress?" |
| 1:57 | | stress replay | edge disappeared | discussion |
| 1:59 | | close | extensions, VPS tutorial, disclosure | |

**Abbreviated route** (behind at 1:00): drop the edge-case table, mocked cycle, paper cells, hypothesis cell (saves ~15 min).
