# Instructor guide

## Before the session (rehearsal, ~45 minutes)

1. Publish the repository and pin a release: `python tools/set_release.py --owner … --repo … --revision …`, commit, tag, push.
2. Open the Colab link from the README in a fresh session and complete `docs/colab_acceptance_checklist.md`. If the room uses Kaggle instead, set up the dataset and notebook as in README "Quickstart A2" and complete the Kaggle section of the same checklist from a second Kaggle account (Copy & Edit, as a student would).
3. Run `notebooks/workshop_instructor.ipynb` yourself once (about 90 seconds of compute; 10 minutes reading) so you know the numbers your data produces. Compare with `docs/verification_report.md`.
4. Create an Alpaca **paper** account for the projector demo (free; no card). Put the keys in Colab Secrets on your own account. Check the market calendar: US regular hours are 09:30–16:00 New York (13:30/14:30–20:00/21:00 UTC depending on DST). Outside those hours the demo shows "market closed"; that is still useful, but plan the talk track accordingly.
5. Decide the room mode: most participants on Colab; a few advanced participants may run locally (`docs/student_prep_checklist.md`).

## Learning objectives → where they are met

| Objective | Section | Exercise |
|---|---|---|
| Investigate a hypothesis on a controlled market | 1–2 | 1, 2 |
| Build a simple predictive model | 3 | 3 |
| Forecasts → allocations with risk constraints | 4 | 4 |
| Evaluate with costs, turnover, drawdown; freeze before test | 5 | 5 |
| Refactor into reusable functions | 6 | 6a, 6b |
| Connect to the supplied engine | 6–7 | none |
| Accelerated simulation; optional paper demo | 7 | none |
| Take home a working application | 6 export, README | none |

## Teaching pattern

Each section follows *Context → Predict → Demonstration → Experiment → Your task → Checkpoint*. The "🔮 Predict"
prompts are meant to be asked aloud with a show of hands **before** running the next cell. The checkpoints print
`[ok]`/`[x]`; ask people to raise a hand at `[x]` instead of debugging silently.

## Expected misconceptions (and where the notebook catches them)

* **Returns across the overnight gap** (Ex. 1): the checker says "Too few NaNs". Teach the session boundary here.
* **Trading at the close you just observed**: the timing contract in section 2. Draw it on the board.
* **"Correlation 0.1 is nothing"** (Ex. 2/3): the bucket-mean panel and the decile plot are the answer.
* **Fitting on validation "to score better"** (Ex. 3): the checker detects a train+validation fit and says so.
* **Renormalising survivors to spend the whole budget** (Ex. 4): the checker rejects it; cash is a position.
* **"Costs are a detail"** (section 5): the cost explorer; ask for the break-even cost before pressing Apply.
* **Peeking at the test set to choose the threshold** (Ex. 5): the notebook only reveals the test after the freeze cell.
* **"The runner retrains every 5 minutes"** (section 6): it loads `model.joblib` once.
* **Replay results vs broker results** (section 7): mode banners; never mix them.
* **"It made money, so the strategy works"** (final test / stress): one synthetic draw; the stress scenario is the antidote.

## Recovery checkpoints

Every exercise has `ex.hint(n)` (three progressive hints), `ex.show_solution(n)` and `ex.use_reference(n)`. Using
the reference is recorded and labelled in the tracker summary and the export manifest. Recommended cut-offs
(minutes from start): Ex 1 → 18, Ex 2 → 33, Ex 3 → 50, Ex 4 → 68, Ex 5 → 80, Ex 6a/6b → 98. At each cut-off say:
"if you are not at `[ok]`, run `ex.use_reference(n)` now and we move on together."

If a participant's runtime disconnected: re-run from the top (Setup is idempotent; the notebook is ~90 s of compute
with the reference route `QSW_AUTO_REFERENCE=1`; you can tell them to set `os.environ["QSW_AUTO_REFERENCE"]="1"`
in the imports cell to fast-forward to where the room is).

## Abbreviated route (if the group is behind by ≥ 10 minutes at minute 60)

Keep: Ex 4 (allocation), the validation backtest, the cost explorer, Ex 5 (freeze + final test), Ex 6a/6b,
export, replay monitor, stress reveal. Drop: the edge-case table, the mocked cycle (run it on the projector), the
paper demo (show one preview on the projector, or skip), the "record hypothesis" cell. This saves about 15 minutes.

## Paper demo talk track (5 minutes, projector only)

1. Show the connectivity cell output: equity, clock, instrument checks. Point out that the keys are never printed.
2. Explain the symbol mapping and why it proves nothing about real predictive validity.
3. Press **Preview**: read the decision log line by line (bars fetched → warm-up or forecasts → targets → differences vs actual positions → proposed orders or a reason for no trade).
4. If the market is open and you choose to, press **Submit** once; press **Preview** again and show the order statuses. Then stop. Continuous operation is for the runner/VPS, not the notebook.

## Safety facts to state out loud

* Nothing in this project can trade real money: the broker adapter hard-codes the paper endpoint.
* "Run all" cannot submit orders; only an explicit click or an explicit flag edit can.
* Every order carries a deterministic id; reruns and restarts reconcile against the broker instead of resubmitting.
* Credentials live in Colab Secrets or a git-ignored `.env`; the export ZIP never contains them.
