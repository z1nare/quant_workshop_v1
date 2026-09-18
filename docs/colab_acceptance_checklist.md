# Colab manual acceptance checklist

Local clean-kernel execution (see `docs/verification_report.md`) does **not** verify the Colab UI. Before the
workshop, an instructor should open the published notebook link in a fresh Colab session and tick every line.

**Setup**
- [ ] Setup cell downloads the release ZIP and prints `Done.` in under ~60 s (first run) and skips the download on a second run.
- [ ] Re-running the setup cell after creating `workspace/strategy.py` does **not** overwrite it.
- [ ] Environment table shows `runtime: Google Colab`, `ipywidgets` present, `alpaca-py` installed.

**Widgets** (each must react within ~2 s and must not re-run on every slider pixel)
- [ ] Asset explorer: switching view / assets redraws.
- [ ] Signal explorer: feature dropdown and bucket slider redraw.
- [ ] Portfolio manager: sliders update only on release; "warm-up (all NaN)" shows 100 % cash.
- [ ] Cost explorer: Apply button redraws; slider alone does not.
- [ ] Replay monitor: Play runs, Pause stops, `< step` / `step >` move one bar, speed dropdown changes the pace; re-running the cell leaves no ghost animation running.
- [ ] State monitor: Refresh updates the table.
- [ ] If any widget fails to render, the static fallback (printed message + figures) appears instead.

**Exercises**
- [ ] A wrong answer prints a specific `[x]` reason; `ex.hint(n)` reveals one hint per call; `ex.show_solution(n)` prints code; `ex.use_reference(n)` prints the `[ref]` banner.
- [ ] Running a downstream checkpoint with an unfinished exercise prints a single-line `[stop]` message (no long traceback).

**Export**
- [ ] The export cell triggers a browser download of `<fund>_strategy.zip`; the ZIP opens and contains `strategy.py`, `model.joblib`, `config.json`, `manifest.json`, `README_RUN_LOCAL.md`, `quantsoc/`, `data/`.
- [ ] `manifest.json` says `"strategy_source": "student"` when you wrote 6a/6b yourself, `"reference"` if you used the reference.

**Paper trading (instructor account only)**
- [ ] With `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` in Colab Secrets and notebook access granted, the connectivity cell prints account equity, clock and instrument checks without printing the keys.
- [ ] Without secrets, the cell prints the OFFLINE banner and nothing later fails.
- [ ] "Run all" never submits orders (Submit button stays disabled; `SUBMIT_FOR_REAL` is `False`).
- [ ] Preview shows proposed orders (or "market closed"); one click on Submit submits once and disables the button.

**Runtime loss**
- [ ] After "Disconnect and delete runtime", re-running from the top recreates everything; the previously downloaded ZIP still runs locally.

**Kaggle (only if participants use Kaggle; see README "Quickstart A2")**
- [ ] The workshop dataset shows a `quantsoc` folder in its file browser (not only a `.zip`).
- [ ] **Copy & Edit** of the shared notebook keeps the dataset attached (it is listed under Input).
- [ ] With Internet **off**, Setup prints `Using the workshop files attached to this notebook`, `running in: Kaggle` and `Done.`; a second run skips the copy.
- [ ] Environment table shows `runtime: Kaggle`; every widget in the list above renders and reacts (if not, `widgets.WIDGETS_AVAILABLE = False` in the imports cell gives the static fallback).
- [ ] The export cell prints the Kaggle download instructions, and the ZIP downloads from Output → `workshop/workspace/`.
