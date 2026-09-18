# Troubleshooting

## Colab

**"The release settings are placeholders"**: the notebook was opened from an unpublished copy. Use the link the
instructor published (after `tools/set_release.py`).

**Setup cell: `HTTP Error 404`**: the tag/revision in the setup cell does not exist on GitHub. Instructor: check
`quantsoc/release.py` and push the tag. Participants: ask for the current link, or upload the release ZIP through
the Files pane, unzip it to `/content/workshop` and re-run Setup.

**`ModuleNotFoundError: quantsoc`**: Setup did not finish. Re-run it; it is safe to repeat.

**Widgets show nothing / a blank box**: run the cell again; if still blank, the notebook falls back to static
figures automatically when `ipywidgets` is unavailable. You can force the fallback: in the imports cell add
`widgets.WIDGETS_AVAILABLE = False`. No learning objective depends on the widgets.

**"Runtime disconnected" / files gone**: Colab runtimes are temporary. Re-run from the top. If you exported your
ZIP, you still have everything; if not, set `os.environ["QSW_AUTO_REFERENCE"] = "1"` before creating the tracker to
fast-forward with reference solutions (labelled), then paste your own code back where you have it.

**Secrets: `SecretNotFoundError` / `NotebookAccessError`**: add the secret names exactly (`APCA_API_KEY_ID`,
`APCA_API_SECRET_KEY`) and switch on "Notebook access" for this notebook. The paper demo is optional; the offline path works without it.

**The animation cell is slow (~10 s)**: normal; it renders 24 frames. Skip it if short on time.

## Kaggle

**Setup cell: `Could not download the workshop files`**: no dataset with the workshop files is attached and Internet
is off. Attach the workshop dataset (**Add Input**) and re-run Setup. If it is attached, open it under Input: you should
see a `quantsoc` folder. If you only see a `.zip`, the instructor should re-create the dataset from the unzipped folder.

**Where is my export ZIP?**: right-hand panel → Output → `workshop/workspace/<fund>_strategy.zip` → ⋮ → Download.
Download it before you stop; the session's files are deleted when it ends.

## Exercises

**`[stop] Exercise n … is not complete`**: a later cell needs an exercise you have not passed. Finish it or run
`ex.use_reference(n)`. Nothing is silently substituted.

**Ex 1 "Too few NaNs"**: group by session: `closes.groupby(sessions).pct_change(fill_method=None)`.

**Ex 3 "match a model fitted on train AND validation"**: call `fit` with `X_train, y_train` only.

**Ex 4 "expected … but got …"**: do not renormalise; unselected assets get 0 (not NaN); ties are broken by symbol name.

**Ex 6 `Writing strategy.py` but the check fails**: re-run 6a *then* 6b (6b appends to the file). Read the `[x]`
message: it names the function and the difference.

## Local

**`pip install` fails on Windows with a compiler error**: use Python 3.11/3.12 from python.org (wheels exist for
numpy/pandas/scikit-learn); avoid Python 3.14 pre-releases.

**`pip install` fails on Windows with `OSError: [Errno 2] No such file or directory` (a long `jupyterlab\...` path)**:
Windows' 260-character path limit. Move the unzipped folder somewhere short (for example `C:\qsw`), delete `.venv` and
install again.

**`.venv\Scripts\Activate.ps1` "cannot be loaded because running scripts is disabled on this system"**: run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once in PowerShell, or use `.venv\Scripts\activate.bat` from `cmd`.

**`streamlit run dashboard.py` shows "No trading state found"**: start the runner first (it writes `./state`).
Both commands must run from the same folder.

**`python run_trader.py --mode paper …` → "No Alpaca credentials found"**: create `.env` from `.env.example` in the
folder you run from, or export the two environment variables.

**`alpaca-py` `APIError: forbidden` / 403**: the keys are live keys or wrong; generate *paper* keys. The adapter only
talks to `paper-api.alpaca.markets`.

**Paper preview says "market closed"**: expected outside 09:30–16:00 New York time (and on US holidays).

**Paper preview says "warm-up: n/13 bars in this session"**: the engine needs 13 completed 5-minute bars of the
current session (features are computed within a session); try again after ~10:40 New York time.

**Paper preview says "stale data"**: the IEX feed returned no recent bar for a symbol (illiquid instrument or feed
delay). The engine excludes that symbol for the cycle; check `QSW_SYMBOL_MAP` uses liquid instruments.

**Orders rejected: "insufficient buying power"**: the paper account has less cash than the strategy assumes
(e.g. positions from another program). The engine scales buys down and logs it; consider a fresh paper account.

**Duplicate-order worry after a crash**: restart the runner. Client order ids are deterministic per bar and symbol;
the registry (`state/orders.json`) and a broker lookup by client id prevent resubmission.

**Stopping**: Ctrl+C once; the runner finishes the cycle, lists open orders and exits. Add
`--cancel-open-orders-on-stop` to cancel them.

## Tests / CI

**`pytest -m slow` fails with a kernel error**: install `ipykernel` (in `requirements-dev.txt`) in the same environment.

**`build_notebooks.py --check` says OUT OF DATE**: someone edited an `.ipynb` directly; edit `workshop_source.py` and rebuild.
