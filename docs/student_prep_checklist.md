# Student preparation checklist (5 minutes, before the session)

**Everyone (browser route)**
- [ ] A Google account that can open Google Colab (free). Test: open https://colab.research.google.com and create an empty notebook.
- [ ] Open the workshop notebook link sent by the organisers and choose `File → Save a copy in Drive`. Do **not** run anything yet.
- [ ] Basic Python comfort: variables, functions, lists/dicts, `for` loops. No pandas, finance or machine-learning knowledge is assumed.
- [ ] A laptop with a keyboard (tablets struggle with Colab's editor).

**If the organisers send a Kaggle link instead of a Colab link**
- [ ] A free Kaggle account (signing in with Google is fine).
- [ ] Open the link and press **Copy & Edit**. Do **not** run anything yet.

**Not required**: a GitHub account, Git, a credit card, a GPU, paid market data, or an Alpaca account.

**Optional: only if you want to run locally instead**
- [ ] Python 3.11 or 3.12 installed (`python3 --version` / `py -3.12 --version`).
- [ ] Download the repository ZIP (repository page → Code → Download ZIP) and unzip it.
- [ ] Follow "Quickstart B" in `README.md` up to `jupyter lab …` and confirm the notebook opens. Installation takes 2–5 minutes.

**Optional: only if you want to try the paper-trading demo yourself**
- [ ] Create a free Alpaca account and generate **Paper Trading** API keys (dashboard → Paper → API keys).
- [ ] Colab: add secrets named `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` (🔑 icon on the left) and enable notebook access.
- [ ] Local: copy `.env.example` to `.env` and paste the keys there. Never put keys in a notebook cell.
- [ ] Note the US market hours: 09:30–16:00 New York time. Outside them the demo can only show "market closed".
