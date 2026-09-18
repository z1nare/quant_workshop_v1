# Optional: running the runner on a small Linux server (VPS)

This is for participants who want their exported application to run unattended during US market hours. It is
optional, costs money (a few dollars a month), and is still **paper trading**: the adapter cannot reach a live account.

## Provider and cost

Any Linux VPS with 1 GB of RAM works. As of September 2026 DigitalOcean's smallest Droplet is listed at
**$4/month** (1 vCPU, 512 MiB, 10 GiB SSD) and the next size at **$6/month** (1 GiB RAM), billed per second
(see [digitalocean.com/pricing/droplets](https://www.digitalocean.com/pricing/droplets); prices change, so check the
page). The 1 GiB size is the safer choice for pandas + scikit-learn. There is **no permanently free VPS** on offer
from the major providers; promotional credits for new accounts exist but expire. Hetzner, Linode/Akamai and Vultr
have comparable entry sizes.

## 1. Environment setup (Ubuntu 24.04)

```bash
# as root, once
adduser trader && usermod -aG sudo trader
# log in as trader
sudo apt update && sudo apt install -y python3.12-venv python3-pip unzip
mkdir -p ~/app && cd ~/app
# upload your export ZIP (scp from your laptop): scp my_fund_strategy.zip trader@SERVER:~/app/
unzip my_fund_strategy.zip && cd my_fund_strategy
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_trader.py --mode replay --cycles 20          # smoke test, offline
```

## 2. Secure credentials

```bash
cp .env.example .env && nano .env                        # paste PAPER keys
chmod 600 .env                                           # readable by the trader user only
```
Never commit `.env`, never put keys in `strategy.py`, never paste them into chat/tickets. Rotate keys from the
Alpaca dashboard if you suspect exposure.

## 3. Process supervision, restart and logs (systemd)

`sudo nano /etc/systemd/system/quant-runner.service`:
```ini
[Unit]
Description=QuantSoc paper-trading runner
After=network-online.target

[Service]
User=trader
WorkingDirectory=/home/trader/app/my_fund_strategy
EnvironmentFile=/home/trader/app/my_fund_strategy/.env
ExecStart=/home/trader/app/my_fund_strategy/.venv/bin/python run_trader.py --mode paper --i-understand-paper-orders
Restart=on-failure
RestartSec=30
KillSignal=SIGINT
TimeoutStopSec=90

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now quant-runner
systemctl status quant-runner
journalctl -u quant-runner -f                            # live logs; state/decisions.log has the decision log
```
`KillSignal=SIGINT` lets the runner finish its cycle and list open orders on stop (graceful stop).

## 4. Persistence

State lives in `state/` (JSON + log) inside the app folder, on the server's disk; it survives restarts and is what
makes restarts safe (the order registry). Back it up with the rest of the folder. The model is `model.joblib` in
the bundle; the runner never retrains.

## 5. Market scheduling and time zones

The runner itself skips cycles when the market is closed (it asks the broker's clock), so it can run 24/7. To save
resources you can instead run it only on weekdays around US hours with a systemd timer; set the server's clock to
UTC and compute the window: regular hours are 09:30–16:00 America/New_York, i.e. 13:30–20:00 UTC in summer (EDT)
and 14:30–21:00 UTC in winter (EST). US market holidays are handled by the broker clock (`is_open` false).

## 6. Graceful stop and outstanding orders

```bash
sudo systemctl stop quant-runner       # sends SIGINT; the runner finishes the cycle and lists open orders
```
Market orders normally fill within seconds; if the log shows open orders at stop, either restart the runner (it
reconciles them by client id) or cancel them in the Alpaca dashboard. Add `--cancel-open-orders-on-stop` to
`ExecStart` if you prefer automatic cancellation. To flatten everything, use the Alpaca dashboard's "close all positions".

## 7. Safe, private dashboard access

Do **not** expose Streamlit to the internet. Keep it bound to localhost (the repository's `.streamlit/config.toml`
does this) and tunnel over SSH from your laptop:
```bash
# on the server (second service or a tmux session):
.venv/bin/streamlit run dashboard.py
# on your laptop:
ssh -L 8501:localhost:8501 trader@SERVER
# then open http://localhost:8501 locally
```
Also: use SSH keys, disable password login, enable the firewall (`sudo ufw allow OpenSSH && sudo ufw enable`).

## 8. Updating the application

1. Export a new bundle from the notebook (new `strategy.py`/`model.joblib`/`config.json`).
2. `sudo systemctl stop quant-runner` (graceful).
3. Upload and unzip into a new folder, copy `.env` and `state/` across, run the offline smoke test.
4. Point the service `WorkingDirectory`/`ExecStart` at the new folder, `daemon-reload`, `start`.

Keep the old folder until the new one has run through a full session.
