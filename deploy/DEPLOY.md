# Deploying to the VPS

Assumes Ubuntu 24.04 (Python 3.12), a system user `bot`, and the code in
`/opt/prediction-cup-bot`. If any of these differ, change them in
`prediction-cup-bot.service` as well.

## 1. One-time setup (as root / with sudo)

```bash
apt update && apt install -y python3.12-venv git
timedatectl set-ntp true            # order expiry timestamps need an accurate clock
useradd --system --no-create-home --home-dir /opt/prediction-cup-bot --shell /usr/sbin/nologin bot
```

## 2. Copy the code

On the VPS:

```bash
git clone https://github.com/karoldudzicz/SIG-predictions-cup-bot.git /opt/prediction-cup-bot
```

(`.env` is not in the repo; it is created in step 3.)

## 3. Python environment, secrets, state directories

```bash
cd /opt/prediction-cup-bot
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
mkdir -p logs data                  # must exist: the unit only allows writes here
cp .env.example .env && nano .env   # paste SIG_API_KEY; leave DRY_RUN=true for now
chmod 600 .env
chown -R bot:bot /opt/prediction-cup-bot
```

## 4. Smoke test as the bot user, before systemd

```bash
sudo -u bot .venv/bin/python exchange_api.py        # account + markets
sudo -u bot .venv/bin/python oracle_api.py          # Polymarket + Kalshi from the VPS
sudo -u bot .venv/bin/python main.py --cycles 2     # two dry-run cycles
```

## 5. Install and start the service

```bash
cp deploy/prediction-cup-bot.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now prediction-cup-bot
systemctl status prediction-cup-bot     # should say "active (running)"
```

## 6. Day-to-day

| What | Command |
|---|---|
| Live log | `journalctl -u prediction-cup-bot -f` |
| Last cycle summary | `cat /opt/prediction-cup-bot/logs/heartbeat.json` |
| Structured log file | `tail -f /opt/prediction-cup-bot/logs/bot.log` |
| Stop (cancels open orders) | `systemctl stop prediction-cup-bot` |
| Restart after changing `.env` | `systemctl restart prediction-cup-bot` |

## 7. Going live

Only after dry-run cycles look right on the VPS **and** the tournament is open
(2026-10-01 12:00 ET):

1. Optional real-order probe (1 share @ 0.005, cancelled immediately):
   `sudo -u bot env DRY_RUN=false .venv/bin/python exchange_api.py order-test <exchange_id>`
2. Set `DRY_RUN=false` in `.env`, then `systemctl restart prediction-cup-bot`.
3. Watch `journalctl -u prediction-cup-bot -f` through the first few cycles.

## If the service stays down

- **Exit code 2** (`systemctl status` shows `status=2`): API key or terms problem.
  Check the log for `TERMS_NOT_ACKNOWLEDGED` / `INVALID_API_KEY`, fix it on the
  website or in `.env`, then `systemctl restart prediction-cup-bot`.
- **"start request repeated too quickly"**: it crashed 10 times in 10 minutes.
  Read `journalctl -u prediction-cup-bot -n 200`, fix, then
  `systemctl reset-failed prediction-cup-bot && systemctl start prediction-cup-bot`.
