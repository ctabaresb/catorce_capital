#!/bin/bash
# deploy/setup_ec2.sh
# Run once on a fresh Amazon Linux 2023 / Ubuntu 22.04 EC2 instance.
# Tested on t3.micro (2 vCPU, 1GB RAM). Sufficient for shadow + paper mode.
# For live trading with sub-second latency, use t3.small or c6i.large.

set -e

echo "=== Bitso Trader EC2 Setup ==="

# 1. System packages
sudo apt-get update -y
sudo apt-get install -y python3.11 python3.11-venv python3-pip git tmux htop

# 2. App directory
APP_DIR="/opt/bitso_trader"
sudo mkdir -p "$APP_DIR"
sudo chown "$USER:$USER" "$APP_DIR"

# 3. Copy code (run from local machine):
# rsync -avz --exclude='__pycache__' --exclude='*.pyc' --exclude='logs/' \
#   ./bitso_trader/ ec2-user@<EC2_IP>:/opt/bitso_trader/

# 4. Python venv
cd "$APP_DIR"
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5. Environment file (DO NOT commit this to git)
cat > "$APP_DIR/.env" <<'EOF'
BITSO_BOOK=btc_usd
EXEC_MODE=shadow
STRATEGY=passive_mm
BITSO_API_KEY=
BITSO_API_SECRET=
MAX_POS_BTC=0.05
MAX_ORDER_BTC=0.01
MAX_DAILY_LOSS=200.0
OBI_THRESHOLD=0.3
FLOW_THRESHOLD=0.4
SPREAD_MAX_BPS=20.0
LOG_LEVEL=INFO
LOG_DIR=/opt/bitso_trader/logs
EOF
echo "Edit $APP_DIR/.env with your API credentials before going live."

# 6. Systemd service
sudo tee /etc/systemd/system/bitso-trader.service > /dev/null <<EOF
[Unit]
Description=Bitso Automated Trader
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/.venv/bin/python main.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=bitso-trader

# Resource limits
LimitNOFILE=65536
MemoryMax=512M

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable bitso-trader

echo ""
echo "=== Setup complete ==="
echo "Start:   sudo systemctl start bitso-trader"
echo "Logs:    sudo journalctl -u bitso-trader -f"
echo "Status:  sudo systemctl status bitso-trader"
echo ""
echo "IMPORTANT: Set EXEC_MODE=shadow first. Validate for >= 1 week before live."
