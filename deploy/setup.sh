#!/usr/bin/env bash
# Atlas Trading Bot — Oracle Cloud (Ubuntu 22.04 ARM) setup script
# Run as root or with sudo:  sudo bash deploy/setup.sh

set -euo pipefail

APP_USER="atlas"
APP_DIR="/opt/atlas-trading-bot"
PYTHON_VERSION="3.11"

echo "=== 1/7 System packages ==="
apt-get update
apt-get install -y \
  software-properties-common \
  python${PYTHON_VERSION} \
  python${PYTHON_VERSION}-venv \
  python${PYTHON_VERSION}-dev \
  docker.io \
  docker-compose \
  git \
  curl \
  build-essential

# Enable Docker
systemctl enable docker
systemctl start docker

echo "=== 2/7 Create app user ==="
if ! id "$APP_USER" &>/dev/null; then
  useradd -r -m -s /bin/bash "$APP_USER"
fi
usermod -aG docker "$APP_USER"

echo "=== 3/7 Clone repo ==="
if [ ! -d "$APP_DIR" ]; then
  mkdir -p "$APP_DIR"
  echo "Copy your project files to $APP_DIR"
  echo "  e.g.: scp -r ./* atlas-server:$APP_DIR/"
fi
chown -R "$APP_USER":"$APP_USER" "$APP_DIR"

echo "=== 4/7 Python venv ==="
sudo -u "$APP_USER" bash -c "
  cd $APP_DIR
  python${PYTHON_VERSION} -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -e '.[dev]'
"

echo "=== 5/7 ChromaDB (Docker) ==="
docker pull chromadb/chroma:latest
docker run -d \
  --name chromadb \
  --restart unless-stopped \
  -p 8000:8000 \
  chromadb/chroma:latest

echo "=== 6/7 Create data directory ==="
sudo -u "$APP_USER" mkdir -p "$APP_DIR/data"

echo "=== 7/7 Install systemd services ==="
cp "$APP_DIR/deploy/atlas-bot.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable atlas-bot

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Copy your .env file:  scp .env atlas-server:$APP_DIR/.env"
echo "  2. Start the bot:        sudo systemctl start atlas-bot"
echo "  3. Check logs:           sudo journalctl -u atlas-bot -f"
echo "  4. Test run:             sudo -u atlas su -c 'cd $APP_DIR && .venv/bin/python -m src.main --run-now'"
echo ""
