#!/usr/bin/env bash
# ============================================================================
# Aura Cloud Server Setup — Oracle Cloud Always Free (ARM/Ampere)
# ============================================================================
# Run this ONCE on a fresh Ubuntu 22.04 ARM VM.
#
#   ssh ubuntu@<YOUR_VM_IP> 'bash -s' < setup_server.sh
#
# What it does:
#   1. System packages & Python 3.11
#   2. Installs Ollama + pulls qwen2.5:14b
#   3. Creates aura user & venv
#   4. Installs Aura dependencies
#   5. Sets up systemd services (Ollama + Aura)
#   6. Configures nginx reverse proxy + firewall
# ============================================================================
set -euo pipefail

AURA_USER="aura"
AURA_HOME="/opt/aura"
AURA_VENV="${AURA_HOME}/venv"
OLLAMA_MODEL="qwen2.5:14b"

echo "╔══════════════════════════════════════════════════════╗"
echo "║    AURA CLOUD SERVER SETUP — Oracle Always Free     ║"
echo "╚══════════════════════════════════════════════════════╝"

# ------------------------------------------------------------------
# 1. System packages
# ------------------------------------------------------------------
echo ">>> [1/7] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip \
    nginx \
    curl wget git \
    build-essential \
    sqlite3 \
    ufw \
    htop tmux

# Make python3.11 the default if it's not python3 already
if ! python3 --version 2>/dev/null | grep -q "3.11"; then
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 || true
fi

# ------------------------------------------------------------------
# 2. Install Ollama
# ------------------------------------------------------------------
echo ">>> [2/7] Installing Ollama..."
# CP126 (critical): "Ollama installer is executed as root without pinning.
# The current response from ollama.ai is piped directly to sh under a
# privileged setup script. There is no version lock, digest, signature
# verification, artifact retention, or package manager involvement."
#
# Same defect and same fix as cloud/hetzner_setup.sh: fetch the script to
# disk, print its digest, keep the artifact so what ran can be read
# afterwards, and execute it only against a pin or an explicit
# acknowledgement. Piping a live URL into a privileged shell hands root to
# whoever controls that URL, and leaves nothing behind to inspect.
if ! command -v ollama &>/dev/null; then
    OLLAMA_INSTALLER="${HOME}/ollama-install.sh"
    curl -fsSL --proto '=https' --tlsv1.2 https://ollama.ai/install.sh -o "$OLLAMA_INSTALLER"
    OLLAMA_SHA="$(sha256sum "$OLLAMA_INSTALLER" | cut -d' ' -f1)"
    echo "    installer sha256: $OLLAMA_SHA"
    echo "    saved to: $OLLAMA_INSTALLER"

    if [ -n "${OLLAMA_INSTALLER_SHA256:-}" ]; then
        if [ "$OLLAMA_SHA" != "$OLLAMA_INSTALLER_SHA256" ]; then
            echo "!!! installer digest mismatch."
            echo "    expected: $OLLAMA_INSTALLER_SHA256"
            echo "    got:      $OLLAMA_SHA"
            echo "    Refusing to execute it. Nothing has been installed."
            exit 1
        fi
        echo "    digest matches the pin; executing."
        sh "$OLLAMA_INSTALLER"
    elif [ "${OLLAMA_INSTALL_UNVERIFIED:-}" = "1" ]; then
        echo "    WARNING: executing an UNVERIFIED remote installer because"
        echo "    OLLAMA_INSTALL_UNVERIFIED=1 was set. Its digest is above."
        sh "$OLLAMA_INSTALLER"
    else
        echo "!!! Not executing the installer."
        echo "    Review $OLLAMA_INSTALLER, then re-run with either:"
        echo "      OLLAMA_INSTALLER_SHA256=$OLLAMA_SHA   (pin what you reviewed)"
        echo "      OLLAMA_INSTALL_UNVERIFIED=1           (accept it unverified)"
        exit 1
    fi
fi

# Start Ollama service
sudo systemctl enable ollama
sudo systemctl start ollama

# Wait for Ollama to be ready
echo "    Waiting for Ollama to start..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "    Ollama is ready."
        break
    fi
    sleep 2
done

# Pull the model (this takes a while on first run)
echo ">>> [3/7] Pulling ${OLLAMA_MODEL} (this may take 10-20 minutes)..."
ollama pull "${OLLAMA_MODEL}" || {
    echo "WARNING: Failed to pull ${OLLAMA_MODEL}. You can pull it manually later:"
    echo "  ollama pull ${OLLAMA_MODEL}"
}

# ------------------------------------------------------------------
# 3. Create aura user & directory
# ------------------------------------------------------------------
echo ">>> [4/7] Creating aura user and directory..."
if ! id "${AURA_USER}" &>/dev/null; then
    sudo useradd -r -m -d "${AURA_HOME}" -s /bin/bash "${AURA_USER}"
fi

sudo mkdir -p "${AURA_HOME}"
sudo mkdir -p "${AURA_HOME}/data/memory"
sudo mkdir -p "${AURA_HOME}/logs"
sudo mkdir -p /home/${AURA_USER}/.aura/data/memory
sudo mkdir -p /home/${AURA_USER}/.aura/data/knowledge_graph
sudo mkdir -p /home/${AURA_USER}/.aura/logs

# ------------------------------------------------------------------
# 4. Python venv & dependencies
# ------------------------------------------------------------------
echo ">>> [5/7] Setting up Python venv..."
sudo -u "${AURA_USER}" python3 -m venv "${AURA_VENV}"

# Install deps (if requirements.txt is already deployed)
if [ -f "${AURA_HOME}/requirements.txt" ]; then
    sudo -u "${AURA_USER}" "${AURA_VENV}/bin/pip" install --upgrade pip
    sudo -u "${AURA_USER}" "${AURA_VENV}/bin/pip" install -r "${AURA_HOME}/requirements.txt"
    # Also install websockets for any WS clients
    sudo -u "${AURA_USER}" "${AURA_VENV}/bin/pip" install websockets
else
    echo "    No requirements.txt yet — will install after first deploy."
fi

# ------------------------------------------------------------------
# 5. Systemd services
# ------------------------------------------------------------------
echo ">>> [6/7] Installing systemd services..."

# --- Aura service ---
sudo tee /etc/systemd/system/aura.service > /dev/null << 'UNIT'
[Unit]
Description=Aura Autonomy Engine
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=aura
Group=aura
WorkingDirectory=/opt/aura
Environment="PATH=/opt/aura/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="HOME=/home/aura"
Environment="AURA_ENV=production"
Environment="LLM_PROVIDER=local"
Environment="AURA_LLM_PROVIDER=local"
Environment="AURA_LLM_MODEL=qwen2.5:14b"
Environment="AURA_LLM_BASE_URL=http://localhost:11434"

ExecStart=/opt/aura/venv/bin/python aura_main.py --server
Restart=always
RestartSec=10
StandardOutput=append:/opt/aura/logs/aura_stdout.log
StandardError=append:/opt/aura/logs/aura_stderr.log

# Resource limits — generous for 24GB RAM VM
LimitNOFILE=65536
MemoryMax=16G

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable aura.service
echo "    aura.service installed (not started yet — deploy code first)."

# ------------------------------------------------------------------
# 6. Nginx reverse proxy
# ------------------------------------------------------------------
echo ">>> [7/7] Configuring Nginx..."

sudo tee /etc/nginx/sites-available/aura > /dev/null << 'NGINX'
# Aura Reverse Proxy — HTTP (port 80) → localhost:8000
# To add HTTPS, use: sudo certbot --nginx -d your-domain.com

upstream aura_backend {
    server 127.0.0.1:8000;
}

server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    # WebSocket support
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # Increase timeouts for long LLM responses
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
    proxy_connect_timeout 60s;

    # Max body size for file uploads
    client_max_body_size 50M;

    location / {
        proxy_pass http://aura_backend;
    }

    # WebSocket endpoints
    location /ws/ {
        proxy_pass http://aura_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400s;  # Keep WS alive for 24h
    }

    # Health check
    location /health {
        proxy_pass http://aura_backend/api/state;
        access_log off;
    }
}
NGINX

sudo rm -f /etc/nginx/sites-enabled/default
sudo ln -sf /etc/nginx/sites-available/aura /etc/nginx/sites-enabled/aura
sudo nginx -t && sudo systemctl enable nginx && sudo systemctl restart nginx

# ------------------------------------------------------------------
# 7. Firewall
# ------------------------------------------------------------------
echo ">>> Configuring firewall..."
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS (for later)
sudo ufw --force enable

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║          SERVER SETUP COMPLETE ✓                    ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║                                                     ║"
echo "║  Next: Deploy Aura's code:                          ║"
echo "║    ./deploy.sh <VM_IP>                              ║"
echo "║                                                     ║"
echo "║  Then start Aura:                                   ║"
echo "║    ssh ubuntu@<VM_IP> sudo systemctl start aura     ║"
echo "║                                                     ║"
echo "║  Access UI:                                         ║"
echo "║    http://<VM_IP>                                    ║"
echo "║                                                     ║"
echo "╚══════════════════════════════════════════════════════╝"
