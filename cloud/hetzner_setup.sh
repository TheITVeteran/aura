#!/bin/bash
# cloud/hetzner_setup.sh
# RUN THIS ON THE HETZNER SERVER (as root)

set -euo pipefail  # exit on error, undefined var, or failed pipe stage

APP_DIR="/opt/aura"
# CP126: "The systemd unit executes /opt/aura/ignition.py, which is absent
# from the tracked source, and deliberately runs the service as root. A
# nominally successful setup creates a restart loop."
#
# Both halves fixed: the unit now starts aura_main.py, which is the actual
# entrypoint in this repository (ignition.py has never existed here), and
# the service runs as a dedicated unprivileged user. A web-facing service
# on port 8000 running as root turns any request-handling bug into a root
# compromise. Override with AURA_SERVICE_USER=root only deliberately.
SERVICE_USER="${AURA_SERVICE_USER:-aura}"

echo ">>> 1. SYSTEM UPDATE (ARM/AMD optimized)..."
apt update && apt upgrade -y
apt install -y python3-pip python3-venv nginx certbot python3-certbot-nginx ufw acl curl

echo ">>> 2. INSTALLING OLLAMA (The Neural Engine)..."
# CP126: "Remote installer executes as root without verification. The script
# pipes the current response from ollama.com directly into sh while the
# documented execution context is root. There is no pinned version, digest,
# signature, downloaded-artifact inspection."
#
# `curl ... | sh` as root grants whoever controls that URL — or anyone who
# can intercept the response — root on this machine, with no record of what
# ran. The script is fetched to disk, its digest printed, and it executes
# only against a pinned digest or an explicit acknowledgement. The artifact
# is kept so what ran can be inspected afterwards.
OLLAMA_INSTALLER="/root/ollama-install.sh"
curl -fsSL --proto '=https' --tlsv1.2 https://ollama.com/install.sh -o "$OLLAMA_INSTALLER"
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
  echo "    WARNING: executing an UNVERIFIED remote installer as root because"
  echo "    OLLAMA_INSTALL_UNVERIFIED=1 was set. Its digest is above."
  sh "$OLLAMA_INSTALLER"
else
  echo "!!! Not executing the installer."
  echo "    Review $OLLAMA_INSTALLER, then re-run with either:"
  echo "      OLLAMA_INSTALLER_SHA256=$OLLAMA_SHA   (pin what you reviewed)"
  echo "      OLLAMA_INSTALL_UNVERIFIED=1           (accept it unverified)"
  exit 1
fi

echo ">>> 3. DOWNLOADING BRAIN MODELS (may take time)..."
echo "    ... Pulling Llama 3 (8B) for fast chat..."
ollama pull llama3 || true
echo "    ... (optional) Pulling nomic-embed-text for embedding services..."
ollama pull nomic-embed-text || true

echo ">>> 4. CONFIGURING FIREWALL (ufw)..."
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp   # SSH
ufw allow 80/tcp   # HTTP
ufw allow 443/tcp  # HTTPS
ufw allow 8000/tcp # Aura Web UI
ufw --force enable

echo ">>> 5. PREPARING DIRECTORY..."
mkdir -p "$APP_DIR"
if [ "$SERVICE_USER" != "root" ]; then
  id -u "$SERVICE_USER" >/dev/null 2>&1 || useradd --system --home "$APP_DIR" --shell /usr/sbin/nologin "$SERVICE_USER"
fi
chown -R "$SERVICE_USER":"$SERVICE_USER" "$APP_DIR" || true

echo ">>> 6. CREATING SYSTEMD SERVICE (Auto-Restart)..."
cat <<'EOF' > /etc/systemd/system/aura.service
[Unit]
Description=Aura AGI Kernel
After=network.target

[Service]
User=__AURA_USER__
WorkingDirectory=/opt/aura
ExecStart=/opt/aura/venv/bin/python3 /opt/aura/aura_main.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
# Add API Keys here or in a .env file
# Environment=OPENAI_API_KEY=your_key_here

[Install]
WantedBy=multi-user.target
EOF

sed -i "s/__AURA_USER__/${SERVICE_USER}/g" /etc/systemd/system/aura.service

systemctl daemon-reload
systemctl enable aura

echo ">>> 7. SETUP COMPLETE.\n    - Brain (Ollama) should be available at http://localhost:11434 if pull succeeded.\n    - Prepare to deploy your code into $APP_DIR and start the service with systemctl start aura" 

exit 0
