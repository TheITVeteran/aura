#!/usr/bin/env bash
# ============================================================================
# Quick status check / management commands for Aura on Oracle Cloud
# ============================================================================
# Usage:
#   ./cloud/manage.sh <VM_IP> status     # Check Aura + Ollama status
#   ./cloud/manage.sh <VM_IP> logs       # Tail Aura logs
#   ./cloud/manage.sh <VM_IP> restart    # Restart Aura
#   ./cloud/manage.sh <VM_IP> stop       # Stop Aura
#   ./cloud/manage.sh <VM_IP> ssh        # SSH into the VM
#   ./cloud/manage.sh <VM_IP> ollama     # Check Ollama status/models
# ============================================================================
set -euo pipefail

VM_IP="${1:?Usage: ./cloud/manage.sh <VM_IP> <command>}"
CMD="${2:-status}"
SSH_USER="ubuntu"
SSH_KEY="${HOME}/.ssh/aura-oracle.key"

# Same class as cloud/deploy.sh (CP126 critical, "Deployment disables SSH
# host verification"). StrictHostKeyChecking=no accepts ANY host key, and
# this script runs sudo commands and reads logs — on a cloud provider IPs
# are recycled, so the machine at a remembered address may not be yours.
#
# accept-new keeps first-connection ergonomics but still refuses when a host
# key CHANGES, which is the case that matters.
SSH_STRICT="${AURA_DEPLOY_SSH_STRICT:-accept-new}"
SSH_OPTS="-i ${SSH_KEY} -o StrictHostKeyChecking=${SSH_STRICT}"
if [[ -n "${AURA_DEPLOY_KNOWN_HOSTS:-}" ]]; then
  SSH_OPTS="${SSH_OPTS} -o UserKnownHostsFile=${AURA_DEPLOY_KNOWN_HOSTS}"
fi

case "${CMD}" in
    status)
        echo "=== Aura Service ==="
        ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" "sudo systemctl status aura --no-pager -l | head -20"
        echo ""
        echo "=== API State ==="
        ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" "curl -s http://127.0.0.1:8000/api/state 2>/dev/null | python3 -m json.tool" || echo "NOT RESPONDING"
        echo ""
        echo "=== Ollama ==="
        ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" "sudo systemctl status ollama --no-pager | head -5"
        ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" "ollama list 2>/dev/null" || echo "Ollama not responding"
        echo ""
        echo "=== Resources ==="
        ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" "free -h | head -2 && echo '' && df -h / | tail -1"
        ;;
    logs)
        ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" "sudo journalctl -u aura -f --no-pager"
        ;;
    restart)
        ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" "sudo systemctl restart aura"
        echo "Aura restarted. Waiting 10s..."
        sleep 10
        ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" "curl -s http://127.0.0.1:8000/api/state 2>/dev/null | python3 -m json.tool" || echo "Still starting..."
        ;;
    stop)
        ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" "sudo systemctl stop aura"
        echo "Aura stopped."
        ;;
    ssh)
        ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}"
        ;;
    ollama)
        echo "=== Ollama Models ==="
        ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" "ollama list"
        echo ""
        echo "=== Ollama Service ==="
        ssh ${SSH_OPTS} "${SSH_USER}@${VM_IP}" "sudo systemctl status ollama --no-pager | head -10"
        ;;
    *)
        echo "Unknown command: ${CMD}"
        echo "Available: status, logs, restart, stop, ssh, ollama"
        exit 1
        ;;
esac
