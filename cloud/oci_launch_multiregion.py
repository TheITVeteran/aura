#!/usr/bin/env python3
"""
OCI ARM Instance Launcher — Multi-Region Edition
Cycles through regions to find Ampere A1 capacity.
Creates networking on-the-fly in each region if needed.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import oci

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.atomic_writer import atomic_write_text  # noqa: E402
from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402

# ─── Configuration ──────────────────────────────────────────
# Configuration
COMPARTMENT_ID = os.environ.get("OCI_COMPARTMENT_ID", "")
SSH_KEY_FILE = os.path.expanduser("~/.ssh/aura-oracle.key.pub")

SHAPE = "VM.Standard.A1.Flex"
OCPUS = 4
MEMORY_GB = 24
BOOT_VOLUME_GB = 200
DISPLAY_NAME = "aura-cloud"

RETRY_INTERVAL = 45   # seconds between attempts (faster with multi-region)
MAX_ATTEMPTS = int(os.environ.get("OCI_MAX_ATTEMPTS", "0"))  # 0 = run until interrupted
STATE_FILE = Path(tempfile.gettempdir()) / "oci_multi_region_state.json"
CLOUD_IP_FILE = Path(tempfile.gettempdir()) / "aura_cloud_ip.txt"
_OCI_RECOVERABLE_ERRORS = (OSError, RuntimeError, ValueError, KeyError, IndexError, json.JSONDecodeError)
_OCI_IMAGE_ERRORS = (oci.exceptions.ServiceError,) + _OCI_RECOVERABLE_ERRORS

# Regions most likely to have free-tier A1 capacity
# Ordered by typical availability (less popular = more capacity)
REGIONS = [
    "us-sanjose-1",
    "us-phoenix-1",
    "us-ashburn-1",
    "ca-toronto-1",
    "ca-montreal-1",
    "eu-frankfurt-1",
    "eu-amsterdam-1",
    "uk-london-1",
    "ap-tokyo-1",
    "ap-osaka-1",
    "ap-sydney-1",
    "ap-melbourne-1",
    "sa-saopaulo-1",
    "me-jeddah-1",
    "af-johannesburg-1",
    "ap-singapore-1",
    "ap-seoul-1",
    "eu-marseille-1",
    "eu-zurich-1",
    "eu-milan-1",
]

# ─── Load/save state (track which regions have networking set up) ───
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except _OCI_RECOVERABLE_ERRORS as exc:
            print(f"[!] Failed to load state {STATE_FILE}: {type(exc).__name__}: {exc}; starting fresh.")
    return {"networks": {}, "total_attempts": 0}

def save_state(state):
    atomic_write_text(STATE_FILE, json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

# ─── Setup ──────────────────────────────────────────────────
config = oci.config.from_file()




def _load_ssh_key() -> str:
    """Read the SSH public key on use, not on import.

    This was a module-level ``with open(...)``, so importing the module on a
    machine without the key raised FileNotFoundError — the same class of
    defect as the provisioning loop: an import must not touch the
    filesystem, and must not be able to fail for environmental reasons.

    Reading it per launch also means a rotated key is picked up without a
    restart.
    """
    with open(SSH_KEY_FILE, encoding="utf-8") as handle:
        return handle.read().strip()


def get_clients(region):
    """Get OCI clients for a specific region."""
    cfg = dict(config)
    cfg["region"] = region
    return (
        oci.core.ComputeClient(cfg),
        oci.core.VirtualNetworkClient(cfg),
        oci.identity.IdentityClient(cfg),
        cfg
    )

def ensure_networking(region, vn_client, identity_client):
    """Create VCN + subnet in a region if not already done."""
    if region in state["networks"]:
        # Verify it still exists
        try:
            vn_client.get_subnet(state["networks"][region]["subnet_id"])
            return state["networks"][region]
        except oci.exceptions.ServiceError:
            print("    [!] Cached network gone, recreating...")
            del state["networks"][region]

    print(f"    [*] Creating networking in {region}...")

    # Create VCN
    vcn = vn_client.create_vcn(
        oci.core.runtime.models.CreateVcnDetails(
            compartment_id=COMPARTMENT_ID,
            cidr_block="10.0.0.0/16",
            display_name=f"aura-vcn-{region}",
        )
    ).data

    # Wait for VCN
    vcn = oci.wait_until(vn_client, vn_client.get_vcn(vcn.id), 'lifecycle_state', 'AVAILABLE').data

    # Create internet gateway
    igw = vn_client.create_internet_gateway(
        oci.core.runtime.models.CreateInternetGatewayDetails(
            compartment_id=COMPARTMENT_ID,
            vcn_id=vcn.id,
            display_name="aura-igw",
            is_enabled=True,
        )
    ).data
    igw = oci.wait_until(vn_client, vn_client.get_internet_gateway(igw.id), 'lifecycle_state', 'AVAILABLE').data

    # Update default route table to use IGW
    rt = vn_client.get_route_table(vcn.default_route_table_id).data
    vn_client.update_route_table(
        rt.id,
        oci.core.runtime.models.UpdateRouteTableDetails(
            route_rules=[
                oci.core.runtime.models.RouteRule(
                    destination="0.0.0.0/0",
                    destination_type="CIDR_BLOCK",
                    network_entity_id=igw.id,
                )
            ]
        )
    )

    # CP126 (critical): "Provisioner exposes SSH and Aura backend globally.
    # The default security list is replaced with ingress from 0.0.0.0/0 to
    # SSH, port 8000, HTTP, and HTTPS, plus unrestricted egress. This exposes
    # the backend directly and broadens SSH before host hardening."
    #
    # Port 8000 is the Aura backend itself — the API that drives the runtime —
    # published to the entire internet at provisioning time, before anything
    # on the box has been hardened. SSH open to 0.0.0.0/0 on a fresh cloud
    # image is the other half.
    #
    # Ingress is now sourced from an explicit allowlist, and port 8000 is NOT
    # opened by default: the backend is reachable through the TLS front door
    # or an SSH tunnel, both of which already work. Opening it requires
    # saying so.
    admin_cidr = os.environ.get("AURA_ADMIN_CIDR", "").strip()
    if not admin_cidr:
        raise SystemExit(
            "AURA_ADMIN_CIDR is required: the CIDR permitted to reach SSH.\n"
            "  Your address:  curl -s https://checkip.amazonaws.com\n"
            "  Then:          AURA_ADMIN_CIDR=<that-ip>/32\n"
            "Refusing to open SSH to 0.0.0.0/0 on a freshly provisioned host."
        )

    public_web = os.environ.get("AURA_PUBLIC_WEB", "1").strip() not in {"0", "false", "no"}
    backend_cidr = os.environ.get("AURA_BACKEND_CIDR", "").strip()

    def _tcp_rule(source: str, port: int):
        return oci.core.runtime.models.IngressSecurityRule(
            protocol="6", source=source,
            tcp_options=oci.core.runtime.models.TcpOptions(
                destination_port_range=oci.core.runtime.models.PortRange(
                    min=port, max=port))
        )

    ingress_rules = [_tcp_rule(admin_cidr, 22)]
    if public_web:
        ingress_rules.append(_tcp_rule("0.0.0.0/0", 80))
        ingress_rules.append(_tcp_rule("0.0.0.0/0", 443))
    if backend_cidr:
        # Deliberate direct exposure of the backend port, scoped to a CIDR.
        ingress_rules.append(_tcp_rule(backend_cidr, 8000))

    print(f"    ingress: SSH<-{admin_cidr}"
          f"{' , 80/443<-0.0.0.0/0' if public_web else ''}"
          f"{f' , 8000<-{backend_cidr}' if backend_cidr else ' , 8000 CLOSED'}")

    sl = vn_client.get_security_list(vcn.default_security_list_id).data
    vn_client.update_security_list(
        sl.id,
        oci.core.runtime.models.UpdateSecurityListDetails(
            ingress_security_rules=ingress_rules,
            egress_security_rules=[
                oci.core.runtime.models.EgressSecurityRule(
                    protocol="all", destination="0.0.0.0/0")
            ]
        )
    )

    # Get availability domain
    ads = identity_client.list_availability_domains(compartment_id=COMPARTMENT_ID).data
    ad_name = ads[0].name  # Use first AD

    # Create subnet
    subnet = vn_client.create_subnet(
        oci.core.runtime.models.CreateSubnetDetails(
            compartment_id=COMPARTMENT_ID,
            vcn_id=vcn.id,
            cidr_block="10.0.1.0/24",
            display_name="aura-subnet",
            availability_domain=ad_name,
            route_table_id=vcn.default_route_table_id,
            security_list_ids=[vcn.default_security_list_id],
        )
    ).data
    subnet = oci.wait_until(vn_client, vn_client.get_subnet(subnet.id), 'lifecycle_state', 'AVAILABLE').data

    info = {
        "vcn_id": vcn.id,
        "subnet_id": subnet.id,
        "ad_name": ad_name,
        "igw_id": igw.id,
    }
    state["networks"][region] = info
    save_state(state)
    print(f"    [✓] Networking ready in {region} (AD: {ad_name})")
    return info

def find_image(compute_client, region):
    """Find Ubuntu 24.04 ARM image in region."""
    try:
        images = compute_client.list_images(
            compartment_id=COMPARTMENT_ID,
            operating_system="Canonical Ubuntu",
            operating_system_version="24.04",
            shape=SHAPE,
            sort_by="TIMECREATED",
            sort_order="DESC",
            limit=1
        ).data
        if images:
            return images[0].id
        # Fallback to 22.04
        images = compute_client.list_images(
            compartment_id=COMPARTMENT_ID,
            operating_system="Canonical Ubuntu",
            operating_system_version="22.04",
            shape=SHAPE,
            sort_by="TIMECREATED",
            sort_order="DESC",
            limit=1
        ).data
        return images[0].id if images else None
    except _OCI_IMAGE_ERRORS:
        return None

def try_launch(region):
    """Attempt to launch instance in a specific region."""
    compute, vn_client, identity, cfg = get_clients(region)

    # Ensure networking
    try:
        net = ensure_networking(region, vn_client, identity)
    except oci.exceptions.ServiceError as e:
        if "NotAuthorizedOrNotFound" in str(e) or "limit" in str(e.message).lower():
            print(f"    [!] Region {region} not subscribed or limited. Skipping.")
            return "skip"
        raise

    # Find image
    image_id = find_image(compute, region)
    if not image_id:
        print(f"    [!] No ARM image in {region}. Skipping.")
        return "skip"

    # Build launch details
    launch_details = oci.core.runtime.models.LaunchInstanceDetails(
        compartment_id=COMPARTMENT_ID,
        availability_domain=net["ad_name"],
        display_name=DISPLAY_NAME,
        shape=SHAPE,
        shape_config=oci.core.runtime.models.LaunchInstanceShapeConfigDetails(
            ocpus=float(OCPUS),
            memory_in_gbs=float(MEMORY_GB)
        ),
        source_details=oci.core.runtime.models.InstanceSourceViaImageDetails(
            image_id=image_id,
            boot_volume_size_in_gbs=BOOT_VOLUME_GB
        ),
        create_vnic_details=oci.core.runtime.models.CreateVnicDetails(
            subnet_id=net["subnet_id"],
            assign_public_ip=True
        ),
        metadata={"ssh_authorized_keys": _load_ssh_key()}
    )

    response = compute.launch_instance(launch_details)
    instance = response.data

    print()
    print("═" * 55)
    print(f"  ✓ INSTANCE CREATED IN {region.upper()}!")
    print("═" * 55)
    print(f"  Instance ID: {instance.id}")

    # Wait for RUNNING
    print("  [*] Waiting for RUNNING state...")
    oci.wait_until(
        compute, compute.get_instance(instance.id),
        'lifecycle_state', 'RUNNING',
        max_interval_seconds=15, max_wait_seconds=600
    )
    print("  [✓] Instance is RUNNING!")

    # Get public IP
    time.sleep(10)
    vnics = compute.list_vnic_attachments(
        compartment_id=COMPARTMENT_ID, instance_id=instance.id
    ).data
    public_ip = None
    for va in vnics:
        if va.lifecycle_state == "ATTACHED":
            vnic = vn_client.get_vnic(va.vnic_id).data
            if vnic.public_ip:
                public_ip = vnic.public_ip
                break

    if public_ip:
        print(f"\n  ✓ PUBLIC IP: {public_ip}")
        print(f"  ✓ REGION: {region}")
        print(f"\n  SSH: ssh -i ~/.ssh/aura-oracle.key ubuntu@{public_ip}\n")
        with open(CLOUD_IP_FILE, "w", encoding="utf-8") as f:
            f.write(f"{public_ip}\n{region}\n")
    else:
        print("  [!] Could not get public IP. Check console.")

    get_subprocess_gateway().run(
        ["afplay", "/System/Library/Sounds/Glass.aiff"],
        cwd=ROOT,
        timeout=10,
        capture_output=True,
        offline_tooling=True,
        source="maintenance_tooling:oci_multiregion_notify",
    )
    return "success"

# ─── Main rotation loop ────────────────────────────────────

skip_regions = set()
region_idx = 0

def main() -> int:
    """Run the multi-region rotation.

    CP126 (high): "Import can provision infrastructure across twenty
    regions." The banner, the SSH-key read, the state load and the entire
    rotation loop sat at module level with no ``__main__`` guard, so
    merely importing this module started launching cloud instances — and
    ended by calling ``sys.exit(1)``, taking the importing process with
    it.

    Test collection, a static analyser, or any ``from cloud import ...``
    was enough. Spending money and killing the interpreter are both
    things an import must never do.
    """
    state = load_state()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   OCI ARM LAUNCHER — MULTI-REGION ROTATION          ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Regions: {len(REGIONS)}")
    print(f"  Previous attempts: {state['total_attempts']}")
    print(f"  Regions with networking: {len(state['networks'])}")
    print()
    print("[*] Starting multi-region rotation. Ctrl+C to stop.\n")
    while MAX_ATTEMPTS <= 0 or state["total_attempts"] < MAX_ATTEMPTS:
        region = REGIONS[region_idx % len(REGIONS)]
        region_idx += 1

        if region in skip_regions:
            continue

        state["total_attempts"] += 1
        attempt = state["total_attempts"]
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] #{attempt} → {region}", end=" ", flush=True)

        try:
            result = try_launch(region)
            if result == "success":
                save_state(state)
                sys.exit(0)
            elif result == "skip":
                skip_regions.add(region)

        except oci.exceptions.ServiceError as e:
            if e.status == 500 and "capacity" in str(e.message).lower():
                print("Out of capacity.")
            elif e.status == 429:
                print("Rate limited. Extra wait...")
                time.sleep(RETRY_INTERVAL)
            elif "limit" in str(e.message).lower() or "quota" in str(e.message).lower():
                print(f"Limit reached: {e.message[:80]}")
                skip_regions.add(region)
            elif "NotAuthorizedOrNotFound" in str(e.code):
                print("Not subscribed. Skipping.")
                skip_regions.add(region)
            else:
                print(f"Error ({e.status}): {e.message[:80]}")

        except _OCI_RECOVERABLE_ERRORS as exc:
            print(f"Error: {type(exc).__name__}: {str(exc)[:80]}")

        save_state(state)

        # Shorter sleep when rotating (we're hitting different regions)
        if len(REGIONS) - len(skip_regions) > 3:
            time.sleep(RETRY_INTERVAL // len(REGIONS) * 3 + 5)  # ~10-15s between regions
        else:
            time.sleep(RETRY_INTERVAL)
    print(f"[!] Max attempts ({MAX_ATTEMPTS}) reached. Giving up.")
    return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
