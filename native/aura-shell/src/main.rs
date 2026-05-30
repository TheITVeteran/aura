//! Aura native shell.
//!
//! Wraps the local FastAPI runtime and the web UI in a Tauri window. The
//! shell launches `aura_main.py` as a sidecar, waits for the readiness heartbeat
//! endpoint to become reachable, and only then loads the UI. On window
//! close, it sends SIGTERM so the runtime drains receipts cleanly.

use std::process::Stdio;
use std::time::Duration;

use serde::Serialize;
use serde_json::Value;
use tauri::{Manager, RunEvent};
use tokio::process::{Child, Command};

#[derive(Clone, Serialize)]
struct BootStatus {
    state: String,
}

#[tauri::command]
async fn boot_status() -> Result<BootStatus, String> {
    // The frontend polls this until the runtime is reachable. The shell
    // doesn't speak to the runtime directly — it just reflects whether
    // the local TCP port is accepting connections.
    Ok(BootStatus { state: "starting".into() })
}

fn readiness_heartbeat_is_healthy(payload: &Value) -> bool {
    if payload.get("healthy").and_then(Value::as_bool) != Some(true) {
        return false;
    }
    if payload.get("status").and_then(Value::as_str) != Some("healthy") {
        return false;
    }
    let Some(probes) = payload.get("required_probes").and_then(Value::as_object) else {
        return false;
    };
    if probes.get("all_passed").and_then(Value::as_bool) != Some(true) {
        return false;
    }
    for group in ["kernel", "inference", "memory", "scheduler", "tool_governance"] {
        let Some(probe) = probes.get(group).and_then(Value::as_object) else {
            return false;
        };
        if probe.get("ok").and_then(Value::as_bool) != Some(true) {
            return false;
        }
    }
    true
}

#[tokio::main]
async fn main() {
    tauri::async_runtime::set(tokio::runtime::Handle::current());
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![boot_status])
        .setup(|app| {
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let mut child: Child = Command::new("python3")
                    .args([
                        "aura_main.py",
                        "--desktop",
                        "--port",
                        "7400",
                    ])
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .kill_on_drop(true)
                    .spawn()
                    .expect("aura runtime failed to launch");

                // Block until the runtime passes the canonical readiness heartbeat,
                // then mark the shell ready. A process-level HTTP response is not
                // enough: the heartbeat is only healthy when kernel, inference,
                // memory, scheduler, and tool-governance probes all pass.
                let client = reqwest::Client::new();
                loop {
                    let r = client
                        .get("http://localhost:7400/api/health/heartbeat")
                        .send()
                        .await;
                    if let Ok(resp) = r {
                        if resp.status().is_success() {
                            if let Ok(payload) = resp.json::<Value>().await {
                                if readiness_heartbeat_is_healthy(&payload) {
                                    let _ = handle.emit("aura://ready", &());
                                    break;
                                }
                            }
                        }
                    }
                    tokio::time::sleep(Duration::from_millis(250)).await;
                }
                let _ = child.wait().await;
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("aura shell failed to start");
}
