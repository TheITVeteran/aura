import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./shell.css";

class ShellErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("Aura shell render failure", error, info);
    try {
      fetch("/api/ui/shell-error", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          error: error instanceof Error ? error.message : String(error),
          stack: error instanceof Error ? error.stack : "",
          component_stack: info?.componentStack || "",
        }),
      }).catch(() => {});
    } catch {
      // The fallback UI must never depend on the backend being reachable.
    }
  }

  render() {
    if (this.state.error) {
      const message =
        this.state.error instanceof Error ? this.state.error.message : String(this.state.error);
      return (
        <main className="shell-crash">
          <section className="shell-crash-card">
            <p className="shell-crash-kicker">AURA SHELL RECOVERY</p>
            <h1>Desktop shell recovered from a render fault.</h1>
            <p>
              The runtime may still be alive. Reload the shell or open the legacy UI while the
              backend health lane continues reporting real status.
            </p>
            <pre>{message}</pre>
            <div className="shell-crash-actions">
              <button type="button" onClick={() => window.location.reload()}>
                Reload Shell
              </button>
              <button type="button" onClick={() => window.location.assign("/")}>
                Open Legacy UI
              </button>
              <button type="button" onClick={() => window.open("/api/health/boot", "_blank")}>
                Boot Health
              </button>
            </div>
          </section>
        </main>
      );
    }

    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ShellErrorBoundary>
      <App />
    </ShellErrorBoundary>
  </React.StrictMode>,
);
