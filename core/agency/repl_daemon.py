import contextlib
import io
import json
import os
import sys
import traceback

from core.runtime.dynamic_execution_gateway import get_dynamic_execution_gateway
from core.runtime.errors import record_degradation

_REPL_EXECUTION_ERRORS = (Exception, KeyboardInterrupt, SystemExit)


def _stateful_execution_enabled() -> bool:
    raw = os.environ.get("AURA_PYTHON_SANDBOX_STATEFUL", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    shared_namespace: dict[str, object] = {}
    stateful = _stateful_execution_enabled()
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(line_buffering=True)
    
    while not sys.stdin.closed:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                size = int(line)
            except ValueError:
                continue
                
            code = sys.stdin.read(size)
            
            out = io.StringIO()
            success = False
            namespace = shared_namespace if stateful else {}
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                try:
                    # Execute in an isolated namespace by default. Stateful
                    # mode is opt-in to avoid contaminating independent tool
                    # calls with hidden prior variables.
                    dynamic_gateway = get_dynamic_execution_gateway()
                    code_object = dynamic_gateway.compile_source(
                        code,
                        filename="<aura_repl>",
                        mode="exec",
                        source="agency.repl_daemon",
                    )
                    dynamic_gateway.execute_code_object(
                        code_object,
                        globals_dict=namespace,
                        source="agency.repl_daemon",
                    )
                    success = True
                except _REPL_EXECUTION_ERRORS:
                    traceback.print_exc(file=out)
            
            result_text = out.getvalue()
            # Send result back
            resp = json.dumps({"success": success, "output": result_text})
            sys.stdout.write(f"{len(resp)}\n{resp}\n")
            sys.stdout.flush()
            
        except (json.JSONDecodeError, TypeError, ValueError) as _e:
            record_degradation('repl_daemon', _e)
            err = json.dumps({"success": False, "output": f"Daemon Error: {str(_e)}"})
            sys.stdout.write(f"{len(err)}\n{err}\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
