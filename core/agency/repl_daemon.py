import contextlib
import io
import json
import sys
import traceback

from core.runtime.errors import record_degradation

_REPL_EXECUTION_ERRORS = (Exception, KeyboardInterrupt, SystemExit)


def main() -> None:
    namespace: dict[str, object] = {}
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
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                try:
                    # Execute the code in the shared namespace
                    exec(code, namespace)  # nosec
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
