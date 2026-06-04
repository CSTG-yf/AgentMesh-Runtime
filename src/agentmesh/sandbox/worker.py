import contextlib
import io
import json
import sys

for line in sys.stdin:
    request = json.loads(line)
    code = request["code"]
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = 0
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(code, {"__name__": "__sandbox__"})
    except Exception as exc:
        exit_code = 1
        print(repr(exc), file=stderr)
    print(
        json.dumps(
            {
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "exit_code": exit_code,
            }
        ),
        flush=True,
    )
