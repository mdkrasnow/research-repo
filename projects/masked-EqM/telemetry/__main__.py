"""``python -m telemetry`` entry point.

Kept to one line of logic so that the CLI's behaviour is entirely testable by
calling :func:`telemetry.cli.main` with an argv list -- a module whose only path
to execution is a subprocess is a module whose error handling never gets tested.
"""

from .cli import main

raise SystemExit(main())
