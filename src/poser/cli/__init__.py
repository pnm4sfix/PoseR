"""cli/__init__.py"""
# Do NOT import poser.cli.main here.
# Doing so puts the module in sys.modules before Python executes it as
# __main__ (via -m poser.cli.main), which causes the RuntimeWarning and
# means app() is never called — the process exits 0 silently.

__all__: list = []
