import os
import sys


def run_app() -> None:
    from streamlit.web import cli as st_cli

    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(st_cli.main())
