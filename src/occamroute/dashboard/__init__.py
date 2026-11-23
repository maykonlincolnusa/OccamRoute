"""OccamRoute observability dashboard package."""
from __future__ import annotations

from occamroute.dashboard.app import app, create_app, export_standalone_html

__all__ = ["app", "create_app", "export_standalone_html"]
