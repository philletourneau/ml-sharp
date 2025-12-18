"""Networking utilities (best-effort).

These helpers are designed to improve robustness in common environments
like Windows corporate networks with HTTPS interception.
"""

from __future__ import annotations

import os


_TRUSTSTORE_INSTALLED = False


def install_system_certificates() -> bool:
    """Best-effort install of OS trust store for TLS verification.

    Returns:
        True if successfully installed, False otherwise.
    """
    global _TRUSTSTORE_INSTALLED
    if _TRUSTSTORE_INSTALLED:
        return True

    if os.environ.get("SHARP_DISABLE_TRUSTSTORE", "").strip() in {"1", "true", "yes"}:
        return False

    try:
        import truststore  # type: ignore[import-not-found]

        truststore.inject_into_ssl()
        _TRUSTSTORE_INSTALLED = True
        return True
    except Exception:
        return False

