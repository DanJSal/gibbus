"""Shared pytest configuration for the gibbus suite.

``GIBBUS_DEBUG=1`` makes unexpected numerical fallbacks fatal while allowing
fallbacks explicitly marked as routine. With ``GIBBUS_DEBUG=1`` enabled, adding
``GIBBUS_DEBUG_STRICT=1`` makes routine fallbacks fatal as well, so strict mode is
useful for targeted diagnostics rather than the ordinary test gate.
"""
