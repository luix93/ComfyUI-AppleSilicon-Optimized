"""User-visible status: text on the Veda node, mirrored to the log.

Uses ComfyUI's native per-node progress text (`send_progress_text`), so no
frontend extension is needed. Safe to call from the sampling thread and
when ComfyUI's server is absent (scripts, tests).
"""

from __future__ import annotations

import logging

_LOG = logging.getLogger('veda')


class NodeStatus:
    """Writes status lines onto one node, skipping exact repeats."""

    def __init__(self, node_id: str | None):
        self.node_id = node_id
        self._last = None

    def show(self, text: str, level: int = logging.INFO) -> None:
        # Status text carries no emoji, so the only non-ASCII left is the
        # ' · ' separator; the log still gets plain ASCII, because Windows
        # consoles and log files may not be UTF-8. The strip stays as the
        # backstop that keeps that promise whatever a caller passes in.
        plain = text.replace(' · ', ' | ').encode('ascii', 'ignore')
        plain = plain.decode().strip()
        _LOG.log(level, 'Veda: %s', plain)
        if text == self._last or self.node_id is None:
            return
        self._last = text
        try:
            from server import PromptServer  # pylint: disable=import-outside-toplevel
            PromptServer.instance.send_progress_text(text, self.node_id)
        except Exception:  # no server (scripts, tests): the log is enough
            pass

    def warn(self, text: str) -> None:
        self.show(text, logging.WARNING)
