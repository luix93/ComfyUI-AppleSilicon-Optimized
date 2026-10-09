"""Runtime diagnostics for the combined H3 node (no separate node/server)."""
import logging

class NodeStatus:
    def __init__(self, node_id=None):
        self._last = None

    def show(self, text, level=logging.INFO):
        plain = text.replace(' · ', ' | ').encode('ascii', 'ignore').decode().strip()
        logging.getLogger('AppleSilicon-Optimized/H3Veda').log(level, '%s', plain)

    def warn(self, text):
        self.show(text, logging.WARNING)
