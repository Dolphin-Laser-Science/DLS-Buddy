"""
gui/stub_module.py
==================

A placeholder widget for a module that is part of the agreed shell but not yet
built. It states what will live there, so the tab structure is visible and
navigable from the start while the modules are filled in one at a time.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class StubModule(QtWidgets.QWidget):
    """A titled placeholder describing a module's planned contents."""

    def __init__(self, title: str, planned: str, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        heading = QtWidgets.QLabel(f'<h2>{title}</h2>')
        layout.addWidget(heading)

        tag = QtWidgets.QLabel('Planned module — not yet implemented.')
        tag.setStyleSheet('color:#b06000; font-style: italic;')
        layout.addWidget(tag)

        body = QtWidgets.QLabel(planned)
        body.setWordWrap(True)
        body.setStyleSheet('color:#444;')
        layout.addWidget(body)

        layout.addStretch(1)
