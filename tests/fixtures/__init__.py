"""Shared helpers for the DLS Buddy regression suite.

These modules are importable from any test as ``from fixtures.<name> import ...``
(the ``tests`` dir is on ``pythonpath`` — see ``pytest.ini``). They exist to hold
the code that was previously copy-pasted across the three retired validation
scripts: the synthetic forward-model builders, the committed-data path helpers, and
the SLS/DDLS forward-model builders.
"""
