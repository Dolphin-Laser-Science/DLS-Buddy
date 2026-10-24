"""
app/result_store.py
===================

`ResultStore`, the one owner of the controller's stored analysis results and of what is
kept beside them: the run stamps a result's freshness is judged by, the two Cross-Sample
candidate memos, and what the DLS sub-tabs draw of a run (its tab stamp, a superseded
run, a refused run's failure line). Every removal goes through one of its methods, so the
tables cannot drift out of step at a site that names only some of them. The controller
binds each table to an attribute of its own (`Controller.results`, `_dls_run_sig`, ...)
once, and never rebinds it, so every method here changes a table in place. Qt-free
(invariant 5).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Hashable, Iterable, Mapping, NamedTuple, Optional


class ShownRun(NamedTuple):
    """A run a DLS sub-tab draws: the result the tab landed, the signature the tab took
    at dispatch (an opaque tuple only that tab compares), what else the tab keeps of the
    run (`extras`, None when nothing), and whether the run is superseded -- taken from
    the controller's results by a Settings clear (`supersede`) or replaced there before
    the tab landed it (`record_shown`)."""
    result: Any
    sig: Any
    extras: Any
    superseded: bool


class ResultStore:
    """The results, their stamp tables and the candidate memos.

    `results` maps a key `('kind', owner_id, ...)` to the stored result, where the owner
    is a sample id for a per-sample result and an item id for a per-measurement one. Each
    stamp table is keyed exactly like the result it describes:

    - `form_factor_run_sig`: the SLS run signature a stored form-factor fit ran under;
    - `dls_run_sig`: the stamp of a stored Γ-vs-q² or D-vs-c run, or of a Correlogram
      cumulant fit;
    - `distribution_stamps`: the never-reused version number of a stored CONTIN, NNLS or
      lognormal fit.

    - `tab_stamps`: `(result, sig, extras)`, the run the DLS sub-tab that ran a stored
      result landed (`record_shown`): that result, the signature the tab took at dispatch
      and what else it keeps to draw it. Until the tab lands its next run, the result here
      is the one the tab draws, while a run in flight may already have replaced the stored
      result.

    The memos are keyed by the run signature, not by a result: the SLS memo
    `(kind, sample_id, fraction, sig, ...)` and the DLS memo `(sample_id, fraction, sig)`.
    A memo entry needs no result behind it, so it goes only with its sample
    (`prune_owners`) or with everything (`clear`).

    Two tables are keyed like the results but outlive them, so neither is a stamp:
    `superseded`, `(result, sig, extras)` for a run a Settings clear took from `results`
    while its DLS sub-tab keeps drawing it, marked out of date (`supersede`); and
    `run_failures`, the message of a run the controller refused, shown beside the
    measurement until the tab's next Run or a clear in its scope. Each goes by its own
    rule (`forget_shown`, `replace_failures`), and with its owner (`prune_owners`) or with
    everything (`clear`). No candidate builder or memo reads either.

    The tables stay parallel dicts rather than one record per key, and the store takes no
    lock: one worker thread writes while the main thread reads, and the order of the
    writes in `put` and `drop` is what keeps a reader from seeing a result without its
    stamp.
    """

    #: The stamp tables, by attribute name; `put`'s `stamps` names them.
    STAMP_SLOTS = ('form_factor_run_sig', 'dls_run_sig', 'distribution_stamps', 'tab_stamps')
    #: The candidate memos, by attribute name, with the key position of the sample id
    #: that owns each entry.
    MEMO_OWNER_POSITION = {'sls_candidate_cache': 1, 'dls_candidate_cache': 0}

    def __init__(self) -> None:
        self.results: Dict[tuple, Any] = {}
        self.form_factor_run_sig: Dict[tuple, tuple] = {}
        self.dls_run_sig: Dict[tuple, tuple] = {}
        self.distribution_stamps: Dict[tuple, int] = {}
        self.tab_stamps: Dict[tuple, tuple] = {}
        self.sls_candidate_cache: Dict[tuple, Any] = {}
        self.dls_candidate_cache: Dict[tuple, Any] = {}
        self.superseded: Dict[tuple, tuple] = {}
        self.run_failures: Dict[tuple, str] = {}

    def _stamp_tables(self):
        return [getattr(self, name) for name in self.STAMP_SLOTS]

    def put(self, key: tuple, value: Any, *,
            stamps: Optional[Mapping[str, Any]] = None) -> None:
        """Store `value` under `key`, writing each stamp in `stamps` (a stamp table's name
        to its value) first and the result last. A candidate builder on the main thread
        reads the result only while its stamp matches; with the stamp written first, a
        result it can see is always stamped, where the reverse order would let it read a
        just-published fit as unstamped and treat it as stale. On a re-run the opposite
        window remains: between the last stamp write and the result write the new stamp
        sits beside the previous result. The tables are looked up first, so those two
        writes are adjacent dict stores with no loop back-edge or call between them, the
        points at which CPython switches threads; a stamp written before the last one has
        no such guarantee, and every publish passes a single stamp. A reader that lands in
        the window reads the previous result under the new stamp; closing it needs one
        atomic record per key, which the store does not keep. A plain
        `results[key] = value` is a `put` with no stamps."""
        stamps = dict(stamps or {})
        unknown = set(stamps) - set(self.STAMP_SLOTS)
        if unknown:
            raise KeyError(f'no stamp table named {sorted(unknown)}; '
                           f'the tables are {self.STAMP_SLOTS}')
        results = self.results
        writes = [(getattr(self, name), stamp) for name, stamp in stamps.items()]
        if not writes:
            results[key] = value
        else:
            *earlier, (last_table, last_stamp) = writes
            for table, stamp in earlier:
                table[key] = stamp
            # No statement may come between these two stores (the docstring's window).
            last_table[key] = last_stamp
            results[key] = value

    def get(self, key: tuple, default: Any = None) -> Any:
        """The result stored under `key`, else `default`."""
        return self.results.get(key, default)

    def drop(self, key: tuple) -> None:
        """Remove the result under `key` and every stamp of that key, the result first, so
        a reader never sees the result without its stamp. Absent entries are skipped."""
        self.results.pop(key, None)
        for table in self._stamp_tables():
            table.pop(key, None)

    def drop_where(self, pred: Callable[[tuple], bool]) -> None:
        """`drop` every key of the results and the stamp tables that `pred` accepts."""
        keys = dict.fromkeys(k for table in (self.results, *self._stamp_tables())
                             for k in list(table) if pred(k))
        for key in keys:
            self.drop(key)

    def prune_owners(self, live: Callable[[Hashable], bool]) -> None:
        """Remove every entry whose owner `live` rejects: a result, stamp, superseded run or
        failure line keyed `('kind', owner, ...)` (a key that is not such a tuple is kept),
        and a memo entry
        whose sample id, at its memo's owner position, is rejected. A memo entry of a
        sample `live` accepts stays whether or not any result is stored for it."""
        for table in (self.results, *self._stamp_tables(), self.superseded,
                      self.run_failures):
            for key in [k for k in list(table)
                        if isinstance(k, tuple) and len(k) > 1 and not live(k[1])]:
                table.pop(key, None)
        for name, pos in self.MEMO_OWNER_POSITION.items():
            memo = getattr(self, name)
            for key in [k for k in list(memo)
                        if isinstance(k, tuple) and len(k) > pos and not live(k[pos])]:
                memo.pop(key, None)

    def clear(self) -> None:
        """Empty every table: the results, the stamps, both memos, the superseded runs and
        the failure lines."""
        self.results.clear()
        for table in self._stamp_tables():
            table.clear()
        for name in self.MEMO_OWNER_POSITION:
            getattr(self, name).clear()
        self.superseded.clear()
        self.run_failures.clear()

    # ---- what the DLS sub-tabs draw of a run ----
    def shown(self, key: tuple) -> Optional[ShownRun]:
        """The run the DLS sub-tab that ran `key` draws: the run its tab stamp recorded
        (`record_shown`), with that run's own signature and extras, else the superseded
        run, else None. A stored result with no tab stamp (no tab ran it) is not drawn.
        While a worker's run has replaced the stored result and the tab has not landed it,
        the tab stamp still serves the run the tab landed before, so a redraw then never
        pairs the new result with the previous run's signature. Each table is read once,
        with `get`, since the worker may drop the key between two reads."""
        stamp = self.tab_stamps.get(key)
        if stamp is not None:
            return ShownRun(*stamp, False)
        kept = self.superseded.get(key)
        return ShownRun(*kept, True) if kept is not None else None

    def shown_of_kind(self, kind: str) -> Dict[tuple, ShownRun]:
        """`shown` for every key of the result kind `kind` a DLS sub-tab draws. Iterates a
        snapshot of each table's keys (the worker may drop a key meanwhile)."""
        keys = dict.fromkeys(k for k in list(self.tab_stamps) + list(self.superseded)
                             if k[0] == kind)
        found = {k: self.shown(k) for k in keys}
        return {k: run for k, run in found.items() if run is not None}

    def record_shown(self, key: tuple, result: Any, sig: Any, extras: Any = None) -> None:
        """The DLS sub-tab that ran `key` landed `result`, dispatched under `sig`: record
        it, with `sig` and `extras`, as the tab stamp beside the stored result while it is
        that result, else keep it as the key's superseded run (the controller no longer
        holds it, and the tab draws it out of date) and take the stamp off the stored
        result, which the tab no longer draws. Either way it replaces what the tab drew of
        `key` before, and `shown` serves it until the tab lands its next run or the stamp
        goes (`drop`, `forget_shown`)."""
        self.superseded.pop(key, None)
        if key in self.results and self.results[key] is result:
            self.tab_stamps[key] = (result, sig, extras)
        else:
            self.tab_stamps.pop(key, None)
            self.superseded[key] = (result, sig, extras)

    def supersede(self, key: tuple) -> None:
        """`drop` the result under `key`, first keeping the run its tab stamp recorded as
        the key's superseded run when a DLS sub-tab draws it (its tab stamp is stored), so
        the tab goes on drawing what it drew."""
        stamp = self.tab_stamps.get(key)
        if key in self.results and stamp is not None:
            self.superseded[key] = stamp
        self.drop(key)

    def forget_shown(self, key: tuple) -> None:
        """Remove what a DLS sub-tab draws of `key` beside the stored result: its tab stamp,
        its superseded run and its failure line. The stored result, if any, stays."""
        self.tab_stamps.pop(key, None)
        self.superseded.pop(key, None)
        self.run_failures.pop(key, None)

    def held(self, key: tuple) -> Optional[tuple]:
        """`(result, stamps)` stored under `key`, else None: what `restore` puts back,
        `stamps` holding every stamp table's entry of `key` by table name, the run stamps
        and the tab stamp alike."""
        if key not in self.results:
            return None
        stamps = {name: getattr(self, name)[key] for name in self.STAMP_SLOTS
                  if key in getattr(self, name)}
        return self.results[key], stamps

    def restore(self, key: tuple, held: Optional[tuple]) -> None:
        """Put back what `held` read under `key`: first remove each stamp table's entry the
        held run did not have, then write its stamps before its result (`put`'s order), so
        the held result is never left beside a stamp of a later run. None restores
        nothing."""
        if held is None:
            return
        result, stamps = held
        for name in self.STAMP_SLOTS:
            if name not in stamps:
                getattr(self, name).pop(key, None)
        self.put(key, result, stamps=stamps)

    def failure(self, key: tuple) -> Optional[str]:
        """The failure line of the refused run `key`, else None."""
        return self.run_failures.get(key)

    def replace_failures(self, kinds: Iterable[str], failures: Mapping[tuple, str]) -> None:
        """Replace every failure line of the result kinds `kinds` with `failures`, as a
        tab's Run does: a line carries no signature, so a kept one could not say whether
        its cause has since been fixed."""
        kinds = set(kinds)
        for key in [k for k in self.run_failures if k[0] in kinds]:
            del self.run_failures[key]
        self.run_failures.update(failures)
