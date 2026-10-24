"""
gui/dls_module.py
=================

One import path for every name the `gui.dls` package defines for the DLS tab, re-exported
for the importers that use it. The code lives in `gui/dls/`, one module per sub-tab.
"""
# ruff: noqa: F401
from __future__ import annotations

from plotting.plots import display_factor as _disp_factor
from gui.dls.region import _AnalysisRegion, _region_fit_kwargs, RegionCheckBox
from gui.dls.fits import (_CORR_METHODS, _DIST_METHODS, _FIT_METHODS,
                          _CORR_RESULT_KINDS, _corr_run_key, _FIT_SHORT_LABELS, _METHOD_LINESTYLE,
                          _fit_label, _failed_without_cause, _neutral_ink, draw_fit_residuals)
from gui.dls.common import (_STALE_HINT, _STALE_UNRUNNABLE, _FOCUSED, _RESULTS_TITLE,
                            _widen_row_gap, _CLASSICAL_WEIGHTING, _SILENT_WEIGHTINGS,
                            _ALL_WEIGHTINGS, _named_weightings, _counted_weightings,
                            _weighting_title, _point_weightings, _title_weightings,
                            _selection_sample_id, _selection_export_names, _sample_label_of,
                            _sample_prefixes, _PICKER_HELP, _PICKER_BULLETS, _GROUP_FIELD_CONC,
                            _GROUP_FIELD_ANGLE, _GROUP_PICKER_BULLETS, _left_out_lines,
                            _union_missing, _needs_lines)
from gui.dls.tables import (_vtable, _qualified, _label_row, _fill_vtable, _repoint_vtable,
                            _reset_vtable, _HIGHER_MOMENT_TOOLTIP, _RESULT_ROW_TOOLTIPS,
                            _RESULT_ROWS, _readonly_table)
from gui.dls.legend import (_KEY_FONT_PT, _EXPORT_FONT_PT, _LEGEND_LABELSPACING, _KEY_SHORT_LABELS,
                            _EXPORT_GROW_CAP_IN, _MARKER_LABELS, _marker_legend_entries,
                            _method_legend_entries, _measurement_legend_entries, _draw_key,
                            export_legend_dressing)
from gui.dls.bases import (_RESID_FLOOR_PX, _RESID_FLOOR_STEPS, _RESID_FLOOR_STEP, _RESID_RATIO_MAX,
                           _DeferredRepaint, _ResidualRatioMixin, _SampleRunTab)
from gui.dls.averaging import (_AVERAGE_METHODS, AverageSources, AverageRequest, AverageDialog,
                               ask_average_method, _grouped_preconditions, _average_note_lines,
                               format_average_summary, show_average_summary)
from gui.dls.correlogram import (_SCALE_MODES, _SCALE_XY, _DecayFrame, _decay_frame,
                                 _delay_window_row, _CUMULANT_ORDER_TOOLTIP, _GLANCE_ROWS,
                                 _GLANCE_VISIBLE_ROWS, _GLANCE_FLOOR_ROWS, _hide_shared_x,
                                 _LAYER_COLS, _GLANCE_COLS, _CorrelogramTab)
from gui.dls.distribution import (_PEAK_RESULTS_TITLE, _PEAK_TABLE_TOOLTIP, _nearest_log_node,
                                  _ordinal_peak, _UNLABELLED_QUALIFIERS, _DistributionTab)
from gui.dls.sample_analysis import _IDENTITY_FORMS, _SampleAnalysisTab
from gui.dls.mode_map import _RECORD_NOT_READY, _SLOPE_ATTRS, _ModeMapTab
from gui.dls.ddls import _ROD_FROM_BOX, _SHAPE_DIMENSION_SES, _SHAPE_ASPECT_SES, _DDLSTab
from gui.dls.summary import (_SUMMARY_PEAK_TIP, _SUMMARY_PICKER_BULLETS, _SOURCE_LABELS, _SE_HEADER,
                             _SAMPLE_FROM_TIP, _SAMPLE_RH_TIP, _ABSENT, _GroupRow, _num, _peak_cell,
                             _MOMENT_MIN_ORDER, _METHOD_SHORT, _detail_quantity, _detail_header,
                             _detail_cell, _with_detail_columns, _last_headline_of,
                             _with_passive_notes, _SummaryTab)
from gui.dls.module import _DLSSubTab, _ReshownSubTab, _ReseededSubTab, DLSModule
