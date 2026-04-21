# Add `.xlsx` support to `pega_io.read_data` and dedupe in ImpactAnalyzer

**Priority:** P2
**Touches:** `python/pdstools/pega_io/File.py`, `python/pdstools/impactanalyzer/ImpactAnalyzer.py`

`ImpactAnalyzer.from_excel` (added in #685) calls `pl.read_excel`
directly and owns the `fastexcel` `MissingDependenciesException` shim.
That logic belongs in `pega_io.read_data` — the project's unified IO
entry point — so any future reader (DA, HC, …) gets Excel support for
free and the optional-dependency dance lives in exactly one place.

## Approach
1. Extend `pega_io.read_data` to recognise `.xlsx` (and `.xls` if
   trivial). On match, call `pl.read_excel(...)` and return the
   result as a `LazyFrame` like every other branch. Wrap the
   `ModuleNotFoundError` raised by polars when `fastexcel` is missing
   and re-raise as `MissingDependenciesException(...) from None`
   (same pattern used today in `from_excel`).
2. Update the supported-extensions table / docstring.
3. Refactor `ImpactAnalyzer.from_excel` to delegate to `read_data`
   for the actual file load; keep IA-specific normalisation
   (column renames, snapshot_time injection, schema validation)
   in `from_excel`. Public signature
   `from_excel(path, snapshot_time=...)` stays the same — no caller
   changes.
4. Tests:
   - Add a small `pega_io` test that round-trips an `.xlsx` fixture
     through `read_data` (reuse
     `python/tests/data/ia/ImpactAnalyzerExport_minimal.xlsx`).
   - Existing `test_ImpactAnalyzer.py::test_from_excel*` tests
     should keep passing unchanged.
5. Docs: mention `.xlsx` in `read_data` docstring's supported-formats
   list.

## Out of scope
- Streaming / lazy Excel reads (fastexcel doesn't support it well).
- Multi-sheet handling beyond what `from_excel` already does.
