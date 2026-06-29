# Benchmark Dashboard Design

## Goal

Create a professional, self-contained static HTML dashboard that presents every
available AgentMesh benchmark result. The dashboard must remain complete and
usable when suites, files, rows, or individual metric fields are missing or
malformed.

## Inputs

The generator reads benchmark artifacts below `runs/latest/benchmarks`:

- `<suite>/benchmark_summary.csv`
- `<suite>/benchmark_detail.jsonl`
- `<suite>/experiment_report.md`

It discovers suites from both the artifact directory and benchmark YAML files
under `examples/benchmarks`. This ensures configured but not-yet-run suites are
still represented.

## Output

The generator writes one `benchmark_dashboard.html` file. CSS, JavaScript, SVG
charts, and normalized benchmark data are embedded in the document. The file
has no CDN, package, server, or network dependency and works when opened through
the local filesystem.

## Information Architecture

1. Header: project identity, generation timestamp, suite count, result
   completeness, and offline status.
2. Executive summary: token saving, communication reduction, latency reduction,
   quality preservation, and memory hit rate KPI cards.
3. Suite navigation: selectable cards for every discovered or configured suite.
4. Comparison charts: Text Mode versus Protocol Mode for tokens, agent I/O
   bytes, wire bytes, and per-task latency.
5. Task results: a searchable, sortable table containing every detail row,
   task metadata, trace ID, and all recorded metrics.
6. Metric catalog: every summary field grouped into efficiency, communication,
   memory, transport/runtime, and feedback/routing sections.
7. Evidence: collapsible report excerpts and source diagnostics.

## Visual System

The page uses a restrained dark navy and neutral gray palette with blue for
Protocol Mode and slate for Text Mode. Large tabular numerals, compact labels,
subtle borders, and generous spacing give the dashboard a technical executive
report appearance. Charts are rendered with native HTML/CSS/SVG to preserve the
single-file offline requirement.

The layout is responsive:

- Wide screens use a fixed summary rail and multi-column content.
- Tablet widths collapse KPI and chart grids.
- Mobile widths use stacked cards and a horizontally scrollable detail table.

## Data Normalization and Missing Data

Missing is represented as `null`, never coerced to zero.

- A configured suite with no artifacts is shown as `Not generated`.
- A missing summary or detail file produces a scoped warning for that source.
- A missing metric displays `N/A` and is omitted from chart calculations.
- A malformed CSV or JSONL row is recorded in diagnostics while valid rows
  continue to render.
- An empty chart displays an explicit no-data state.
- Unknown metrics remain visible in the complete metric catalog.
- Values are escaped before insertion into HTML and JSON is serialized safely
  so artifact text cannot break the generated page.

## Generator Boundaries

The implementation consists of:

- A Python generator responsible for discovery, parsing, normalization, and
  safe HTML serialization.
- An HTML template with embedded styles and dependency-free client-side
  rendering for suite selection, filtering, sorting, charts, and details.

The generator does not run benchmarks or mutate benchmark artifacts.

## Verification

Automated tests cover:

- complete multi-suite artifacts;
- no benchmark output directory;
- missing files and missing fields;
- malformed JSONL rows;
- embedded script-closing text;
- inclusion of every summary metric and detail result.

The generated HTML is also checked for responsive structure, local-file
operation, and absence of external resource references.
