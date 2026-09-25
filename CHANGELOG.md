# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Settings panel: import historical data from CSV or Excel with automatic
  delimiter, encoding and column detection (including swapped headers);
  switch back to the demo dataset at any time.
- Configurable SUNAT source (registry page, direct ZIP URL or local file) and
  optional extra SUNAT registries as model features: retention agents, good
  taxpayers and perception agents, with a public snapshot for offline runs.
- MongoDB connection editable from the panel with a connection test; panel
  settings take precedence over `.env` and registry flags are uploaded too.
- Interactive dashboard: tooltips, click-to-filter charts, search, CSV export,
  invoice detail sheet, registry risk chart and a hide-names switch. Local
  runs show company names and fiscal addresses.
- Tests for the importer, settings, registry parser and dashboard filters.
- New line-art logo: an F drawn with two curved strokes and the risk dot, with
  thicker strokes at favicon sizes; model names auto-fit their cards.
- Redesigned desktop interface: new visual identity (logo, favicon, icon set,
  palette), responsive layout that reflows at any window size, and a risk
  dashboard module with KPIs, charts, a priority table and the model chart
  gallery.
- Dark theme inspired by modern analytics dashboards: blue / violet / pink
  risk scale, ring gauge, segmented meter, status pills, amount-vs-probability
  quadrant chart, type-by-month heatmap and lollipop chart of risk factors.
- Pipeline charts (training, preparation and prediction) use the dark palette.
- SUNAT data-source indicator in the machine learning module.
- Visual redesign: violet-black palette, bundled Outfit typeface (OFL), solid
  color icon tiles, antialiased rounded cards and pill buttons, hero banner,
  rounded sidebar selection and a segmented risk filter for the priority table.
- Windows taskbar and window use the FactuRisk icon (explicit AppUserModelID)
  and `scripts/crear_acceso_directo.py` creates a desktop shortcut.

### Fixed

- The desktop app no longer closes silently when started with pythonw.exe
  (no console available for pipeline log messages).

## [1.0.0] - 2026-09-25

### Added

- Public release of the academic project as FactuRisk SUNAT, with the desktop
  interface, the data preparation steps, the training pipeline and the
  documentation generator.
- SUNAT snapshot without supplier/customer names or addresses, so the project
  can be reproduced and run fully offline.
- Local MongoDB via `docker-compose.yml` for users without a MongoDB Atlas
  account.

### Changed

- Dataset converted to Parquet (no longer CSV), keeping supplier and customer
  names out of the distributed data.
- Prediction now always uses the model whose metrics are reported; the
  hardcoded replacement thresholds were removed so the reported scores always
  match the artifact that scores new cases.
- Feature engineering is faster: the 30/90-day supplier windows are
  vectorized instead of iterating row by row.
- The smoothing prior of the supplier incidence rate now uses only invoices
  from earlier days (it previously used the global rate of the whole dataset,
  including future outcomes).
- MongoDB is optional: without configuration the pipeline uses the local
  scraping backup or the bundled SUNAT snapshot instead of failing.
- SUNAT encoding and delimiter detection read only a sample instead of the
  whole ~1.5 GB file.
- Predictions include the emission date and currency.
- SUNAT ZIP files are reused based on their age instead of being downloaded
  again on every run.
- Probability calibration updated for scikit-learn 1.6+.
- CatBoost categorical feature handling fixed; the custom transformer lives in
  its own module so saved models load from any entry point.
- Charts of pending-invoice risk are generated only from fresh predictions.
- Test suite (leakage, temporal split, threshold, SUNAT fallback) and a
  GitHub Actions workflow.

### Removed

- Duplicated legacy files that were kept in parallel with their current
  equivalents.
