# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[PEP 440](https://peps.python.org/pep-0440/).

## Unreleased

Private v0 preview (pre-alpha). Tested on synthetic clips and the `fake` test connector only.

### Added

- Reconstruction from video, frame folders or frame arrays, with CPU baselines, uncertainty,
  held-out residuals and a stage cache.
- Pluggable perception stages and optional SAM 2, CoTracker and Hugging Face model adapters.
- Schemas and validators for scenes, templates, run logs, content catalogs, package requests,
  score reports and world sidecars; world I/O using the 4DCodeBench layout.
- Compiler with run-up synthesis, frame-0 alignment and connector dialects.
- Connector negotiation, replayable hash-chained ledgers, offline audit, the `fake` test
  connector and the cassette-tested `openbeam` connector (documented MCP tools only).
- Scoring for paths, attitude, silhouettes, events, targets and repeat statistics.
- Seven shared CLI/stdio MCP tools, plus CLI audit and connector commands.
- Four-family synthetic benchmark, generated controller programs, per-family walkthroughs,
  a connector quickstart and a custom perception model example.
- Offline CI, wheel smoke checks, and all-four-family general-compiler contract tests.
- Apache-2.0 licence for code, CC BY 4.0 for data, and layout tests.

### Fixed

- Single-axis rotation batch fixtures support current SciPy as well as older supported versions.
- Position-only run logs explicitly declare unobserved attitude and can be read without
  fabricating orientations. Full-pose export refuses logs with missing attitude.
- Generated benchmark programs run through the shared CLI/MCP controller factory.
- Benchmark target names match the package scorer where supported; event-time targets read
  the named run event, and absent events count as unmet rather than zero error.
- Source distributions include the benchmark guides linked from the README.
- Background segmentation tracks backwards from the first prompt with signed-time prediction,
  uses prompt-guided background exclusions for objects that stop, preserves silhouettes at image
  edges, and handles the automatic second pass on a downscaled grid. Exclusion boxes are not masks.
- The benchmark cruise driver follows the line through its specified frame-0 position at any
  heading, rather than applying a global-Y correction to rotated run-ups.
- Later segmentation prompts can recover missing, image-supported masks after a visibility gap
  or stop. Reverse tracking preserves forward observations, resets prediction at annotations,
  respects annotated pixel ownership, and can use unsampled prompts without inventing masks.
- Pose fitting compares only the image-visible hull projection at image borders. Ground tracks
  seed from an untruncated fitting mask and revisit earlier masks backwards with a weak,
  configurable adjacent-heading prior; reversing motion and held-out evidence remain separate.
- Optional ballistic refinement uses sustained-clearance intervals rather than missing-event
  whole-clip fallbacks, excludes held-out/interpolated poses from measurement, screens inconsistent
  or ground-penetrating proposals, and refreshes final silhouette residuals after accepted changes.
- Sampled contact detection restores uncapped neighbouring distances around candidate minima,
  so the broad-phase cutoff no longer hides fast between-frame rebounds. Unknown poses, contact
  thresholds and named-event matching remain unchanged.

### Known limitations

- Multi-family baseline reconstruction has large errors; see [validation](docs/validation.md).
- Native OpenBeam runs and independent real footage remain unqualified.
