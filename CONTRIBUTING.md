# Contributing to OpenBeam V2E

Thank you for helping. This project is licensed under Apache-2.0 (code) and CC BY 4.0 (data we
make); contributions are accepted under the same terms ("inbound = outbound").

## Sign your commits (DCO)

Every commit must carry a `Signed-off-by:` line certifying the
[Developer Certificate of Origin 1.1](https://developercertificate.org/):

```sh
git commit -s
```

## Development setup

```sh
python -m pip install -e ".[test]"
python -m pytest
```

Tests run offline on the `fake` connector. Tests that need a live engine are marked (for example
`-m openbeam`) and are never required in CI.

## Ground rules for changes

- **Connectors only install, run and report.** Reconstruction, scoring and search stay in the
  engine-free core.
- **No physics bypass.** No force, impulse, pose, velocity, gravity or mass setters, teleports or
  in-run resets, and no initial velocity at spawn.
- **Capabilities are declared, not assumed.** Engine-specific actuator channels, timing regimes
  and recording classes come from a connector's manifest or a plugin, never from the core.
- **Keep the base install light.** Core modules may import only the standard library, NumPy and
   SciPy at import time. Optional model libraries are imported lazily inside the
  code that needs them.
- **Perception models are pluggable.** New models implement the interfaces in
  `openbeam_v2e.perception` and record their model id, version and weights hash. Do not vendor
  third-party model code.
- **Data needs provenance.** Every binary or media file in examples and benchmarks needs a
  provenance record with its source, licence and how to regenerate it. Strip media metadata.
- **Every source file starts with the SPDX header:**

  ```python
  # SPDX-FileCopyrightText: 2026 OpenBeam, LLC
  # SPDX-License-Identifier: Apache-2.0
  ```

## Review

Maintainers run additional review checks before merging. A pull request from outside the
maintainer team may show a neutral "maintainer review required" status until a maintainer has
looked at it; that status says nothing about the content of your change.
