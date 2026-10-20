# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Allow ``python -m openbeam_v2e`` as an alias of the ``openbeam-v2e`` command."""

from openbeam_v2e.cli import main

raise SystemExit(main())
