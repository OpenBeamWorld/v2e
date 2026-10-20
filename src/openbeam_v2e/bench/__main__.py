# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""``python -m openbeam_v2e.bench``: the benchmark command line (see :mod:`openbeam_v2e.bench.cli`)."""

import sys

from openbeam_v2e.bench.cli import main

sys.exit(main())
