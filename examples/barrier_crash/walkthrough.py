# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Walk the barrier_crash example through the V2E pipeline (offline, CPU, fake connector).

Run ``python walkthrough.py --out <folder>``; see README.md in this folder for what each step does.
"""

import sys

from openbeam_v2e.bench.walkthrough import main

if __name__ == "__main__":
    sys.exit(main("barrier_crash", sys.argv[1:]))
