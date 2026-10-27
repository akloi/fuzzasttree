#!/usr/bin/env python3
"""Example command adapter: return the historical subtree unchanged."""

import json
import sys


request = json.load(sys.stdin)
json.dump({"mutated_subtree": request["historical_subtree"]}, sys.stdout, ensure_ascii=False)

