"""
EditableAlpha Foundry Nuke Plugin Registration (init.py)
"""

import os
import sys
import nuke

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

nuke.pluginAddPath(_PKG_DIR)
