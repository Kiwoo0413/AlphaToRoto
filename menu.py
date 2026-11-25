"""
EditableAlpha Foundry Nuke Menu & Hotkey Registration (menu.py)
"""

import nuke
import editablealpha_bridge

toolbar = nuke.menu("Nodes")
ea_menu = toolbar.addMenu("EditableAlpha")
ea_menu.addCommand(
    "EditableAlpha (Mask to Roto)",
    "editablealpha_bridge.create_editablealpha_gizmo()",
    shortcut="Ctrl+Alt+E",
)

top_menu = nuke.menu("Nuke").addMenu("&EditableAlpha")
top_menu.addCommand(
    "Create EditableAlpha Gizmo",
    "editablealpha_bridge.create_editablealpha_gizmo()",
    shortcut="Ctrl+Alt+E",
)
top_menu.addCommand(
    "Check Worker Python Environment",
    "editablealpha_bridge.check_environment_status()",
)
