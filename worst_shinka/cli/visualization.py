from __future__ import annotations

import json
import webbrowser
from html import escape
from pathlib import Path
from typing import Any


def _safe_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and abs(value) >= 1000:
            return f"{value:.2f}"
        return str(value)
    return str(value)


def _status_for(node: dict[str, Any]) -> str:
    return str(node.get("status") or "pending").lower()


def _get_best_node_id(nodes: list[dict[str, Any]]) -> str | None:
    for node in nodes:
        if _status_for(node) == "best":
            return str(node.get("id"))
    best_id = None
    best_score = float("-inf")
    for node in nodes:
        score = node.get("score")
        if isinstance(score, (int, float)) and score > best_score:
            best_score = score
            best_id = str(node.get("id"))
    return best_id


def _node_color(node: dict[str, Any], is_best: bool) -> str:
    if is_best:
        return "#f5b700"
    status = _status_for(node)
    if status == "incorrect":
        return "#ef4444"
    return "#3b82f6"


def _node_shape(node: dict[str, Any], is_best: bool) -> str:
    if is_best:
        return "diamond"
    status = _status_for(node)
    if status == "incorrect":
        return "rect"
    return "circle"


def _node_label(node: dict[str, Any]) -> str:
    node_id = str(node.get("id", "?"))
    if node_id.startswith("model-"):
        return node_id[6:]
    return node_id


def _node_parent_id(node: dict[str, Any]) -> str | None:
    parent_id = node.get("parent_id")
    if parent_id:
        return str(parent_id)
    parent_ids = node.get("parent_ids")
    if isinstance(parent_ids, list) and parent_ids:
        return str(parent_ids[0])
    return None


def _as_nodes(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        nodes = payload.get("nodes")
        if isinstance(nodes, list):
            return nodes
    raise ValueError("lineage.json must contain a list of nodes or a {'nodes': [...]} object")


def _sorted_roots(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(node.get("id")): node for node in nodes if node.get("id") is not None}
    roots: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in nodes:
        node_id = str(node.get("id"))
        if node_id in seen:
            continue
        parent_id = _node_parent_id(node)
        if parent_id is None or parent_id not in by_id:
            roots.append(node)
            seen.add(node_id)
    return roots


def _build_children_map(nodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    children: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        parent_id = _node_parent_id(node)
        if parent_id is None:
            continue
        children.setdefault(str(parent_id), []).append(node)
    return children


def _layout_tree_horizontal(
    nodes: list[dict[str, Any]], 
    children: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, float]]:
    by_id = {str(node.get("id")): node for node in nodes if node.get("id") is not None}
    roots = _sorted_roots(nodes)
    positions: dict[str, dict[str, float]] = {}

    X_STEP = 180.0
    Y_STEP = 90.0

    current_leaf_y = 0.0

    def layout_node(node_id: str, depth: int) -> float:
        nonlocal current_leaf_y
        node = by_id[node_id]
        x = depth * X_STEP + 80.0

        children_for_node = children.get(node_id, [])

        if not children_for_node:
            y = current_leaf_y * Y_STEP + 80.0
            current_leaf_y += 1.0
        else:
            child_y_sum = sum(layout_node(str(child.get("id")), depth + 1) for child in children_for_node)
            y = child_y_sum / len(children_for_node)

        positions[node_id] = {"x": x, "y": y}
        return y

    for root in roots:
        root_id = str(root.get("id"))
        layout_node(root_id, 0)

    if not positions:
        for node in nodes:
            node_id = str(node.get("id"))
            positions[node_id] = {"x": 80.0, "y": 80.0}

    return positions


def _node_summary(node: dict[str, Any]) -> str:
    lines = [
        f"ID: {node.get('id', '-')}",
        f"Generation: {node.get('generation', '-')}",
        f"Status: {node.get('status', '-')}",
        f"Score: {_safe_number(node.get('score'))}",
        f"Avg Score: {_safe_number(node.get('average_training_score'))}",
        f"Elo: {_safe_number(node.get('elo'))}",
        f"Wins: {_safe_number(node.get('wins'))}",
        f"Losses: {_safe_number(node.get('losses'))}",
        f"Draws: {_safe_number(node.get('draws'))}",
        f"Matches: {_safe_number(node.get('matches'))}",
        f"Win Rate: {_safe_number(node.get('win_rate'))}",
        f"Time: {_safe_number(node.get('time'))}",
    ]
    if node.get("parent_id"):
        lines.append(f"Parent: {node.get('parent_id')}")
    if node.get("parent_ids"):
        lines.append(f"Parents: {', '.join(str(item) for item in node.get('parent_ids'))}")
    return "<br>".join(lines)


def _render_html(nodes: list[dict[str, Any]], *, title: str) -> str:
    children = _build_children_map(nodes)
    positions = _layout_tree_horizontal(nodes, children)
    best_id = _get_best_node_id(nodes)
    roots = _sorted_roots(nodes)
    root_id = str(roots[0].get("id")) if roots else (nodes[0].get("id") if nodes else None)

    node_data_js = json.dumps(nodes, ensure_ascii=False)
    edges = []
    
    for node in nodes:
        node_id = str(node.get("id"))
        parent_id = _node_parent_id(node)
        if not parent_id or parent_id not in positions or node_id not in positions:
            continue

        x1, y1 = positions[parent_id]["x"], positions[parent_id]["y"]
        x2, y2 = positions[node_id]["x"], positions[node_id]["y"]

        start_x = x1 + 22
        end_x = x2 - 22

        ctrl_x1 = start_x + (end_x - start_x) * 0.5
        ctrl_y1 = y1
        ctrl_x2 = start_x + (end_x - start_x) * 0.5
        ctrl_y2 = y2

        edges.append(
            f'<path class="edge" data-parent="{escape(parent_id)}" data-child="{escape(node_id)}" '
            f'd="M {start_x} {y1} C {ctrl_x1} {ctrl_y1}, {ctrl_x2} {ctrl_y2}, {end_x} {y2}" />'
        )

    start_arrow_markup = ""
    if root_id and root_id in positions:
        rx, ry = positions[root_id]["x"], positions[root_id]["y"]
        arrow_start_x = rx - 70
        arrow_end_x = rx - 26
        start_arrow_markup = (
            f'<path class="start-arrow" d="M {arrow_start_x} {ry} L {arrow_end_x} {ry}" marker-end="url(#arrowhead)" />'
            f'<text x="{arrow_start_x - 5}" y="{ry - 8}" class="start-label">START</text>'
        )

    node_groups = []
    for node in nodes:
        node_id = str(node.get("id"))
        if node_id not in positions:
            continue
        x = positions[node_id]["x"]
        y = positions[node_id]["y"]
        is_best = (node_id == best_id)
        is_root = (node_id == root_id)
        status = _status_for(node)
        color = _node_color(node, is_best)
        label = _node_label(node)
        shape = _node_shape(node, is_best)
        stats = escape(_node_summary(node), quote=False)

        badge_markup = ""
        if is_best:
            badge_markup = f'<text x="{x}" y="{y - 30}" text-anchor="middle" font-size="10" fill="#f5b700" font-weight="bold">BEST</text>'

        root_class = "root-initial-node" if is_root else ""

        if shape == "circle":
            node_markup = (
                f'<circle class="node-element node-{status} {root_class}" data-node-id="{escape(node_id)}" cx="{x}" cy="{y}" r="22" fill="{color}" stroke="#0f172a" stroke-width="2.5" />'
                f'<text x="{x}" y="{y + 4}" text-anchor="middle" font-size="12" fill="#fff" font-weight="700">{escape(label)}</text>'
            )
        elif shape == "rect":
            node_markup = (
                f'<rect class="node-element node-{status} {root_class}" data-node-id="{escape(node_id)}" x="{x - 20}" y="{y - 20}" width="40" height="40" rx="8" fill="{color}" stroke="#0f172a" stroke-width="2.5" />'
                f'<text x="{x}" y="{y + 4}" text-anchor="middle" font-size="12" fill="#fff" font-weight="700">{escape(label)}</text>'
            )
        else:
            node_markup = (
                f'<polygon class="node-element node-{status} {root_class}" data-node-id="{escape(node_id)}" points="{x},{y - 26} {x + 26},{y} {x},{y + 26} {x - 26},{y}" fill="{color}" stroke="#0f172a" stroke-width="2.5" />'
                f'<text x="{x}" y="{y + 4}" text-anchor="middle" font-size="12" fill="#fff" font-weight="700">{escape(label)}</text>'
            )

        node_groups.append(
            f'<g class="node-item {"best-node" if is_best else ""}" data-node-id="{escape(node_id)}" tabindex="0" data-stats="{stats}">'
            f'{badge_markup}{node_markup}'
            f'</g>'
        )

    ascii_logo = r"""██╗   ██╗    ██╗  ██████╗  ██████╗   ███████╗ ████████╗
██║   ██║    ██║ ██╔═══██╗ ██╔══██╗  ██╔════╝ ╚══██╔══╝
██║   ██║    ██║ ██║   ██║ ██████╔╝  ███████╗    ██║   
██║   ██║    ██║ ██║   ██║ ██╔══██╗  ╚════██║    ██║   
╚██████╔██████╔╝ ╚██████╔╝ ██║  ██║  ███████║    ██║   
 ╚═════╝ ╚════╝   ╚═════╝  ╚═╝  ╚═╝  ╚══════╝    ╚═╝   """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #1e293b;
      --panel-border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --edge: #475569;
      --edge-selected: #38bdf8;
      --edge-best-path: #f5b700;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      overflow: hidden;
    }}
    .layout {{
      position: relative;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      height: 100vh;
      width: 100vw;
      transition: grid-template-columns 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    .layout.collapsed {{
      grid-template-columns: minmax(0, 1fr) 48px;
    }}
    .tree-pane {{
      position: relative;
      background-color: var(--bg);
      background-image: 
        radial-gradient(rgba(255, 255, 255, 0.1) 1px, transparent 1px);
      background-size: 24px 24px;
      overflow: hidden;
      cursor: grab;
      user-select: none;
    }}
    .tree-pane:active {{
      cursor: grabbing;
    }}
    svg {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    .controls-vertical {{
      position: absolute;
      bottom: 24px;
      left: 24px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      z-index: 10;
    }}
    .btn-circle {{
      background: var(--panel);
      border: 1px solid var(--panel-border);
      color: var(--text);
      width: 44px;
      height: 44px;
      border-radius: 50%;
      font-weight: 700;
      font-size: 18px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 4px 14px rgba(0,0,0,0.4);
      transition: all 0.2s ease;
      padding: 0;
    }}
    .btn-circle svg {{
      width: 20px;
      height: 20px;
      fill: none;
      stroke: currentColor;
      stroke-width: 2;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    .btn-circle:hover {{
      background: #334155;
      border-color: #64748b;
      transform: scale(1.08);
    }}
    .btn-circle:active {{
      transform: scale(0.96);
    }}
    .toggle-sidebar-btn {{
      position: absolute;
      top: 32px;
      right: 341px;
      width: 38px;
      height: 38px;
      background: var(--panel);
      border: 1px solid var(--panel-border);
      color: var(--text);
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      z-index: 100;
      box-shadow: -2px 4px 12px rgba(0,0,0,0.5);
      transition: right 0.3s cubic-bezier(0.4, 0, 0.2, 1), background 0.2s ease, border-color 0.2s ease;
    }}
    .layout.collapsed .toggle-sidebar-btn {{
      right: 29px;
    }}
    .toggle-sidebar-btn:hover {{
      background: #334155;
      border-color: #64748b;
    }}
    .toggle-sidebar-btn svg {{
      width: 20px;
      height: 20px;
      fill: none;
      stroke: currentColor;
      stroke-width: 2.5;
      stroke-linecap: round;
      stroke-linejoin: round;
      transition: transform 0.3s ease;
    }}
    .layout.collapsed .toggle-sidebar-btn svg {{
      transform: rotate(180deg);
    }}
    .edge {{
      fill: none;
      stroke: var(--edge);
      stroke-width: 2;
      opacity: 0.4;
      transition: stroke 0.25s ease, opacity 0.25s ease, stroke-width 0.25s ease;
    }}
    .edge.best-path {{
      stroke: var(--edge-best-path) !important;
      stroke-width: 3.5px !important;
      opacity: 1 !important;
      filter: drop-shadow(0 0 6px rgba(245, 183, 0, 0.6));
    }}
    .edge.highlight {{
      stroke: var(--edge-selected);
      stroke-width: 3.5px;
      opacity: 1;
      filter: drop-shadow(0 0 6px rgba(56, 189, 248, 0.6));
    }}
    .start-arrow {{
      stroke: #ffffff;
      stroke-width: 2.5;
      fill: none;
      stroke-dasharray: 4 2;
    }}
    .start-label {{
      fill: #ffffff;
      font-size: 10px;
      font-weight: 800;
      letter-spacing: 0.05em;
    }}
    .node-element.root-initial-node {{
      stroke: #ffffff !important;
      stroke-width: 3.5px !important;
      filter: drop-shadow(0 0 8px rgba(255, 255, 255, 0.8));
    }}
    .node-item {{
      cursor: pointer;
      outline: none;
    }}
    .node-item:hover .node-element {{
      filter: brightness(1.25);
    }}
    .node-item.selected .node-element {{
      stroke: #38bdf8;
      stroke-width: 4;
      filter: drop-shadow(0 0 12px rgba(56, 189, 248, 0.8));
    }}
    .best-node .node-element {{
      animation: pulse-border 2s infinite ease-in-out;
    }}
    @keyframes pulse-border {{
      0% {{ filter: drop-shadow(0 0 4px rgba(245, 183, 0, 0.4)); }}
      50% {{ filter: drop-shadow(0 0 12px rgba(245, 183, 0, 0.9)); }}
      100% {{ filter: drop-shadow(0 0 4px rgba(245, 183, 0, 0.4)); }}
    }}
    .sidebar {{
      background: var(--panel);
      border-left: 1px solid var(--panel-border);
      padding: 24px;
      overflow-y: auto;
      overflow-x: hidden;
      display: flex;
      flex-direction: column;
      gap: 20px;
      z-index: 5;
      width: 100%;
      box-sizing: border-box;
    }}
    .sidebar-content {{
      display: flex;
      flex-direction: column;
      gap: 20px;
      width: 312px;
      transition: opacity 0.2s ease;
    }}
    .layout.collapsed .sidebar-content {{
      opacity: 0;
      pointer-events: none;
    }}
    .logo-container {{
      background: #090d16;
      border: 1px solid var(--panel-border);
      border-radius: 10px;
      padding: 12px;
      overflow-x: auto;
      text-align: center;
    }}
    .logo-ascii {{
      font-family: monospace;
      font-size: 8.5px;
      line-height: 1.1;
      color: #38bdf8;
      white-space: pre;
      margin: 0;
      display: inline-block;
    }}
    .sidebar h3 {{
      margin: 0;
      font-size: 16px;
      font-weight: 600;
      color: #f1f5f9;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }}
    .stat-box {{
      background: var(--bg);
      border: 1px solid var(--panel-border);
      border-radius: 12px;
      padding: 18px;
      color: #cbd5e1;
      font-size: 13.5px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      flex-grow: 1;
    }}
    .stat-row {{
      display: flex;
      justify-content: space-between;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      padding-bottom: 8px;
    }}
    .stat-row:last-child {{
      border-bottom: none;
      padding-bottom: 0;
    }}
    .stat-label {{
      color: var(--muted);
      font-weight: 500;
    }}
    .stat-val {{
      color: #f8fafc;
      font-weight: 600;
    }}
    .legend {{
      display: flex;
      gap: 16px;
      justify-content: center;
      align-items: center;
      font-size: 12.5px;
      color: var(--muted);
      background: var(--bg);
      padding: 12px 16px;
      border-radius: 10px;
      border: 1px solid var(--panel-border);
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
      display: inline-block;
    }}
    .swatch.square {{ border-radius: 3px; }}
    .swatch.diamond {{ transform: rotate(45deg); border-radius: 2px; }}
  </style>
</head>
<body>
  <div class="layout" id="app-layout">
    <button class="toggle-sidebar-btn" id="btn-toggle-sidebar" title="Toggle Sidebar">
      <svg viewBox="0 0 24 24">
        <polyline points="9 18 15 12 9 6"></polyline>
      </svg>
    </button>

    <div class="tree-pane" id="pane">
      <div class="controls-vertical">
        <button class="btn-circle" id="btn-zoom-in" title="Zoom In">+</button>
        <button class="btn-circle" id="btn-fit" title="Center / Fit View">
          <svg viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="7" />
            <line x1="12" y1="2" x2="12" y2="5" />
            <line x1="12" y1="19" x2="12" y2="22" />
            <line x1="2" y1="12" x2="5" y2="12" />
            <line x1="19" y1="12" x2="22" y2="12" />
            <circle cx="12" cy="12" r="1.5" fill="currentColor" />
          </svg>
        </button>
        <button class="btn-circle" id="btn-zoom-out" title="Zoom Out">-</button>
      </div>
      <svg id="svg-tree">
        <defs>
          <marker id="arrowhead" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#ffffff" />
          </marker>
        </defs>
        <g id="viewport">
          {''.join(edges)}
          {start_arrow_markup}
          {''.join(node_groups)}
        </g>
      </svg>
    </div>
    <aside class="sidebar">
      <div class="sidebar-content">
        <div class="logo-container">
          <pre class="logo-ascii">{escape(ascii_logo)}</pre>
        </div>
        
        <div class="legend">
          <span class="legend-item"><span class="swatch" style="background:#3b82f6"></span>Candidate</span>
          <span class="legend-item"><span class="swatch square" style="background:#ef4444"></span>Incorrect</span>
          <span class="legend-item"><span class="swatch diamond" style="background:#f5b700"></span>Best</span>
        </div>

        <h3>Selected Node</h3>
        <div id="node-stats" class="stat-box">Click a node to inspect metrics.</div>
      </div>
    </aside>
  </div>

  <script>
    const rawNodes = {node_data_js};
    const parentByNode = {{}};
    for (const node of rawNodes) {{
      const nodeId = String(node.id);
      const parentId = node.parent_id || (Array.isArray(node.parent_ids) && node.parent_ids.length ? String(node.parent_ids[0]) : null);
      if (parentId) parentByNode[nodeId] = String(parentId);
    }}
    const nodeLookup = Object.fromEntries(rawNodes.map(node => [String(node.id), node]));
    const statsPanel = document.getElementById('node-stats');
    const layout = document.getElementById('app-layout');
    const toggleSidebarBtn = document.getElementById('btn-toggle-sidebar');

    const bestNodeId = "{best_id or ''}";

    // Sidebar toggle logic
    toggleSidebarBtn.addEventListener('click', () => {{
      layout.classList.toggle('collapsed');
      setTimeout(fitToScreen, 310);
    }});

    // Canvas Zoom & Pan logic
    const svg = document.getElementById('svg-tree');
    const viewport = document.getElementById('viewport');
    const pane = document.getElementById('pane');

    let scale = 1;
    let pointX = 0;
    let pointY = 0;
    let start = {{ x: 0, y: 0 }};
    let isPanning = false;

    function setTransform() {{
      viewport.setAttribute('transform', `translate(${{pointX}}, ${{pointY}}) scale(${{scale}})`);
      pane.style.backgroundPosition = `${{pointX}}px ${{pointY}}px`;
    }}

    function fitToScreen() {{
      const bbox = viewport.getBBox();
      if (!bbox.width || !bbox.height) return;

      const containerWidth = pane.clientWidth;
      const containerHeight = pane.clientHeight;
      const padding = 100;

      const scaleX = (containerWidth - padding) / bbox.width;
      const scaleY = (containerHeight - padding) / bbox.height;
      scale = Math.min(scaleX, scaleY, 1.2);

      pointX = (containerWidth - bbox.width * scale) / 2 - bbox.x * scale;
      pointY = (containerHeight - bbox.height * scale) / 2 - bbox.y * scale;
      setTransform();
    }}

    pane.addEventListener('mousedown', (e) => {{
      if (e.target.closest('.controls-vertical') || e.target.closest('.toggle-sidebar-btn') || e.target.closest('.node-item')) return;
      isPanning = true;
      start = {{ x: e.clientX - pointX, y: e.clientY - pointY }};
    }});

    window.addEventListener('mousemove', (e) => {{
      if (!isPanning) return;
      pointX = e.clientX - start.x;
      pointY = e.clientY - start.y;
      setTransform();
    }});

    window.addEventListener('mouseup', () => {{ isPanning = false; }});

    pane.addEventListener('wheel', (e) => {{
      e.preventDefault();
      const xs = (e.clientX - pointX) / scale;
      const ys = (e.clientY - pointY) / scale;
      const delta = -e.deltaY;
      (delta > 0) ? (scale *= 1.1) : (scale /= 1.1);
      pointX = e.clientX - xs * scale;
      pointY = e.clientY - ys * scale;
      setTransform();
    }}, {{ passive: false }});

    document.getElementById('btn-zoom-in').addEventListener('click', () => {{
      scale *= 1.2;
      setTransform();
    }});

    document.getElementById('btn-zoom-out').addEventListener('click', () => {{
      scale /= 1.2;
      setTransform();
    }});

    document.getElementById('btn-fit').addEventListener('click', fitToScreen);

    // Path calculation
    const pathTo = (nodeId) => {{
      const chain = new Set();
      let current = nodeId;
      while (current) {{
        chain.add(current);
        current = parentByNode[current] || null;
      }}
      return chain;
    }};

    // Resets selection and activates default Best Path
    const resetToBestPath = () => {{
      document.querySelectorAll('.node-item').forEach(group => group.classList.remove('selected'));
      document.querySelectorAll('.edge').forEach(edge => edge.classList.remove('highlight'));

      if (!bestNodeId) {{
        statsPanel.innerHTML = 'Click a node to inspect metrics.';
        return;
      }}

      const bestPathSet = pathTo(bestNodeId);
      document.querySelectorAll('.edge').forEach(edge => {{
        const parentId = edge.dataset.parent;
        const childId = edge.dataset.child;
        const isBestEdge = bestPathSet.has(parentId) && bestPathSet.has(childId);
        edge.classList.toggle('best-path', isBestEdge);
      }});

      renderStats(bestNodeId);
    }};

    const renderStats = (nodeId) => {{
      const node = nodeLookup[nodeId];
      if (!node) return;
      const entries = [
        ['ID', node.id ?? '-'],
        ['Generation', node.generation ?? '-'],
        ['Status', node.status ?? '-'],
        ['Score', node.score ?? '-'],
        ['Avg Score', node.average_training_score ?? '-'],
        ['Elo', node.elo ?? '-'],
        ['Wins', node.wins ?? '-'],
        ['Losses', node.losses ?? '-'],
        ['Draws', node.draws ?? '-'],
        ['Matches', node.matches ?? '-'],
        ['Win Rate', node.win_rate ?? '-'],
        ['Time', node.time ?? '-'],
        ['Parent', node.parent_id ?? (Array.isArray(node.parent_ids) ? node.parent_ids.join(', ') : '-')],
      ];
      statsPanel.innerHTML = entries.map(([label, value]) => 
        `<div class="stat-row"><span class="stat-label">${{label}}</span><span class="stat-val">${{value}}</span></div>`
      ).join('');
    }};

    const selectNode = (nodeId) => {{
      document.querySelectorAll('.edge').forEach(edge => edge.classList.remove('best-path'));

      const selected = pathTo(nodeId);
      document.querySelectorAll('.node-item').forEach(group => {{
        const id = group.dataset.nodeId;
        group.classList.toggle('selected', id === nodeId);
      }});

      document.querySelectorAll('.edge').forEach(edge => {{
        const parentId = edge.dataset.parent;
        const childId = edge.dataset.child;
        const isActive = selected.has(parentId) && selected.has(childId);
        edge.classList.toggle('highlight', isActive);
      }});

      renderStats(nodeId);
    }};

    document.querySelectorAll('.node-item').forEach(group => {{
      group.addEventListener('click', (e) => {{
        e.stopPropagation();
        selectNode(group.dataset.nodeId);
      }});
    }});

    pane.addEventListener('click', (e) => {{
      if (e.target.closest('.controls-vertical') || e.target.closest('.toggle-sidebar-btn') || e.target.closest('.node-item')) return;
      resetToBestPath();
    }});

    window.addEventListener('load', () => {{
      setTimeout(fitToScreen, 50);
      resetToBestPath();
    }});
    
    window.addEventListener('resize', fitToScreen);
  </script>
</body>
</html>
"""


def visualize_run(run_path: str | Path, *, save_to: str | Path | None = None, open_browser: bool = True) -> Path:
    run_dir = Path(run_path).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    lineage_path = run_dir / "lineage.json"
    if not lineage_path.is_file():
        raise FileNotFoundError(f"No lineage.json found in run directory: {lineage_path}")

    with lineage_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    nodes = _as_nodes(payload)
    if not nodes:
        raise ValueError(f"No lineage nodes found in {lineage_path}")

    output_dir = Path(save_to).expanduser().resolve() if save_to is not None else run_dir / "visualize"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evolution_tree.html"
    output_path.write_text(_render_html(nodes, title=f"Evolution tree — {run_dir.name}"), encoding="utf-8")

    if open_browser:
        target_url = output_path.resolve().as_uri()
        try:
            webbrowser.open(target_url)
        except Exception:
            try:
                import os
                os.system(f'xdg-open "{output_path}" >/dev/null 2>&1 || true')
            except Exception:
                pass

    return output_path