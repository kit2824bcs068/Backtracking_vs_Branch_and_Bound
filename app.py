import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import time
import pandas as pd
import numpy as np

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Backtracking vs Branch & Bound",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
}
.main-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #94a3b8;
    margin: 0.5rem 0 0;
    font-size: 1rem;
    font-weight: 300;
}

.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #f1f5f9;
    line-height: 1.2;
}
.metric-card .value.blue { color: #38bdf8; }
.metric-card .value.green { color: #34d399; }
.metric-card .value.amber { color: #fbbf24; }

.algo-header-bt {
    background: #0c1a2e;
    border-left: 4px solid #38bdf8;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1rem;
}
.algo-header-bb {
    background: #0a1f16;
    border-left: 4px solid #34d399;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1rem;
}
.algo-header-bt h3 { color: #38bdf8; margin: 0; font-size: 1rem; }
.algo-header-bb h3 { color: #34d399; margin: 0; font-size: 1rem; }

.step-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    max-height: 250px;
    overflow-y: auto;
    color: #94a3b8;
    line-height: 1.8;
}
.step-box .include { color: #34d399; }
.step-box .prune   { color: #f87171; }
.step-box .best    { color: #fbbf24; }

.winner-badge {
    display: inline-block;
    background: #fbbf24;
    color: #0f172a;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-left: 8px;
}

.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label { color: #94a3b8 !important; font-size: 0.85rem; }

.stButton > button {
    width: 100%;
    border-radius: 8px;
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    border: none;
    padding: 0.6rem 1rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

# ─── 0/1 Knapsack — Backtracking ──────────────────────────────────────────────
def knapsack_backtracking(weights, values, capacity):
    n = len(weights)
    best = {"value": 0, "items": []}
    steps = []
    nodes = [0]

    def bt(idx, cur_weight, cur_value, chosen):
        nodes[0] += 1
        if idx == n:
            if cur_value > best["value"]:
                best["value"] = cur_value
                best["items"] = chosen[:]
                steps.append(("best", f"New best = {cur_value}  (items: {chosen})"))
            return
        # Include
        if cur_weight + weights[idx] <= capacity:
            steps.append(("include", f"Level {idx}: Include item {idx} (w={weights[idx]}, v={values[idx]})"))
            chosen.append(idx)
            bt(idx + 1, cur_weight + weights[idx], cur_value + values[idx], chosen)
            chosen.pop()
        else:
            steps.append(("prune", f"Level {idx}: Skip item {idx} — weight exceeds capacity"))
        # Exclude
        steps.append(("normal", f"Level {idx}: Exclude item {idx}"))
        bt(idx + 1, cur_weight, cur_value, chosen)

    bt(0, 0, 0, [])
    return best["value"], best["items"], nodes[0], steps


# ─── 0/1 Knapsack — Branch & Bound ────────────────────────────────────────────
def knapsack_branch_bound(weights, values, capacity):
    n = len(weights)
    # Sort by value/weight ratio descending
    items = sorted(range(n), key=lambda i: values[i] / weights[i], reverse=True)
    w = [weights[i] for i in items]
    v = [values[i] for i in items]
    steps = []
    nodes = [0]
    pruned = [0]
    best = {"value": 0, "items": []}

    def upper_bound(idx, cur_w, cur_v):
        ub = cur_v
        rem = capacity - cur_w
        for i in range(idx, n):
            if w[i] <= rem:
                rem -= w[i]
                ub += v[i]
            else:
                ub += v[i] * rem / w[i]
                break
        return ub

    def bb(idx, cur_w, cur_v, chosen):
        nodes[0] += 1
        ub = upper_bound(idx, cur_w, cur_v)
        if ub <= best["value"]:
            pruned[0] += 1
            steps.append(("prune", f"Level {idx}: PRUNED — UB={ub:.1f} ≤ best={best['value']}"))
            return
        if idx == n:
            if cur_v > best["value"]:
                best["value"] = cur_v
                best["items"] = [items[i] for i in chosen]
                steps.append(("best", f"New best = {cur_v}  (items: {best['items']})"))
            return
        # Include
        if cur_w + w[idx] <= capacity:
            steps.append(("include", f"Level {idx}: Include item {items[idx]} (UB={ub:.1f})"))
            chosen.append(idx)
            bb(idx + 1, cur_w + w[idx], cur_v + v[idx], chosen)
            chosen.pop()
        # Exclude
        steps.append(("normal", f"Level {idx}: Exclude item {items[idx]}"))
        bb(idx + 1, cur_w, cur_v, chosen)

    bb(0, 0, 0, [])
    return best["value"], best["items"], nodes[0], pruned[0], steps


# ─── N-Queens — Backtracking ──────────────────────────────────────────────────
def nqueens_backtracking(n):
    solutions = []
    steps = []
    nodes = [0]

    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col or abs(board[i] - col) == abs(i - row):
                return False
        return True

    def bt(board, row):
        nodes[0] += 1
        if row == n:
            solutions.append(board[:])
            steps.append(("best", f"Solution found: {board}"))
            return
        for col in range(n):
            if is_safe(board, row, col):
                steps.append(("include", f"Place queen at row {row}, col {col}"))
                board.append(col)
                bt(board, row + 1)
                board.pop()
            else:
                steps.append(("prune", f"Conflict at row {row}, col {col} — backtrack"))

    bt([], 0)
    return solutions, nodes[0], steps


# ─── TSP — Backtracking ───────────────────────────────────────────────────────
def tsp_backtracking(dist):
    n = len(dist)
    best = {"cost": float("inf"), "path": []}
    nodes = [0]
    steps = []

    def bt(path, visited, cost):
        nodes[0] += 1
        if len(path) == n:
            total = cost + dist[path[-1]][path[0]]
            if total < best["cost"]:
                best["cost"] = total
                best["path"] = path[:] + [path[0]]
                steps.append(("best", f"New best path: {best['path']}  cost={total:.1f}"))
            return
        cur = path[-1]
        for nxt in range(n):
            if nxt not in visited:
                steps.append(("include", f"Visit city {nxt} from {cur}  (edge={dist[cur][nxt]:.1f})"))
                path.append(nxt)
                visited.add(nxt)
                bt(path, visited, cost + dist[cur][nxt])
                path.pop()
                visited.discard(nxt)

    bt([0], {0}, 0)
    return best["cost"], best["path"], nodes[0], steps


# ─── TSP — Branch & Bound ─────────────────────────────────────────────────────
def tsp_branch_bound(dist):
    n = len(dist)
    best = {"cost": float("inf"), "path": []}
    nodes = [0]
    pruned = [0]
    steps = []

    def lower_bound(path, visited, cost):
        lb = cost
        for i in range(n):
            if i not in visited and i != path[-1]:
                row = [dist[i][j] for j in range(n) if j != i]
                lb += min(row) if row else 0
        return lb

    def bb(path, visited, cost):
        nodes[0] += 1
        lb = lower_bound(path, visited, cost)
        if lb >= best["cost"]:
            pruned[0] += 1
            steps.append(("prune", f"PRUNED at city {path[-1]} — LB={lb:.1f} ≥ best={best['cost']:.1f}"))
            return
        if len(path) == n:
            total = cost + dist[path[-1]][path[0]]
            if total < best["cost"]:
                best["cost"] = total
                best["path"] = path[:] + [path[0]]
                steps.append(("best", f"New best: {best['path']}  cost={total:.1f}"))
            return
        cur = path[-1]
        neighbors = sorted([j for j in range(n) if j not in visited], key=lambda j: dist[cur][j])
        for nxt in neighbors:
            steps.append(("include", f"Try city {nxt} from {cur}  LB={lb:.1f}"))
            path.append(nxt)
            visited.add(nxt)
            bb(path, visited, cost + dist[cur][nxt])
            path.pop()
            visited.discard(nxt)

    bb([0], {0}, 0)
    return best["cost"], best["path"], nodes[0], pruned[0], steps


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison_bar(bt_nodes, bb_nodes, bt_time, bb_time, problem):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("#0f172a")

    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    # Nodes bar
    bars1 = axes[0].bar(
        ["Backtracking", "Branch & Bound"],
        [bt_nodes, bb_nodes],
        color=["#38bdf8", "#34d399"],
        width=0.5, edgecolor="none"
    )
    axes[0].set_title("Nodes Explored", color="#e2e8f0", fontsize=11, pad=10)
    axes[0].set_ylabel("Count", color="#64748b", fontsize=9)
    for bar, val in zip(bars1, [bt_nodes, bb_nodes]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(val), ha="center", va="bottom", color="#f1f5f9",
                     fontsize=11, fontweight="bold")
    axes[0].tick_params(axis="x", colors="#94a3b8")
    axes[0].tick_params(axis="y", colors="#64748b")

    # Time bar
    bars2 = axes[1].bar(
        ["Backtracking", "Branch & Bound"],
        [bt_time * 1000, bb_time * 1000],
        color=["#38bdf8", "#34d399"],
        width=0.5, edgecolor="none"
    )
    axes[1].set_title("Execution Time (ms)", color="#e2e8f0", fontsize=11, pad=10)
    axes[1].set_ylabel("Milliseconds", color="#64748b", fontsize=9)
    for bar, val in zip(bars2, [bt_time * 1000, bb_time * 1000]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                     f"{val:.2f}", ha="center", va="bottom", color="#f1f5f9",
                     fontsize=11, fontweight="bold")
    axes[1].tick_params(axis="x", colors="#94a3b8")
    axes[1].tick_params(axis="y", colors="#64748b")

    plt.suptitle(f"{problem} — Algorithm Comparison", color="#e2e8f0", fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def plot_nqueens_board(solution, n):
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    for r in range(n):
        for c in range(n):
            color = "#2d3f55" if (r + c) % 2 == 0 else "#1a2840"
            ax.add_patch(plt.Rectangle((c, r), 1, 1, color=color))
    if solution:
        for row, col in enumerate(solution):
            ax.text(col + 0.5, row + 0.5, "♛", ha="center", va="center",
                    fontsize=22, color="#fbbf24")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([chr(65 + i) for i in range(n)], color="#64748b")
    ax.set_yticklabels(range(1, n + 1), color="#64748b")
    ax.set_title("N-Queens Solution", color="#e2e8f0", fontsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    plt.tight_layout()
    return fig


def plot_tsp_path(path, coords, cost, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.tick_params(colors="#64748b")

    if path:
        xs = [coords[i][0] for i in path]
        ys = [coords[i][1] for i in path]
        ax.plot(xs, ys, "-o", color="#38bdf8", linewidth=2, markersize=8,
                markerfacecolor="#fbbf24", markeredgecolor="#0f172a", markeredgewidth=1.5)
        for i, (x, y) in enumerate(coords):
            ax.annotate(f"C{i}", (x, y), textcoords="offset points",
                        xytext=(8, 5), color="#e2e8f0", fontsize=9)

    ax.set_title(f"{title}\nCost: {cost:.2f}", color="#e2e8f0", fontsize=10)
    ax.set_xlabel("X", color="#64748b", fontsize=9)
    ax.set_ylabel("Y", color="#64748b", fontsize=9)
    plt.tight_layout()
    return fig


def render_steps(steps, max_steps=40):
    lines = []
    for kind, msg in steps[:max_steps]:
        if kind == "include":
            lines.append(f'<span class="include">✓ {msg}</span>')
        elif kind == "prune":
            lines.append(f'<span class="prune">✗ {msg}</span>')
        elif kind == "best":
            lines.append(f'<span class="best">★ {msg}</span>')
        else:
            lines.append(f'<span>{msg}</span>')
    if len(steps) > max_steps:
        lines.append(f'<span style="color:#475569">... {len(steps)-max_steps} more steps</span>')
    return '<div class="step-box">' + "<br>".join(lines) + "</div>"


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    problem = st.selectbox(
        "Select Problem",
        ["0/1 Knapsack", "N-Queens", "Travelling Salesman (TSP)"]
    )
    st.divider()

    if problem == "0/1 Knapsack":
        st.markdown("**Knapsack Settings**")
        num_items = st.slider("Number of Items", 3, 8, 5)
        capacity = st.slider("Knapsack Capacity", 10, 50, 20)
        st.markdown("**Item Weights**")
        weights = []
        values = []
        default_w = [3, 5, 2, 7, 4, 6, 1, 8]
        default_v = [20, 35, 15, 45, 28, 40, 10, 50]
        for i in range(num_items):
            c1, c2 = st.columns(2)
            with c1:
                w = st.number_input(f"W{i+1}", 1, 15, default_w[i], key=f"w{i}")
            with c2:
                v = st.number_input(f"V{i+1}", 1, 60, default_v[i], key=f"v{i}")
            weights.append(w)
            values.append(v)

    elif problem == "N-Queens":
        st.markdown("**N-Queens Settings**")
        n_queens = st.slider("Board Size (N)", 4, 8, 6)

    else:
        st.markdown("**TSP Settings**")
        num_cities = st.slider("Number of Cities", 4, 8, 5)
        st.info("Random city coordinates will be generated")
        seed = st.number_input("Random Seed", 1, 100, 42)

    st.divider()
    run_btn = st.button("▶  Run Algorithms", type="primary")
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#475569;text-align:center'>"
        "DAA Mini Project<br>Backtracking vs Branch & Bound</div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
  <h1>🔍 Backtracking vs Branch &amp; Bound</h1>
  <p>Complex Optimization Problems — DAA Mini Project</p>
</div>
""", unsafe_allow_html=True)

# ─── Concept Cards ────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="metric-card">
      <div class="label">Backtracking</div>
      <div class="value blue" style="font-size:1rem;margin-top:6px">DFS + Feasibility Pruning</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="metric-card">
      <div class="label">Branch &amp; Bound</div>
      <div class="value green" style="font-size:1rem;margin-top:6px">DFS/BFS + Bound Pruning</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="metric-card">
      <div class="label">Key Difference</div>
      <div class="value amber" style="font-size:1rem;margin-top:6px">Upper/Lower Bound ← B&amp;B</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Run Algorithms ───────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running algorithms..."):

        # ── KNAPSACK ──────────────────────────────────────────────────────────
        if problem == "0/1 Knapsack":
            t0 = time.perf_counter()
            bt_val, bt_items, bt_nodes, bt_steps = knapsack_backtracking(weights, values, capacity)
            bt_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            bb_val, bb_items, bb_nodes, bb_pruned, bb_steps = knapsack_branch_bound(weights, values, capacity)
            bb_time = time.perf_counter() - t0

            # Metrics
            st.markdown("### 📊 Results")
            m1, m2, m3, m4 = st.columns(4)
            winner = "BT" if bt_nodes < bb_nodes else "B&B"
            with m1:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">BT Nodes</div>
                    <div class="value blue">{bt_nodes}</div></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">B&B Nodes</div>
                    <div class="value green">{bb_nodes}</div></div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">B&B Pruned</div>
                    <div class="value amber">{bb_pruned}</div></div>""", unsafe_allow_html=True)
            with m4:
                reduction = round((1 - bb_nodes / max(bt_nodes, 1)) * 100, 1)
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Node Reduction</div>
                    <div class="value green">{reduction}%</div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Side by side
            col_bt, col_bb = st.columns(2)
            with col_bt:
                st.markdown('<div class="algo-header-bt"><h3>🔵 Backtracking</h3></div>', unsafe_allow_html=True)
                st.markdown(f"**Optimal Value:** `{bt_val}`")
                st.markdown(f"**Selected Items:** `{bt_items}`")
                st.markdown(f"**Time:** `{bt_time*1000:.3f} ms`")
                st.markdown("**Execution Steps:**")
                st.markdown(render_steps(bt_steps), unsafe_allow_html=True)

            with col_bb:
                st.markdown('<div class="algo-header-bb"><h3>🟢 Branch &amp; Bound</h3></div>', unsafe_allow_html=True)
                st.markdown(f"**Optimal Value:** `{bb_val}`")
                st.markdown(f"**Selected Items:** `{bb_items}`")
                st.markdown(f"**Time:** `{bb_time*1000:.3f} ms`")
                st.markdown(f"**Pruned Branches:** `{bb_pruned}`")
                st.markdown("**Execution Steps:**")
                st.markdown(render_steps(bb_steps), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📈 Performance Comparison")
            fig = plot_comparison_bar(bt_nodes, bb_nodes, bt_time, bb_time, "0/1 Knapsack")
            st.pyplot(fig)

            # Items table
            st.markdown("### 🎒 Item Details")
            df = pd.DataFrame({
                "Item": [f"Item {i+1}" for i in range(len(weights))],
                "Weight": weights,
                "Value": values,
                "Ratio (V/W)": [round(v/w, 2) for v, w in zip(values, weights)],
                "In BT Solution": ["✅" if i in bt_items else "❌" for i in range(len(weights))],
                "In B&B Solution": ["✅" if i in bb_items else "❌" for i in range(len(weights))],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

        # ── N-QUEENS ──────────────────────────────────────────────────────────
        elif problem == "N-Queens":
            t0 = time.perf_counter()
            solutions, bt_nodes, bt_steps = nqueens_backtracking(n_queens)
            bt_time = time.perf_counter() - t0

            st.markdown("### 📊 Results")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Solutions Found</div>
                    <div class="value amber">{len(solutions)}</div></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Nodes Explored</div>
                    <div class="value blue">{bt_nodes}</div></div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Time (ms)</div>
                    <div class="value green">{bt_time*1000:.2f}</div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.info("ℹ️ N-Queens is a constraint satisfaction problem — Backtracking is the primary technique. Branch & Bound adds cost bounds for optimization variants.")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown('<div class="algo-header-bt"><h3>🔵 Backtracking — N-Queens</h3></div>', unsafe_allow_html=True)
                st.markdown(f"**Board Size:** `{n_queens}×{n_queens}`")
                st.markdown(f"**Total Solutions:** `{len(solutions)}`")
                st.markdown("**Steps:**")
                st.markdown(render_steps(bt_steps, 30), unsafe_allow_html=True)

            with col2:
                if solutions:
                    fig = plot_nqueens_board(solutions[0], n_queens)
                    st.pyplot(fig)
                    st.caption(f"Showing solution 1 of {len(solutions)}: {solutions[0]}")

            # Show multiple boards
            if len(solutions) > 1:
                st.markdown("### 👑 All Solutions (first 6)")
                cols = st.columns(min(len(solutions), 3))
                for i, (sol, col) in enumerate(zip(solutions[:6], cols * 2)):
                    with col:
                        fig = plot_nqueens_board(sol, n_queens)
                        st.pyplot(fig)
                        st.caption(f"Solution {i+1}")

        # ── TSP ───────────────────────────────────────────────────────────────
        else:
            np.random.seed(int(seed))
            coords = [(float(np.random.uniform(0, 100)), float(np.random.uniform(0, 100)))
                      for _ in range(num_cities)]
            dist = [[((coords[i][0]-coords[j][0])**2 + (coords[i][1]-coords[j][1])**2)**0.5
                     for j in range(num_cities)] for i in range(num_cities)]

            t0 = time.perf_counter()
            bt_cost, bt_path, bt_nodes, bt_steps = tsp_backtracking(dist)
            bt_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            bb_cost, bb_path, bb_nodes, bb_pruned, bb_steps = tsp_branch_bound(dist)
            bb_time = time.perf_counter() - t0

            st.markdown("### 📊 Results")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">BT Nodes</div>
                    <div class="value blue">{bt_nodes}</div></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">B&B Nodes</div>
                    <div class="value green">{bb_nodes}</div></div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="metric-card">
                    <div class="label">B&B Pruned</div>
                    <div class="value amber">{bb_pruned}</div></div>""", unsafe_allow_html=True)
            with m4:
                reduction = round((1 - bb_nodes / max(bt_nodes, 1)) * 100, 1)
                st.markdown(f"""<div class="metric-card">
                    <div class="label">Node Reduction</div>
                    <div class="value green">{reduction}%</div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_bt, col_bb = st.columns(2)
            with col_bt:
                st.markdown('<div class="algo-header-bt"><h3>🔵 Backtracking — TSP</h3></div>', unsafe_allow_html=True)
                st.markdown(f"**Best Cost:** `{bt_cost:.2f}`")
                st.markdown(f"**Path:** `{' → '.join(map(str, bt_path))}`")
                st.markdown(f"**Time:** `{bt_time*1000:.3f} ms`")
                st.markdown("**Steps:**")
                st.markdown(render_steps(bt_steps), unsafe_allow_html=True)
                fig1 = plot_tsp_path(bt_path, coords, bt_cost, "Backtracking TSP")
                st.pyplot(fig1)

            with col_bb:
                st.markdown('<div class="algo-header-bb"><h3>🟢 Branch &amp; Bound — TSP</h3></div>', unsafe_allow_html=True)
                st.markdown(f"**Best Cost:** `{bb_cost:.2f}`")
                st.markdown(f"**Path:** `{' → '.join(map(str, bb_path))}`")
                st.markdown(f"**Time:** `{bb_time*1000:.3f} ms`")
                st.markdown(f"**Pruned:** `{bb_pruned}`")
                st.markdown("**Steps:**")
                st.markdown(render_steps(bb_steps), unsafe_allow_html=True)
                fig2 = plot_tsp_path(bb_path, coords, bb_cost, "Branch & Bound TSP")
                st.pyplot(fig2)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📈 Performance Comparison")
            fig = plot_comparison_bar(bt_nodes, bb_nodes, bt_time, bb_time, "TSP")
            st.pyplot(fig)

            # Distance matrix
            st.markdown("### 🗺️ Distance Matrix")
            df_dist = pd.DataFrame(
                [[round(dist[i][j], 1) for j in range(num_cities)] for i in range(num_cities)],
                columns=[f"C{i}" for i in range(num_cities)],
                index=[f"C{i}" for i in range(num_cities)]
            )
            st.dataframe(df_dist, use_container_width=True)

else:
    # Default landing state
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;color:#475569">
        <div style="font-size:3rem;margin-bottom:1rem">⚙️</div>
        <div style="font-size:1.1rem;color:#64748b">Select a problem from the sidebar and click <b style="color:#38bdf8">Run Algorithms</b></div>
        <div style="font-size:0.9rem;margin-top:0.5rem">Choose between 0/1 Knapsack, N-Queens, or TSP</div>
    </div>
    """, unsafe_allow_html=True)

    # Theory section
    st.markdown("---")
    st.markdown("### 📚 How They Work")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown('<div class="algo-header-bt"><h3>🔵 Backtracking</h3></div>', unsafe_allow_html=True)
        st.markdown("""
        1. **Build** solution incrementally step by step
        2. **Check** feasibility at each step (constraint)
        3. **Abandon** (backtrack) if constraint is violated
        4. **Continue** exploring all remaining possibilities
        5. **Return** all valid solutions

        **Pruning:** Only removes *infeasible* branches  
        **Use when:** You need all solutions or have hard constraints
        """)
    with t2:
        st.markdown('<div class="algo-header-bb"><h3>🟢 Branch &amp; Bound</h3></div>', unsafe_allow_html=True)
        st.markdown("""
        1. **Branch** — divide problem into sub-problems
        2. **Bound** — calculate upper/lower bound for sub-problem
        3. **Prune** — discard if bound ≤ current best solution
        4. **Select** — pick most promising branch next
        5. **Return** optimal solution

        **Pruning:** Removes *suboptimal* branches using bounds  
        **Use when:** You need the single best optimal solution
        """)