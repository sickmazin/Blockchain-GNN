"""Bounded, reproducible visualisations for Study A community partitions."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import math
from pathlib import Path
from typing import Any, Mapping

import networkx as nx
import pandas as pd

_MAX_PLOT_COMMUNITIES = 12


def _membership_mapping(partition: Mapping[Any, Any] | pd.DataFrame) -> dict[Any, Any]:
    """Normalize either a node-to-community mapping or compute_communities table."""
    if isinstance(partition, pd.DataFrame):
        required = {"node", "community"}
        if not required.issubset(partition.columns):
            raise ValueError(f"partition DataFrame must contain {sorted(required)}")
        if partition["node"].duplicated().any():
            raise ValueError("partition has multiple community assignments for a node")
        return partition.set_index("node")["community"].to_dict()
    return dict(partition)


def _select_plot_nodes(
    graph: nx.Graph,
    memberships: Mapping[Any, Any],
    max_nodes: int,
    weight: str,
    highlight_nodes: list[Any],
) -> list[Any]:
    """Select nodes from the largest communities in the largest component."""
    if max_nodes < 1:
        raise ValueError("max_nodes must be at least 1")
    eligible_nodes = [node for node in memberships if node in graph]
    if not eligible_nodes:
        return []
    eligible_graph = graph.subgraph(eligible_nodes)
    components = list(nx.connected_components(eligible_graph))
    largest_component = max(
        components,
        key=lambda component: (len(component), min(map(str, component))),
    )

    def rank(node: Any) -> tuple[float, str]:
        strength = float(graph.degree(node, weight=weight))
        return (-strength, str(node))

    highlight_order = [node for node in highlight_nodes if node in largest_component]
    highlighted_set = set(highlight_order)
    by_community: dict[Any, list[Any]] = {}
    for node in largest_component:
        by_community.setdefault(memberships[node], []).append(node)
    for nodes in by_community.values():
        nodes.sort(key=rank)
    highlight_communities = list(dict.fromkeys(memberships[node] for node in highlight_order))
    community_order = sorted(
        by_community,
        key=lambda community: (
            0 if community in highlight_communities else 1,
            -len(by_community[community]), str(community),
        ),
    )[:_MAX_PLOT_COMMUNITIES]

    # Include the rank highlights first, then sample evenly from the largest
    # communities so the map shows distinct groups instead of a hub-and-spoke
    # neighbourhood dominated by one high-degree address.
    selected: list[Any] = []
    selected_set: set[Any] = set()
    for node in highlight_order:
        if memberships[node] in community_order and node not in selected_set:
            selected.append(node)
            selected_set.add(node)
    remaining = {
        community: [node for node in by_community[community] if node not in selected_set]
        for community in community_order
    }
    depth = 0
    while len(selected) < max_nodes:
        added = False
        for community in community_order:
            candidates = remaining[community]
            if depth < len(candidates):
                node = candidates[depth]
                selected.append(node)
                selected_set.add(node)
                added = True
                if len(selected) == max_nodes:
                    break
        if not added:
            break
        depth += 1
    return selected


def _normalise_highlights(
    highlight_nodes: pd.DataFrame | Iterable[Any] | None,
    memberships: Mapping[Any, Any],
    graph: nx.Graph,
    measure: str,
) -> list[Any]:
    """Keep highlighted nodes in deterministic order, balanced by community."""
    if highlight_nodes is None:
        return []
    if isinstance(highlight_nodes, pd.DataFrame):
        if "node" not in highlight_nodes.columns:
            raise ValueError("highlight_nodes DataFrame must contain a 'node' column")
        if "community" in highlight_nodes.columns:
            groups = {
                community: (part.sort_values(measure, ascending=False, kind="stable")["node"].tolist()
                            if measure in part.columns else part["node"].tolist())
                for community, part in highlight_nodes.groupby("community", sort=True)
            }
            selected: list[Any] = []
            communities = sorted(groups, key=str)
            depth = 0
            while True:
                added = False
                for community in communities:
                    if depth < len(groups[community]):
                        selected.append(groups[community][depth])
                        added = True
                if not added:
                    break
                depth += 1
        else:
            selected = highlight_nodes["node"].tolist()
    else:
        selected = list(highlight_nodes)

    result: list[Any] = []
    seen: set[Any] = set()
    for node in selected:
        if node in memberships and node in graph and node not in seen:
            result.append(node)
            seen.add(node)
    return result


def _draw_partition(
    graph: nx.Graph,
    partition: Mapping[Any, Any] | pd.DataFrame,
    *,
    algorithm: str,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    max_nodes: int = 400,
    seed: int = 42,
    weight: str = "weight",
    highlight_nodes: pd.DataFrame | Iterable[Any] | None = None,
    highlight_measure: str = "pagerank",
) -> Any:
    """Draw a deterministic induced sample; full layouts are unsafe for large graphs."""
    if graph.is_directed():
        raise TypeError("community drawing expects the undirected projection")
    memberships = _membership_mapping(partition)
    highlights = _normalise_highlights(highlight_nodes, memberships, graph, highlight_measure)
    selected_nodes = _select_plot_nodes(graph, memberships, max_nodes, weight, highlights)
    if not selected_nodes:
        raise ValueError("no graph nodes have community assignments to draw")

    # Import plotting support lazily so graph analysis can be imported without
    # Matplotlib; the caller gets a clear message if it is not installed.
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Community plots require matplotlib to be installed") from exc

    plot_graph = graph.subgraph(selected_nodes).copy()
    assigned_graph = graph.subgraph([node for node in memberships if node in graph])
    largest_component = max(
        nx.connected_components(assigned_graph),
        key=lambda component: (len(component), min(map(str, component))),
        default=set(),
    )
    selected_membership = {node: memberships[node] for node in selected_nodes}
    community_values = sorted(set(selected_membership.values()), key=str)
    community_index = {community: index for index, community in enumerate(community_values)}
    color_ids = [community_index[selected_membership[node]] for node in plot_graph.nodes]

    # Lay out each detected community as a compact group, then arrange those
    # groups around a circle. This avoids isolated sampled outliers stretching
    # the overall layout and keeps community structure legible.
    nodes_by_community: dict[Any, list[Any]] = {}
    for node in plot_graph:
        nodes_by_community.setdefault(selected_membership[node], []).append(node)
    community_centers = nx.circular_layout(
        list(nodes_by_community), scale=max(2.5, 0.5 * len(nodes_by_community)),
    )
    positions: dict[Any, Any] = {}
    for index, (community, community_nodes) in enumerate(nodes_by_community.items()):
        subgraph = plot_graph.subgraph(community_nodes)
        if len(community_nodes) == 1:
            local_positions = {community_nodes[0]: (0.0, 0.0)}
        else:
            local_positions = nx.spring_layout(
                subgraph, seed=seed + index, weight=None, iterations=30, scale=1.0,
            )
        center = community_centers[community]
        cluster_radius = 0.45 + 0.07 * math.sqrt(len(community_nodes))
        for node, local_position in local_positions.items():
            positions[node] = (
                center[0] + local_position[0] * cluster_radius,
                center[1] + local_position[1] * cluster_radius,
            )
    fig, ax = plt.subplots(figsize=(12, 9))
    cmap = plt.get_cmap("turbo", max(len(community_values), 1))
    nx.draw_networkx_edges(plot_graph, positions, ax=ax, alpha=0.16, width=0.35,
                           edge_color="#667085")
    nx.draw_networkx_nodes(
        plot_graph,
        positions,
        ax=ax,
        node_size=22,
        node_color=color_ids,
        cmap=cmap,
        vmin=0,
        vmax=max(len(community_values) - 1, 1),
        linewidths=0,
    )
    visible_highlights = [node for node in highlights if node in plot_graph]
    if visible_highlights:
        nx.draw_networkx_nodes(
            plot_graph,
            positions,
            ax=ax,
            nodelist=visible_highlights,
            node_size=72,
            node_color="#e53935",
            edgecolors="#111827",
            linewidths=0.7,
            label=f"Top {len(highlights)} by {highlight_measure}",
        )
        ax.legend(loc="upper right", frameon=True, fontsize=9)
    visible_communities = len(set(selected_membership.values()))
    component_communities = len({memberships[node] for node in largest_component})
    ax.set_title(title or f"{algorithm} communities — induced sample")
    ax.text(
        0.01, 0.01,
        f"{len(selected_nodes):,}/{len(largest_component):,} largest-component nodes shown "
        f"(graph total={graph.number_of_nodes():,}); "
        f"{visible_communities:,}/{component_communities:,} largest-component communities shown "
        f"(up to {_MAX_PLOT_COMMUNITIES} selected); "
        f"{len(visible_highlights):,}/{len(highlights):,} highlighted nodes shown; "
        f"seed={seed}",
        transform=ax.transAxes,
        fontsize=9,
        color="#344054",
    )
    ax.set_axis_off()
    fig.tight_layout()

    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def draw_louvain(
    graph: nx.Graph,
    partition: Mapping[Any, Any] | pd.DataFrame,
    **kwargs: Any,
) -> Any:
    """Draw Louvain memberships; accepts a mapping or returned membership table."""
    return _draw_partition(graph, partition, algorithm="Louvain", **kwargs)


def draw_leiden(
    graph: nx.Graph,
    partition: Mapping[Any, Any] | pd.DataFrame,
    **kwargs: Any,
) -> Any:
    """Draw Leiden memberships; accepts a mapping or returned membership table."""
    return _draw_partition(graph, partition, algorithm="Leiden", **kwargs)


def draw_overlap(
    summary: pd.DataFrame,
    overlaps: pd.DataFrame,
    *,
    metric: str = "overlap_coefficient_on_shared_graph_nodes",
    top_k: int = 40,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Any:
    """Plot community-to-community overlap from ``community_overlap`` outputs.

    The figure is bipartite: left nodes are communities from token A, right
    nodes are communities from token B, and edges represent shared addresses.
    Only the strongest ``top_k`` overlap edges are drawn for readability.
    """
    if summary.empty:
        raise ValueError("summary must contain the one-row community_overlap summary")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    required_summary = {"token_a", "token_b", "window_a", "window_b", "shared_node_count"}
    if not required_summary.issubset(summary.columns):
        raise ValueError(f"summary must contain {sorted(required_summary)}")
    required_overlaps = {
        "community_a", "community_b", "overlapping_address_count", metric,
        "shared_addresses_in_community_a", "shared_addresses_in_community_b",
    }
    if not required_overlaps.issubset(overlaps.columns) and not overlaps.empty:
        raise ValueError(f"overlaps must contain {sorted(required_overlaps)}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Overlap plots require matplotlib to be installed") from exc

    info = summary.iloc[0]
    fig, ax = plt.subplots(figsize=(13, 8))
    heading = title or (
        f"Community overlap: {info['token_a']} ({info['window_a']}) ↔ "
        f"{info['token_b']} ({info['window_b']})"
    )
    ax.set_title(heading)

    if int(info["shared_node_count"]) == 0 or overlaps.empty:
        ax.text(
            0.5, 0.52,
            "No community overlap found\n"
            f"Shared addresses between graphs: {int(info['shared_node_count']):,}",
            ha="center", va="center", transform=ax.transAxes, fontsize=14,
        )
        ax.text(0.5, 0.42, f"Score shown: {metric}", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#475467")
        ax.set_axis_off()
    else:
        selected = overlaps.sort_values(
            [metric, "overlapping_address_count"], ascending=[False, False], kind="stable",
        ).head(top_k)
        graph = nx.Graph()
        side_a: list[tuple[str, Any]] = []
        side_b: list[tuple[str, Any]] = []
        labels: dict[tuple[str, Any], str] = {}
        sizes: dict[tuple[str, Any], float] = {}
        for row in selected.itertuples(index=False):
            a = ("A", row.community_a)
            b = ("B", row.community_b)
            if a not in graph:
                side_a.append(a)
                graph.add_node(a, bipartite=0)
                labels[a] = f"A · C{row.community_a}\n(n={row.shared_addresses_in_community_a:,})"
                sizes[a] = float(row.shared_addresses_in_community_a)
            if b not in graph:
                side_b.append(b)
                graph.add_node(b, bipartite=1)
                labels[b] = f"B · C{row.community_b}\n(n={row.shared_addresses_in_community_b:,})"
                sizes[b] = float(row.shared_addresses_in_community_b)
            graph.add_edge(a, b, score=float(getattr(row, metric)),
                           shared=int(row.overlapping_address_count))

        positions = nx.bipartite_layout(graph, side_a, align="vertical", scale=1.0)
        max_size = max(sizes.values(), default=1.0)
        node_size_map = {node: 260 + 1000 * math.sqrt(sizes[node] / max_size) for node in graph.nodes}
        max_score = max((attrs["score"] for _, _, attrs in graph.edges(data=True)), default=1.0)
        edge_widths = [0.8 + 4.2 * attrs["score"] / max_score
                       for _, _, attrs in graph.edges(data=True)]
        nx.draw_networkx_edges(graph, positions, ax=ax, width=edge_widths,
                               alpha=0.55, edge_color="#667085")
        nx.draw_networkx_edge_labels(
            graph, positions,
            edge_labels={(u, v): f"n={attrs['shared']}" for u, v, attrs in graph.edges(data=True)},
            ax=ax, font_size=7, rotate=False, label_pos=0.5,
        )
        nx.draw_networkx_nodes(graph, positions, ax=ax, nodelist=side_a,
                               node_color="#4e79a7", node_size=[node_size_map[n] for n in side_a])
        nx.draw_networkx_nodes(graph, positions, ax=ax, nodelist=side_b,
                               node_color="#59a14f", node_size=[node_size_map[n] for n in side_b])
        nx.draw_networkx_labels(graph, positions, labels=labels, ax=ax, font_size=7)
        ax.text(
            0.01, 0.01,
            f"Shared addresses between graphs: {int(info['shared_node_count']):,}; "
            f"edges shown: {len(selected):,}/{len(overlaps):,}; score: {metric}",
            transform=ax.transAxes, fontsize=9, color="#344054",
        )
        ax.set_axis_off()

    fig.tight_layout()
    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
