"""Graph construction and descriptive SNA for Study A (ERC-20 TGS data).

Raw transfer rows retain their event meaning. Graphs are directed aggregate
views; no address is interpreted as a real-world person or organization.
Derived outputs should be written outside ``datasets/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Mapping
import scipy
import networkx as nx
import pandas as pd
import leidenalg
import igraph

try:
    from .preprocessing import ZERO_ADDRESS, aggregate_window, load_token_events
    from .draw import draw_leiden, draw_louvain
except ImportError:
    from preprocessing import ZERO_ADDRESS, aggregate_window, load_token_events
    from draw import draw_leiden, draw_louvain


WEIGHT_COLUMNS = {
    "frequency": "event_count",
    "raw_volume": "volume",
    "log1p_volume": "log1p_volume",
}


@dataclass
class GraphView:
    """A NetworkX graph plus the identity and filtering audit for that view."""

    graph: nx.DiGraph
    name: str
    weight_column: str
    metadata: dict[str, Any] = field(default_factory=dict)
    filter_audit: dict[str, Any] = field(default_factory=dict)


def _networkx_version() -> str:
    try:
        return version("networkx")
    except PackageNotFoundError:
        return "unknown"


def _finite_numeric(series: pd.Series, column: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    invalid = values.isna() | ~values.map(lambda x: math.isfinite(float(x)) if pd.notna(x) else False)
    if invalid.any():
        raise ValueError(f"{column} contains {int(invalid.sum())} missing or non-finite values")
    return values


def build_graphs(
    aggregates: pd.DataFrame,
    *,
    variants: list[str] | tuple[str, ...] | None = None,
    graph_info: Mapping[str, Any] | None = None,
    exclude_self_loops: bool = True,
    exclude_zero_address: bool = True,
    zero_address: str = ZERO_ADDRESS,
) -> dict[str, GraphView]:
    """Build selected directed frequency, raw-volume, and log1p-volume views.

    ``aggregates`` is typically the output of ``preprocessing.aggregate_window``.
    Audits are computed before filters. A negative transfer amount is rejected:
    it cannot be interpreted as a positive tie strength or transformed by log1p.
    """
    data = aggregates.copy()
    #  PROB DA RIMUOVERE
    #data["source"] = data["source"].astype(str).str.strip()
    #data["destination"] = data["destination"].astype(str).str.strip()
    #data["event_count"] = _finite_numeric(data["event_count"], "event_count")
    #data["volume"] = _finite_numeric(data["volume"], "volume")
    #if (data["event_count"] < 0).any() or (data["volume"] < 0).any():
    #    raise ValueError("event_count and volume must be non-negative")
    #if data.duplicated(["source", "destination"]).any():
    #    raise ValueError("aggregates must contain one row per ordered pair (and window)")
    data["log1p_volume"] = data["volume"].map(math.log1p)

    is_loop = data["source"].eq(data["destination"])
    is_zero = data["source"].eq(zero_address) | data["destination"].eq(zero_address)
    before = {
        "aggregate_edges": int(len(data)),
        "events": int(data["event_count"].sum()),
        "raw_volume": float(data["volume"].sum()),
        "nodes": int(pd.unique(pd.concat([data["source"], data["destination"]])).size),
        "self_loop_edges": int(is_loop.sum()),
        "self_loop_events": int(data.loc[is_loop, "event_count"].sum()),
        "self_loop_volume": float(data.loc[is_loop, "volume"].sum()),
        "zero_address_edges": int(is_zero.sum()),
        "zero_address_events": int(data.loc[is_zero, "event_count"].sum()),
        "zero_address_volume": float(data.loc[is_zero, "volume"].sum()),
    }

    keep = pd.Series(True, index=data.index)
    removed_by_filter: dict[str, Any] = {}
    if exclude_self_loops:
        removed_by_filter["self_loops"] = {
            "edges": int((keep & is_loop).sum()),
            "events": int(data.loc[keep & is_loop, "event_count"].sum()),
            "volume": float(data.loc[keep & is_loop, "volume"].sum()),
        }
        keep &= ~is_loop #aggiorna la maschera booleana keep mantenendo solo le righe che non sono self-loop.
    if exclude_zero_address:
        removed_by_filter["zero_address"] = {
            "edges": int((keep & is_zero).sum()),
            "events": int(data.loc[keep & is_zero, "event_count"].sum()),
            "volume": float(data.loc[keep & is_zero, "volume"].sum()),
        }
        keep &= ~is_zero #aggiorna la maschera booleana keep mantenendo solo le righe che non sono con zero address.
    filtered = data.loc[keep].copy()
    after = {
        "aggregate_edges": int(len(filtered)),
        "events": int(filtered["event_count"].sum()),
        "raw_volume": float(filtered["volume"].sum()),
        "nodes": int(pd.unique(pd.concat([filtered["source"], filtered["destination"]])).size)
        if len(filtered) else 0,
    }
    audit = {"before_filters": before, "removed_by_filter": removed_by_filter,
             "after_filters": after, "exclude_self_loops": exclude_self_loops,
             "exclude_zero_address": exclude_zero_address}

    metadata = dict(graph_info or {})
    metadata.setdefault("networkx_version", _networkx_version())
    metadata.setdefault("directed", True)
    metadata.setdefault("edge_semantics", "ERC-20 token transfer between addresses")
    metadata.setdefault("filter_audit", audit)
    views: dict[str, GraphView] = {}
    selected_variants = list(WEIGHT_COLUMNS) if variants is None else list(dict.fromkeys(variants))
    unknown_variants = [variant for variant in selected_variants if variant not in WEIGHT_COLUMNS]
    if unknown_variants:
        raise ValueError(f"unknown graph variants {unknown_variants}; choose from {list(WEIGHT_COLUMNS)}")
    if not selected_variants:
        raise ValueError("select at least one graph variant")
    for name in selected_variants:
        weight_column = WEIGHT_COLUMNS[name]
        graph = nx.DiGraph()
        for row in filtered.itertuples(index=False):
            graph.add_edge(
                row.source,
                row.destination,
                weight=float(getattr(row, weight_column)),
                event_count=int(row.event_count),
                raw_volume=float(row.volume),
                log1p_volume=float(row.log1p_volume),
            )
        view_metadata = {**metadata, "variant": name, "weight_definition": weight_column}
        views[name] = GraphView(graph, name, weight_column, view_metadata, audit)
    return views


def build_graphs_from_events(
    events: pd.DataFrame,
    *,
    variants: list[str] | tuple[str, ...] | None = None,
    graph_info: Mapping[str, Any] | None = None,
    exclude_self_loops: bool = True,
    exclude_zero_address: bool = True,
) -> tuple[dict[str, GraphView], pd.DataFrame]:
    """Aggregate events, then build only the requested named graph views."""
    aggregates = aggregate_window(events)
    return build_graphs(
        aggregates,
        variants=variants,
        graph_info=graph_info,
        exclude_self_loops=exclude_self_loops,
        exclude_zero_address=exclude_zero_address,
    ), aggregates


def _coerce_view(graph: GraphView | nx.DiGraph, graph_info: Mapping[str, Any] | None = None) -> GraphView:
    if isinstance(graph, GraphView):
        return graph
    if not isinstance(graph, nx.DiGraph):
        raise TypeError("graph must be a GraphView or networkx.DiGraph")
    info = dict(graph_info or {})
    name = str(info.get("variant", "unspecified"))
    weight_column = str(info.get("weight_definition", "weight"))
    return GraphView(graph, name, weight_column, info, dict(info.get("filter_audit", {})))

# info about graph density, degree, num node & edge etc
def describe_graph(
    graph: GraphView | nx.DiGraph,
    graph_info: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return graph-level directed descriptors and one row per node."""
    view = _coerce_view(graph, graph_info)
    g = view.graph
    node, edge = g.number_of_nodes(), g.number_of_edges()
    loop_count = nx.number_of_selfloops(g)
    non_loop_edges = edge - loop_count
    # SOME METRICS
    density = non_loop_edges / (node * (node - 1))
    reciprocal_edges = sum(1 for u, v in g.edges if u != v and g.has_edge(v, u))
    reciprocity = reciprocal_edges / edge
    weak_components = list(nx.weakly_connected_components(g))
    strong_count = nx.number_strongly_connected_components(g)
    largest_weak = max((len(c) for c in weak_components), default=0)
    summary: dict[str, Any] = {
        **view.metadata,
        "graph": view.name,
        "weight_definition": view.weight_column,
        "nodes": node,
        "directed_edges": edge,
        "directed_non_loop_edges": non_loop_edges,
        "density_directed_no_loops_denominator": node * (node - 1),
        "density": density,
        "sparsity": 1 - density,
        "reciprocity_reciprocal_arcs_over_all_arcs": reciprocity,
        "reciprocal_directed_arcs": reciprocal_edges,
        "self_loops_in_analytical_graph": loop_count,
        "weak_component_count": len(weak_components),
        "strong_component_count": strong_count,
        "largest_weak_component_nodes": largest_weak,
        "largest_weak_component_share": largest_weak / node,
        "in_strength_total": float(sum(dict(g.in_degree(weight="weight")).values())),
        "out_strength_total": float(sum(dict(g.out_degree(weight="weight")).values())),
    }
    rows = []
    for node in sorted(g.nodes, key=str):
        rows.append({
            "node": node,
            "in_degree": int(g.in_degree(node)),
            "out_degree": int(g.out_degree(node)),
            "in_strength": float(g.in_degree(node, weight="weight")),
            "out_strength": float(g.out_degree(node, weight="weight")),
            **{k: v for k, v in view.metadata.items() if isinstance(v, (str, int, float, bool)) or v is None},
            "graph": view.name,
            "weight_definition": view.weight_column,
        })
    return summary, pd.DataFrame(rows)


def _distance_graph(g: nx.DiGraph) -> nx.DiGraph:
    """Return a copy with inverse-log-strength distances for positive strengths."""
    result = nx.DiGraph()
    result.add_nodes_from(g.nodes(data=True))
    for u, v, attrs in g.edges(data=True):
        strength = float(attrs.get("weight", 1.0))
        if not math.isfinite(strength) or strength <= 0:
            raise ValueError("distance conversion requires finite positive edge weights")
        result.add_edge(u, v, **attrs, distance=1.0 / math.log1p(strength))
    return result


def compute_centralities(
    graph: GraphView | nx.DiGraph,
    graph_info: Mapping[str, Any] | None = None,
    *,
    seed: int = 42,
    betweenness_k: int | None = None,
    top_k: int = 20,
    max_iter: int = 5000,
    tolerance: float = 1e-8,
    calcolate_betweenness : bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute directed centralities and return full, top-k, and run-status tables.

    Betweenness is exact unless ``betweenness_k`` requests deterministic source
    sampling. Weighted path measures use ``distance = 1/log1p(strength)``.
    Individual algorithm failures are recorded in the status table.
    """
    view = _coerce_view(graph, graph_info)
    g = view.graph
    nodes = sorted(g.nodes, key=str)
    logger = logging.getLogger("study_A")
    values: dict[str, dict[Any, float]] = {name: {} for name in (
        "in_degree_centrality", "out_degree_centrality", "betweenness",
        "closeness_in", "closeness_out", "eigenvector_in", "eigenvector_out",
        "pagerank",
    )}
    statuses: list[dict[str, Any]] = []

    def record(name: str, action: Any, *, approximate: bool = False) -> None:
        started = time.perf_counter()
        logger.info("Centralità %s | %s: avvio", view.name, name)
        try:
            result = action()
            values[name] = {node: float(result.get(node, 0.0)) for node in nodes}
            statuses.append({"measure": name, "status": "ok", "approximate": approximate,
                             "seed": seed, "error": None})
            logger.info("Centralità %s | %s: completata in %.1f s",
                        view.name, name, time.perf_counter() - started)
        except Exception as exc:  # Keep failures visible; never replace by zero silently.
            statuses.append({"measure": name, "status": "omitted", "approximate": approximate,
                             "seed": seed, "error": f"{type(exc).__name__}: {exc}"})
            logger.exception("Centralità %s | %s: fallita dopo %.1f s",
                             view.name, name, time.perf_counter() - started)

    record("in_degree_centrality", lambda: nx.in_degree_centrality(g))
    record("out_degree_centrality", lambda: nx.out_degree_centrality(g))
    dg = None
    if calcolate_betweenness:
        try:
            dg = _distance_graph(g)
            logger.info("Centralità %s | trasformazione pesi in distanze completata", view.name)
        except Exception as exc:
            statuses.append({"measure": "distance_transform", "status": "omitted", "approximate": False,
                             "seed": seed, "error": f"{type(exc).__name__}: {exc}"})
            logger.exception("Centralità %s | trasformazione pesi in distanze fallita", view.name)
    if dg is not None:
        if calcolate_betweenness:
            record("betweenness", lambda: nx.betweenness_centrality(
                dg, k=betweenness_k if  g.number_of_nodes() > betweenness_k else None, normalized=True, weight="distance", seed=seed,
            ), approximate=True)

            # NetworkX closeness on a directed graph measures incoming distances.
            record("closeness_in", lambda: nx.closeness_centrality(dg, distance="distance"))
            record("closeness_out", lambda: nx.closeness_centrality(dg.reverse(copy=False), distance="distance"))
    elif calcolate_betweenness:
        statuses.append({"measure": "betweenness", "status": "omitted", "approximate": False,
                         "seed": seed, "error": "valid distance transform unavailable"})
        statuses.extend({"measure": measure, "status": "omitted", "approximate": False,
                         "seed": seed, "error": "valid distance transform unavailable"}
                        for measure in ("closeness_in", "closeness_out"))
    record("eigenvector_in", lambda: nx.eigenvector_centrality(
        g, max_iter=max_iter, tol=tolerance, weight="weight",
    ))
    record("eigenvector_out", lambda: nx.eigenvector_centrality(
        g.reverse(copy=False), max_iter=max_iter, tol=tolerance, weight="weight",
    ))
    record("pagerank", lambda: nx.pagerank(g, alpha=0.85, max_iter=max_iter,
                                             tol=tolerance, weight="weight"))

    common_metadata = {k: v for k, v in view.metadata.items()
                       if isinstance(v, (str, int, float, bool)) or v is None}
    rows = []
    for node in nodes:
        row = {"node": node, **common_metadata, "graph": view.name,
               "weight_definition": view.weight_column}
        row.update({measure: scores.get(node, math.nan) for measure, scores in values.items()})
        rows.append(row)
    full = pd.DataFrame(rows)
    top_rows = []
    for measure in values:
        if measure not in full:
            continue
        ordered = full.sort_values([measure, "node"], ascending=[False, True], kind="stable").head(top_k)
        for rank, (_, row) in enumerate(ordered.iterrows(), 1):
            top_rows.append({"measure": measure, "rank": rank, "node": row["node"], "score": row[measure],
                             "graph": view.name, "weight_definition": view.weight_column})
    return full, pd.DataFrame(top_rows), pd.DataFrame(statuses)


def make_undirected_projection(
    graph: GraphView | nx.DiGraph,
    *,
    weight: str = "weight",
    name: str | None = None,
    keep_self_loops: bool = False,
) -> nx.Graph:
    """Sum reciprocal directed weights into a named undirected projection."""
    view = _coerce_view(graph)
    directed = view.graph
    projection = nx.Graph()
    projection.graph.update({**view.metadata, "projection": name or f"undirected_sum_{weight}",
                             "source_graph": view.name, "projection_rule": "sum reciprocal arc weights",
                             "self_loops_kept": keep_self_loops})
    projection.add_nodes_from(directed.nodes(data=True))
    for u, v, attrs in directed.edges(data=True):
        if u == v and not keep_self_loops:
            continue
        value = float(attrs.get(weight, 1.0))
        if projection.has_edge(u, v):
            projection[u][v]["weight"] += value
        else:
            projection.add_edge(u, v, weight=value)
    return projection


def describe_projection(projection: nx.Graph) -> dict[str, Any]:
    """Describe clustering, assortativity, and path length on largest component."""
    if projection.is_directed():
        raise TypeError("projection must be undirected")

    node, edge = projection.number_of_nodes(), projection.number_of_edges()
    largest = max(nx.connected_components(projection), key=len, default=set())
    component = projection.subgraph(largest).copy()
    try:
        assortativity = float(nx.degree_assortativity_coefficient(projection)) if edge else math.nan
    except Exception:
        assortativity = math.nan
    try:
        avg_path = float(nx.average_shortest_path_length(component)) if len(component) > 1 else 0.0
        diameter = int(nx.diameter(component)) if len(component) > 1 else 0
        path_error = None
    except Exception as exc:
        avg_path, diameter, path_error = math.nan, math.nan, f"{type(exc).__name__}: {exc}"
    return {
        **projection.graph,
        "nodes": node,
        "edges": edge,
        "density": nx.density(projection),
        "clustering_binary_average": float(nx.average_clustering(projection)) if node else 0.0,
        "clustering_binary_by_node": nx.clustering(projection),
        "degree_assortativity": assortativity,
        "largest_component_nodes": len(component),
        "largest_component_share": len(component) / node if node else 0.0,
        "largest_component_average_path_length_unweighted": avg_path,
        "largest_component_diameter_unweighted": diameter,
        "path_metric_error": path_error,
    }


def compute_communities(
    graph_und: nx.Graph,
    *,
    seed: int = 42,
    resolution: float = 1.0,
    weight: str = "weight",
) -> tuple[pd.DataFrame,pd.DataFrame, dict[str, Any]]:
    """Use Leiden when installed, otherwise NetworkX Louvain; save membership."""
    if graph_und.is_directed():
        raise TypeError("community detection requires a named undirected projection")

    # LEIDEN ALGORITHM
    nodes = sorted(graph_und.nodes, key=str)
    index = {node: i for i, node in enumerate(nodes)}
    edges = [(index[u], index[v]) for u, v in graph_und.edges]
    ig = igraph.Graph(n=len(nodes), edges=edges, directed=False)
    ig.es["weight"] = [float(graph_und[u][v].get(weight, 1.0)) for u, v in graph_und.edges]
    partition = leidenalg.find_partition(
        ig, leidenalg.RBConfigurationVertexPartition, weights="weight",
        resolution_parameter=resolution, seed=seed,
    )
    leiden_communities = [set(nodes[i] for i in members) for members in partition]
    # Evaluate both partitions with NetworkX's same resolution-aware definition.
    leiden_modularity = float(nx.algorithms.community.modularity(
        graph_und, leiden_communities, weight=weight, resolution=resolution,
    ))
    leiden_version = f"leidenalg={version('leidenalg')}; igraph={version('igraph')}"

    leiden_communities = sorted((set(c) for c in leiden_communities), key=lambda c: (-len(c), min(map(str, c), default="")))
    leiden_membership = {node: community_id for community_id, community in enumerate(leiden_communities)
                          for node in community}
    leiden_rows = [{"node": node, "community": leiden_membership[node], "community_size": len(leiden_communities[leiden_membership[node]])}
                    for node in sorted(leiden_membership, key=str)]

    # LOUVAIN
    louvain = getattr(nx.algorithms.community, "louvain_communities", None)
    louvain_communities = louvain(graph_und, weight=weight, resolution=resolution, seed=seed)
    louvain_modularity = float(nx.algorithms.community.modularity(graph_und, louvain_communities, weight=weight,
                                                           resolution=resolution))
    louvain_version = f"networkx={_networkx_version()}"
    louvain_communities = sorted((set(c) for c in louvain_communities), key=lambda c: (-len(c), min(map(str, c), default="")))
    louvain_membership = {node: community_id for community_id, community in enumerate(louvain_communities)
                  for node in community}
    louvain_rows = [{"node": node, "community": louvain_membership[node], "community_size": len(louvain_communities[louvain_membership[node]])}
            for node in sorted(louvain_membership, key=str)]
    metadata = {
        **graph_und.graph,
        "seed": seed,
        "resolution": resolution,
        "weight": weight,
        "leiden_version": leiden_version,
        "leiden_modularity": leiden_modularity,
        "leiden_community_count": len(leiden_communities),
        "leiden_community_sizes": [len(c) for c in leiden_communities],
        "louvain_version": louvain_version,
        "louvain_modularity": louvain_modularity,
        "louvain_community_count": len(louvain_communities),
        "louvain_community_sizes": [len(c) for c in louvain_communities],

    }
    return pd.DataFrame(louvain_rows), pd.DataFrame(leiden_rows), metadata


def community_central_nodes(
    memberships: pd.DataFrame,
    centralities: pd.DataFrame,
    *,
    measure: str = "pagerank",
    top_k: int = 5,
) -> pd.DataFrame:
    """Return top-k structurally central addresses within each community."""
    required_members = {"node", "community"}
    if not required_members.issubset(memberships.columns):
        raise ValueError(f"memberships must contain {sorted(required_members)}")
    if not {"node", measure}.issubset(centralities.columns):
        raise ValueError(f"centralities must contain node and {measure!r}")
    joined = memberships.merge(centralities[["node", measure]], on="node", how="left", validate="one_to_one")
    joined = joined.sort_values(["community", measure, "node"], ascending=[True, False, True], kind="stable")
    return joined.groupby("community", sort=True, group_keys=False).head(top_k).reset_index(drop=True)


def community_persistence(
    previous: pd.DataFrame,
    current: pd.DataFrame,
) -> pd.DataFrame:
    """Compare adjacent-window communities by Jaccard overlap of addresses."""
    required = {"node", "community"}
    if not required.issubset(previous.columns) or not required.issubset(current.columns):
        raise ValueError("both membership tables must contain node and community")
    prev_sets = {c: set(g["node"]) for c, g in previous.groupby("community")}
    curr_sets = {c: set(g["node"]) for c, g in current.groupby("community")}
    rows = []
    for old_id, old_nodes in prev_sets.items():
        for new_id, new_nodes in curr_sets.items():
            union = old_nodes | new_nodes
            intersection = old_nodes & new_nodes
            if intersection:
                rows.append({"previous_community": old_id, "current_community": new_id,
                             "intersection_nodes": len(intersection), "jaccard": len(intersection) / len(union),
                             "previous_size": len(old_nodes), "current_size": len(new_nodes)})
    return pd.DataFrame(rows, columns=["previous_community", "current_community", "intersection_nodes",
                                       "jaccard", "previous_size", "current_size"])


def community_overlap(
    memberships_a: pd.DataFrame,
    memberships_b: pd.DataFrame,
    *,
    token_a: str,
    token_b: str,
    window_a: str,
    window_b: str,
    algorithm_a: str = "unspecified",
    algorithm_b: str = "unspecified",
    variant_a: str = "unspecified",
    variant_b: str = "unspecified",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare communities across two token graphs using shared addresses first.

    Community sets are restricted to nodes present in both graphs before pairwise
    overlap is calculated. Returns (one-row graph summary, non-zero community
    overlaps, shared-address membership table). ``window_a`` and ``window_b``
    are required labels so the temporal scope of each graph is explicit; use
    comparable date ranges when the goal is a token-to-token comparison.
    """
    #required = {"node", "community"}
    #for label, memberships in (("memberships_a", memberships_a), ("memberships_b", memberships_b)):
    #    if not required.issubset(memberships.columns):
    #        raise ValueError(f"{label} must contain columns {sorted(required)}")
    #    if memberships["node"].isna().any() or memberships["community"].isna().any():
    #        raise ValueError(f"{label} contains missing node/community values")
    #    if memberships["node"].duplicated().any():
    #        raise ValueError(f"{label} must contain exactly one community assignment per node")
    #if not token_a or not token_b:
    #    raise ValueError("token_a and token_b must be non-empty")
    #if not window_a or not window_b:
    #    raise ValueError("window_a and window_b must be non-empty temporal labels")

    assignment_a = memberships_a.set_index("node")["community"].to_dict()
    assignment_b = memberships_b.set_index("node")["community"].to_dict()
    nodes_a, nodes_b = set(assignment_a), set(assignment_b)
    shared_nodes = sorted(nodes_a & nodes_b, key=str)
    common_count = len(shared_nodes)

    context = {
        "token_a": token_a,
        "token_b": token_b,
        "window_a": window_a,
        "window_b": window_b,
        "algorithm_a": algorithm_a,
        "algorithm_b": algorithm_b,
        "variant_a": variant_a,
        "variant_b": variant_b,
    }
    summary = pd.DataFrame([{
        **context,
        "nodes_a": len(nodes_a),
        "nodes_b": len(nodes_b),
        "shared_node_count": common_count,
        "shared_fraction_of_a": common_count / len(nodes_a) if nodes_a else 0.0,
        "shared_fraction_of_b": common_count / len(nodes_b) if nodes_b else 0.0,
        "community_count_a": memberships_a["community"].nunique(),
        "community_count_b": memberships_b["community"].nunique(),
    }])

    shared_addresses = pd.DataFrame([
        {**context, "node": node, "community_a": assignment_a[node], "community_b": assignment_b[node]}
        for node in shared_nodes
    ], columns=[*context.keys(), "node", "community_a", "community_b"])

    shared_communities_a: dict[Any, set[Any]] = {}
    shared_communities_b: dict[Any, set[Any]] = {}
    for node in shared_nodes:
        shared_communities_a.setdefault(assignment_a[node], set()).add(node)
        shared_communities_b.setdefault(assignment_b[node], set()).add(node)

    overlap_rows = []
    for community_a, addresses_a in shared_communities_a.items():
        for community_b, addresses_b in shared_communities_b.items():
            intersection = addresses_a & addresses_b
            if not intersection:
                continue
            union = addresses_a | addresses_b
            smaller_size = min(len(addresses_a), len(addresses_b))
            overlap_rows.append({
                **context,
                "community_a": community_a,
                "community_b": community_b,
                "shared_addresses_in_community_a": len(addresses_a),
                "shared_addresses_in_community_b": len(addresses_b),
                "overlapping_address_count": len(intersection),
                "jaccard_on_shared_graph_nodes": len(intersection) / len(union),
                "overlap_coefficient_on_shared_graph_nodes": len(intersection) / smaller_size,
                "addresses_a_covered": len(intersection) / len(addresses_a),
                "addresses_b_covered": len(intersection) / len(addresses_b),
            })
    overlap_columns = [
        *context.keys(), "community_a", "community_b",
        "shared_addresses_in_community_a", "shared_addresses_in_community_b",
        "overlapping_address_count", "jaccard_on_shared_graph_nodes",
        "overlap_coefficient_on_shared_graph_nodes", "addresses_a_covered", "addresses_b_covered",
    ]
    overlaps = pd.DataFrame(overlap_rows, columns=overlap_columns)
    if not overlaps.empty:
        overlaps = overlaps.sort_values(
            ["overlap_coefficient_on_shared_graph_nodes", "jaccard_on_shared_graph_nodes",
             "overlapping_address_count", "community_a", "community_b"],
            ascending=[False, False, False, True, True],
            kind="stable",
        ).reset_index(drop=True)
    return summary, overlaps, shared_addresses




def analyze_windows(
    events: pd.DataFrame,
    *,
    variants: list[str] | tuple[str, ...] | None = None,
    graph_info: Mapping[str, Any] | None = None,
    exclude_self_loops: bool = True,
    exclude_zero_address: bool = True,
    centrality_seed: int = 42,
    betweenness_k: int | None = None,
    calculate_path_centralities: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run graph construction and core summaries independently per window.

    Events without a ``window`` column are treated as one explicitly named
    ``all_events`` window. This function does not create or infer time cutoffs.
    """
    if "window" in events.columns:
        groups = list(events.loc[events["window"].notna()].groupby("window", sort=True))
    else:
        groups = [("all_events", events)]
    logger = logging.getLogger("study_A")
    selected_variants = list(WEIGHT_COLUMNS) if variants is None else list(dict.fromkeys(variants))
    unknown_variants = [variant for variant in selected_variants if variant not in WEIGHT_COLUMNS]
    if unknown_variants:
        raise ValueError(f"unknown graph variants {unknown_variants}; choose from {list(WEIGHT_COLUMNS)}")
    if not selected_variants:
        raise ValueError("select at least one graph variant")
    logger.info("Analisi diretta: %d finestre, tipi arco=%s",
                len(groups), ", ".join(selected_variants))
    summaries, node_rows, centrality_rows, top_rows, status_rows = [], [], [], [], []
    for window_index, (window_name, subset) in enumerate(groups, start=1):
        info = {**dict(graph_info or {}), "window": str(window_name)}
        logger.info("Analisi diretta: finestra %d/%d %s | %s eventi",
                    window_index, len(groups), window_name, f"{len(subset):,}")
        views, aggregates = build_graphs_from_events(
            subset, variants=selected_variants, graph_info=info, exclude_self_loops=exclude_self_loops,
            exclude_zero_address=exclude_zero_address,
        )
        views = {name: view for name, view in views.items() if name in selected_variants}
        for variant_index, (variant, view) in enumerate(views.items(), start=1):
            variant_started = time.perf_counter()
            logger.info("Centralità: finestra %s | arco %s (%d/%d) avviata | nodi=%s archi=%s",
                        window_name, variant, variant_index, len(views),
                        f"{view.graph.number_of_nodes():,}", f"{view.graph.number_of_edges():,}")
            summary, nodes = describe_graph(view)
            summaries.append(summary)
            if not nodes.empty:
                node_rows.append(nodes)
            full, top, statuses = compute_centralities(
                view,
                seed=centrality_seed,
                betweenness_k=betweenness_k,
                calcolate_betweenness=calculate_path_centralities,
            )
            if not calculate_path_centralities:
                statuses = pd.concat([
                    statuses,
                    pd.DataFrame([
                        {"measure": measure, "status": "disabled_by_config", "approximate": False,
                         "seed": centrality_seed, "error": "expensive shortest-path centrality disabled by main configuration"}
                        for measure in ("betweenness", "closeness_in", "closeness_out")
                    ]),
                ], ignore_index=True)
            if not full.empty:
                centrality_rows.append(full)
            if not top.empty:
                top_rows.append(top.assign(window=str(window_name), variant=variant))
            if not statuses.empty:
                status_rows.append(statuses.assign(window=str(window_name), variant=variant))
            logger.info("Centralità: finestra %s | arco %s completata in %.1f s",
                        window_name, variant, time.perf_counter() - variant_started)
    return {
        "graph_summary": pd.DataFrame(summaries),
        "node_descriptives": pd.concat(node_rows, ignore_index=True) if node_rows else pd.DataFrame(),
        "centralities": pd.concat(centrality_rows, ignore_index=True) if centrality_rows else pd.DataFrame(),
        "centrality_top_k": pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame(),
        "centrality_status": pd.concat(status_rows, ignore_index=True) if status_rows else pd.DataFrame(),
    }
