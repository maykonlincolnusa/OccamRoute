"""Deterministic Python AST parser and NetworkX dependency graph builder."""
from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import networkx as nx

SymbolKind = Literal["class", "function", "method"]

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODING = None


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base with a deterministic regex fallback."""
    if not text:
        return 0
    if _ENCODING is not None:
        return len(_ENCODING.encode(text, disallowed_special=()))
    return max(1, len(re.findall(r"\w+|[^\s\w]", text)))


def safe_pagerank(
    graph: nx.DiGraph,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Compute PageRank via NetworkX, falling back to pure-Python power iteration."""
    n = graph.number_of_nodes()
    if n == 0:
        return {}
    try:
        return nx.pagerank(graph, alpha=alpha, max_iter=max_iter, tol=tol)
    except Exception:
        nodes = list(graph.nodes())
        rank: dict[str, float] = {u: 1.0 / n for u in nodes}
        out_deg: dict[str, int] = {u: graph.out_degree(u) for u in nodes}
        base = (1.0 - alpha) / n
        for _ in range(max_iter):
            dangling_sum = alpha * sum(rank[u] for u in nodes if out_deg[u] == 0) / n
            new_rank: dict[str, float] = {}
            diff = 0.0
            for v in nodes:
                in_sum = sum(rank[u] / out_deg[u] for u in graph.predecessors(v) if out_deg[u] > 0)
                val = base + dangling_sum + alpha * in_sum
                new_rank[v] = val
                diff += abs(val - rank[v])
            rank = new_rank
            if diff < n * tol:
                break
        return rank


@dataclass
class SymbolNode:
    """AST-extracted symbol node stored inside RepoGraph."""

    node_id: str
    file_path: str
    name: str
    qualname: str
    kind: SymbolKind
    parent_class: str | None
    start_line: int
    end_line: int
    signature: str
    docstring: str
    source: str
    token_count: int
    raw_calls: set[str] = field(default_factory=set)
    imports: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    cyclomatic_complexity: int = 1
    is_leaf: bool = True


class _ASTSymbolVisitor(ast.NodeVisitor):
    """Collect outgoing call names, raised exceptions, and McCabe cyclomatic complexity."""

    _BRANCH_NODES = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.ExceptHandler,
        ast.With,
        ast.AsyncWith,
        ast.IfExp,
        ast.comprehension,
    )

    def __init__(self) -> None:
        self.calls: set[str] = set()
        self.complexity: int = 1

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id in ("self", "cls"):
                self.calls.add(f"self.{node.func.attr}")
            else:
                self.calls.add(node.func.attr)
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        if isinstance(node.exc, ast.Name):
            self.calls.add(node.exc.id)
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.complexity += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, self._BRANCH_NODES):
            self.complexity += 1
        super().generic_visit(node)


def _format_signature(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        args_str = ast.unparse(node.args)
        ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
        return f"{prefix} {node.name}({args_str}){ret}"
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    return ""


def _extract_module_imports(tree: ast.Module) -> list[str]:
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                imports.append(f"{mod}.{alias.name}" if mod else alias.name)
    return imports


@dataclass
class RepoGraph:
    """Repository-level AST symbol and dependency graph."""

    root_dir: Path
    nx_graph: nx.DiGraph
    nodes: dict[str, SymbolNode]
    leaf_ids: set[str]
    file_to_nodes: dict[str, list[str]]
    pagerank: dict[str, float]
    total_raw_tokens: int

    def k_hop_neighbors(
        self,
        seeds: list[str],
        k: int = 2,
        undirected: bool = False,
    ) -> dict[str, int]:
        """Return shortest hop distance <= k from any seed node."""
        view = self.nx_graph.to_undirected(as_view=True) if undirected else self.nx_graph
        distances: dict[str, int] = {}
        for seed in seeds:
            if seed not in view:
                continue
            sp = nx.single_source_shortest_path_length(view, seed, cutoff=k)
            for node_id, dist in sp.items():
                if node_id not in distances or dist < distances[node_id]:
                    distances[node_id] = dist
        return distances

    def summary_stats(self) -> dict[str, int | float]:
        """Return compact graph statistics."""
        return {
            "num_files": len(self.file_to_nodes),
            "num_nodes": self.nx_graph.number_of_nodes(),
            "num_leaves": len(self.leaf_ids),
            "num_edges": self.nx_graph.number_of_edges(),
            "total_raw_tokens": self.total_raw_tokens,
        }


class RepoGraphBuilder:
    """Parses Python source files into a directed NetworkX AST dependency graph."""

    DEFAULT_EXCLUDES: tuple[str, ...] = (
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        "build",
        "dist",
    )

    def build_from_directory(
        self,
        root_dir: Path | str,
        exclude_dirs: tuple[str, ...] | None = None,
    ) -> RepoGraph:
        root = Path(root_dir).resolve()
        excludes = set(self.DEFAULT_EXCLUDES if exclude_dirs is None else exclude_dirs)

        nx_graph = nx.DiGraph()
        nodes: dict[str, SymbolNode] = {}
        leaf_ids: set[str] = set()
        file_to_nodes: dict[str, list[str]] = defaultdict(list)
        class_bases: dict[str, list[str]] = {}
        total_raw_tokens = 0

        py_files = sorted(
            fp
            for fp in root.rglob("*.py")
            if not any(part in excludes for part in fp.relative_to(root).parts)
        )

        for fp in py_files:
            rel_path = fp.relative_to(root).as_posix()
            src = fp.read_text(encoding="utf-8")
            total_raw_tokens += count_tokens(src)
            try:
                tree = ast.parse(src, filename=rel_path)
            except SyntaxError:
                continue
            self._index_module(
                rel_path=rel_path,
                lines=src.splitlines(),
                tree=tree,
                nx_graph=nx_graph,
                nodes=nodes,
                leaf_ids=leaf_ids,
                file_to_nodes=file_to_nodes,
                class_bases=class_bases,
            )

        self._resolve_edges(nx_graph, nodes, class_bases)
        pagerank = safe_pagerank(nx_graph)
        return RepoGraph(
            root_dir=root,
            nx_graph=nx_graph,
            nodes=nodes,
            leaf_ids=leaf_ids,
            file_to_nodes=dict(file_to_nodes),
            pagerank=pagerank,
            total_raw_tokens=total_raw_tokens,
        )

    def _extract_source(self, lines: list[str], node: ast.AST) -> tuple[int, int, str]:
        start = getattr(node, "lineno", 1)
        decs = getattr(node, "decorator_list", [])
        if decs:
            start = min(start, *(getattr(d, "lineno", start) for d in decs))
        end = getattr(node, "end_lineno", start)
        return start, end, "\n".join(lines[start - 1 : end])

    def _add_or_merge(
        self,
        sym: SymbolNode,
        nx_graph: nx.DiGraph,
        nodes: dict[str, SymbolNode],
        leaf_ids: set[str],
        file_to_nodes: dict[str, list[str]],
    ) -> None:
        if sym.node_id in nodes:
            existing = nodes[sym.node_id]
            existing.source = f"{existing.source}\n\n{sym.source}"
            existing.end_line = max(existing.end_line, sym.end_line)
            existing.token_count = count_tokens(existing.source)
            existing.raw_calls.update(sym.raw_calls)
            existing.cyclomatic_complexity = max(existing.cyclomatic_complexity, sym.cyclomatic_complexity)
            nx_graph.nodes[sym.node_id]["token_count"] = existing.token_count
            nx_graph.nodes[sym.node_id]["cyclomatic"] = existing.cyclomatic_complexity
            return

        nodes[sym.node_id] = sym
        file_to_nodes[sym.file_path].append(sym.node_id)
        if sym.is_leaf:
            leaf_ids.add(sym.node_id)
        nx_graph.add_node(
            sym.node_id,
            file_path=sym.file_path,
            kind=sym.kind,
            token_count=sym.token_count,
            cyclomatic=sym.cyclomatic_complexity,
            is_leaf=sym.is_leaf,
            parent_class=sym.parent_class or "",
        )

    def _index_module(
        self,
        rel_path: str,
        lines: list[str],
        tree: ast.Module,
        nx_graph: nx.DiGraph,
        nodes: dict[str, SymbolNode],
        leaf_ids: set[str],
        file_to_nodes: dict[str, list[str]],
        class_bases: dict[str, list[str]],
    ) -> None:
        mod_imports = _extract_module_imports(tree)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start, end, source = self._extract_source(lines, node)
                visitor = _ASTSymbolVisitor()
                visitor.visit(node)
                doc = ast.get_docstring(node) or ""
                self._add_or_merge(
                    SymbolNode(
                        node_id=f"{rel_path}::{node.name}",
                        file_path=rel_path,
                        name=node.name,
                        qualname=node.name,
                        kind="function",
                        parent_class=None,
                        start_line=start,
                        end_line=end,
                        signature=_format_signature(node),
                        docstring=doc.splitlines()[0] if doc else "",
                        source=source,
                        token_count=count_tokens(source),
                        raw_calls=visitor.calls,
                        imports=mod_imports,
                        bases=[],
                        cyclomatic_complexity=visitor.complexity,
                        is_leaf=True,
                    ),
                    nx_graph,
                    nodes,
                    leaf_ids,
                    file_to_nodes,
                )
            elif isinstance(node, ast.ClassDef):
                bases = [ast.unparse(b) for b in node.bases]
                class_bases[node.name] = bases
                methods = [m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                c_start, c_end, c_source = self._extract_source(lines, node)
                c_visitor = _ASTSymbolVisitor()
                c_visitor.visit(node)
                c_doc = ast.get_docstring(node) or ""
                self._add_or_merge(
                    SymbolNode(
                        node_id=f"{rel_path}::{node.name}",
                        file_path=rel_path,
                        name=node.name,
                        qualname=node.name,
                        kind="class",
                        parent_class=None,
                        start_line=c_start,
                        end_line=c_end,
                        signature=_format_signature(node),
                        docstring=c_doc.splitlines()[0] if c_doc else "",
                        source=c_source,
                        token_count=count_tokens(c_source),
                        raw_calls=c_visitor.calls,
                        imports=mod_imports,
                        bases=bases,
                        cyclomatic_complexity=c_visitor.complexity,
                        is_leaf=(len(methods) == 0),
                    ),
                    nx_graph,
                    nodes,
                    leaf_ids,
                    file_to_nodes,
                )
                for m in methods:
                    m_start, m_end, m_source = self._extract_source(lines, m)
                    m_visitor = _ASTSymbolVisitor()
                    m_visitor.visit(m)
                    m_doc = ast.get_docstring(m) or ""
                    self._add_or_merge(
                        SymbolNode(
                            node_id=f"{rel_path}::{node.name}.{m.name}",
                            file_path=rel_path,
                            name=m.name,
                            qualname=f"{node.name}.{m.name}",
                            kind="method",
                            parent_class=node.name,
                            start_line=m_start,
                            end_line=m_end,
                            signature=_format_signature(m),
                            docstring=m_doc.splitlines()[0] if m_doc else "",
                            source=m_source,
                            token_count=count_tokens(m_source),
                            raw_calls=m_visitor.calls,
                            imports=mod_imports,
                            bases=[],
                            cyclomatic_complexity=m_visitor.complexity,
                            is_leaf=True,
                        ),
                        nx_graph,
                        nodes,
                        leaf_ids,
                        file_to_nodes,
                    )

    def _resolve_edges(
        self,
        nx_graph: nx.DiGraph,
        nodes: dict[str, SymbolNode],
        class_bases: dict[str, list[str]],
    ) -> None:
        by_name: dict[str, list[str]] = defaultdict(list)
        by_class_method: dict[tuple[str, str], str] = {}
        class_nodes_by_name: dict[str, list[str]] = defaultdict(list)

        for sid, sym in nodes.items():
            by_name[sym.name].append(sid)
            if sym.parent_class:
                by_class_method[(sym.parent_class, sym.name)] = sid
            if sym.kind == "class":
                class_nodes_by_name[sym.name].append(sid)

        # Inheritance edges between indexed classes
        for sid, sym in nodes.items():
            if sym.kind == "class":
                for base in sym.bases:
                    base_name = base.split(".")[-1]
                    for target_cls in class_nodes_by_name.get(base_name, []):
                        if target_cls != sid:
                            nx_graph.add_edge(sid, target_cls, edge_type="inherits")

        # Call & exception edges from executable symbols
        for sid, sym in nodes.items():
            if not sym.is_leaf:
                continue
            for call in sym.raw_calls:
                target: str | None = None
                if call.startswith("self.") and sym.parent_class:
                    m_name = call.split(".", 1)[1]
                    if (sym.parent_class, m_name) in by_class_method:
                        target = by_class_method[(sym.parent_class, m_name)]
                    else:
                        for base in class_bases.get(sym.parent_class, []):
                            base_plain = base.split(".")[-1]
                            if (base_plain, m_name) in by_class_method:
                                target = by_class_method[(base_plain, m_name)]
                                break
                else:
                    plain = call.split(".")[-1]
                    cands = by_name.get(plain, [])
                    same_file = [c for c in cands if nodes[c].file_path == sym.file_path]
                    if len(same_file) == 1:
                        target = same_file[0]
                    elif len(cands) == 1:
                        target = cands[0]
                    elif len(cands) > 1:
                        leaf_cands = [c for c in cands if nodes[c].is_leaf]
                        if len(leaf_cands) == 1:
                            target = leaf_cands[0]

                if target and target != sid:
                    nx_graph.add_edge(sid, target, edge_type="calls")
