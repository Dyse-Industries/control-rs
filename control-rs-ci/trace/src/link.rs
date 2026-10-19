//! The link graph and cycle-safe status derivation.
//!
//! Evaluates link targets, cycle detection, and topological status derivation
//! in two states: [`LinkGraph<Resolved>`] and [`LinkGraph<Acyclic>`].

use std::collections::{BTreeMap, BTreeSet};

use super::status::{ConditionStatus, ItemOutcome, Status};
use super::{Defect, Row};

type Adjacency = Vec<Vec<usize>>;

/// An acyclic link graph and its cycle defects.
pub type AcyclicGraph<'a> = (LinkGraph<'a, Acyclic>, Vec<Defect>);

/// Condition statuses by condition ID.
pub type ConditionMap<'a> = BTreeMap<&'a str, ConditionStatus>;

type NodeMap<'a> = BTreeMap<&'a str, usize>;

/// A resolved link graph and its target-resolution defects.
pub type ResolvedGraph<'a> = (LinkGraph<'a, Resolved>, Vec<Defect>);

type SingleResolution = (TargetOutcome, Option<Defect>);
type TargetDefects = (TargetOutcomes, Vec<Defect>);
type TargetOutcomes = Vec<Option<Vec<TargetOutcome>>>;

/// State where cycles have been detected, cyclic link edges eliminated,
/// and a topological evaluation order established.
pub struct Acyclic {
    order: Vec<usize>,
    cyclic: BTreeSet<usize>,
    link_targets: TargetOutcomes,
}

/// The link graph over `rows` in state `S`.
pub struct LinkGraph<'a, S> {
    rows: &'a [Row],
    vertices: Vec<Vertex<'a>>,
    ids: NodeMap<'a>,
    adj: Adjacency,
    state: S,
}

struct ResolutionContext<'a, 'b> {
    ids: &'b NodeMap<'a>,
    adj: &'b mut Adjacency,
    link_methods: &'b BTreeSet<String>,
    all_docs: &'b BTreeSet<&'a str>,
    docs_with_reqs: &'b BTreeSet<&'a str>,
}

/// State where all targets have been resolved or classified as defects.
pub struct Resolved {
    link_targets: TargetOutcomes,
}

#[derive(Clone, Copy)]
enum TargetOutcome {
    Resolved(usize),
    Unresolved,
    EmptyDoc,
}

enum Vertex<'a> {
    Doc,
    Req(&'a Row),
    Cond(&'a Row),
}

struct TarjanState<'a> {
    adj: &'a Adjacency,
    index: Vec<usize>,
    lowlink: Vec<usize>,
    on_stack: Vec<bool>,
    stack: Vec<usize>,
    comp: Vec<usize>,
    comp_count: usize,
    clock: usize,
}

impl<'a> TarjanState<'a> {
    /// The component index of every vertex; dependencies get lower indices.
    fn run(adj: &'a Adjacency) -> Vec<usize> {
        let n = adj.len();
        let mut state = Self {
            adj,
            index: vec![0; n],
            lowlink: vec![0; n],
            on_stack: vec![false; n],
            stack: Vec::with_capacity(n),
            comp: vec![0; n],
            comp_count: 0,
            clock: 1,
        };
        for v in 0..n {
            if state.index.get(v).copied().unwrap_or(0) == 0 {
                state.visit(v);
            }
        }
        state.comp
    }

    /// Depth-first visit from `root` with an explicit call stack of
    /// `(vertex, next neighbor position)` frames.
    fn visit(&mut self, root: usize) {
        let mut frames = vec![(root, 0_usize)];
        self.open(root);
        while let Some(&(v, pos)) = frames.last() {
            let next = self.adj.get(v).and_then(|a| a.get(pos)).copied();
            if let Some(frame) = frames.last_mut() {
                frame.1 = pos.saturating_add(1);
            }
            match next {
                Some(w) if self.index.get(w).copied().unwrap_or(0) == 0 => {
                    self.open(w);
                    frames.push((w, 0));
                }
                Some(w) => {
                    if self.on_stack.get(w).copied().unwrap_or(false) {
                        let w_idx = self.index.get(w).copied().unwrap_or(0);
                        self.lower(v, w_idx);
                    }
                }
                None => {
                    frames.pop();
                    self.close(v);
                    if let Some(&(parent, _)) = frames.last() {
                        let low = self.lowlink.get(v).copied().unwrap_or(0);
                        self.lower(parent, low);
                    }
                }
            }
        }
    }

    /// Numbers `v` and pushes it on the component stack.
    fn open(&mut self, v: usize) {
        if let Some(slot) = self.index.get_mut(v) {
            *slot = self.clock;
        }
        if let Some(slot) = self.lowlink.get_mut(v) {
            *slot = self.clock;
        }
        self.clock = self.clock.saturating_add(1);
        self.stack.push(v);
        if let Some(slot) = self.on_stack.get_mut(v) {
            *slot = true;
        }
    }

    /// Lowers the lowest reachable discovery number of `v` to `to`.
    fn lower(&mut self, v: usize, to: usize) {
        if let Some(low) = self.lowlink.get_mut(v) {
            *low = (*low).min(to);
        }
    }

    /// Pops the component rooted at `v` when `v` is its root.
    fn close(&mut self, v: usize) {
        if self.lowlink.get(v) != self.index.get(v) {
            return;
        }
        let cid = self.comp_count;
        self.comp_count = self.comp_count.saturating_add(1);
        while let Some(w) = self.stack.pop() {
            if let Some(slot) = self.on_stack.get_mut(w) {
                *slot = false;
            }
            if let Some(slot) = self.comp.get_mut(w) {
                *slot = cid;
            }
            if w == v {
                break;
            }
        }
    }
}

impl<'a> LinkGraph<'a, Resolved> {
    /// Constructs the resolved graph from `rows` and link method names.
    ///
    /// Interns document, requirement, and condition vertices, connects
    /// structural hierarchy edges, and maps link targets to vertices or
    /// records defects.
    #[must_use]
    pub fn resolve(
        rows: &'a [Row],
        link_methods: &BTreeSet<String>,
    ) -> ResolvedGraph<'a> {
        let mut ids = NodeMap::new();
        let mut vertices = Vec::new();
        let mut adj = Adjacency::new();
        let mut docs_with_reqs = BTreeSet::new();
        let mut all_docs = BTreeSet::new();

        for row in rows {
            let doc = super::doc_of(&row.id);
            all_docs.insert(doc);
            let vertex = if row.kind == super::DEFINITION {
                docs_with_reqs.insert(doc);
                Vertex::Req(row)
            } else if row.kind == super::CONDITION {
                Vertex::Cond(row)
            } else {
                continue;
            };
            if !ids.contains_key(row.id.as_str()) {
                ids.insert(row.id.as_str(), vertices.len());
                vertices.push(vertex);
                adj.push(Vec::new());
            }
        }
        for &doc in &all_docs {
            if !ids.contains_key(doc) {
                ids.insert(doc, vertices.len());
                vertices.push(Vertex::Doc);
                adj.push(Vec::new());
            }
        }

        connect_structural_edges(&vertices, &ids, &mut adj);
        let (link_targets, defects) = resolve_link_targets(
            &vertices,
            ResolutionContext {
                ids: &ids,
                adj: &mut adj,
                link_methods,
                all_docs: &all_docs,
                docs_with_reqs: &docs_with_reqs,
            },
        );

        (
            Self {
                rows,
                vertices,
                ids,
                adj,
                state: Resolved { link_targets },
            },
            defects,
        )
    }

    /// Runs Tarjan's SCC algorithm, identifies cycles, eliminates cyclic link
    /// edges, and transitions the graph to [`Acyclic`].
    #[must_use]
    pub fn condense(mut self) -> AcyclicGraph<'a> {
        let n = self.vertices.len();
        let comp = TarjanState::run(&self.adj);

        let mut defects = Vec::new();
        let mut cyclic = BTreeSet::new();

        for u in 0..n {
            let Some(Vertex::Cond(r)) = self.vertices.get(u) else {
                continue;
            };
            if self
                .state
                .link_targets
                .get(u)
                .and_then(Option::as_ref)
                .is_none()
            {
                continue;
            }
            let comp_u = comp.get(u).copied().unwrap_or(0);
            let mut is_cyclic = false;
            if let Some(neighbors) = self.adj.get_mut(u) {
                neighbors.retain(|&v| {
                    let same = comp.get(v).copied().unwrap_or(0) == comp_u;
                    is_cyclic |= same;
                    !same
                });
            }
            if is_cyclic {
                cyclic.insert(u);
                defects.push(Defect::at(r, super::cycle_message(&r.id)));
            }
        }

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&v| {
            let rank = match self.vertices.get(v) {
                Some(Vertex::Cond(_)) => 0,
                Some(Vertex::Req(_)) => 1,
                _ => 2,
            };
            (comp.get(v).copied().unwrap_or(0), rank)
        });

        (
            LinkGraph {
                rows: self.rows,
                vertices: self.vertices,
                ids: self.ids,
                adj: self.adj,
                state: Acyclic {
                    order,
                    cyclic,
                    link_targets: self.state.link_targets,
                },
            },
            defects,
        )
    }
}

impl<'a> LinkGraph<'a, Acyclic> {
    /// Evaluates condition, requirement, and document statuses along the
    /// topological order.
    pub fn derive(
        &self,
        mut leaf: impl FnMut(&Row) -> ConditionStatus,
    ) -> ConditionMap<'a> {
        let mut statuses = BTreeMap::new();
        let mut vertex_status = vec![Status::Pass; self.vertices.len()];

        for &v in &self.state.order {
            let status = match self.vertices.get(v) {
                Some(Vertex::Cond(row)) => {
                    let cond = self
                        .state
                        .link_targets
                        .get(v)
                        .and_then(Option::as_ref)
                        .map_or_else(
                            || leaf(row),
                            |targets| {
                                derive_link_status(
                                    row,
                                    targets,
                                    &vertex_status,
                                    self.state.cyclic.contains(&v),
                                )
                            },
                        );
                    let status = cond.status;
                    statuses.insert(row.id.as_str(), cond);
                    status
                }
                Some(Vertex::Req(_) | Vertex::Doc) => {
                    let children =
                        self.adj.get(v).map_or(&[] as &[usize], Vec::as_slice);
                    children
                        .iter()
                        .map(|&c| {
                            vertex_status
                                .get(c)
                                .copied()
                                .unwrap_or(Status::Uncovered)
                        })
                        .max()
                        .unwrap_or(Status::Uncovered)
                }
                None => continue,
            };
            if let Some(slot) = vertex_status.get_mut(v) {
                *slot = status;
            }
        }
        statuses
    }
}

fn connect_structural_edges(
    vertices: &[Vertex<'_>],
    ids: &NodeMap<'_>,
    adj: &mut Adjacency,
) {
    for (idx, vertex) in vertices.iter().enumerate() {
        let parents: Vec<&str> = match vertex {
            Vertex::Req(r) => vec![super::doc_of(&r.id)],
            Vertex::Cond(r) => r.parents.iter().map(String::as_str).collect(),
            Vertex::Doc => continue,
        };
        for parent in parents {
            if let Some(&p) = ids.get(parent)
                && let Some(neighbors) = adj.get_mut(p)
            {
                neighbors.push(idx);
            }
        }
    }
}

fn resolve_link_targets(
    vertices: &[Vertex<'_>],
    mut ctx: ResolutionContext<'_, '_>,
) -> TargetDefects {
    let mut defects = Vec::new();
    let mut link_targets = vec![None; vertices.len()];

    for (idx, vertex) in vertices.iter().enumerate() {
        let Vertex::Cond(r) = vertex else {
            continue;
        };
        if !r
            .method
            .as_ref()
            .is_some_and(|m| ctx.link_methods.contains(m))
        {
            continue;
        }
        if r.targets.is_empty() {
            defects.push(Defect::at(r, super::no_target_message(&r.id)));
        }
        let mut outcomes = Vec::with_capacity(r.targets.len());
        for target in &r.targets {
            let (outcome, defect) = resolve_target(r, idx, target, &mut ctx);
            defects.extend(defect);
            outcomes.push(outcome);
        }
        if let Some(slot) = link_targets.get_mut(idx) {
            *slot = Some(outcomes);
        }
    }
    (link_targets, defects)
}

/// Resolves `target` of link condition `r` at vertex `idx`, adding the link
/// edge when it names a vertex.
fn resolve_target(
    r: &Row,
    idx: usize,
    target: &str,
    ctx: &mut ResolutionContext<'_, '_>,
) -> SingleResolution {
    let named = if target.contains('#') || ctx.docs_with_reqs.contains(target) {
        ctx.ids.get(target).copied()
    } else {
        None
    };
    if let Some(t) = named {
        if let Some(neighbors) = ctx.adj.get_mut(idx) {
            neighbors.push(t);
        }
        return (TargetOutcome::Resolved(t), None);
    }
    if !target.contains('#') && ctx.all_docs.contains(target) {
        (
            TargetOutcome::EmptyDoc,
            Some(Defect::at(r, super::empty_document_message(&r.id, target))),
        )
    } else {
        (
            TargetOutcome::Unresolved,
            Some(Defect::at(
                r,
                super::unresolved_target_message(&r.id, target),
            )),
        )
    }
}

fn derive_link_status(
    row: &Row,
    targets: &[TargetOutcome],
    vertex_status: &[Status],
    is_cyclic: bool,
) -> ConditionStatus {
    let mut worst = if is_cyclic {
        Status::Fail
    } else if targets.is_empty() {
        Status::Uncovered
    } else {
        Status::Pass
    };
    let mut items = Vec::new();
    if !is_cyclic {
        for (outcome, target) in targets.iter().zip(&row.targets) {
            let status = match outcome {
                TargetOutcome::Resolved(t) => {
                    vertex_status.get(*t).copied().unwrap_or(Status::Unrun)
                }
                TargetOutcome::Unresolved => Status::Unrun,
                TargetOutcome::EmptyDoc => Status::Uncovered,
            };
            worst = worst.max(status);
            items.push(ItemOutcome {
                target: target.clone(),
                passed: !super::status::fails_gate(status),
                message: Some(format!("status {status:?}")),
            });
        }
    }
    ConditionStatus {
        id: row.id.clone(),
        parents: row.parents.clone(),
        method: row.method.clone().unwrap_or_default(),
        status: worst,
        items,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CONDITION, DEFINITION, SCHEMA};

    fn row(id: &str, kind: &str) -> Row {
        Row {
            schema: SCHEMA,
            id: id.to_string(),
            kind: kind.to_string(),
            parents: Vec::new(),
            method: None,
            targets: Vec::new(),
            file: "doc.md".to_string(),
            line: 1,
            text: String::new(),
        }
    }

    fn req(id: &str) -> Row {
        row(id, DEFINITION)
    }

    fn cond(id: &str, parent: &str, method: &str, targets: &[&str]) -> Row {
        let mut r = row(id, CONDITION);
        r.parents = vec![parent.to_string()];
        r.method = Some(method.to_string());
        r.targets = targets.iter().map(|t| (*t).to_string()).collect();
        r
    }

    fn links() -> BTreeSet<String> {
        BTreeSet::from(["link".to_string()])
    }

    fn leaf(row: &Row) -> ConditionStatus {
        ConditionStatus {
            id: row.id.clone(),
            parents: row.parents.clone(),
            method: row.method.clone().unwrap_or_default(),
            status: Status::Pass,
            items: Vec::new(),
        }
    }

    fn messages(defects: &[Defect]) -> Vec<&str> {
        defects.iter().map(|d| d.message.as_str()).collect()
    }

    #[test]
    fn link_without_a_target_is_reported() {
        let rows = [req("a#R1"), cond("a#V1", "a#R1", "link", &[])];
        let (_, defects) = LinkGraph::resolve(&rows, &links());
        assert_eq!(messages(&defects), ["condition a#V1 names no target"]);
    }

    #[test]
    fn link_to_an_undefined_node_is_reported() {
        let rows = [
            req("a#R1"),
            cond("a#V1", "a#R1", "link", &["a#R9", "nodoc"]),
        ];
        let (_, defects) = LinkGraph::resolve(&rows, &links());
        assert_eq!(
            messages(&defects),
            [
                "condition a#V1 target 'a#R9' names no selected document, requirement or condition",
                "condition a#V1 target 'nodoc' names no selected document, requirement or condition",
            ]
        );
    }

    #[test]
    fn link_to_a_document_without_requirements_is_reported() {
        let rows = [
            req("a#R1"),
            cond("a#V1", "a#R1", "link", &["b"]),
            cond("b#V1", "b#R1", "libtest", &[]),
        ];
        let (_, defects) = LinkGraph::resolve(&rows, &links());
        assert_eq!(
            messages(&defects),
            ["condition a#V1 target 'b' defines no requirement"]
        );
    }

    #[test]
    fn a_self_link_is_a_cycle_and_its_edge_is_removed() {
        let rows = [req("a#R1"), cond("a#V1", "a#R1", "link", &["a#V1"])];
        let (graph, resolve_defects) = LinkGraph::resolve(&rows, &links());
        assert_eq!(resolve_defects, Vec::new());
        let (acyclic, defects) = graph.condense();
        assert_eq!(messages(&defects), ["condition a#V1 lies on a link cycle"]);
        let cond = acyclic.ids.get("a#V1").copied().unwrap_or(usize::MAX);
        assert!(acyclic.adj.get(cond).is_some_and(Vec::is_empty));
    }

    #[test]
    fn an_acyclic_graph_has_no_defects() {
        let rows = [
            req("a#R1"),
            cond("a#V1", "a#R1", "link", &["b"]),
            req("b#R1"),
            cond("b#V1", "b#R1", "libtest", &["t"]),
        ];
        let (graph, resolve_defects) = LinkGraph::resolve(&rows, &links());
        let (_, defects) = graph.condense();
        assert_eq!(resolve_defects, Vec::new());
        assert_eq!(defects, Vec::new());
    }

    #[test]
    fn a_long_link_chain_is_condensed_without_overflow() {
        const DEPTH: usize = 100_000;
        let mut rows = Vec::with_capacity(DEPTH * 2);
        for i in 0..DEPTH {
            let next = format!("d{}", i + 1);
            let targets: &[&str] =
                if i + 1 < DEPTH { &[next.as_str()] } else { &[] };
            rows.push(req(&format!("d{i}#R1")));
            rows.push(cond(
                &format!("d{i}#V1"),
                &format!("d{i}#R1"),
                if targets.is_empty() {
                    "libtest"
                } else {
                    "link"
                },
                targets,
            ));
        }
        let (graph, resolve_defects) = LinkGraph::resolve(&rows, &links());
        let (_, defects) = graph.condense();
        assert_eq!(resolve_defects, Vec::new());
        assert_eq!(defects, Vec::new());
    }

    #[test]
    fn derive_takes_the_worst_status_of_the_targets() {
        let rows = [
            req("a#R1"),
            cond("a#V1", "a#R1", "link", &["b", "c"]),
            req("b#R1"),
            cond("b#V1", "b#R1", "libtest", &["t"]),
            req("c#R1"),
            cond("c#V1", "c#R1", "libtest", &["u"]),
        ];
        let (graph, _) = LinkGraph::resolve(&rows, &links());
        let (acyclic, _) = graph.condense();
        let statuses = acyclic.derive(|row| {
            let mut s = leaf(row);
            if row.id == "c#V1" {
                s.status = Status::Fail;
            }
            s
        });
        let link = statuses.get("a#V1").expect("link status");
        assert_eq!(link.status, Status::Fail);
        assert_eq!(link.items.len(), 2);
        assert!(link.items.first().is_some_and(|i| i.passed));
        assert!(link.items.get(1).is_some_and(|i| !i.passed));
    }

    #[test]
    fn derive_fails_a_link_on_a_cycle() {
        let rows = [req("a#R1"), cond("a#V1", "a#R1", "link", &["a"])];
        let (graph, _) = LinkGraph::resolve(&rows, &links());
        let (acyclic, _) = graph.condense();
        let statuses = acyclic.derive(leaf);
        assert_eq!(statuses.get("a#V1").map(|s| s.status), Some(Status::Fail));
    }

    #[test]
    fn derive_marks_a_link_without_a_target_uncovered() {
        let rows = [req("a#R1"), cond("a#V1", "a#R1", "link", &[])];
        let (graph, _) = LinkGraph::resolve(&rows, &links());
        let (acyclic, _) = graph.condense();
        let statuses = acyclic.derive(leaf);
        assert_eq!(
            statuses.get("a#V1").map(|s| s.status),
            Some(Status::Uncovered)
        );
    }

    #[test]
    fn derive_evaluates_a_chain_of_documents_dependencies_first() {
        let rows = [
            req("a#R1"),
            cond("a#V1", "a#R1", "link", &["b"]),
            req("b#R1"),
            cond("b#V1", "b#R1", "link", &["c"]),
            req("c#R1"),
            cond("c#V1", "c#R1", "libtest", &["t"]),
        ];
        let (graph, _) = LinkGraph::resolve(&rows, &links());
        let (acyclic, _) = graph.condense();
        let statuses = acyclic.derive(|row| {
            let mut s = leaf(row);
            s.status = Status::Unrun;
            s
        });
        for id in ["a#V1", "b#V1", "c#V1"] {
            assert_eq!(statuses.get(id).map(|s| s.status), Some(Status::Unrun));
        }
    }
}
