//! The link graph and cycle-safe status derivation.
//!
//! A link condition names documents, requirements or conditions and takes
//! their status. [`LinkGraph`] has one vertex per document, requirement and
//! condition, and an edge from each vertex to every vertex its status depends
//! on. The graph passes through three steps, each producing the input of the
//! next: [`LinkGraph::resolve`] builds it, [`LinkGraph::condense`] finds and
//! removes link cycles and fixes the evaluation order, and
//! [`LinkGraph::derive`] and [`LinkGraph::render`] read the acyclic result.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::reqs::LINK_METHOD;
use super::status::{ConditionStatus, ItemOutcome, Status};
use super::{CONDITION, DEFINITION, Defect, Defects, QUALIFIER, Row};

/// Stroke of a removed cyclic edge.
const CYCLE_STROKE: &str = "#d32f2f";

/// Fill of a vertex with a failing status.
const FILL_FAIL: &str = "#ffcdd2";

/// Fill of a vertex without coverage or with a target that did not run.
const FILL_OPEN: &str = "#ffe0b2";

/// Fill of a vertex with a passing status.
const FILL_PASS: &str = "#c8e6c9";

/// Fill of a vertex that awaits sign-off or is deferred.
const FILL_PENDING: &str = "#e0e0e0";

/// Every status, in the order the Mermaid classes are defined.
const STATUSES: [Status; 6] = [
    Status::Pass,
    Status::Review,
    Status::Deferred,
    Status::Uncovered,
    Status::Unrun,
    Status::Fail,
];

/// Out-edges per vertex: the vertices its status depends on.
type Adjacency = Vec<Vec<usize>>;

/// A graph in state `S` and the defects found producing it.
pub type Built<'a, S> = (LinkGraph<'a, S>, Defects);

/// Condition statuses by condition ID.
pub type ConditionMap<'a> = BTreeMap<&'a str, ConditionStatus>;

/// A directed edge as `(from, to)` vertex indices.
type Edge = (usize, usize);

/// Status of every vertex, when statuses were given.
type Rolled = Option<Vec<Status>>;

/// A resolved target and the defect it caused, if any.
type Resolution = (Target, Option<Defect>);

/// Vertices by name.
type Index<'a> = HashMap<&'a str, usize>;

/// Targets of each link condition, by condition vertex.
type Links = BTreeMap<usize, Vec<Target>>;

/// Output format of [`LinkGraph::render`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphFormat {
    /// A Mermaid `flowchart`.
    Mermaid,
    /// A Graphviz `digraph`.
    Dot,
}

/// What a vertex stands for, in evaluation order within a component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    /// A verification condition.
    Condition,
    /// A requirement definition.
    Requirement,
    /// A design document.
    Document,
}

/// What one target of a link condition resolved to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target {
    /// A vertex the target names.
    Resolved(usize),
    /// A target that names nothing defined.
    Unresolved,
}

/// The link graph over `rows` in state `S`: one vertex per document,
/// requirement and condition, with the structural and the resolved link edges.
#[derive(Debug, Clone)]
pub struct LinkGraph<'a, S> {
    /// Out-edges per vertex.
    edges: Adjacency,
    /// Targets of each link condition.
    links: Links,
    /// The rows the graph was built from.
    rows: &'a [Row],
    /// The state this graph is in.
    state: S,
    /// Vertices in interning order.
    vertices: Vec<Vertex<'a>>,
}

/// No cycle remains, and every vertex has a place in the evaluation order.
#[derive(Debug, Clone)]
pub struct Acyclic {
    /// Link conditions that lie on a cycle.
    cyclic: BTreeSet<usize>,
    /// Vertices, each after every vertex its status depends on.
    order: Vec<usize>,
    /// Link edges removed as cyclic.
    removed: Vec<Edge>,
}

/// Every link target is resolved to a vertex, an unresolved target or an
/// empty document.
#[derive(Debug, Clone, Copy)]
pub struct Resolved;

/// Accumulates the vertices and edges while rows are read.
#[derive(Default)]
struct Builder<'a> {
    /// Document vertices by name.
    docs: Index<'a>,
    /// Out-edges per vertex.
    edges: Adjacency,
    /// Requirement and condition vertices by qualified ID.
    ids: Index<'a>,
    /// Vertices in interning order.
    vertices: Vec<Vertex<'a>>,
}

/// Iterative depth-first search that numbers strongly connected components.
struct Search<'g> {
    /// Out-edges per vertex.
    adj: &'g Adjacency,
    /// Component of each vertex, numbered in completion order.
    comp: Vec<usize>,
    /// Number of components completed.
    count: usize,
    /// Next discovery number; 0 marks an unvisited vertex.
    clock: usize,
    /// Smallest discovery number reachable from each vertex.
    low: Vec<usize>,
    /// Discovery number of each vertex.
    number: Vec<usize>,
    /// Whether each vertex is on the component stack.
    on_stack: Vec<bool>,
    /// Vertices of components not yet completed.
    stack: Vec<usize>,
}

/// One vertex of the graph.
#[derive(Debug, Clone, Copy)]
struct Vertex<'a> {
    /// What the vertex stands for.
    kind: Kind,
    /// Qualified ID, or the document name of a document.
    name: &'a str,
    /// Index of the first row that introduces the vertex.
    row: usize,
}

/// What [`LinkGraph::render`] draws.
struct View<'v> {
    /// Removed cyclic edges between shown vertices.
    cut: &'v [Edge],
    /// Edges between shown vertices.
    kept: &'v [Edge],
    /// Shown vertices.
    shown: &'v BTreeSet<usize>,
    /// Status of every vertex, when statuses were given.
    status: Rolled,
}

impl<'a> Builder<'a> {
    /// Adds the edge `from -> to`.
    fn connect(&mut self, from: usize, to: usize) {
        if let Some(out) = self.edges.get_mut(from) {
            out.push(to);
        }
    }

    /// Adds a vertex and its empty edge list, and returns its index.
    fn intern(&mut self, kind: Kind, name: &'a str, row: usize) -> usize {
        self.vertices.push(Vertex { kind, name, row });
        self.edges.push(Vec::new());
        self.vertices.len().saturating_sub(1)
    }

    /// The kind of vertex `v`.
    fn kind(&self, v: usize) -> Option<Kind> {
        self.vertices.get(v).map(|x| x.kind)
    }

    /// The vertex that row `i` introduces as a condition, if any.
    fn condition_of(&self, row: &Row, i: usize) -> Option<usize> {
        let v = *self.ids.get(row.id.as_str())?;
        let x = self.vertices.get(v)?;
        (x.kind == Kind::Condition && x.row == i).then_some(v)
    }

    /// Adds the edges from each requirement to its conditions.
    fn parents(&mut self, rows: &[Row]) {
        for (i, row) in rows.iter().enumerate() {
            let Some(v) = self.condition_of(row, i) else {
                continue;
            };
            for parent in &row.parents {
                let r = self.ids.get(parent.as_str()).copied();
                if let Some(r) =
                    r.filter(|&r| self.kind(r) == Some(Kind::Requirement))
                {
                    self.connect(r, v);
                }
            }
        }
    }

    /// Resolves the targets of every link condition and adds its edges.
    fn links(&mut self, rows: &[Row]) -> (Links, Defects) {
        let mut links = Links::new();
        let mut defects = Defects::new();
        for (i, row) in rows.iter().enumerate() {
            let is_link =
                row.method.as_deref().is_some_and(|m| m == LINK_METHOD);
            let Some(v) = self.condition_of(row, i).filter(|_| is_link) else {
                continue;
            };
            if row.targets.is_empty() {
                let message =
                    format!("condition {} names no link target", row.id);
                defects.push(Defect::at(row, message));
            }
            let mut targets = Vec::with_capacity(row.targets.len());
            for name in &row.targets {
                let (target, defect) = self.target(row, v, name);
                targets.push(target);
                defects.extend(defect);
            }
            links.insert(v, targets);
        }
        (links, defects)
    }

    /// Reads the rows: a document vertex at the first row of each document,
    /// and a vertex and the document's edge for each requirement and
    /// condition.
    fn scan(&mut self, rows: &'a [Row]) {
        for (i, row) in rows.iter().enumerate() {
            let Some((doc, _)) = row.id.split_once(QUALIFIER) else {
                continue;
            };
            let kind = match row.kind.as_str() {
                DEFINITION => Kind::Requirement,
                CONDITION => Kind::Condition,
                _ => continue,
            };
            let d = if let Some(&d) = self.docs.get(doc) {
                d
            } else {
                let d = self.intern(Kind::Document, doc, i);
                self.docs.insert(doc, d);
                d
            };
            if self.ids.contains_key(row.id.as_str()) {
                continue;
            }
            let v = self.intern(kind, &row.id, i);
            self.ids.insert(&row.id, v);
            if kind == Kind::Requirement {
                self.connect(d, v);
            }
        }
    }

    /// Resolves one target `name` of link condition `v` of `row`, and adds the
    /// edge when it names a vertex.
    fn target(&mut self, row: &Row, v: usize, name: &str) -> Resolution {
        let index = if name.contains(QUALIFIER) {
            &self.ids
        } else {
            &self.docs
        };
        let Some(&w) = index.get(name) else {
            let message = format!(
                "condition {} target '{name}' names no selected document, \
                 requirement or condition",
                row.id
            );
            return (Target::Unresolved, Some(Defect::at(row, message)));
        };
        self.connect(v, w);
        let empty = self.kind(w) == Some(Kind::Document)
            && self.edges.get(w).is_none_or(Vec::is_empty);
        let defect = empty.then(|| {
            let message = format!(
                "condition {} target '{name}' defines no requirement",
                row.id
            );
            Defect::at(row, message)
        });
        (Target::Resolved(w), defect)
    }
}

impl<'g> Search<'g> {
    /// Numbers `v` and pushes it on the component stack.
    fn discover(&mut self, v: usize) {
        put(&mut self.number, v, self.clock);
        put(&mut self.low, v, self.clock);
        self.clock = self.clock.saturating_add(1);
        self.stack.push(v);
        if let Some(flag) = self.on_stack.get_mut(v) {
            *flag = true;
        }
    }

    /// Pops the component rooted at `v` when `v` is its root.
    fn finish(&mut self, v: usize) {
        if at(&self.low, v) != at(&self.number, v) {
            return;
        }
        let id = self.count;
        self.count = self.count.saturating_add(1);
        while let Some(w) = self.stack.pop() {
            if let Some(flag) = self.on_stack.get_mut(w) {
                *flag = false;
            }
            put(&mut self.comp, w, id);
            if w == v {
                break;
            }
        }
    }

    /// Lowers the smallest reachable discovery number of `v` to `to`.
    fn lower(&mut self, v: usize, to: usize) {
        let low = at(&self.low, v).min(to);
        put(&mut self.low, v, low);
    }

    /// The component of every vertex; a component gets a lower number than
    /// every component that reaches it.
    fn run(adj: &'g Adjacency) -> Vec<usize> {
        let n = adj.len();
        let mut search = Self {
            adj,
            comp: vec![0; n],
            count: 0,
            clock: 1,
            low: vec![0; n],
            number: vec![0; n],
            on_stack: vec![false; n],
            stack: Vec::with_capacity(n),
        };
        for v in 0..n {
            if at(&search.number, v) == 0 {
                search.visit(v);
            }
        }
        search.comp
    }

    /// Depth-first visit from `root` with an explicit stack of
    /// `(vertex, next edge)` frames, so the depth of a chain does not bound the
    /// call stack.
    fn visit(&mut self, root: usize) {
        self.discover(root);
        let mut frames = vec![(root, 0_usize)];
        while let Some(&(v, pos)) = frames.last() {
            if let Some(frame) = frames.last_mut() {
                frame.1 = pos.saturating_add(1);
            }
            match self.adj.get(v).and_then(|out| out.get(pos)).copied() {
                Some(w) if at(&self.number, w) == 0 => {
                    self.discover(w);
                    frames.push((w, 0));
                }
                Some(w) => {
                    if self.on_stack.get(w).copied().unwrap_or(false) {
                        self.lower(v, at(&self.number, w));
                    }
                }
                None => {
                    frames.pop();
                    self.finish(v);
                    if let Some(&(parent, _)) = frames.last() {
                        self.lower(parent, at(&self.low, v));
                    }
                }
            }
        }
    }
}

impl View<'_> {
    /// The status of vertex `v`, if statuses were given.
    fn status_of(&self, v: usize) -> Option<Status> {
        self.status.as_ref().and_then(|all| all.get(v).copied())
    }
}

impl<'a> LinkGraph<'a, Resolved> {
    /// Finds the link conditions on a cycle, removes their cyclic edges and
    /// fixes the evaluation order. Reports one defect per cyclic condition,
    /// in row order.
    #[must_use]
    pub fn condense(self) -> Built<'a, Acyclic> {
        let Self {
            mut edges,
            links,
            rows,
            state: Resolved,
            vertices,
        } = self;
        let comp = Search::run(&edges);

        let mut cyclic = BTreeSet::new();
        let mut removed = Vec::new();
        for &v in links.keys() {
            let home = at(&comp, v);
            if let Some(out) = edges.get_mut(v) {
                out.retain(|&w| {
                    let inside = at(&comp, w) == home;
                    if inside {
                        cyclic.insert(v);
                        removed.push((v, w));
                    }
                    !inside
                });
            }
        }
        removed.sort_unstable();
        removed.dedup();

        let defects = cyclic
            .iter()
            .filter_map(|&v| vertices.get(v).and_then(|x| rows.get(x.row)))
            .map(|row| {
                let message =
                    format!("condition {} lies on a link cycle", row.id);
                Defect::at(row, message)
            })
            .collect();

        let mut order: Vec<usize> = (0..vertices.len()).collect();
        order.sort_by_key(|&v| {
            (at(&comp, v), vertices.get(v).map(|x| x.kind), v)
        });

        let state = Acyclic {
            cyclic,
            order,
            removed,
        };
        let graph = LinkGraph {
            edges,
            links,
            rows,
            state,
            vertices,
        };
        (graph, defects)
    }

    /// Builds the graph and reports the resolution defects, in row order.
    ///
    /// A row whose method is `link` is a link condition. A target
    /// is a qualified ID `<doc>#<id>` or a bare document name `<doc>`.
    #[must_use]
    pub fn resolve(rows: &'a [Row]) -> Built<'a, Resolved> {
        let mut builder = Builder::default();
        builder.scan(rows);
        builder.parents(rows);
        let (links, defects) = builder.links(rows);
        let graph = Self {
            edges: builder.edges,
            links,
            rows,
            state: Resolved,
            vertices: builder.vertices,
        };
        (graph, defects)
    }
}

impl<'a> LinkGraph<'a, Acyclic> {
    /// Derives the status of every condition row, keyed by condition ID.
    /// `leaf` supplies the status of a row whose method is not `link`.
    ///
    /// A link condition on a cycle is `Fail` and one with no target is
    /// `Uncovered`. Any other takes the worst status of its targets, a target
    /// that resolved to nothing counting as `Unrun`. A requirement or a
    /// document takes the worst status of the vertices it depends on, and is
    /// `Uncovered` when it depends on none.
    #[must_use]
    pub fn derive(
        &self,
        mut leaf: impl FnMut(&Row) -> ConditionStatus,
    ) -> ConditionMap<'a> {
        let mut derived = ConditionMap::new();
        self.fold(|v, status| {
            let Some(row) = self.row_of(v) else {
                return Status::Uncovered;
            };
            let condition = self
                .link_status(v, row, status)
                .unwrap_or_else(|| leaf(row));
            let result = condition.status;
            derived.insert(row.id.as_str(), condition);
            result
        });
        derived
    }

    /// Renders the link view of the graph as text in `format`. `statuses`
    /// is the result of [`LinkGraph::derive`]; without it the vertices carry
    /// no status.
    ///
    /// The link view holds the link conditions, their parent requirements, the
    /// documents defining those requirements and every vertex a link names.
    /// A removed cyclic edge is drawn dashed and red.
    #[must_use]
    pub fn render(
        &self,
        format: GraphFormat,
        statuses: Option<&ConditionMap<'a>>,
    ) -> String {
        let rolled = statuses.map(|derived| {
            self.fold(|v, _| {
                self.row_of(v)
                    .and_then(|row| derived.get(row.id.as_str()))
                    .map_or(Status::Uncovered, |c| c.status)
            })
        });
        let shown = self.link_view();
        let kept: Vec<Edge> = shown
            .iter()
            .flat_map(|&u| {
                let out = self.edges.get(u).into_iter().flatten();
                out.filter(|w| shown.contains(w)).map(move |&w| (u, w))
            })
            .collect();
        let cut: Vec<Edge> = (self.state.removed.iter().copied())
            .filter(|(u, w)| shown.contains(u) && shown.contains(w))
            .collect();
        let view = View {
            cut: &cut,
            kept: &kept,
            shown: &shown,
            status: rolled,
        };
        match format {
            GraphFormat::Mermaid => self.mermaid(&view),
            GraphFormat::Dot => self.dot(&view),
        }
    }

    /// The link view as a Graphviz `digraph`.
    fn dot(&self, view: &View<'_>) -> String {
        let mut lines =
            vec!["digraph links {".to_string(), "    rankdir=TB;".to_string()];
        for &v in view.shown {
            let Some(vertex) = self.vertices.get(v) else {
                continue;
            };
            let status = view.status_of(v);
            let (shape, rounded) = match vertex.kind {
                Kind::Document => ("ellipse", false),
                Kind::Requirement => ("box", false),
                Kind::Condition => ("box", true),
            };
            let style = match (rounded, status) {
                (false, None) => String::new(),
                (true, None) => ", style=\"rounded\"".to_string(),
                (false, Some(s)) => {
                    format!(", style=\"filled\", fillcolor=\"{}\"", fill_of(s))
                }
                (true, Some(s)) => {
                    format!(
                        ", style=\"rounded,filled\", fillcolor=\"{}\"",
                        fill_of(s)
                    )
                }
            };
            let text = label(vertex.name, status).replace('"', "\\\"");
            lines.push(format!(
                "    n{v} [label=\"{text}\", shape={shape}{style}];"
            ));
        }
        for &(u, w) in view.kept {
            lines.push(format!("    n{u} -> n{w};"));
        }
        for &(u, w) in view.cut {
            let attrs = format!("style=dashed, color=\"{CYCLE_STROKE}\"");
            lines.push(format!("    n{u} -> n{w} [{attrs}];"));
        }
        lines.push("}".to_string());
        lines.push(String::new());
        lines.join("\n")
    }

    /// Visits every vertex once, each after the vertices it depends on, and
    /// stores what `condition` returns for a condition and the worst status
    /// of the dependencies for any other vertex.
    fn fold(
        &self,
        mut condition: impl FnMut(usize, &[Status]) -> Status,
    ) -> Vec<Status> {
        let mut status = vec![Status::Pass; self.vertices.len()];
        for &v in &self.state.order {
            let Some(vertex) = self.vertices.get(v) else {
                continue;
            };
            let value = match vertex.kind {
                Kind::Condition => condition(v, &status),
                Kind::Requirement | Kind::Document => self
                    .edges
                    .get(v)
                    .and_then(|out| {
                        out.iter().filter_map(|&w| status.get(w).copied()).max()
                    })
                    .unwrap_or(Status::Uncovered),
            };
            put_status(&mut status, v, value);
        }
        status
    }

    /// The vertices of the link view.
    fn link_view(&self) -> BTreeSet<usize> {
        let mut shown = BTreeSet::new();
        for (&condition, targets) in &self.links {
            shown.insert(condition);
            shown.extend(targets.iter().filter_map(|t| match t {
                Target::Resolved(w) => Some(*w),
                Target::Unresolved => None,
            }));
        }
        let parents: Vec<usize> = (0..self.vertices.len())
            .filter(|&v| kind_at(&self.vertices, v) == Some(Kind::Requirement))
            .filter(|&v| self.depends_on(v, |c| self.links.contains_key(c)))
            .collect();
        let documents = (0..self.vertices.len())
            .filter(|&v| kind_at(&self.vertices, v) == Some(Kind::Document))
            .filter(|&v| self.depends_on(v, |r| parents.contains(r)));
        shown.extend(documents);
        shown.extend(parents.iter().copied());
        shown
    }

    /// Whether vertex `v` depends on a vertex that satisfies `pred`.
    fn depends_on(&self, v: usize, pred: impl Fn(&usize) -> bool) -> bool {
        self.edges.get(v).is_some_and(|out| out.iter().any(pred))
    }

    /// The status of link condition `v` of `row`, given the statuses of the
    /// vertices it depends on, or `None` when `v` is not a link condition.
    fn link_status(
        &self,
        v: usize,
        row: &Row,
        status: &[Status],
    ) -> Option<ConditionStatus> {
        let targets = self.links.get(&v)?;
        let mut items = Vec::new();
        let result = if self.state.cyclic.contains(&v) {
            Status::Fail
        } else if targets.is_empty() {
            Status::Uncovered
        } else {
            let mut worst = Status::Pass;
            for (name, target) in row.targets.iter().zip(targets) {
                let seen = match target {
                    Target::Resolved(w) => {
                        status.get(*w).copied().unwrap_or(Status::Unrun)
                    }
                    Target::Unresolved => Status::Unrun,
                };
                worst = worst.max(seen);
                items.push(ItemOutcome {
                    target: name.clone(),
                    passed: !fails_gate(seen),
                    message: None,
                });
            }
            worst
        };
        Some(ConditionStatus {
            id: row.id.clone(),
            parents: row.parents.clone(),
            method: row.method.clone().unwrap_or_default(),
            status: result,
            items,
        })
    }

    /// The link view as a Mermaid `flowchart`.
    fn mermaid(&self, view: &View<'_>) -> String {
        let mut lines = vec!["flowchart TD".to_string()];
        for &v in view.shown {
            let Some(vertex) = self.vertices.get(v) else {
                continue;
            };
            let status = view.status_of(v);
            let text = label(vertex.name, status).replace('"', "#quot;");
            let node = match vertex.kind {
                Kind::Document => format!("n{v}([\"{text}\"])"),
                Kind::Requirement => format!("n{v}[\"{text}\"]"),
                Kind::Condition => format!("n{v}(\"{text}\")"),
            };
            let class = status
                .map_or_else(String::new, |s| format!(":::{}", class_of(s)));
            lines.push(format!("    {node}{class}"));
        }
        for &(u, w) in view.kept {
            lines.push(format!("    n{u} --> n{w}"));
        }
        for &(u, w) in view.cut {
            lines.push(format!("    n{u} -.-> n{w}"));
        }
        if view.status.is_some() {
            for s in STATUSES {
                let (class, fill) = (class_of(s), fill_of(s));
                lines.push(format!(
                    "    classDef {class} fill:{fill},stroke:#555,color:#000"
                ));
            }
        }
        for i in 0..view.cut.len() {
            let index = view.kept.len().saturating_add(i);
            let attrs = format!("stroke:{CYCLE_STROKE},stroke-dasharray:5 5");
            lines.push(format!("    linkStyle {index} {attrs}"));
        }
        lines.push(String::new());
        lines.join("\n")
    }

    /// The row that introduces vertex `v`.
    fn row_of(&self, v: usize) -> Option<&'a Row> {
        self.vertices.get(v).and_then(|x| self.rows.get(x.row))
    }
}

/// `values[i]`, or 0 outside the slice; every index comes from a vertex of the
/// graph the slice was sized for.
fn at(values: &[usize], i: usize) -> usize {
    values.get(i).copied().unwrap_or(0)
}

/// The Mermaid class of `status`.
const fn class_of(status: Status) -> &'static str {
    match status {
        Status::Pass => "pass",
        Status::Review => "review",
        Status::Deferred => "deferred",
        Status::Uncovered => "uncovered",
        Status::Unrun => "unrun",
        Status::Fail => "fail",
    }
}

/// Whether a status fails the gate.
const fn fails_gate(status: Status) -> bool {
    matches!(status, Status::Uncovered | Status::Unrun | Status::Fail)
}

/// The fill of `status`.
const fn fill_of(status: Status) -> &'static str {
    match status {
        Status::Pass => FILL_PASS,
        Status::Review | Status::Deferred => FILL_PENDING,
        Status::Uncovered | Status::Unrun => FILL_OPEN,
        Status::Fail => FILL_FAIL,
    }
}

/// The kind of vertex `v`.
fn kind_at(vertices: &[Vertex<'_>], v: usize) -> Option<Kind> {
    vertices.get(v).map(|x| x.kind)
}

/// The label of a vertex: its name and, with a status, the status name.
fn label(name: &str, status: Option<Status>) -> String {
    status.map_or_else(|| name.to_string(), |s| format!("{name} ({s:?})"))
}

/// Sets `values[i]`; an index outside the slice is ignored.
fn put(values: &mut [usize], i: usize, value: usize) {
    if let Some(slot) = values.get_mut(i) {
        *slot = value;
    }
}

/// Sets `values[i]`; an index outside the slice is ignored.
fn put_status(values: &mut [Status], i: usize, value: Status) {
    if let Some(slot) = values.get_mut(i) {
        *slot = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::SCHEMA;

    /// A target and whether it passed.
    type Outcome<'t> = (&'t str, bool);

    /// Statuses by condition ID.
    type Table<'t> = &'t [(&'t str, Status)];

    fn def(id: &str, line: usize) -> Row {
        Row {
            schema: SCHEMA,
            id: id.to_string(),
            kind: DEFINITION.to_string(),
            parents: Vec::new(),
            method: None,
            targets: Vec::new(),
            file: "docs/t-design.md".to_string(),
            line,
            text: String::new(),
        }
    }

    fn cond(
        ids: [&str; 2],
        method: &str,
        targets: &[&str],
        line: usize,
    ) -> Row {
        let [id, parent] = ids;
        Row {
            schema: SCHEMA,
            id: id.to_string(),
            kind: CONDITION.to_string(),
            parents: vec![parent.to_string()],
            method: Some(method.to_string()),
            targets: targets.iter().map(ToString::to_string).collect(),
            file: "docs/t-design.md".to_string(),
            line,
            text: String::new(),
        }
    }

    fn link(id: &str, parent: &str, targets: &[&str], line: usize) -> Row {
        cond([id, parent], "link", targets, line)
    }

    fn messages(defects: &[Defect]) -> Vec<String> {
        defects.iter().map(|d| d.message.clone()).collect()
    }

    fn acyclic(rows: &[Row]) -> Built<'_, Acyclic> {
        let (graph, mut defects) = LinkGraph::resolve(rows);
        let (graph, cycles) = graph.condense();
        defects.extend(cycles);
        (graph, defects)
    }

    /// A leaf evaluation that takes each row's status from `table` by ID and
    /// is `Pass` for a row the table lacks.
    fn leaf_status(table: Table<'_>, row: &Row) -> ConditionStatus {
        ConditionStatus {
            id: row.id.clone(),
            parents: row.parents.clone(),
            method: row.method.clone().unwrap_or_default(),
            status: table
                .iter()
                .find(|(id, _)| *id == row.id)
                .map_or(Status::Pass, |(_, s)| *s),
            items: Vec::new(),
        }
    }

    fn status_of(rows: &[Row], leaf_table: Table<'_>, id: &str) -> Status {
        let (graph, _) = acyclic(rows);
        graph
            .derive(|row| leaf_status(leaf_table, row))
            .get(id)
            .map(|c| c.status)
            .unwrap()
    }

    fn out_degree(rows: &[Row], name: &str) -> usize {
        let (graph, _) = LinkGraph::resolve(rows);
        let v = graph.vertices.iter().position(|x| x.name == name).unwrap();
        graph.edges.get(v).map_or(0, Vec::len)
    }

    #[test]
    fn link_without_a_target_is_reported() {
        let rows = [def("a#FR-1", 1), link("a#VC-1.1", "a#FR-1", &[], 3)];
        let (_, defects) = LinkGraph::resolve(&rows);
        assert_eq!(
            messages(&defects),
            ["condition a#VC-1.1 names no link target"]
        );
        assert_eq!(defects.first().map(|d| d.line), Some(3));
    }

    #[test]
    fn link_to_an_undefined_item_is_reported() {
        let rows = [
            def("a#FR-1", 1),
            link("a#VC-1.1", "a#FR-1", &["b#FR-9", "nowhere"], 3),
        ];
        let (_, defects) = LinkGraph::resolve(&rows);
        assert_eq!(
            messages(&defects),
            [
                "condition a#VC-1.1 target 'b#FR-9' names no selected document, \
                 requirement or condition",
                "condition a#VC-1.1 target 'nowhere' names no selected document, \
                 requirement or condition",
            ]
        );
    }

    #[test]
    fn link_to_a_document_without_requirements_is_reported() {
        let rows = [
            def("a#FR-1", 1),
            link("a#VC-1.1", "a#FR-1", &["b"], 3),
            cond(["b#VC-1.1", "b#FR-1"], "libtest", &["t"], 5),
        ];
        let (_, defects) = LinkGraph::resolve(&rows);
        assert_eq!(
            messages(&defects),
            ["condition a#VC-1.1 target 'b' defines no requirement"]
        );
    }

    #[test]
    fn targets_naming_a_document_requirement_or_condition_resolve() {
        let rows = [
            def("a#FR-1", 1),
            cond(["a#VC-1.1", "a#FR-1"], "libtest", &["t"], 3),
            def("b#FR-1", 5),
            link("a#VC-2.1", "a#FR-1", &["b", "b#FR-1", "a#VC-1.1"], 7),
        ];
        let (_, defects) = LinkGraph::resolve(&rows);
        assert!(defects.is_empty(), "{defects:?}");
    }

    #[test]
    fn a_self_link_is_a_cycle() {
        let rows = [
            def("a#FR-1", 1),
            link("a#VC-1.1", "a#FR-1", &["a#VC-1.1"], 3),
        ];
        let (_, defects) = acyclic(&rows);
        assert_eq!(
            messages(&defects),
            ["condition a#VC-1.1 lies on a link cycle"]
        );
    }

    #[test]
    fn a_link_to_the_parent_requirement_is_a_cycle() {
        let rows =
            [def("a#FR-1", 1), link("a#VC-1.1", "a#FR-1", &["a#FR-1"], 3)];
        let (_, defects) = acyclic(&rows);
        assert_eq!(
            messages(&defects),
            ["condition a#VC-1.1 lies on a link cycle"]
        );
    }

    #[test]
    fn a_link_to_the_own_document_is_a_cycle() {
        let rows = [def("a#FR-1", 1), link("a#VC-1.1", "a#FR-1", &["a"], 3)];
        let (_, defects) = acyclic(&rows);
        assert_eq!(
            messages(&defects),
            ["condition a#VC-1.1 lies on a link cycle"]
        );
    }

    #[test]
    fn a_two_document_cycle_reports_both_conditions() {
        let rows = [
            def("a#FR-1", 1),
            link("a#VC-1.1", "a#FR-1", &["b"], 3),
            def("b#FR-1", 5),
            link("b#VC-1.1", "b#FR-1", &["a"], 7),
        ];
        let (_, defects) = acyclic(&rows);
        assert_eq!(
            messages(&defects),
            [
                "condition a#VC-1.1 lies on a link cycle",
                "condition b#VC-1.1 lies on a link cycle",
            ]
        );
        assert_eq!(defects.iter().map(|d| d.line).collect::<Vec<_>>(), [3, 7]);
    }

    #[test]
    fn a_link_to_a_sibling_requirement_is_no_cycle() {
        let rows = [
            def("a#FR-1", 1),
            link("a#VC-1.1", "a#FR-1", &["a#FR-2"], 3),
            def("a#FR-2", 5),
            cond(["a#VC-2.1", "a#FR-2"], "libtest", &["t"], 7),
        ];
        let (_, defects) = acyclic(&rows);
        assert!(defects.is_empty(), "{defects:?}");
    }

    #[test]
    fn an_acyclic_graph_has_no_defects() {
        let rows = [
            def("a#FR-1", 1),
            cond(["a#VC-1.1", "a#FR-1"], "libtest", &["t"], 3),
            def("b#FR-1", 5),
            link("b#VC-1.1", "b#FR-1", &["a", "a#VC-1.1"], 7),
        ];
        let (_, defects) = acyclic(&rows);
        assert!(defects.is_empty(), "{defects:?}");
    }

    #[test]
    fn derive_takes_the_worst_status_of_the_targets() {
        let rows = [
            def("a#FR-1", 1),
            link("a#VC-1.1", "a#FR-1", &["b#VC-1.1", "c#VC-1.1"], 3),
            def("b#FR-1", 5),
            cond(["b#VC-1.1", "b#FR-1"], "libtest", &["t"], 7),
            def("c#FR-1", 9),
            cond(["c#VC-1.1", "c#FR-1"], "libtest", &["t"], 11),
        ];
        let (graph, _) = acyclic(&rows);
        let table = [("c#VC-1.1", Status::Unrun)];
        let derived = graph.derive(|row| leaf_status(&table, row));
        let linked = derived.get("a#VC-1.1").unwrap();
        assert_eq!(linked.status, Status::Unrun);
        let outcomes: Vec<Outcome<'_>> = linked
            .items
            .iter()
            .map(|i| (i.target.as_str(), i.passed))
            .collect();
        assert_eq!(outcomes, [("b#VC-1.1", true), ("c#VC-1.1", false)]);
    }

    #[test]
    fn derive_gives_a_requirement_or_condition_target_its_status() {
        let rows = [
            def("a#FR-1", 1),
            link("a#VC-1.1", "a#FR-1", &["b#FR-1"], 3),
            link("a#VC-2.1", "a#FR-1", &["b#VC-2.1"], 4),
            def("b#FR-1", 5),
            cond(["b#VC-1.1", "b#FR-1"], "libtest", &["t"], 7),
            cond(["b#VC-2.1", "b#FR-1"], "review", &[], 8),
        ];
        let table = [("b#VC-1.1", Status::Fail), ("b#VC-2.1", Status::Review)];
        assert_eq!(status_of(&rows, &table, "a#VC-1.1"), Status::Fail);
        assert_eq!(status_of(&rows, &table, "a#VC-2.1"), Status::Review);
    }

    #[test]
    fn derive_gives_a_document_the_worst_status_of_its_requirements() {
        let rows = [
            def("a#FR-1", 1),
            link("a#VC-1.1", "a#FR-1", &["b"], 3),
            def("b#FR-1", 5),
            cond(["b#VC-1.1", "b#FR-1"], "libtest", &["t"], 7),
            def("b#FR-2", 9),
            cond(["b#VC-2.1", "b#FR-2"], "libtest", &["t"], 11),
        ];
        let table =
            [("b#VC-1.1", Status::Deferred), ("b#VC-2.1", Status::Unrun)];
        assert_eq!(status_of(&rows, &table, "a#VC-1.1"), Status::Unrun);
    }

    #[test]
    fn derive_evaluates_a_chain_of_links_dependencies_first() {
        let rows = [
            def("a#FR-1", 1),
            link("a#VC-1.1", "a#FR-1", &["b"], 3),
            def("b#FR-1", 5),
            link("b#VC-1.1", "b#FR-1", &["c"], 7),
            def("c#FR-1", 9),
            cond(["c#VC-1.1", "c#FR-1"], "libtest", &["t"], 11),
        ];
        let table = [("c#VC-1.1", Status::Fail)];
        assert_eq!(status_of(&rows, &table, "b#VC-1.1"), Status::Fail);
        assert_eq!(status_of(&rows, &table, "a#VC-1.1"), Status::Fail);
        assert_eq!(status_of(&rows, &[], "a#VC-1.1"), Status::Pass);
    }

    #[test]
    fn derive_fails_a_link_on_a_cycle() {
        let rows =
            [def("a#FR-1", 1), link("a#VC-1.1", "a#FR-1", &["a#FR-1"], 3)];
        assert_eq!(status_of(&rows, &[], "a#VC-1.1"), Status::Fail);
    }

    #[test]
    fn derive_marks_a_link_without_a_target_uncovered() {
        let rows = [def("a#FR-1", 1), link("a#VC-1.1", "a#FR-1", &[], 3)];
        assert_eq!(status_of(&rows, &[], "a#VC-1.1"), Status::Uncovered);
    }

    #[test]
    fn derive_marks_an_unresolved_target_unrun() {
        let rows =
            [def("a#FR-1", 1), link("a#VC-1.1", "a#FR-1", &["b#FR-9"], 3)];
        assert_eq!(status_of(&rows, &[], "a#VC-1.1"), Status::Unrun);
    }

    #[test]
    fn derive_marks_an_empty_document_uncovered() {
        let rows = [
            def("a#FR-1", 1),
            link("a#VC-1.1", "a#FR-1", &["b"], 3),
            cond(["b#VC-1.1", "b#FR-1"], "libtest", &["t"], 5),
        ];
        assert_eq!(status_of(&rows, &[], "a#VC-1.1"), Status::Uncovered);
    }

    #[test]
    fn derive_marks_a_requirement_without_conditions_uncovered() {
        let rows = [
            def("a#FR-1", 1),
            link("a#VC-1.1", "a#FR-1", &["b#FR-1"], 3),
            def("b#FR-1", 5),
        ];
        assert_eq!(status_of(&rows, &[], "a#VC-1.1"), Status::Uncovered);
    }

    #[test]
    fn a_document_target_adds_one_edge_per_link() {
        let (m, n) = (4, 3);
        let mut rows = vec![def("a#FR-1", 1)];
        for i in 0..m {
            let id = format!("a#VC-1.{i}");
            rows.push(link(&id, "a#FR-1", &["b"], 10 + i));
        }
        for i in 0..n {
            let req = format!("b#FR-{i}");
            rows.push(def(&req, 20 + i));
            let id = format!("b#VC-{i}.1");
            rows.push(cond([&id, &req], "libtest", &["t"], 30 + i));
        }
        for i in 0..m {
            assert_eq!(out_degree(&rows, &format!("a#VC-1.{i}")), 1);
        }
        assert_eq!(out_degree(&rows, "b"), n);
    }

    #[test]
    fn a_long_link_chain_is_processed_on_the_default_stack() {
        let docs = 100_000_usize;
        let mut rows = Vec::with_capacity(docs * 2);
        for i in 0..docs {
            rows.push(def(&format!("d{i}#FR-1"), 1));
            let next = format!("d{}", i + 1);
            if i + 1 < docs {
                rows.push(link(
                    &format!("d{i}#VC-1.1"),
                    &format!("d{i}#FR-1"),
                    &[&next],
                    2,
                ));
            } else {
                let ids = [format!("d{i}#VC-1.1"), format!("d{i}#FR-1")];
                rows.push(cond([&ids[0], &ids[1]], "libtest", &["t"], 2));
            }
        }
        let (graph, defects) = acyclic(&rows);
        assert!(defects.is_empty(), "{defects:?}");
        let derived = graph.derive(|row| leaf_status(&[], row));
        assert_eq!(derived.len(), docs);
        assert!(derived.values().all(|c| c.status == Status::Pass));
    }

    /// Two documents, `a` linking to `b`; `b` is not part of the link view
    /// beyond its document vertex.
    fn linked_rows() -> Vec<Row> {
        vec![
            def("a#FR-1", 1),
            link("a#VC-1.1", "a#FR-1", &["b"], 3),
            def("b#FR-1", 5),
            cond(["b#VC-1.1", "b#FR-1"], "libtest", &["t"], 7),
        ]
    }

    fn cycle_rows() -> Vec<Row> {
        vec![def("a#FR-1", 1), link("a#VC-1.1", "a#FR-1", &["a#FR-1"], 3)]
    }

    fn rendered(rows: &[Row], format: GraphFormat) -> String {
        let (graph, _) = acyclic(rows);
        let derived = graph.derive(|row| leaf_status(&[], row));
        graph.render(format, Some(&derived))
    }

    #[test]
    fn render_mermaid_shows_status_and_marks_cycle_edges() {
        let text = rendered(&cycle_rows(), GraphFormat::Mermaid);
        let expected = "flowchart TD\n\
            \x20   n0([\"a (Fail)\"]):::fail\n\
            \x20   n1[\"a#FR-1 (Fail)\"]:::fail\n\
            \x20   n2(\"a#VC-1.1 (Fail)\"):::fail\n\
            \x20   n0 --> n1\n\
            \x20   n1 --> n2\n\
            \x20   n2 -.-> n1\n\
            \x20   classDef pass fill:#c8e6c9,stroke:#555,color:#000\n\
            \x20   classDef review fill:#e0e0e0,stroke:#555,color:#000\n\
            \x20   classDef deferred fill:#e0e0e0,stroke:#555,color:#000\n\
            \x20   classDef uncovered fill:#ffe0b2,stroke:#555,color:#000\n\
            \x20   classDef unrun fill:#ffe0b2,stroke:#555,color:#000\n\
            \x20   classDef fail fill:#ffcdd2,stroke:#555,color:#000\n\
            \x20   linkStyle 2 stroke:#d32f2f,stroke-dasharray:5 5\n";
        assert_eq!(text, expected);
    }

    #[test]
    fn render_dot_shows_status_and_marks_cycle_edges() {
        let text = rendered(&cycle_rows(), GraphFormat::Dot);
        let expected = "digraph links {\n\
            \x20   rankdir=TB;\n\
            \x20   n0 [label=\"a (Fail)\", shape=ellipse, style=\"filled\", fillcolor=\"#ffcdd2\"];\n\
            \x20   n1 [label=\"a#FR-1 (Fail)\", shape=box, style=\"filled\", fillcolor=\"#ffcdd2\"];\n\
            \x20   n2 [label=\"a#VC-1.1 (Fail)\", shape=box, style=\"rounded,filled\", fillcolor=\"#ffcdd2\"];\n\
            \x20   n0 -> n1;\n\
            \x20   n1 -> n2;\n\
            \x20   n2 -> n1 [style=dashed, color=\"#d32f2f\"];\n\
            }\n";
        assert_eq!(text, expected);
    }

    #[test]
    fn render_draws_only_the_link_view() {
        let rows = linked_rows();
        for format in [GraphFormat::Mermaid, GraphFormat::Dot] {
            let text = rendered(&rows, format);
            assert!(text.contains("a#VC-1.1 (Pass)"), "{text}");
            assert!(text.contains("a#FR-1 (Pass)"), "{text}");
            assert!(text.contains("\"b (Pass)\""), "{text}");
            assert!(!text.contains("b#FR-1"), "{text}");
            assert!(!text.contains("b#VC-1.1"), "{text}");
        }
        let (graph, _) = acyclic(&rows);
        let bare = graph.render(GraphFormat::Mermaid, None);
        assert!(
            !bare.contains("Pass") && !bare.contains("classDef"),
            "{bare}"
        );
    }

    #[test]
    fn render_is_byte_identical_across_runs() {
        for rows in [linked_rows(), cycle_rows()] {
            for format in [GraphFormat::Mermaid, GraphFormat::Dot] {
                assert_eq!(rendered(&rows, format), rendered(&rows, format));
            }
        }
    }
}
