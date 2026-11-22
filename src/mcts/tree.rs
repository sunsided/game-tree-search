//! The search tree of Monte Carlo tree search: nodes in one arena, children stored next to
//! each other, positions replayed from the root rather than stored. Every node is expanded
//! when it is added to a simulation's path, the root when the search starts.

use alloc::vec::Vec;

use super::policy::{NodeStats, Selection};
use super::{MctsResult, MoveStats, TreeStats};
use crate::{Game, Side};

/// One position of the tree.
#[derive(Debug, Clone)]
struct Node<M> {
    /// The move that leads here; `None` at the root.
    mv: Option<M>,
    /// The index of the first child in the arena; the children follow it.
    first: usize,
    /// The number of children; `0` until expanded, or where the game ends.
    children: usize,
    /// What the simulations through the node found.
    stats: NodeStats,
    /// The side to move; `None` until expanded, or where the game ends.
    side: Option<Side>,
    /// Moves from the root to the node.
    depth: u32,
}

/// Where a simulation's descent ended.
pub(super) struct Leaf<P> {
    /// The position reached; `None` if the descent stayed at the root.
    pub(super) position: Option<P>,
    /// Whether the game ends there, so its reward needs no evaluation.
    pub(super) finished: bool,
}

/// The tree of one search, with buffers kept between searches.
#[derive(Debug, Clone)]
pub(super) struct Tree<M> {
    /// Every node; the root is at index `0`.
    nodes: Vec<Node<M>>,
    /// The nodes the current simulation passed through, from the root.
    path: Vec<usize>,
    /// The move list for expansions, kept for its capacity.
    moves: Vec<M>,
    /// The counters of the current search.
    pub(super) stats: TreeStats,
}

impl<M: Clone> Tree<M> {
    /// An empty tree.
    pub(super) const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            path: Vec::new(),
            moves: Vec::new(),
            stats: TreeStats {
                simulations: 0,
                nodes: 0,
                depth: 0,
            },
        }
    }

    /// Starts a new search at `root`, with the root expanded so its moves are listed even
    /// before the first simulation.
    pub(super) fn reset<G: Game<Move = M>>(&mut self, game: &G, root: &G::Position) {
        self.nodes.clear();
        self.stats = TreeStats::default();
        self.nodes.push(Node {
            mv: None,
            first: 0,
            children: 0,
            stats: NodeStats {
                visits: 0,
                value: 0.0,
                prior: 1.0,
            },
            side: None,
            depth: 0,
        });
        self.expand(game, root, 0);
    }

    /// Descends from the root: through visited children chosen by `selection`, into the
    /// first unvisited child, which it expands to tell whether the game ends there, or to a
    /// node where the game ends. Records the path.
    pub(super) fn select<G: Game<Move = M>>(
        &mut self,
        game: &G,
        root: &G::Position,
        selection: &impl Selection,
    ) -> Leaf<G::Position> {
        self.path.clear();
        self.path.push(0);
        let (mut index, mut current) = (0, None);
        loop {
            let here = current.as_ref().unwrap_or(root);
            let Some(child) = self.choose(index, selection) else {
                return Leaf {
                    position: current,
                    finished: true,
                };
            };

            let mv = self.nodes[child].mv.as_ref().expect("a child has a move");
            let next = game.play(here, mv);
            self.path.push(child);
            if self.nodes[child].stats.visits == 0 {
                self.expand(game, &next, child);
                return Leaf {
                    finished: self.nodes[child].children == 0,
                    position: Some(next),
                };
            }

            (index, current) = (child, Some(next));
        }
    }

    /// The child of `index` a simulation descends into: the first unvisited one, else the
    /// best by `selection` for the side to move, ties going to the earliest. `None` if the
    /// game ends at `index`.
    fn choose(&self, index: usize, selection: &impl Selection) -> Option<usize> {
        let node = &self.nodes[index];
        let side = node.side?;
        let children = node.first..node.first + node.children;
        if let Some(unvisited) = children
            .clone()
            .find(|&child| self.nodes[child].stats.visits == 0)
        {
            return Some(unvisited);
        }

        let scored = children.map(|child| {
            let score = selection.score(side, &node.stats, &self.nodes[child].stats);
            (child, score)
        });
        let best = scored.reduce(|best, candidate| {
            if candidate.1 > best.1 {
                candidate
            } else {
                best
            }
        });

        best.map(|(child, _)| child)
    }

    /// Adds the children of `index`, the moves of `position`, in [`Game::order_moves`] order.
    fn expand<G: Game<Move = M>>(&mut self, game: &G, position: &G::Position, index: usize) {
        let mut moves = core::mem::take(&mut self.moves);
        moves.clear();
        game.moves(position, &mut moves);
        game.order_moves(position, &mut moves);

        let (first, children) = (self.nodes.len(), moves.len());
        let depth = self.nodes[index].depth + 1;
        let prior = 1.0 / f64::from(u32::try_from(children.max(1)).unwrap_or(u32::MAX));
        for mv in moves.drain(..) {
            self.nodes.push(Node {
                mv: Some(mv),
                first: 0,
                children: 0,
                stats: NodeStats {
                    visits: 0,
                    value: 0.0,
                    prior,
                },
                side: None,
                depth,
            });
        }

        self.moves = moves;
        let side = (children > 0).then(|| game.side_to_move(position));
        let node = &mut self.nodes[index];
        (node.first, node.children, node.side) = (first, children, side);
        self.stats.nodes = u64::try_from(self.nodes.len()).unwrap_or(u64::MAX);
        if children > 0 {
            self.stats.depth = self.stats.depth.max(depth);
        }
    }

    /// Adds `value`, [`Side::Max`]'s reward, to every node of the last simulation's path.
    pub(super) fn backup(&mut self, value: f64) {
        for &index in &self.path {
            let stats = &mut self.nodes[index].stats;
            stats.visits = stats.visits.saturating_add(1);
            stats.value += value;
        }

        self.stats.simulations += 1;
    }

    /// The result of the search so far.
    pub(super) fn result(&self) -> MctsResult<M> {
        let root = &self.nodes[0];
        let moves = (root.first..root.first + root.children).map(|child| {
            let node = &self.nodes[child];
            MoveStats {
                mv: node.mv.clone().expect("a child has a move"),
                visits: node.stats.visits,
                value: node.stats.value,
            }
        });
        let root: Vec<_> = moves.collect();

        MctsResult {
            best: self
                .most_visited(0)
                .and_then(|child| self.nodes[child].mv.clone()),
            root,
            pv: self.line(),
            stats: self.stats,
        }
    }

    /// The child of `index` with the most visits, ties going to the earliest.
    fn most_visited(&self, index: usize) -> Option<usize> {
        let node = &self.nodes[index];
        let visits = |child: usize| self.nodes[child].stats.visits;

        (node.first..node.first + node.children).reduce(|best, child| {
            if visits(child) > visits(best) {
                child
            } else {
                best
            }
        })
    }

    /// The most visited line from the root, while its nodes were visited.
    fn line(&self) -> Vec<M> {
        let mut line = Vec::new();
        let mut index = 0;
        while let Some(child) = self.most_visited(index) {
            if self.nodes[child].stats.visits == 0 {
                break;
            }

            line.extend(self.nodes[child].mv.clone());
            index = child;
        }

        line
    }
}
