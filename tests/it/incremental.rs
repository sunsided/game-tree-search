//! Make/unmake: in-place searches equal copying searches to the last counter, and every
//! `make` is undone by its `unmake`.

use core::cell::Cell;

use game_tree_search::{
    Bound, Deepening, Game, Hashed, Incremental, SearchResult, Searcher, Side, SlotTable, Stats,
    Window,
};

use crate::algorithms::ALGORITHMS;
use crate::games::RandomTree;

/// Seeds of the random trees compared on.
const SEEDS: core::ops::Range<u64> = 0..60;

/// A [`RandomTree`] whose position is the path of tree positions from the root, so a move
/// can be made by pushing and undone by popping. Counts the moves made and not yet undone.
struct Stacked {
    /// The tree.
    tree: RandomTree,
    /// Moves made and not yet undone.
    open: Cell<i64>,
}

impl Stacked {
    /// `tree`, with nothing made.
    const fn new(tree: RandomTree) -> Self {
        Self {
            tree,
            open: Cell::new(0),
        }
    }

    /// The tree position at the end of `path`.
    fn at(path: &[u64]) -> u64 {
        *path.last().expect("a path starts at the root")
    }
}

impl Game for Stacked {
    /// The tree positions from the root.
    type Position = Vec<u64>;
    /// The tree's moves.
    type Move = u64;
    /// The tree's scores.
    type Score = i32;

    fn side_to_move(&self, path: &Vec<u64>) -> Side {
        self.tree.side_to_move(&Self::at(path))
    }

    fn moves(&self, path: &Vec<u64>, moves: &mut Vec<u64>) {
        self.tree.moves(&Self::at(path), moves);
    }

    fn play(&self, path: &Vec<u64>, mv: &u64) -> Vec<u64> {
        let mut next = path.clone();
        next.push(self.tree.play(&Self::at(path), mv));
        next
    }

    fn evaluate(&self, path: &Vec<u64>) -> i32 {
        self.tree.evaluate(&Self::at(path))
    }

    fn noisy_moves(&self, path: &Vec<u64>, moves: &mut Vec<u64>) {
        self.tree.noisy_moves(&Self::at(path), moves);
    }

    fn order_moves(&self, path: &Vec<u64>, moves: &mut [u64]) {
        self.tree.order_moves(&Self::at(path), moves);
    }

    fn satisfices(&self, side: Side, score: i32) -> bool {
        self.tree.satisfices(side, score)
    }
}

impl Incremental for Stacked {
    /// The move made, checked when it is undone.
    type Undo = u64;

    fn make(&self, path: &mut Vec<u64>, mv: &u64) -> u64 {
        self.open.set(self.open.get() + 1);
        path.push(self.tree.play(&Self::at(path), mv));
        *mv
    }

    fn unmake(&self, path: &mut Vec<u64>, mv: &u64, made: u64) {
        assert_eq!(*mv, made, "a move is undone with its own undo");
        self.open.set(self.open.get() - 1);
        path.pop();
    }
}

impl Hashed for Stacked {
    /// The tree's key of the position reached.
    type Key = u64;

    fn key(&self, path: &Vec<u64>) -> u64 {
        self.tree.key(&Self::at(path))
    }
}

/// Never stops a search early.
fn unlimited(_: &Stats) -> bool {
    false
}

/// The tree shapes compared on: plain, with transpositions, with quiescence, and with
/// both sides satisficing.
fn trees() -> [RandomTree; 4] {
    [
        RandomTree::new(4),
        RandomTree {
            width: Some(5),
            ..RandomTree::new(4)
        },
        RandomTree {
            noisy: true,
            ..RandomTree::new(4)
        },
        RandomTree {
            min_satisfied_below: Some(-20),
            max_satisfied_above: Some(40),
            ..RandomTree::new(4)
        },
    ]
}

/// Every algorithm returns the same result in place as by copying, statistics included,
/// with and without a table, and leaves no move made.
#[test]
fn in_place_searches_equal_copying_searches() {
    for tree in trees() {
        let game = Stacked::new(tree);
        for algorithm in ALGORITHMS {
            for seed in SEEDS {
                let root = vec![seed];
                let copied = Searcher::new()
                    .with_algorithm(algorithm)
                    .search(&game, &root, 4);
                let in_place = Searcher::new()
                    .with_algorithm(algorithm)
                    .search_in_place(&game, &root, 4);
                assert_eq!(in_place, copied, "{algorithm:?}, seed {seed}");

                let copied = Searcher::with_table(SlotTable::new(64))
                    .with_algorithm(algorithm)
                    .deepen(&game, &root, 5, unlimited);
                let in_place = Searcher::with_table(SlotTable::new(64))
                    .with_algorithm(algorithm)
                    .deepen_in_place(&game, &root, 5, unlimited);
                assert_eq!(in_place, copied, "{algorithm:?} deepening, seed {seed}");
                assert_eq!(game.open.get(), 0, "{algorithm:?}, seed {seed}");
            }
        }
    }
}

/// A search the budget stops partway agrees too, and still undoes every move it made.
#[test]
fn an_aborted_search_undoes_its_moves() {
    let game = Stacked::new(RandomTree::new(5));
    for seed in SEEDS {
        let root = vec![seed];
        let budget = |stats: &Stats| stats.nodes >= 150;
        let copied = Searcher::new().deepen(&game, &root, 12, budget);
        let in_place = Searcher::new().deepen_in_place(&game, &root, 12, budget);
        assert_eq!(in_place, copied, "seed {seed}");
        assert_eq!(game.open.get(), 0, "seed {seed}");
    }
}

/// A hook that asks for one narrow window every time, so iterations fail and search again.
struct Narrow;

impl<M> Deepening<M, i32> for Narrow {
    fn proceed(&mut self, _: &SearchResult<M, i32>) -> bool {
        true
    }

    fn window(&mut self, previous: &SearchResult<M, i32>, _: u32) -> Window<i32> {
        Window {
            alpha: Some(previous.score - 1),
            beta: Some(previous.score + 1),
        }
    }
}

/// Aspiration windows that fail and search again, and PVS re-searches, agree in place too.
#[test]
fn researches_agree_in_place() {
    let game = Stacked::new(RandomTree::new(4));
    for algorithm in ALGORITHMS {
        for seed in SEEDS {
            let root = vec![seed];
            let mut copying = Searcher::new().with_algorithm(algorithm);
            let copied = copying.deepen_with(&game, &root, 5, unlimited, Narrow);
            let mut in_place = Searcher::new().with_algorithm(algorithm);
            let made = in_place.deepen_in_place_with(&game, &root, 5, unlimited, Narrow);
            assert_eq!(made, copied, "{algorithm:?}, seed {seed}");
            assert_eq!(made.bound, Bound::Exact, "{algorithm:?}, seed {seed}");
            assert_eq!(game.open.get(), 0);
        }
    }
}
