//! Sample games for the tests: [`Nim`] with alternating sides and [`RandomTree`], whose side
//! to move is random, so one side often moves several times in a row.

use game_tree_search::{Game, Hashed, Side};

/// One-pile Nim: take one or two stones; whoever cannot move has lost.
pub struct Nim;

impl Game for Nim {
    /// The stones left and the side to move.
    type Position = (u32, Side);
    /// The stones taken.
    type Move = u32;
    /// `1` if [`Side::Max`] has won, `-1` if it has lost, `0` while the game goes on.
    type Score = i32;

    fn side_to_move(&self, &(_, side): &Self::Position) -> Side {
        side
    }

    fn moves(&self, &(stones, _): &Self::Position, moves: &mut Vec<u32>) {
        moves.extend((1..=2).filter(|&take| take <= stones));
    }

    fn play(&self, &(stones, side): &Self::Position, take: &u32) -> Self::Position {
        let next = match side {
            Side::Max => Side::Min,
            Side::Min => Side::Max,
        };

        (stones - take, next)
    }

    fn evaluate(&self, &(stones, side): &Self::Position) -> i32 {
        match (stones, side) {
            (0, Side::Max) => -1,
            (0, Side::Min) => 1,
            _ => 0,
        }
    }
}

/// A game tree derived from a seed: branching, side to move and scores all come from
/// hashing the position, so the same seed always gives the same tree.
#[derive(Debug)]
pub struct RandomTree {
    /// The most moves a position has; a position may have none and end the game.
    pub max_branching: u64,
    /// Whether [`Game::order_moves`] reverses the generated order.
    pub reversed: bool,
    /// [`Side::Min`] is satisfied by any score below this, as a game might be by any line
    /// that wins; `None` never satisfies.
    pub min_satisfied_below: Option<i32>,
    /// [`Side::Max`] is satisfied by any score above this; `None` never satisfies.
    pub max_satisfied_above: Option<i32>,
    /// With `Some(width)`, every ply has only `width` distinct positions, so different move
    /// orders often reach the same position (a transposition); `None` makes every path its
    /// own position.
    pub width: Option<u64>,
    /// Whether some moves are noisy and searched on past the horizon. A noisy move always
    /// lowers the position's heat, so quiescence ends within a few plies.
    pub noisy: bool,
}

impl RandomTree {
    /// The number of moves of `position`, at most [`RandomTree::max_branching`].
    pub fn branching(&self, position: u64) -> u64 {
        mix(position ^ BRANCHING) % (self.max_branching + 1)
    }

    /// Trees with up to `max_branching` moves per position, in generated order, never
    /// satisfied.
    pub const fn new(max_branching: u64) -> Self {
        Self {
            max_branching,
            reversed: false,
            min_satisfied_below: None,
            max_satisfied_above: None,
            width: None,
            noisy: false,
        }
    }
}

/// Salt for the branching factor of a position.
const BRANCHING: u64 = 0x1;
/// Salt for the side to move in a position.
const SIDE: u64 = 0x2;
/// Salt for the score of a position.
const SCORE: u64 = 0x3;
/// Salt for the heat of a position, which every noisy move lowers.
const HEAT: u64 = 0x4;
/// Salt for which moves are noisy.
const NOISE: u64 = 0x5;

impl Game for RandomTree {
    /// A hash of the path from the root; the root is the seed.
    type Position = u64;
    /// The index of the move among its position's moves.
    type Move = u64;
    /// A score in `-100..=100`.
    type Score = i32;

    fn side_to_move(&self, &position: &u64) -> Side {
        if mix(position ^ SIDE) % 2 == 0 {
            Side::Max
        } else {
            Side::Min
        }
    }

    fn moves(&self, &position: &u64, moves: &mut Vec<u64>) {
        moves.extend(0..self.branching(position));
    }

    fn play(&self, &position: &u64, &index: &u64) -> u64 {
        let next = mix(position.rotate_left(17) ^ index.wrapping_add(1));
        match self.width {
            None => next,
            Some(width) => (((position >> 32) + 1) << 32) | (next % width),
        }
    }

    fn evaluate(&self, &position: &u64) -> i32 {
        let offset = i32::try_from(mix(position ^ SCORE) % 201).expect("below 201");

        offset - 100
    }

    fn noisy_moves(&self, &position: &u64, moves: &mut Vec<u64>) {
        if !self.noisy {
            return;
        }

        let heat = |position: u64| mix(position ^ HEAT) % 6;
        let mut all = Vec::new();
        self.moves(&position, &mut all);
        moves.extend(all.into_iter().filter(|&index| {
            let cooler = heat(self.play(&position, &index)) < heat(position);
            cooler && mix(position ^ NOISE ^ index) % 2 == 0
        }));
    }

    fn order_moves(&self, _: &u64, moves: &mut [u64]) {
        if self.reversed {
            moves.reverse();
        }
    }

    fn satisfices(&self, side: Side, score: i32) -> bool {
        match side {
            Side::Min => self.min_satisfied_below.is_some_and(|limit| score < limit),
            Side::Max => self.max_satisfied_above.is_some_and(|limit| score > limit),
        }
    }
}

/// The `SplitMix64` finalizer: a bijective mix of all 64 bits.
fn mix(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);

    z ^ (z >> 31)
}

impl Hashed for RandomTree {
    /// The position itself: it already determines everything below it.
    type Key = u64;

    fn key(&self, &position: &u64) -> u64 {
        mix(position)
    }
}

impl Hashed for Nim {
    /// Twice the stones, plus one when [`Side::Max`] is to move.
    type Key = u64;

    fn key(&self, &(stones, side): &(u32, Side)) -> u64 {
        u64::from(stones) * 2 + u64::from(side == Side::Max)
    }
}

/// A [`RandomTree`] with its scores mapped through `rescore`, a non-decreasing function, so
/// the same trees run with float scores or scores at the ends of an integer range.
pub struct Rescored<S> {
    /// The tree whose moves and sides are used.
    pub tree: RandomTree,
    /// Maps the tree's scores, keeping their order.
    pub rescore: fn(i32) -> S,
}

impl<S: game_tree_search::Score> Game for Rescored<S> {
    /// The tree's positions.
    type Position = u64;
    /// The tree's moves.
    type Move = u64;
    /// The mapped scores.
    type Score = S;

    fn side_to_move(&self, position: &u64) -> Side {
        self.tree.side_to_move(position)
    }

    fn moves(&self, position: &u64, moves: &mut Vec<u64>) {
        self.tree.moves(position, moves);
    }

    fn play(&self, position: &u64, mv: &u64) -> u64 {
        self.tree.play(position, mv)
    }

    fn evaluate(&self, position: &u64) -> S {
        (self.rescore)(self.tree.evaluate(position))
    }
}

/// A [`RandomTree`] with float scores where some positions are chance positions: their
/// moves become outcomes, each weighted 1, 2 or 3 by a hash.
pub struct Dicey {
    /// The tree whose moves, sides and scores are used.
    pub tree: RandomTree,
    /// One in `chance` positions is a chance position; `0` makes none.
    pub chance: u64,
    /// Multiplies every outcome weight, to check that only their ratios matter.
    pub scale: f64,
}

/// Salt for whether a position is a chance position.
const CHANCE: u64 = 0x6;

impl Game for Dicey {
    /// The tree's positions.
    type Position = u64;
    /// The tree's moves, or the outcomes at a chance position.
    type Move = u64;
    /// The tree's scores as floats.
    type Score = game_tree_search::TotalF64;

    fn side_to_move(&self, position: &u64) -> Side {
        self.tree.side_to_move(position)
    }

    fn moves(&self, position: &u64, moves: &mut Vec<u64>) {
        self.tree.moves(position, moves);
    }

    fn play(&self, position: &u64, mv: &u64) -> u64 {
        self.tree.play(position, mv)
    }

    fn evaluate(&self, position: &u64) -> game_tree_search::TotalF64 {
        game_tree_search::TotalF64::new(f64::from(self.tree.evaluate(position))).expect("finite")
    }

    fn noisy_moves(&self, position: &u64, moves: &mut Vec<u64>) {
        self.tree.noisy_moves(position, moves);
    }
}

impl game_tree_search::Stochastic for Dicey {
    fn outcomes(&self, &position: &u64, outcomes: &mut Vec<(u64, f64)>) {
        if self.chance == 0 || mix(position ^ CHANCE) % self.chance != 0 {
            return;
        }

        let mut moves = Vec::new();
        self.tree.moves(&position, &mut moves);
        outcomes.extend(moves.into_iter().map(|mv| {
            let weight = mix(position ^ CHANCE ^ mv) % 3 + 1;
            (
                mv,
                f64::from(u32::try_from(weight).expect("below 4")) * self.scale,
            )
        }));
    }
}

impl game_tree_search::Rewarded for Nim {
    /// A win `1`, a loss `0`.
    fn reward(&self, score: i32) -> f64 {
        f64::from(score + 1) / 2.0
    }
}

impl game_tree_search::Rewarded for RandomTree {
    /// Scores `-100..=100` mapped linearly onto `[0, 1]`.
    fn reward(&self, score: i32) -> f64 {
        f64::from(score + 100) / 200.0
    }
}

/// A [`RandomTree`] in its narrow form, cut off after `levels` plies: deep enough to need a
/// search, small enough to solve exactly.
pub struct Shallow {
    /// The tree, with [`RandomTree::width`] set, so a position knows its ply.
    pub tree: RandomTree,
    /// Positions at this ply end the game.
    pub levels: u64,
}

impl Game for Shallow {
    /// The tree's positions.
    type Position = u64;
    /// The tree's moves.
    type Move = u64;
    /// The tree's scores.
    type Score = i32;

    fn side_to_move(&self, position: &u64) -> Side {
        self.tree.side_to_move(position)
    }

    fn moves(&self, &position: &u64, moves: &mut Vec<u64>) {
        if position >> 32 < self.levels {
            self.tree.moves(&position, moves);
        }
    }

    fn play(&self, position: &u64, mv: &u64) -> u64 {
        self.tree.play(position, mv)
    }

    fn evaluate(&self, position: &u64) -> i32 {
        self.tree.evaluate(position)
    }
}

impl game_tree_search::Rewarded for Shallow {
    /// The tree's rewards.
    fn reward(&self, score: i32) -> f64 {
        self.tree.reward(score)
    }
}

/// A [`RandomTree`] whose moves are `u8`, so its entries fit a
/// [`PackedTable`](game_tree_search::PackedTable): the tree's move indices stay below 256.
#[cfg(feature = "parallel")]
#[derive(Debug)]
pub struct Narrow(pub RandomTree);

#[cfg(feature = "parallel")]
impl Game for Narrow {
    /// The tree's positions.
    type Position = u64;
    /// The tree's move index, narrowed.
    type Move = u8;
    /// The tree's scores.
    type Score = i32;

    fn side_to_move(&self, position: &u64) -> Side {
        self.0.side_to_move(position)
    }

    fn moves(&self, &position: &u64, moves: &mut Vec<u8>) {
        let count = u8::try_from(self.0.branching(position)).expect("few moves");
        moves.extend(0..count);
    }

    // Only quiescence asks, so the scratch vector stays off the main path.
    fn noisy_moves(&self, position: &u64, moves: &mut Vec<u8>) {
        let mut wide = Vec::new();
        self.0.noisy_moves(position, &mut wide);
        moves.extend(
            wide.into_iter()
                .map(|index| u8::try_from(index).expect("few moves")),
        );
    }

    fn play(&self, position: &u64, &index: &u8) -> u64 {
        self.0.play(position, &u64::from(index))
    }

    fn evaluate(&self, position: &u64) -> i32 {
        self.0.evaluate(position)
    }

    fn order_moves(&self, _: &u64, moves: &mut [u8]) {
        if self.0.reversed {
            moves.reverse();
        }
    }

    fn satisfices(&self, side: Side, score: i32) -> bool {
        self.0.satisfices(side, score)
    }
}

#[cfg(feature = "parallel")]
impl Hashed for Narrow {
    /// The tree's key.
    type Key = u64;

    fn key(&self, position: &u64) -> u64 {
        self.0.key(position)
    }
}

/// A tree whose score accumulates along the path, each move counting less than the one
/// before, so a child's own score predicts its subtree and ordering moves by it pays off as
/// in real games. Branching and side to move come from a [`RandomTree`]; the move at ply `p`
/// adds up to `±20 * (16 - p)` to the score, never less than `±20`.
#[cfg(feature = "parallel")]
#[derive(Debug)]
pub struct Smooth {
    /// The tree that gives branching and side to move.
    pub tree: RandomTree,
    /// Whether [`Game::order_moves`] puts the moves with the best child score first.
    pub ordered: bool,
}

#[cfg(feature = "parallel")]
impl Smooth {
    /// The position `index` leads to from `position`.
    fn step(&self, &(id, score, ply): &(u64, i32, i32), index: u8) -> (u64, i32, i32) {
        let next = self.tree.play(&id, &u64::from(index));
        let change = i32::try_from(mix(next ^ SCORE) % 41).expect("below 41") - 20;

        (next, score + change * (16 - ply).max(1), ply + 1)
    }
}

#[cfg(feature = "parallel")]
impl Game for Smooth {
    /// The tree position, the score so far and the ply.
    type Position = (u64, i32, i32);
    /// The index of the move.
    type Move = u8;
    /// The score so far.
    type Score = i32;

    fn side_to_move(&self, (id, _, _): &(u64, i32, i32)) -> Side {
        self.tree.side_to_move(id)
    }

    fn moves(&self, &(id, _, _): &(u64, i32, i32), moves: &mut Vec<u8>) {
        let count = u8::try_from(self.tree.branching(id)).expect("few moves");
        moves.extend(0..count);
    }

    fn play(&self, position: &(u64, i32, i32), &index: &u8) -> (u64, i32, i32) {
        self.step(position, index)
    }

    fn evaluate(&self, &(_, score, _): &(u64, i32, i32)) -> i32 {
        score
    }

    fn order_moves(&self, position: &(u64, i32, i32), moves: &mut [u8]) {
        if !self.ordered {
            return;
        }

        let side = self.side_to_move(position);
        moves.sort_by_key(|&index| {
            let score = self.step(position, index).1;
            if side == Side::Max { -score } else { score }
        });
    }

    fn satisfices(&self, side: Side, score: i32) -> bool {
        self.tree.satisfices(side, score)
    }
}

#[cfg(feature = "parallel")]
impl Hashed for Smooth {
    /// The tree position, which fixes the path and with it the score.
    type Key = u64;

    fn key(&self, &(id, _, _): &(u64, i32, i32)) -> u64 {
        mix(id)
    }
}
