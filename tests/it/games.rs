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
        moves.extend(0..mix(position ^ BRANCHING) % (self.max_branching + 1));
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
