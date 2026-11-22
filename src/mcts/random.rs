//! The random numbers Monte Carlo tree search draws: the [`Random`] trait and the seeded
//! [`SplitMix64`] generator.

/// A source of random bits for rollouts.
///
/// Implemented by [`SplitMix64`], the default. A game that needs its own generator, for
/// example to share one sequence with the rest of a program, implements it in one line.
pub trait Random {
    /// The next 64 random bits.
    fn next_u64(&mut self) -> u64;

    /// A uniform index below `len`, which must not be `0`.
    ///
    /// Uses the remainder of a 64-bit draw, whose bias is below `len / 2^64`, too small to
    /// matter for move choice.
    fn below(&mut self, len: usize) -> usize {
        let len = u64::try_from(len).expect("a slice length fits in u64");
        usize::try_from(self.next_u64() % len).expect("an index below a length fits in usize")
    }
}

/// The `SplitMix64` generator (Steele, Lea and Flood 2014): fast, with a full `2^64` period.
///
/// An own generator rather than a dependency, so a seed reproduces a search across releases.
/// Not cryptographically secure, which rollouts do not need.
///
/// # Examples
///
/// ```
/// use game_tree_search::{Random, SplitMix64};
///
/// let (mut a, mut b) = (SplitMix64::new(7), SplitMix64::new(7));
/// assert_eq!(a.next_u64(), b.next_u64());
/// assert!(a.below(6) < 6);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitMix64 {
    /// The generator state, advanced by a fixed odd increment per draw.
    state: u64,
}

impl SplitMix64 {
    /// A generator from `seed`; every seed, `0` included, is valid.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl Random for SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);

        z ^ (z >> 31)
    }
}

/// Unit tests for the generator.
#[cfg(test)]
mod tests {
    use super::{Random, SplitMix64};

    /// The first draw from seed 0 matches the published `SplitMix64` reference value.
    #[test]
    fn matches_the_reference_sequence() {
        assert_eq!(SplitMix64::new(0).next_u64(), 0xe220_a839_7b1d_cdaf);
    }

    /// Indices stay below the length.
    #[test]
    fn indices_stay_in_range() {
        let mut rng = SplitMix64::new(3);
        assert!((0..1000).all(|_| rng.below(7) < 7));
        assert_eq!(rng.below(1), 0);
    }
}
