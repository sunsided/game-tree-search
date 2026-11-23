//! [`PackedTable`]: a lock-free transposition table for several threads (feature
//! `parallel`), its entries packed into atomic words.

use alloc::vec::Vec;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicU64, Ordering};

use super::{Entry, Table, TableKey};
use crate::{Bound, TotalF32};

/// A move or score that fits into a few bits, so a [`PackedTable`] can store it in an
/// atomic word.
///
/// Implemented for the integers of up to 32 bits and [`TotalF32`]. `unpack(pack(x))` must
/// give back `x`.
pub trait Packed: Sized {
    /// How many low bits [`Packed::pack`] uses; the rest are zero.
    const BITS: u32;

    /// The value in the low [`Packed::BITS`] bits.
    fn pack(&self) -> u64;

    /// The value [`Packed::pack`] turned into `bits`.
    fn unpack(bits: u64) -> Self;
}

/// Implements [`Packed`] for integers through their unsigned bit pattern.
macro_rules! packed_int {
    ($($int:ty => $bits:ty),* $(,)?) => {
        $(
            impl Packed for $int {
                const BITS: u32 = <$int>::BITS;

                fn pack(&self) -> u64 {
                    u64::from(<$bits>::from_ne_bytes(self.to_ne_bytes()))
                }

                fn unpack(bits: u64) -> Self {
                    let bits = <$bits>::try_from(bits & low_bits(Self::BITS))
                        .expect("the masked bits fit the type");
                    <$int>::from_ne_bytes(bits.to_ne_bytes())
                }
            }
        )*
    };
}

packed_int!(i8 => u8, i16 => u16, i32 => u32, u8 => u8, u16 => u16, u32 => u32);

impl Packed for TotalF32 {
    const BITS: u32 = 32;

    fn pack(&self) -> u64 {
        u64::from(self.get().to_bits())
    }

    fn unpack(bits: u64) -> Self {
        let bits = u32::try_from(bits & low_bits(32)).expect("the masked bits fit in 32");
        // Only packed scores are unpacked, and those were never NaN.
        Self::new(f32::from_bits(bits)).expect("a packed score is not NaN")
    }
}

/// A mask of the low `bits` bits, `bits` at most 63.
const fn low_bits(bits: u32) -> u64 {
    (1 << bits) - 1
}

/// The bits an entry's fields take besides its move and score.
const META_BITS: u32 = 16;
/// Where the bound starts: two bits.
const BOUND_SHIFT: u32 = 0;
/// The bit that says the entry is proven.
const PROVEN_SHIFT: u32 = 2;
/// Where the depth starts: six bits.
const DEPTH_SHIFT: u32 = 3;
/// Where the reach starts: six bits.
const REACH_SHIFT: u32 = 9;
/// The bit that says the entry has a best move.
const HAS_MOVE_SHIFT: u32 = 15;
/// The width of the depth and reach fields.
const DEPTH_BITS: u32 = 6;
/// The deepest entry stored; deeper ones are dropped.
pub(crate) const MAX_DEPTH: u32 = 62;
/// The depth field of an empty slot, one above [`MAX_DEPTH`].
const EMPTY_DEPTH: u64 = 63;
/// The data word of an empty slot.
const EMPTY: u64 = EMPTY_DEPTH << DEPTH_SHIFT;

/// One slot: the data word and the key XOR-ed into it, written separately.
#[derive(Debug)]
struct Slot {
    /// The slot bits of the key, XOR-ed with [`Slot::data`].
    check: AtomicU64,
    /// The packed entry.
    data: AtomicU64,
}

impl Slot {
    /// An empty slot.
    const fn empty() -> Self {
        Self {
            check: AtomicU64::new(EMPTY),
            data: AtomicU64::new(EMPTY),
        }
    }
}

/// The types a [`PackedTable`] holds, behind a function pointer so the table is `Send` and
/// `Sync` whatever they are: it stores only their packed bits.
type Holds<K, M, S> = fn() -> (K, M, S);

/// A transposition table several threads search with at once, without locks: each entry is
/// packed into one 64-bit word, stored next to that word XOR-ed with the key (Hyatt and
/// Mann's lockless hashing).
///
/// A probe reads both words without a lock; if another thread wrote the slot in between,
/// the XOR no longer gives the key, and the probe misses instead of returning a torn
/// entry. Like [`SharedTable`](crate::SharedTable), it is used through a shared reference:
/// `&PackedTable` implements [`Table`], so each thread builds its own
/// [`Searcher::with_table`](crate::Searcher::with_table)`(&table)`, and
/// [`deepen_shared`](crate::deepen_shared()) takes it for Lazy SMP.
///
/// The packing has limits that [`SharedTable`](crate::SharedTable) does not:
///
/// - The move and the score must be [`Packed`] and fit 48 bits together, such as an `i32`
///   score with a `u16` move; [`PackedTable::new`] fails to compile otherwise.
/// - Entries more than 62 plies deep are not stored.
/// - A slot checks the 64 bits of [`TableKey::slot`], not the whole key: for a `(u64, u64)`
///   key only the first half, which also picks the slot. With well-mixed keys, a wrong hit
///   stays as unlikely as for any 64-bit hash.
///
/// A store always replaces the slot's entry, as in [`SlotTable`](crate::SlotTable).
///
/// Available with the `parallel` feature, which needs `std`.
///
/// # Examples
///
/// ```
/// # use game_tree_search::{Game, Hashed, Side};
/// # struct Nim;
/// # impl Game for Nim {
/// #     type Position = (u32, Side);
/// #     type Move = u32;
/// #     type Score = i8;
/// #     fn side_to_move(&self, &(_, side): &Self::Position) -> Side { side }
/// #     fn moves(&self, &(stones, _): &Self::Position, moves: &mut Vec<u32>) {
/// #         moves.extend((1..=2).filter(|&take| take <= stones));
/// #     }
/// #     fn play(&self, &(stones, side): &Self::Position, take: &u32) -> Self::Position {
/// #         (stones - take, if side == Side::Max { Side::Min } else { Side::Max })
/// #     }
/// #     fn evaluate(&self, &(stones, side): &Self::Position) -> i8 {
/// #         match (stones, side) { (0, Side::Max) => -1, (0, Side::Min) => 1, _ => 0 }
/// #     }
/// # }
/// # impl Hashed for Nim {
/// #     type Key = u64;
/// #     fn key(&self, &(stones, side): &Self::Position) -> u64 {
/// #         u64::from(stones) * 2 + u64::from(side == Side::Max)
/// #     }
/// # }
/// use game_tree_search::{Algorithm, PackedTable, Stats, deepen_shared};
///
/// let table = PackedTable::new(1 << 12);
/// let unlimited = |_: &Stats| false;
/// let result = deepen_shared(&Nim, &(20, Side::Max), 21, (4, Algorithm::AlphaBeta), &table, unlimited);
/// assert_eq!(result.best(), Some(&2));
/// ```
#[derive(Debug)]
pub struct PackedTable<K, M, S> {
    /// The slots; their number is a power of two.
    slots: Vec<Slot>,
    /// The key, move and score types the words hold.
    types: PhantomData<Holds<K, M, S>>,
}

impl<K: TableKey, M: Packed, S: Packed> PackedTable<K, M, S> {
    /// A table with `capacity` slots, rounded up to a power of two (at least one).
    ///
    /// Fails to compile if `M` and `S` take more than 48 bits together.
    ///
    /// # Panics
    ///
    /// Panics if the rounded capacity does not fit in `usize`.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        const {
            assert!(
                M::BITS + S::BITS <= 64 - META_BITS,
                "a packed move and score must fit 48 bits together"
            );
        }

        let slots = capacity
            .max(1)
            .checked_next_power_of_two()
            .expect("the capacity fits in usize when rounded up");

        Self {
            slots: (0..slots).map(|_| Slot::empty()).collect(),
            types: PhantomData,
        }
    }

    /// The number of slots.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Empties every slot, for example before searching a different game. Run it while no
    /// search uses the table.
    pub fn clear(&self) {
        for slot in &self.slots {
            slot.check.store(EMPTY, Ordering::Relaxed);
            slot.data.store(EMPTY, Ordering::Relaxed);
        }
    }

    /// The slot of `key`: its low bits, as many as the slot count needs.
    fn slot(&self, key: &K) -> &Slot {
        let mask = u64::try_from(self.slots.len() - 1).expect("a slot count fits in u64");
        let index = usize::try_from(key.slot() & mask).expect("a masked slot is below the count");

        &self.slots[index]
    }
}

impl<K: TableKey, M: Packed, S: Packed> Table<K, M, S> for &PackedTable<K, M, S> {
    fn probe(&mut self, key: &K) -> Option<Entry<M, S>> {
        let slot = self.slot(key);
        let check = slot.check.load(Ordering::Relaxed);
        let data = slot.data.load(Ordering::Relaxed);

        (check ^ data == key.slot()).then(|| unpack(data)).flatten()
    }

    fn store(&mut self, key: K, entry: Entry<M, S>) {
        let Some(data) = pack(&entry) else {
            return;
        };

        let slot = self.slot(&key);
        slot.check.store(key.slot() ^ data, Ordering::Relaxed);
        slot.data.store(data, Ordering::Relaxed);
    }
}

/// `entry` as one word, or `None` if it is too deep to store.
pub(crate) fn pack<M: Packed, S: Packed>(entry: &Entry<M, S>) -> Option<u64> {
    // The reach only matters for proven entries.
    let reach = if entry.proven { entry.reach } else { 0 };
    if entry.depth > MAX_DEPTH || reach > MAX_DEPTH {
        return None;
    }

    let bound: u64 = match entry.bound {
        Bound::Exact => 0,
        Bound::Lower => 1,
        Bound::Upper => 2,
        Bound::Approximate => 3,
    };
    let best = entry.best.as_ref().map_or(0, |mv| (mv.pack() << 1) | 1);
    let meta = (bound << BOUND_SHIFT)
        | (u64::from(entry.proven) << PROVEN_SHIFT)
        | (u64::from(entry.depth) << DEPTH_SHIFT)
        | (u64::from(reach) << REACH_SHIFT);

    Some(meta | (best << HAS_MOVE_SHIFT) | (entry.score.pack() << (META_BITS + M::BITS)))
}

/// The entry in `data`, or `None` for an empty slot.
pub(crate) fn unpack<M: Packed, S: Packed>(data: u64) -> Option<Entry<M, S>> {
    let field = |shift: u32, bits: u32| (data >> shift) & low_bits(bits);
    let depth = field(DEPTH_SHIFT, DEPTH_BITS);
    if depth == EMPTY_DEPTH {
        return None;
    }

    let bound = match field(BOUND_SHIFT, 2) {
        0 => Bound::Exact,
        1 => Bound::Lower,
        2 => Bound::Upper,
        _ => Bound::Approximate,
    };
    let best = (field(HAS_MOVE_SHIFT, 1) == 1).then(|| M::unpack(field(META_BITS, M::BITS)));
    let small = |bits: u64| u32::try_from(bits).expect("six bits fit in u32");

    Some(Entry {
        depth: small(depth),
        score: S::unpack(data >> (META_BITS + M::BITS)),
        bound,
        best,
        proven: field(PROVEN_SHIFT, 1) == 1,
        reach: small(field(REACH_SHIFT, DEPTH_BITS)),
    })
}

/// Unit tests for packing and the lockless check.
#[cfg(test)]
mod tests {
    use super::{EMPTY, MAX_DEPTH, PackedTable, Slot, pack, unpack};
    use crate::table::{Entry, Table};
    use crate::{Bound, TotalF32};

    /// An entry with the given fields.
    fn entry<M, S>(score: S, best: Option<M>, bound: Bound, depth: u32) -> Entry<M, S> {
        Entry {
            depth,
            score,
            bound,
            best,
            proven: false,
            reach: 0,
        }
    }

    /// Every bound, both ends of the integer ranges, a missing move and the depth and reach
    /// limits come back unchanged.
    #[test]
    fn entries_round_trip() {
        for bound in [Bound::Exact, Bound::Lower, Bound::Upper, Bound::Approximate] {
            for (score, best) in [(i32::MIN, Some(u16::MAX)), (i32::MAX, None), (-1, Some(0))] {
                for depth in [0, MAX_DEPTH] {
                    let mut original = entry(score, best, bound, depth);
                    original.proven = depth > 0;
                    original.reach = if original.proven { MAX_DEPTH } else { 0 };
                    let packed = pack(&original).expect("within the limits");
                    assert_eq!(unpack::<u16, i32>(packed), Some(original));
                }
            }
        }

        let float = entry(
            TotalF32::new(-0.5).expect("not NaN"),
            Some(7u8),
            Bound::Lower,
            3,
        );
        assert_eq!(unpack(pack(&float).expect("shallow")), Some(float));
    }

    proptest::proptest! {
        /// Any entry within the depth limit packs and unpacks to itself, the reach of an
        /// unproven entry aside, which is not stored.
        #[test]
        fn any_entry_round_trips(
            score in proptest::prelude::any::<i32>(),
            best in proptest::option::of(proptest::prelude::any::<u16>()),
            bound in 0..4usize,
            depth in 0..=MAX_DEPTH,
            proven in proptest::prelude::any::<bool>(),
            reach in 0..=MAX_DEPTH,
        ) {
            let bound = [Bound::Exact, Bound::Lower, Bound::Upper, Bound::Approximate][bound];
            let mut original = entry(score, best, bound, depth);
            original.proven = proven;
            original.reach = if proven { reach } else { 0 };
            let packed = pack(&original).expect("within the limits");
            proptest::prop_assert_eq!(unpack::<u16, i32>(packed), Some(original));
        }
    }

    /// Entries deeper than the depth field allows are dropped, and an unproven entry's
    /// reach is not stored.
    #[test]
    fn deep_entries_are_dropped() {
        let deep = entry(0i8, None::<u8>, Bound::Exact, MAX_DEPTH + 1);
        assert_eq!(pack(&deep), None);

        let mut unproven = entry(0i8, None::<u8>, Bound::Exact, 4);
        unproven.reach = 99;
        let back = unpack::<u8, i8>(pack(&unproven).expect("reach ignored")).expect("stored");
        assert_eq!(back.reach, 0);

        unproven.proven = true;
        assert_eq!(pack(&unproven), None);
    }

    /// A stored entry comes back for its key only, and an empty slot answers nothing, not
    /// even for key `0`.
    #[test]
    fn keys_check_their_slot() {
        let table = PackedTable::<u64, u8, i16>::new(4);
        let mut handle = &table;
        assert_eq!(handle.probe(&0), None);

        let stored = entry(-3i16, Some(2u8), Bound::Upper, 5);
        handle.store(5, stored.clone());
        assert_eq!(handle.probe(&5), Some(stored.clone()));
        assert_eq!(handle.probe(&1), None, "same slot, other key");

        // An entry too deep to pack is dropped, leaving the slot as it was.
        handle.store(5, entry(0i16, None, Bound::Exact, MAX_DEPTH + 1));
        assert_eq!(handle.probe(&5), Some(stored));

        table.clear();
        assert_eq!(handle.probe(&5), None);
        assert_eq!(table.capacity(), 4);
    }

    /// A slot whose words come from two different stores, as a write racing a probe leaves
    /// it, is detected and ignored.
    #[test]
    fn torn_entries_are_ignored() {
        let table = PackedTable::<u64, u8, i16>::new(1);
        let mut handle = &table;
        let (first, second) = (
            entry(1i16, None, Bound::Exact, 1),
            entry(2i16, None, Bound::Exact, 2),
        );
        handle.store(7, first);
        let first_check = table.slots[0]
            .check
            .load(core::sync::atomic::Ordering::Relaxed);
        handle.store(7, second);
        // The check word of the first store next to the data word of the second.
        table.slots[0]
            .check
            .store(first_check, core::sync::atomic::Ordering::Relaxed);
        assert_eq!(handle.probe(&7), None);

        let empty = Slot::empty();
        assert_eq!(empty.data.into_inner(), EMPTY);
    }
}
