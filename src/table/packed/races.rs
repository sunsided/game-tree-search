//! Loom models of racing stores and probes on a one-slot [`PackedTable`]: every
//! interleaving of two writers and a reader, up to the preemption bound.
//!
//! # Memory orderings
//!
//! Every load and store of a slot is `Relaxed`, and these models are why that is enough.
//! An entry is plain bits in two words, with no pointer and no other memory to publish, so
//! there is nothing for `Acquire`/`Release` to order: a probe only needs each word to be
//! one that some store wrote, which every atomic access guarantees. What can go wrong is a
//! probe, or a later store, seeing the check word of one store next to the data word of
//! another, in any order. The check word is the key XOR-ed with the data, so such a pair
//! fails the check, and the probe misses ([`a_probe_never_returns_a_torn_entry`],
//! [`racing_stores_of_one_key_give_one_of_them`]). Stores always replace, so after racing
//! stores the slot holds one whole entry, or a torn pair that every key misses
//! ([`racing_stores_leave_one_entry_or_none`]).
//!
//! A torn pair passes the check only if the two data words differ by exactly the XOR of the
//! two keys; with well-mixed 64-bit keys that is as unlikely as any key collision. The
//! entries here are chosen so it cannot happen.

use loom::model::Builder;
use loom::sync::Arc;
use loom::thread;

use super::PackedTable;
use crate::Bound;
use crate::table::{Entry, Table};

/// Two keys that share the one slot of a one-slot table.
const KEYS: [u64; 2] = [1, 3];

/// The entry each writer stores. They differ in score and depth, so their data words never
/// differ by exactly `KEYS[0] ^ KEYS[1]`.
fn entries() -> [Entry<u8, i16>; 2] {
    let entry = |score, depth| Entry {
        depth,
        score,
        bound: Bound::Exact,
        best: Some(1),
        proven: false,
        reach: 0,
    };

    [entry(-7, 4), entry(9, 11)]
}

/// Runs `model` over every interleaving, with at most three preemptions unless
/// `LOOM_MAX_PREEMPTIONS` asks for another bound.
fn check(model: impl Fn() + Sync + Send + 'static) {
    let mut builder = Builder::new();
    if builder.preemption_bound.is_none() {
        builder.preemption_bound = Some(3);
    }

    builder.check(model);
}

/// A thread storing `entry` under `key` in `table`.
fn writer(
    table: &Arc<PackedTable<u64, u8, i16>>,
    key: u64,
    entry: Entry<u8, i16>,
) -> thread::JoinHandle<()> {
    let table = Arc::clone(table);
    thread::spawn(move || (&*table).store(key, entry))
}

/// Two writers store different keys in one slot while a reader probes both keys: each
/// probe misses or returns its own key's entry, never the other key's or a torn mix.
#[test]
fn a_probe_never_returns_a_torn_entry() {
    check(|| {
        let table = Arc::new(PackedTable::new(1));
        let [first, second] = entries();
        let writers = [
            writer(&table, KEYS[0], first.clone()),
            writer(&table, KEYS[1], second.clone()),
        ];

        let mut handle = &*table;
        let probes = [handle.probe(&KEYS[0]), handle.probe(&KEYS[1])];
        for writer in writers {
            writer.join().expect("a writer never panics");
        }

        assert!(
            probes[0].as_ref().is_none_or(|entry| *entry == first),
            "{probes:?}"
        );
        assert!(
            probes[1].as_ref().is_none_or(|entry| *entry == second),
            "{probes:?}"
        );
    });
}

/// Two writers store different entries under one key while a reader probes it: the probe
/// misses or returns exactly one of them.
#[test]
fn racing_stores_of_one_key_give_one_of_them() {
    check(|| {
        let table = Arc::new(PackedTable::new(1));
        let [first, second] = entries();
        let writers = [
            writer(&table, KEYS[0], first.clone()),
            writer(&table, KEYS[0], second.clone()),
        ];

        let probe = (&*table).probe(&KEYS[0]);
        for writer in writers {
            writer.join().expect("a writer never panics");
        }

        assert!(
            probe.is_none() || probe == Some(first.clone()) || probe == Some(second.clone()),
            "{probe:?}"
        );
    });
}

/// Once two racing stores of different keys are done, the slot holds one of them whole,
/// as always-replace promises, or a torn pair that both keys miss: never both entries, and
/// never one key's entry under the other key.
#[test]
fn racing_stores_leave_one_entry_or_none() {
    check(|| {
        let table = Arc::new(PackedTable::new(1));
        let [first, second] = entries();
        let writers = [
            writer(&table, KEYS[0], first.clone()),
            writer(&table, KEYS[1], second.clone()),
        ];
        for writer in writers {
            writer.join().expect("a writer never panics");
        }

        let mut handle = &*table;
        match (handle.probe(&KEYS[0]), handle.probe(&KEYS[1])) {
            (Some(entry), None) => assert_eq!(entry, first),
            (None, Some(entry)) => assert_eq!(entry, second),
            (None, None) => {}
            both => panic!("one slot holds two entries: {both:?}"),
        }
    });
}
