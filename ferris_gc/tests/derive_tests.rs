use ferris_gc::{Finalize, Gc, Trace};
use ferris_gc_proc_macro::{Finalize, Trace};

// --- Enum support ---

#[derive(Trace, Finalize)]
enum SimpleEnum {
    Empty,
    Leaf(i32),
    Branch {
        left: Gc<SimpleEnum>,
        right: Gc<SimpleEnum>,
    },
}

#[test]
fn derive_trace_on_enum_compiles() {
    // Just verify it compiles and the trait methods are callable
    let node = SimpleEnum::Empty;
    node.reset_root();
    node.trace();
    node.reset();
}

#[derive(Trace, Finalize)]
enum TupleEnum {
    One(i32),
    Two(i32, i32),
}

#[test]
fn derive_trace_on_tuple_enum_compiles() {
    let node = TupleEnum::Two(1, 2);
    node.reset_root();
    node.trace();
    node.reset();
}

// --- Generics support ---

#[derive(Trace, Finalize)]
struct Container<T> {
    item: T,
}

#[test]
fn derive_trace_on_generic_struct_compiles() {
    let c = Container { item: 42i32 };
    c.reset_root();
    c.trace();
    c.reset();
}

#[derive(Trace, Finalize)]
struct Pair<T, U> {
    first: T,
    second: U,
}

#[test]
fn derive_trace_on_multi_generic_compiles() {
    let p = Pair {
        first: 1i32,
        second: 2i32,
    };
    p.reset_root();
    p.trace();
    p.reset();
}

// --- Unit struct support ---

#[derive(Trace, Finalize)]
struct UnitStruct;

#[test]
fn derive_trace_on_unit_struct_compiles() {
    let u = UnitStruct;
    u.reset_root();
    u.trace();
    u.reset();
}

// --- Tuple struct with generics ---

#[derive(Trace, Finalize)]
struct Wrapper<T>(T);

#[test]
fn derive_trace_on_generic_tuple_struct_compiles() {
    let w = Wrapper(42i32);
    w.reset_root();
    w.trace();
    w.reset();
}

// --- Enum with unsafe_ignore_trace ---

#[derive(Trace, Finalize)]
enum MixedEnum {
    WithIgnored {
        tracked: i32,
        #[unsafe_ignore_trace]
        _ignored: String,
    },
    Plain(i32),
}

#[test]
fn derive_trace_on_enum_with_ignore_compiles() {
    let m = MixedEnum::WithIgnored {
        tracked: 1,
        _ignored: String::from("skip"),
    };
    m.reset_root();
    m.trace();
    m.reset();
}

// --- Generic enum ---

#[derive(Trace, Finalize)]
enum GenericEnum<T> {
    None,
    Some(T),
}

#[test]
fn derive_trace_on_generic_enum_compiles() {
    let g: GenericEnum<i32> = GenericEnum::Some(42);
    g.reset_root();
    g.trace();
    g.reset();
}
