use crate::gc::{Finalize, Trace};
use std::collections::{BinaryHeap, BTreeSet, HashMap, BTreeMap, HashSet, LinkedList, VecDeque};

macro_rules! trivial_types {
    ($($prm:ident),*) => {
        $(
            impl Trace for $prm {
                fn is_root(&self) -> bool {
                    unreachable!("is_root should never be called on primitive type !!");
                }
                fn reset_root(&self) {
                }
                fn trace(&self) {
                }
                fn reset(&self) {
                }
                fn is_traceable(&self) -> bool {
                    unreachable!("is_traceable should never be called on primitive type !!");
                }
            }

            impl Finalize for $prm {
                fn finalize(&self) {
                }
            }
        )*
    };
    ($(&$prm:ident),*) => {
        $(
            impl Trace for &$prm {
                fn is_root(&self) -> bool {
                    unreachable!("is_root should never be called on primitive type !!");
                }
                fn reset_root(&self) {
                }
                fn trace(&self) {
                }
                fn reset(&self) {
                }
                fn is_traceable(&self) -> bool {
                    unreachable!("is_traceable should never be called on primitive type !!");
                }
            }

            impl Finalize for &$prm {
                fn finalize(&self) {
                }
            }
        )*
    };
    ($($std:ident<T>),*) => {
        $(
            impl<T> Trace for $std<T> where T: 'static + Sized + Trace {
                fn is_root(&self) -> bool {
                    unreachable!("is_root should never be called on primitive type !!");
                }
                fn reset_root(&self) {
                }
                fn trace(&self) {
                }
                fn reset(&self) {
                }
                fn is_traceable(&self) -> bool {
                    unreachable!("is_traceable should never be called on primitive type !!");
                }
            }

            impl<T> Finalize for $std<T> where T: 'static + Sized + Trace {
                fn finalize(&self) {
                }
            }
        )*
    };
}

trivial_types!(
    u8, i8, u16, i16, u32, i32, u64, i64, u128, i128,
    usize, isize,
    f32, f64,
    bool,
    String
);

trivial_types!(
    &str, &String
);

trivial_types!(
    Box<T>
);

macro_rules! collection_types {
    ($($std:ident<T>),*) => {
        $(
            impl<T> Trace for $std<T> where T: 'static + Sized + Trace {
                fn is_root(&self) -> bool {
                    unreachable!("is_root should never be called on primitive type !!");
                }
                fn reset_root(&self) {
                    for child in self {
                        child.reset_root();
                    }
                }
                fn trace(&self) {
                    for child in self {
                        child.trace();
                    }
                }
                fn reset(&self) {
                    for child in self {
                        child.reset();
                    }
                }
                fn is_traceable(&self) -> bool {
                    unreachable!("is_traceable should never be called on primitive type !!");
                }
            }

            impl<T> Finalize for $std<T> where T: 'static + Sized + Trace {
                fn finalize(&self) {
                    for child in self {
                        child.finalize();
                    }
                }
            }
        )*
    };
    ($($std:ident<K, T>),*) => {
        $(
            impl<K, T> Trace for $std<K, T> where T: 'static + Sized + Trace {
                fn is_root(&self) -> bool {
                    unreachable!("is_root should never be called on primitive type !!");
                }
                fn reset_root(&self) {
                    for (key, child) in self {
                        child.reset_root();
                    }
                }
                fn trace(&self) {
                    for (key, child) in self {
                        child.trace();
                    }
                }
                fn reset(&self) {
                    for (key, child) in self {
                        child.reset();
                    }
                }
                fn is_traceable(&self) -> bool {
                    unreachable!("is_traceable should never be called on primitive type !!");
                }
            }

            impl<K, T> Finalize for $std<K, T> where T: 'static + Sized + Trace {
                fn finalize(&self) {
                    for (key, child) in self {
                        child.finalize();
                    }
                }
            }
        )*
    };
}

collection_types!(
    Vec<T>, VecDeque<T>, LinkedList<T>,
    HashSet<T>, BTreeSet<T>, BinaryHeap<T>
);

collection_types!(
    HashMap<K, T>, BTreeMap<K, T>
);

impl<T> Trace for Option<T> where T: 'static + Sized + Trace {
    fn is_root(&self) -> bool {
        unreachable!("is_root should never be called on primitive type !!");
    }
    fn reset_root(&self) {
        if let Some(obj) = self.as_ref() {
            obj.reset_root();
        }
    }
    fn trace(&self) {
        if let Some(obj) = self.as_ref() {
            obj.trace();
        }
    }
    fn reset(&self) {
        if let Some(obj) = self.as_ref() {
            obj.reset();
        }
    }
    fn is_traceable(&self) -> bool {
        unreachable!("is_traceable should never be called on primitive type !!");
    }
}

impl<T> Finalize for Option<T> where T: 'static + Sized + Trace {
    fn finalize(&self) {
        if let Some(obj) = self.as_ref() {
            obj.finalize();
        }
    }
}
