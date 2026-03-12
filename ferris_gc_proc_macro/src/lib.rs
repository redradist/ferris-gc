use proc_macro::TokenStream;
use quote::quote;
use syn::{ReturnType, Data, Fields, NestedMeta, Meta, Lit};

#[proc_macro_derive(Trace, attributes(unsafe_ignore_trace))]
pub fn derive_trace(item: TokenStream) -> TokenStream {
    let derive_input = syn::parse_macro_input!(item as syn::DeriveInput);

    let ident = &derive_input.ident;
    let data_type = &derive_input.data;
    let trace_impl = match data_type {
        Data::Struct(data_struct) => {
            match &data_struct.fields {
                Fields::Named(named_fields) => {
                    let mut fields = Vec::new();
                    for field in &named_fields.named {
                        let attrs = &field.attrs;
                        if !attrs.into_iter().any(|attr| attr.path.get_ident().unwrap() == "unsafe_ignore_trace") {
                            let ident = &field.ident;
                            fields.push(ident);
                        }
                    }
                    quote! {
                        impl Trace for #ident {
                            fn is_root(&self) -> bool {
                                unreachable!("is_root should never be called on user-defined type !!");
                            }

                            fn reset_root(&self) {
                                #(self.#fields.reset_root();)*
                            }

                            fn trace(&self) {
                                #(self.#fields.trace();)*
                            }

                            fn reset(&self) {
                                #(self.#fields.reset();)*
                            }

                            fn is_traceable(&self) -> bool {
                                unreachable!("is_traceable should never be called on user-defined type !!");
                            }
                        }
                    }
                },
                Fields::Unnamed(unnamed_fields) => {
                    let mut fields = Vec::new();
                    let mut idx = 0;
                    for field in &unnamed_fields.unnamed {
                        let attrs = &field.attrs;
                        if !attrs.into_iter().any(|attr| attr.path.get_ident().unwrap() == "unsafe_ignore_trace") {
                            fields.push(idx);
                        }
                        idx += 1;
                    }
                    quote! {
                        impl Trace for #ident {
                            fn is_root(&self) -> bool {
                                unreachable!("is_root should never be called on user-defined type !!");
                            }

                            fn reset_root(&self) {
                                #(self.#fields.reset_root();)*
                            }

                            fn trace(&self) {
                                #(self.#fields.trace();)*
                            }

                            fn reset(&self) {
                                #(self.#fields.reset();)*
                            }

                            fn is_traceable(&self) -> bool {
                                unreachable!("is_traceable should never be called on user-defined type !!");
                            }
                        }
                    }
                },
                Fields::Unit => {
                    panic!("Unit type is not supported !!");
                },
            }
        },
        Data::Enum(_data_enum) => {
            panic!("Enum type is not supported !!");
        },
        Data::Union(_data_union) => {
            panic!("Union type is not supported !!");
        },
    };

    let print_tokens = Into::<TokenStream>::into(trace_impl.clone());
    println!("Result Trace Impl is {}", print_tokens.to_string());
    trace_impl.into()
}

#[proc_macro_derive(Finalize)]
pub fn derive_finalize(item: TokenStream) -> TokenStream {
    let derive_input = syn::parse_macro_input!(item as syn::DeriveInput);

    let ident = &derive_input.ident;
    let finalizer_impl = quote! {
        impl Finalize for #ident {
            fn finalize(&self) {
            }
        }
    };

    let print_tokens = Into::<TokenStream>::into(finalizer_impl.clone());
    println!("Result Finalizer Impl is {}", print_tokens.to_string());
    finalizer_impl.into()
}

/// Attribute macro for the application entry point.
///
/// # Usage
/// ```ignore
/// #[ferris_gc_main]                          // basic strategy (default)
/// #[ferris_gc_main(strategy = "threshold")]  // threshold-based generational GC
/// #[ferris_gc_main(strategy = "adaptive")]   // adaptive auto-tuning generational GC
/// fn main() { }
/// ```
#[proc_macro_attribute]
pub fn ferris_gc_main(attrs: TokenStream, item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as syn::ItemFn);
    let attr_args = syn::parse_macro_input!(attrs as syn::AttributeArgs);

    // Parse strategy attribute
    let mut strategy = String::from("basic");
    for arg in &attr_args {
        if let NestedMeta::Meta(Meta::NameValue(nv)) = arg {
            if nv.path.is_ident("strategy") {
                if let Lit::Str(s) = &nv.lit {
                    strategy = s.value();
                } else {
                    panic!("strategy value must be a string literal");
                }
            } else {
                let name = nv.path.get_ident().map(|i| i.to_string()).unwrap_or_default();
                panic!("Unknown attribute '{}'. Supported: strategy", name);
            }
        } else {
            panic!("Expected strategy = \"...\". Supported strategies: basic, threshold, adaptive");
        }
    }

    let strategy_setup = match strategy.as_str() {
        "basic" => quote! {},
        "threshold" => quote! {
            ferris_gc::BASIC_STRATEGY_DISABLED.store(true, std::sync::atomic::Ordering::Release);
            ferris_gc::LOCAL_GC_STRATEGY.with(|s| {
                let (start, stop) = ferris_gc::threshold_local_start(ferris_gc::ThresholdConfig::default());
                s.borrow().change_strategy(start, stop);
            });
            {
                let (start, stop) = ferris_gc::threshold_global_start(ferris_gc::ThresholdConfig::default());
                ferris_gc::sync::GLOBAL_GC_STRATEGY.change_strategy(start, stop);
            }
        },
        "adaptive" => quote! {
            ferris_gc::BASIC_STRATEGY_DISABLED.store(true, std::sync::atomic::Ordering::Release);
            ferris_gc::LOCAL_GC_STRATEGY.with(|s| {
                let (start, stop) = ferris_gc::adaptive_local_start(ferris_gc::AdaptiveConfig::default());
                s.borrow().change_strategy(start, stop);
            });
            {
                let (start, stop) = ferris_gc::adaptive_global_start(ferris_gc::AdaptiveConfig::default());
                ferris_gc::sync::GLOBAL_GC_STRATEGY.change_strategy(start, stop);
            }
        },
        other => panic!(
            "Unknown GC strategy '{}'. Supported: basic, threshold, adaptive",
            other
        ),
    };

    let _sig = &input.sig;
    let vis = input.vis;
    let name = &input.sig.ident;
    if name != "main" {
        panic!("#[ferris_gc_main] is applied only for main function")
    }
    let mut args = Vec::new();
    for arg in &input.sig.inputs {
        args.push(arg);
    }
    let ret = match &input.sig.output {
        ReturnType::Default => {
            quote! {
            }
        },
        ReturnType::Type(_arrow, box_type) => {
            quote! {
                -> #box_type
            }
        }
    };
    let body = &input.block;
    let attrs = &input.attrs;

    let res_fun = quote! {
        #(#attrs)*
        #vis fn #name (#(#args),*) #ret {
            let _cleanup = ferris_gc::ApplicationCleanup;
            #strategy_setup
            {
                #body
            }
        }
    };

    let print_tokens = Into::<TokenStream>::into(res_fun.clone());
    println!("Result Function is {}", print_tokens.to_string());
    res_fun.into()
}
