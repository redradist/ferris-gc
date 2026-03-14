use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, Fields, Lit, Meta, NestedMeta, ReturnType};

fn has_ignore_trace(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path
            .get_ident()
            .map(|id| id == "unsafe_ignore_trace")
            .unwrap_or(false)
    })
}

/// Generate method bodies (reset_root, trace, reset, trace_children) for struct fields.
fn struct_bodies(
    fields: &Fields,
) -> (
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
    proc_macro2::TokenStream,
) {
    match fields {
        Fields::Named(named) => {
            let traced: Vec<_> = named
                .named
                .iter()
                .filter(|f| !has_ignore_trace(&f.attrs))
                .filter_map(|f| f.ident.as_ref())
                .collect();
            (
                quote! { #(self.#traced.reset_root();)* },
                quote! { #(self.#traced.trace();)* },
                quote! { #(self.#traced.reset();)* },
                quote! { #(self.#traced.trace_children(children);)* },
            )
        }
        Fields::Unnamed(unnamed) => {
            let indices: Vec<syn::Index> = unnamed
                .unnamed
                .iter()
                .enumerate()
                .filter(|(_, f)| !has_ignore_trace(&f.attrs))
                .map(|(i, _)| syn::Index::from(i))
                .collect();
            (
                quote! { #(self.#indices.reset_root();)* },
                quote! { #(self.#indices.trace();)* },
                quote! { #(self.#indices.reset();)* },
                quote! { #(self.#indices.trace_children(children);)* },
            )
        }
        Fields::Unit => (quote! {}, quote! {}, quote! {}, quote! {}),
    }
}

/// Generate a `match self { ... }` expression for an enum, calling `method` on traced fields.
fn enum_match(
    ident: &syn::Ident,
    data_enum: &syn::DataEnum,
    method: &str,
) -> proc_macro2::TokenStream {
    let method_ident = syn::Ident::new(method, proc_macro2::Span::call_site());

    let arms: Vec<_> = data_enum
        .variants
        .iter()
        .map(|variant| {
            let var_ident = &variant.ident;
            match &variant.fields {
                Fields::Named(named) => {
                    let traced: Vec<_> = named
                        .named
                        .iter()
                        .filter(|f| !has_ignore_trace(&f.attrs))
                        .filter_map(|f| f.ident.as_ref())
                        .collect();
                    quote! {
                        #ident::#var_ident { #(#traced,)* .. } => {
                            #(#traced.#method_ident();)*
                        }
                    }
                }
                Fields::Unnamed(unnamed) => {
                    let total = unnamed.unnamed.len();
                    let is_traced: Vec<bool> = unnamed
                        .unnamed
                        .iter()
                        .map(|f| !has_ignore_trace(&f.attrs))
                        .collect();
                    let binding_names: Vec<syn::Ident> = (0..total)
                        .map(|i| {
                            syn::Ident::new(&format!("_{}", i), proc_macro2::Span::call_site())
                        })
                        .collect();
                    let pattern: Vec<proc_macro2::TokenStream> = (0..total)
                        .map(|i| {
                            if is_traced[i] {
                                let name = &binding_names[i];
                                quote! { #name }
                            } else {
                                quote! { _ }
                            }
                        })
                        .collect();
                    let traced_names: Vec<&syn::Ident> = (0..total)
                        .filter(|&i| is_traced[i])
                        .map(|i| &binding_names[i])
                        .collect();
                    quote! {
                        #ident::#var_ident(#(#pattern),*) => {
                            #(#traced_names.#method_ident();)*
                        }
                    }
                }
                Fields::Unit => {
                    quote! { #ident::#var_ident => {} }
                }
            }
        })
        .collect();

    quote! {
        match self {
            #(#arms)*
        }
    }
}

/// Generate a `match self { ... }` expression for an enum, calling `trace_children(children)` on traced fields.
fn enum_match_children(ident: &syn::Ident, data_enum: &syn::DataEnum) -> proc_macro2::TokenStream {
    let arms: Vec<_> = data_enum
        .variants
        .iter()
        .map(|variant| {
            let var_ident = &variant.ident;
            match &variant.fields {
                Fields::Named(named) => {
                    let traced: Vec<_> = named
                        .named
                        .iter()
                        .filter(|f| !has_ignore_trace(&f.attrs))
                        .filter_map(|f| f.ident.as_ref())
                        .collect();
                    quote! {
                        #ident::#var_ident { #(#traced,)* .. } => {
                            #(#traced.trace_children(children);)*
                        }
                    }
                }
                Fields::Unnamed(unnamed) => {
                    let total = unnamed.unnamed.len();
                    let is_traced: Vec<bool> = unnamed
                        .unnamed
                        .iter()
                        .map(|f| !has_ignore_trace(&f.attrs))
                        .collect();
                    let binding_names: Vec<syn::Ident> = (0..total)
                        .map(|i| {
                            syn::Ident::new(&format!("_{}", i), proc_macro2::Span::call_site())
                        })
                        .collect();
                    let pattern: Vec<proc_macro2::TokenStream> = (0..total)
                        .map(|i| {
                            if is_traced[i] {
                                let name = &binding_names[i];
                                quote! { #name }
                            } else {
                                quote! { _ }
                            }
                        })
                        .collect();
                    let traced_names: Vec<&syn::Ident> = (0..total)
                        .filter(|&i| is_traced[i])
                        .map(|i| &binding_names[i])
                        .collect();
                    quote! {
                        #ident::#var_ident(#(#pattern),*) => {
                            #(#traced_names.trace_children(children);)*
                        }
                    }
                }
                Fields::Unit => {
                    quote! { #ident::#var_ident => {} }
                }
            }
        })
        .collect();

    quote! {
        match self {
            #(#arms)*
        }
    }
}

#[proc_macro_derive(Trace, attributes(unsafe_ignore_trace))]
pub fn derive_trace(item: TokenStream) -> TokenStream {
    let derive_input = syn::parse_macro_input!(item as syn::DeriveInput);

    let ident = &derive_input.ident;

    // Add Trace bound to all type parameters
    let mut generics = derive_input.generics.clone();
    for param in &mut generics.params {
        if let syn::GenericParam::Type(ref mut type_param) = *param {
            type_param.bounds.push(syn::parse_quote!(Trace));
        }
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let (reset_root_body, trace_body, reset_body, trace_children_body) = match &derive_input.data {
        Data::Struct(data_struct) => struct_bodies(&data_struct.fields),
        Data::Enum(data_enum) => (
            enum_match(ident, data_enum, "reset_root"),
            enum_match(ident, data_enum, "trace"),
            enum_match(ident, data_enum, "reset"),
            enum_match_children(ident, data_enum),
        ),
        Data::Union(_) => {
            return syn::Error::new_spanned(
                &derive_input,
                "#[derive(Trace)] is not supported for unions. Use a struct or enum instead.",
            )
            .to_compile_error()
            .into();
        }
    };

    let trace_impl = quote! {
        impl #impl_generics Trace for #ident #ty_generics #where_clause {
            fn is_root(&self) -> bool {
                unreachable!("is_root should never be called on user-defined type !!");
            }

            fn reset_root(&self) {
                #reset_root_body
            }

            fn trace(&self) {
                #trace_body
            }

            fn reset(&self) {
                #reset_body
            }

            fn trace_children(&self, children: &mut std::vec::Vec<*const dyn Trace>) {
                #trace_children_body
            }

            fn is_traceable(&self) -> bool {
                unreachable!("is_traceable should never be called on user-defined type !!");
            }
        }
    };

    trace_impl.into()
}

#[proc_macro_derive(Finalize)]
pub fn derive_finalize(item: TokenStream) -> TokenStream {
    let derive_input = syn::parse_macro_input!(item as syn::DeriveInput);

    let ident = &derive_input.ident;

    // Add Finalize bound to all type parameters
    let mut generics = derive_input.generics.clone();
    for param in &mut generics.params {
        if let syn::GenericParam::Type(ref mut type_param) = *param {
            type_param.bounds.push(syn::parse_quote!(Finalize));
        }
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let finalizer_impl = quote! {
        impl #impl_generics Finalize for #ident #ty_generics #where_clause {
            fn finalize(&self) {
            }
        }
    };

    finalizer_impl.into()
}

/// Attribute macro for the application entry point.
///
/// # Usage
/// ```ignore
/// #[ferris_gc_main]                          // basic strategy (default, 500ms poll)
/// #[ferris_gc_main(strategy = "basic", poll_interval_ms = 200)]
/// #[ferris_gc_main(strategy = "threshold")]  // threshold with defaults
/// #[ferris_gc_main(strategy = "threshold", gen0_threshold = 200, poll_interval_ms = 100)]
/// #[ferris_gc_main(strategy = "adaptive", min_threshold = 30, max_threshold = 5000)]
/// fn main() { }
/// ```
///
/// ## Parameters
///
/// **All strategies:**
/// - `poll_interval_ms` — background thread polling interval in milliseconds
///
/// **`threshold` strategy:**
/// - `gen0_threshold` — allocations before triggering Gen0 collection (default: 100)
/// - `gen0_collections_per_gen1` — Gen0 collections between each Gen1 (default: 5)
/// - `gen1_collections_per_gen2` — Gen1 collections between each Gen2 (default: 5)
///
/// **`adaptive` strategy** (all `threshold` params plus):
/// - `initial_gen0_threshold` — starting allocation threshold (default: 100)
/// - `min_threshold` — minimum threshold floor (default: 50)
/// - `max_threshold` — maximum threshold ceiling (default: 10000)
/// - `high_ratio` — collection ratio above which threshold decreases (default: 0.5)
/// - `low_ratio` — collection ratio below which threshold increases (default: 0.1)
/// - `adjust_factor` — multiplicative factor for threshold adjustment (default: 1.5)
#[proc_macro_attribute]
pub fn ferris_gc_main(attrs: TokenStream, item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as syn::ItemFn);
    let attr_args = syn::parse_macro_input!(attrs as syn::AttributeArgs);

    // Parse all name = value attributes
    let mut strategy = String::from("basic");
    let mut strategy_span: Option<proc_macro2::Span> = None;
    let mut config_params: Vec<(String, Lit, proc_macro2::Span)> = Vec::new();

    for arg in &attr_args {
        if let NestedMeta::Meta(Meta::NameValue(nv)) = arg {
            let name = nv
                .path
                .get_ident()
                .map(|i| i.to_string())
                .unwrap_or_default();
            if name == "strategy" {
                if let Lit::Str(s) = &nv.lit {
                    strategy = s.value();
                    strategy_span = Some(s.span());
                } else {
                    return syn::Error::new_spanned(
                        &nv.lit,
                        "strategy value must be a string literal, e.g. strategy = \"basic\"",
                    )
                    .to_compile_error()
                    .into();
                }
            } else {
                let span = nv
                    .path
                    .get_ident()
                    .map(|i| i.span())
                    .unwrap_or_else(proc_macro2::Span::call_site);
                config_params.push((name, nv.lit.clone(), span));
            }
        } else {
            return syn::Error::new(
                proc_macro2::Span::call_site(),
                "expected key = value. Example: #[ferris_gc_main(strategy = \"threshold\", gen0_threshold = 200)]",
            )
            .to_compile_error()
            .into();
        }
    }

    // Validate config params for the chosen strategy
    let valid_basic = &["poll_interval_ms"];
    let valid_threshold = &[
        "gen0_threshold",
        "gen0_collections_per_gen1",
        "gen1_collections_per_gen2",
        "poll_interval_ms",
    ];
    let valid_adaptive = &[
        "initial_gen0_threshold",
        "min_threshold",
        "max_threshold",
        "high_ratio",
        "low_ratio",
        "adjust_factor",
        "gen0_collections_per_gen1",
        "gen1_collections_per_gen2",
        "poll_interval_ms",
    ];

    let valid_params: &[&str] = match strategy.as_str() {
        "basic" => valid_basic,
        "threshold" => valid_threshold,
        "adaptive" => valid_adaptive,
        _ => &[], // will error below
    };

    if strategy == "basic" || strategy == "threshold" || strategy == "adaptive" {
        for (name, _, span) in &config_params {
            if !valid_params.contains(&name.as_str()) {
                return syn::Error::new(
                    *span,
                    format!(
                        "unknown parameter '{}' for strategy '{}'. Valid: {}",
                        name,
                        strategy,
                        valid_params.join(", "),
                    ),
                )
                .to_compile_error()
                .into();
            }
        }
    }

    // Helper: find a config param by name and return its literal
    let find_param = |name: &str| -> Option<&Lit> {
        config_params
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, lit, _)| lit)
    };

    // Generate strategy setup code
    let strategy_setup = match strategy.as_str() {
        "basic" => {
            if let Some(lit) = find_param("poll_interval_ms") {
                quote! {
                    ferris_gc::BASIC_POLL_INTERVAL_MS.store(#lit, std::sync::atomic::Ordering::Release);
                }
            } else {
                quote! {}
            }
        }
        "threshold" => {
            let gen0_threshold = find_param("gen0_threshold");
            let gen0_per_gen1 = find_param("gen0_collections_per_gen1");
            let gen1_per_gen2 = find_param("gen1_collections_per_gen2");
            let poll_ms = find_param("poll_interval_ms");

            // Build field overrides — only emit fields that were explicitly set
            let mut overrides = Vec::new();
            if let Some(v) = gen0_threshold {
                overrides.push(quote! { gen0_threshold: #v });
            }
            if let Some(v) = gen0_per_gen1 {
                overrides.push(quote! { gen0_collections_per_gen1: #v });
            }
            if let Some(v) = gen1_per_gen2 {
                overrides.push(quote! { gen1_collections_per_gen2: #v });
            }
            if let Some(v) = poll_ms {
                overrides.push(quote! { poll_interval: std::time::Duration::from_millis(#v) });
            }

            quote! {
                ferris_gc::BASIC_STRATEGY_DISABLED.store(true, std::sync::atomic::Ordering::Release);
                let __ferris_gc_config = ferris_gc::ThresholdConfig {
                    #(#overrides,)*
                    ..ferris_gc::ThresholdConfig::default()
                };
                ferris_gc::LOCAL_GC_STRATEGY.with(|s| {
                    let (start, stop) = ferris_gc::threshold_local_start(ferris_gc::ThresholdConfig {
                        #(#overrides,)*
                        ..ferris_gc::ThresholdConfig::default()
                    });
                    s.borrow().change_strategy(start, stop);
                });
                {
                    let (start, stop) = ferris_gc::threshold_global_start(__ferris_gc_config);
                    ferris_gc::sync::GLOBAL_GC_STRATEGY.change_strategy(start, stop);
                }
            }
        }
        "adaptive" => {
            let initial = find_param("initial_gen0_threshold");
            let min_t = find_param("min_threshold");
            let max_t = find_param("max_threshold");
            let high_r = find_param("high_ratio");
            let low_r = find_param("low_ratio");
            let adjust = find_param("adjust_factor");
            let gen0_per_gen1 = find_param("gen0_collections_per_gen1");
            let gen1_per_gen2 = find_param("gen1_collections_per_gen2");
            let poll_ms = find_param("poll_interval_ms");

            let mut overrides = Vec::new();
            if let Some(v) = initial {
                overrides.push(quote! { initial_gen0_threshold: #v });
            }
            if let Some(v) = min_t {
                overrides.push(quote! { min_threshold: #v });
            }
            if let Some(v) = max_t {
                overrides.push(quote! { max_threshold: #v });
            }
            if let Some(v) = high_r {
                overrides.push(quote! { high_ratio: #v });
            }
            if let Some(v) = low_r {
                overrides.push(quote! { low_ratio: #v });
            }
            if let Some(v) = adjust {
                overrides.push(quote! { adjust_factor: #v });
            }
            if let Some(v) = gen0_per_gen1 {
                overrides.push(quote! { gen0_collections_per_gen1: #v });
            }
            if let Some(v) = gen1_per_gen2 {
                overrides.push(quote! { gen1_collections_per_gen2: #v });
            }
            if let Some(v) = poll_ms {
                overrides.push(quote! { poll_interval: std::time::Duration::from_millis(#v) });
            }

            quote! {
                ferris_gc::BASIC_STRATEGY_DISABLED.store(true, std::sync::atomic::Ordering::Release);
                let __ferris_gc_config = ferris_gc::AdaptiveConfig {
                    #(#overrides,)*
                    ..ferris_gc::AdaptiveConfig::default()
                };
                ferris_gc::LOCAL_GC_STRATEGY.with(|s| {
                    let (start, stop) = ferris_gc::adaptive_local_start(ferris_gc::AdaptiveConfig {
                        #(#overrides,)*
                        ..ferris_gc::AdaptiveConfig::default()
                    });
                    s.borrow().change_strategy(start, stop);
                });
                {
                    let (start, stop) = ferris_gc::adaptive_global_start(__ferris_gc_config);
                    ferris_gc::sync::GLOBAL_GC_STRATEGY.change_strategy(start, stop);
                }
            }
        }
        other => {
            let span = strategy_span.unwrap_or_else(proc_macro2::Span::call_site);
            return syn::Error::new(
                span,
                format!(
                    "unknown GC strategy '{}'. Supported: basic, threshold, adaptive",
                    other
                ),
            )
            .to_compile_error()
            .into();
        }
    };

    let _sig = &input.sig;
    let vis = input.vis;
    let name = &input.sig.ident;
    if name != "main" {
        return syn::Error::new_spanned(
            &input.sig.ident,
            "#[ferris_gc_main] can only be applied to the main function",
        )
        .to_compile_error()
        .into();
    }
    let mut args = Vec::new();
    for arg in &input.sig.inputs {
        args.push(arg);
    }
    let ret = match &input.sig.output {
        ReturnType::Default => {
            quote! {}
        }
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

    res_fun.into()
}
