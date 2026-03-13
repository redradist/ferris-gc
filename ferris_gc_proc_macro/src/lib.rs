use proc_macro::TokenStream;
use quote::quote;
use syn::{ReturnType, Data, Fields, NestedMeta, Meta, Lit};

fn has_ignore_trace(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path.get_ident()
            .map(|id| id == "unsafe_ignore_trace")
            .unwrap_or(false)
    })
}

/// Generate method bodies (reset_root, trace, reset, trace_children) for struct fields.
fn struct_bodies(fields: &Fields) -> (proc_macro2::TokenStream, proc_macro2::TokenStream, proc_macro2::TokenStream, proc_macro2::TokenStream) {
    match fields {
        Fields::Named(named) => {
            let traced: Vec<_> = named.named.iter()
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
            let indices: Vec<syn::Index> = unnamed.unnamed.iter()
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
fn enum_match(ident: &syn::Ident, data_enum: &syn::DataEnum, method: &str) -> proc_macro2::TokenStream {
    let method_ident = syn::Ident::new(method, proc_macro2::Span::call_site());

    let arms: Vec<_> = data_enum.variants.iter().map(|variant| {
        let var_ident = &variant.ident;
        match &variant.fields {
            Fields::Named(named) => {
                let traced: Vec<_> = named.named.iter()
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
                let is_traced: Vec<bool> = unnamed.unnamed.iter()
                    .map(|f| !has_ignore_trace(&f.attrs))
                    .collect();
                let binding_names: Vec<syn::Ident> = (0..total)
                    .map(|i| syn::Ident::new(&format!("_{}", i), proc_macro2::Span::call_site()))
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
    }).collect();

    quote! {
        match self {
            #(#arms)*
        }
    }
}

/// Generate a `match self { ... }` expression for an enum, calling `trace_children(children)` on traced fields.
fn enum_match_children(ident: &syn::Ident, data_enum: &syn::DataEnum) -> proc_macro2::TokenStream {
    let arms: Vec<_> = data_enum.variants.iter().map(|variant| {
        let var_ident = &variant.ident;
        match &variant.fields {
            Fields::Named(named) => {
                let traced: Vec<_> = named.named.iter()
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
                let is_traced: Vec<bool> = unnamed.unnamed.iter()
                    .map(|f| !has_ignore_trace(&f.attrs))
                    .collect();
                let binding_names: Vec<syn::Ident> = (0..total)
                    .map(|i| syn::Ident::new(&format!("_{}", i), proc_macro2::Span::call_site()))
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
    }).collect();

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
                "#[derive(Trace)] is not supported for unions. Use a struct or enum instead."
            ).to_compile_error().into();
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
    let mut strategy_span: Option<proc_macro2::Span> = None;
    for arg in &attr_args {
        if let NestedMeta::Meta(Meta::NameValue(nv)) = arg {
            if nv.path.is_ident("strategy") {
                if let Lit::Str(s) = &nv.lit {
                    strategy = s.value();
                    strategy_span = Some(s.span());
                } else {
                    return syn::Error::new_spanned(
                        &nv.lit,
                        "strategy value must be a string literal, e.g. strategy = \"basic\""
                    ).to_compile_error().into();
                }
            } else {
                let name = nv.path.get_ident().map(|i| i.to_string()).unwrap_or_default();
                return syn::Error::new_spanned(
                    &nv.path,
                    format!("unknown attribute '{}'. Only 'strategy' is supported", name)
                ).to_compile_error().into();
            }
        } else {
            return syn::Error::new(
                proc_macro2::Span::call_site(),
                "expected strategy = \"...\". Supported strategies: basic, threshold, adaptive"
            ).to_compile_error().into();
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
        other => {
            let span = strategy_span.unwrap_or_else(proc_macro2::Span::call_site);
            return syn::Error::new(
                span,
                format!("unknown GC strategy '{}'. Supported: basic, threshold, adaptive", other)
            ).to_compile_error().into();
        }
    };

    let _sig = &input.sig;
    let vis = input.vis;
    let name = &input.sig.ident;
    if name != "main" {
        return syn::Error::new_spanned(
            &input.sig.ident,
            "#[ferris_gc_main] can only be applied to the main function"
        ).to_compile_error().into();
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

    res_fun.into()
}
