use proc_macro::TokenStream;
use std::borrow::{Borrow, BorrowMut};
use quote::quote;
use syn::{ReturnType, Type, Data, Fields};

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
        Data::Enum(data_enum) => {
            panic!("Enum type is not supported !!");
        },
        Data::Union(data_union) => {
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

#[proc_macro_attribute]
pub fn ferris_gc_main(attrs: TokenStream, item: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(item as syn::ItemFn);
    let attr_args = syn::parse_macro_input!(attrs as syn::AttributeArgs);

    let sig = &input.sig;
    let vis = input.vis;
    let name = &input.sig.ident;
    let mut args = Vec::new();
    for arg in &input.sig.inputs {
        args.push(arg);
    }
    let ret = match &input.sig.output {
        ReturnType::Default => {
            quote! {
            }
        },
        ReturnType::Type(arrow, box_type) => {
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
            // Should be added proper closing background threads
            {
               let cleanup = ApplicationCleanUp;
                #body
            }
        }
    };

    let print_tokens = Into::<TokenStream>::into(res_fun.clone());
    println!("Result Function is {}", print_tokens.to_string());
    res_fun.into()
}
