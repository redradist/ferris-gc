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
                                unreachable!();
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
                                unreachable!();
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
                                unreachable!();
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
                                unreachable!();
                            }
                        }
                    }
                },
                Fields::Unit => {
                    panic!("Enum type is not supported !!");
                    // quote! {
                    //     #(#attrs)*
                    //     #vis struct #ident {
                    //         #(#data_struct.fields)*
                    //     }
                    // }
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

#[proc_macro_derive(Finalizer)]
pub fn derive_finalizer(item: TokenStream) -> TokenStream {
    let derive_input = syn::parse_macro_input!(item as syn::DeriveInput);

    let ident = &derive_input.ident;
    let finalizer_impl = quote! {
        impl Finalizer for #ident {
            fn finalize(&self) {
            }
        }
    };

    let print_tokens = Into::<TokenStream>::into(finalizer_impl.clone());
    println!("Result Finalizer Impl is {}", print_tokens.to_string());
    finalizer_impl.into()
}
