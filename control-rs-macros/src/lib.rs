//! Procedural macros for `control-rs` Embedded Test Server (ETS) firmware.
//!
//! | Macro | Generates |
//! |:--|:--|
//! | `#[ets_suite]` | Suite descriptor and case registration in `.ets_test_suites` |
//! | `#[ets_setup]` | Target `main` that runs the server with the returned `Context` |
//! | `ets_entrypoint!` | Target `main` for a given setup function |
//! | `ets_panic!` | Panic handler that reports the failure to the host and resets |
//! | `ets_exception!` | Exception handler |
//!
//! Firmware usage: `examples/qemu` and `examples/teensy4` in the repository.

#![allow(unused_extern_crates)]

extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{Item, ItemFn, ItemMod, ItemStatic, parse_quote};

/// Type alias for a test function's identifier and its description.
type TestFnInfo = (syn::Ident, String);

/// The lifecycle task of a suite: the implementing type and its description.
type TaskInfo = (syn::Type, String);

/// The cases and the optional lifecycle task of a suite.
type Executables<'a> = (&'a [TestFnInfo], Option<&'a TaskInfo>);

/// The context generics and the optional board type of a setup return type.
type SetupTypes = (syn::Type, syn::Type, Option<syn::Type>);

/// Helper to extract doc comments from syn attributes, strip compiler-injected leading space,
/// and truncate to a maximum of 160 characters (appending `...` if truncated).
fn extract_doc_string(attrs: &[syn::Attribute]) -> String {
    let mut docs = Vec::new();
    for attr in attrs {
        if attr.path().is_ident("doc")
            && let syn::Meta::NameValue(syn::MetaNameValue {
                value:
                    syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(lit_str),
                        ..
                    }),
                ..
            }) = &attr.meta
        {
            let val = lit_str.value();
            let trimmed = val.strip_prefix(' ').unwrap_or(&val);
            docs.push(trimmed.to_string());
        }
    }
    let full_doc = docs.join("\n");
    let full_doc = full_doc.trim();
    if full_doc.chars().count() > 160 {
        let truncated: String = full_doc.chars().take(157).collect();
        format!("{truncated}...")
    } else {
        full_doc.to_string()
    }
}

/// Checks if a given `syn::Type` is a supported primitive parameter.
/// Returns the name of the matching `ParamEntry` constructor.
fn get_param_constructor(ty: &syn::Type) -> Option<&'static str> {
    // Ensure the type is a standard path (for example, `u8` or `std::primitive::u8`)
    let path = match ty {
        syn::Type::Path(type_path) if type_path.qself.is_none() => {
            &type_path.path
        }
        _ => return None,
    };

    // Extract the last segment (the actual type name)
    let ident = &path.segments.last()?.ident;

    // Convert to string and match in O(1)
    match ident.to_string().as_str() {
        "u8" => Some("u8"),
        "u16" => Some("u16"),
        "u32" => Some("u32"),
        "i32" => Some("i32"),
        "bool" => Some("bool"),
        "f32" => Some("f32"),
        _ => None,
    }
}

/// Reads `#[param(min = A, max = B)]` and returns the `.bounded(..)` call, or nothing.
fn param_bounds(
    attrs: &[syn::Attribute],
    constructor: &str,
) -> proc_macro2::TokenStream {
    let mut min = None;
    let mut max = None;
    for attr in attrs.iter().filter(|a| a.path().is_ident("param")) {
        let _ = attr.parse_nested_meta(|meta| {
            let value: syn::Expr = meta.value()?.parse()?;
            if meta.path.is_ident("min") {
                min = Some(value);
            } else if meta.path.is_ident("max") {
                max = Some(value);
            }
            Ok(())
        });
    }
    let variant = format_ident!(
        "{}",
        match constructor {
            "bool" => "Bool".to_string(),
            other => other.to_uppercase(),
        }
    );
    match (min, max) {
        (Some(lo), Some(hi)) => quote! {
            .bounded(
                ::control_rs_ets::wire::Param::#variant(#lo),
                ::control_rs_ets::wire::Param::#variant(#hi),
            )
        },
        _ => proc_macro2::TokenStream::new(),
    }
}

/// If the static item has a supported primitive type, rewrites it into a
/// `ParamEntry` static named after the item and returns its identifier.
fn process_static_param(item_static: &mut ItemStatic) -> Option<syn::Ident> {
    // 1. Search phase: Delegate to the helper function
    let constructor = get_param_constructor(&item_static.ty)?;

    // 2. Mutation phase: clone required fields and overwrite
    let vis = item_static.vis.clone();
    let name_ident = item_static.ident.clone();
    let init_expr = item_static.expr.clone();
    let attrs = item_static.attrs.clone();

    let param_doc = extract_doc_string(&attrs);
    let constructor_ident = format_ident!("{}", constructor);
    let bounds = param_bounds(&attrs, constructor);
    let attrs: Vec<_> = attrs
        .into_iter()
        .filter(|a| !a.path().is_ident("param"))
        .collect();

    let new_static: ItemStatic = parse_quote! {
        #(#attrs)*
        #vis static #name_ident: ::control_rs_ets::params::ParamEntry =
            ::control_rs_ets::params::ParamEntry::#constructor_ident(stringify!(#name_ident), #param_doc, #init_expr) #bounds;
    };

    *item_static = new_static;

    Some(name_ident)
}

/// If the function is a test executable (`fn()` and does not start with `_`),
/// extracts its identifier and doc comments.
///
/// Functions with parameters or a return value are helpers: `ExecDescriptor::test_fn` is `fn()`, so
/// registering them fails QEMU/`no_std` compilation with E0308.
fn process_test_fn(item_fn: &ItemFn) -> Option<TestFnInfo> {
    let fn_name = &item_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    if fn_name_str.starts_with('_')
        || !item_fn.sig.inputs.is_empty()
        || !matches!(item_fn.sig.output, syn::ReturnType::Default)
    {
        None
    } else {
        let test_doc = extract_doc_string(&item_fn.attrs);
        Some((fn_name.clone(), test_doc))
    }
}

/// Generates the case step wrappers.
fn case_wrappers(tests: &[TestFnInfo]) -> Vec<Item> {
    tests
        .iter()
        .map(|(t, _)| {
            let wrapper = format_ident!("__ets_case_{}", t);
            parse_quote! {
                fn #wrapper(
                    _: &mut ::control_rs_ets::exec::TaskIo<'_>,
                ) -> ::control_rs_ets::exec::TaskOutcome {
                    #t();
                    ::control_rs_ets::exec::TaskOutcome::Finish(
                        ::control_rs_ets::wire::Verdict::Pass,
                        None,
                    )
                }
            }
        })
        .collect()
}

/// Generates the slot and the five monomorphic wrappers of a lifecycle task.
fn task_wrappers(ty: &syn::Type) -> Vec<Item> {
    let exec = quote!(::control_rs_ets::exec);
    vec![
        parse_quote! {
            static __ETS_TASK: #exec::TaskSlot<#ty> = #exec::TaskSlot::new();
        },
        parse_quote! {
            fn __ets_task_setup() -> #exec::HookOf {
                #exec::task_setup(&__ETS_TASK)
            }
        },
        parse_quote! {
            fn __ets_task_step(io: &mut #exec::TaskIo<'_>) -> #exec::TaskOutcome {
                #exec::task_step(&__ETS_TASK, io)
            }
        },
        parse_quote! {
            fn __ets_task_reset() -> #exec::HookOf {
                #exec::task_reset(&__ETS_TASK)
            }
        },
        parse_quote! {
            fn __ets_task_teardown() -> #exec::HookOf {
                #exec::task_teardown(&__ETS_TASK)
            }
        },
        parse_quote! {
            fn __ets_task_safe_state() {
                <#ty as #exec::LifecycleTask>::safe_state();
            }
        },
    ]
}

/// Generates the static descriptor items to append to the module.
fn generate_suite_descriptors(
    suite_name: &str,
    suite_doc: &str,
    (tests, task): Executables<'_>,
    params: &[syn::Ident],
) -> Vec<Item> {
    let mut items = case_wrappers(tests);
    let mut descriptors: Vec<proc_macro2::TokenStream> = tests
        .iter()
        .map(|(t, doc)| {
            let wrapper = format_ident!("__ets_case_{}", t);
            quote! {
                ::control_rs_ets::ExecDescriptor::case(stringify!(#t), #doc, #wrapper)
            }
        })
        .collect();
    if let Some((ty, doc)) = task {
        items.extend(task_wrappers(ty));
        let name = quote!(#ty).to_string().replace(' ', "");
        descriptors.push(quote! {
            ::control_rs_ets::exec::task_descriptor::<#ty>(
                #name,
                #doc,
                (
                    __ets_task_setup,
                    __ets_task_step,
                    __ets_task_reset,
                    __ets_task_teardown,
                    __ets_task_safe_state,
                ),
            )
        });
    }

    let param_ptrs = params.iter().map(|s| {
        quote! {
            &#s
        }
    });

    items.extend([
        parse_quote! {
            static EXECUTABLES: &[::control_rs_ets::ExecDescriptor] = &[
                #(#descriptors),*
            ];
        },
        parse_quote! {
            static PARAMS: &[&::control_rs_ets::params::ParamEntry] = &[
                #(#param_ptrs),*
            ];
        },
        parse_quote! {
            static SUITE_DESCRIPTOR: ::control_rs_ets::SuiteDescriptor = ::control_rs_ets::SuiteDescriptor {
                name: #suite_name,
                description: #suite_doc,
                executables: EXECUTABLES,
                params: PARAMS,
            };
        },
        parse_quote! {
            /// Pointer to the suite descriptor, linked into the ETS test suites section.
            #[cfg_attr(target_vendor = "apple", unsafe(link_section = "__DATA,__ets_suites"))]
            #[cfg_attr(not(target_vendor = "apple"), unsafe(link_section = ".ets_test_suites"))]
            #[used]
            pub static SUITE_DESCRIPTOR_PTR: &::control_rs_ets::SuiteDescriptor = &SUITE_DESCRIPTOR;
        },
    ]);
    items
}

/// The implementing type and description when `item_impl` is `impl LifecycleTask for X`.
fn lifecycle_impl(item_impl: &syn::ItemImpl) -> Option<TaskInfo> {
    let (_, path, _) = item_impl.trait_.as_ref()?;
    if path.segments.last()?.ident != "LifecycleTask" {
        return None;
    }
    Some((
        (*item_impl.self_ty).clone(),
        extract_doc_string(&item_impl.attrs),
    ))
}

/// Expand `#[ets_suite]` for an inline module.
fn ets_suite_impl(item: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let mut item_mod: ItemMod = match syn::parse2(item) {
        Ok(m) => m,
        Err(e) => return e.to_compile_error(),
    };
    if item_mod.content.is_none() {
        return syn::Error::new_spanned(
            &item_mod,
            "#[ets_suite] attribute is only supported on inline modules (e.g., mod foo { ... })",
        )
        .to_compile_error();
    }
    let suite_name = item_mod.ident.to_string();
    let suite_doc = extract_doc_string(&item_mod.attrs);

    let mut tests = Vec::new();
    let mut params = Vec::new();
    let mut task: Option<TaskInfo> = None;

    if let Some((_, ref mut items)) = item_mod.content {
        for inner_item in items.iter_mut() {
            match inner_item {
                Item::Static(item_static) => {
                    let Some(param) = process_static_param(item_static) else {
                        return syn::Error::new_spanned(
                            &*item_static,
                            "suite statics must be u8, u16, u32, i32, f32 or bool",
                        )
                        .to_compile_error();
                    };
                    params.push(param);
                }
                Item::Fn(item_fn) => {
                    if let Some(test) = process_test_fn(item_fn) {
                        tests.push(test);
                    }
                }
                Item::Impl(item_impl) => {
                    if let Some(found) = lifecycle_impl(item_impl) {
                        if task.is_some() {
                            return syn::Error::new_spanned(
                                &*item_impl,
                                "at most one lifecycle task per suite",
                            )
                            .to_compile_error();
                        }
                        task = Some(found);
                    }
                }
                _ => {}
            }
        }

        let suite_desc_code = generate_suite_descriptors(
            &suite_name,
            &suite_doc,
            (&tests, task.as_ref()),
            &params,
        );
        items.extend(suite_desc_code);
    }

    quote! {
        #item_mod
    }
}

/// Attribute macro for declaring a ETS test suite.
///
/// Converts statics to parameter table entries and registers functions as test executables.
#[proc_macro_attribute]
pub fn ets_suite(_attr: TokenStream, item: TokenStream) -> TokenStream {
    TokenStream::from(ets_suite_impl(item.into()))
}

/// Helper to extract C and P generic type arguments from a Type of form `PathSegment<C, P>`.
#[allow(clippy::type_complexity)]
fn extract_context_generics(ty: &syn::Type) -> Option<(syn::Type, syn::Type)> {
    let syn::Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if segment.ident != "Context" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(generic_args) = &segment.arguments
    else {
        return None;
    };
    let mut args = generic_args.args.iter();
    let syn::GenericArgument::Type(c_ty) = args.next()? else {
        return None;
    };
    let syn::GenericArgument::Type(p_ty) = args.next()? else {
        return None;
    };
    Some((c_ty.clone(), p_ty.clone()))
}

/// Splits a setup return type into the `Context` type and the optional board type.
fn setup_types(ty: &syn::Type) -> Option<SetupTypes> {
    if let syn::Type::Tuple(tuple) = ty {
        let mut elems = tuple.elems.iter();
        let (context, board) = (elems.next()?, elems.next()?);
        if elems.next().is_some() {
            return None;
        }
        let (c_ty, p_ty) = extract_context_generics(context)?;
        return Some((c_ty, p_ty, Some(board.clone())));
    }
    let (c_ty, p_ty) = extract_context_generics(ty)?;
    Some((c_ty, p_ty, None))
}

/// Expand `#[derive(Packet)]`: a Postcard packet over the type's serde implementations.
fn packet_derive_impl(
    input: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let item: syn::DeriveInput = match syn::parse2(input) {
        Ok(i) => i,
        Err(e) => return e.to_compile_error(),
    };
    let name = &item.ident;
    let (impl_generics, ty_generics, where_clause) =
        item.generics.split_for_impl();
    quote! {
        impl #impl_generics ::control_rs_ets::exec::Packet for #name #ty_generics #where_clause {
            const INFO: ::control_rs_ets::wire::PacketInfo<'static> =
                ::control_rs_ets::wire::PacketInfo {
                    type_name: stringify!(#name),
                    encoding: ::control_rs_ets::wire::PayloadEncoding::Postcard,
                };

            fn decode(bytes: &[u8]) -> Option<Self> {
                ::control_rs_ets::exec::postcard_decode(bytes)
            }

            fn encode(&self, out: &mut [u8]) -> Option<usize> {
                ::control_rs_ets::exec::postcard_encode(self, out)
            }
        }
    }
}

/// Derives `Packet` for a type that implements serde's `Serialize` and `Deserialize`.
#[proc_macro_derive(Packet)]
pub fn derive_packet(input: TokenStream) -> TokenStream {
    TokenStream::from(packet_derive_impl(input.into()))
}

/// Expand `#[ets_setup]` for a function returning `Context<C, P>` or `(Context<C, P>, B)`.
fn ets_setup_impl(item: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let setup_fn: ItemFn = match syn::parse2(item) {
        Ok(f) => f,
        Err(e) => return e.to_compile_error(),
    };
    let setup_name = &setup_fn.sig.ident;

    let syn::ReturnType::Type(_, return_type) = &setup_fn.sig.output else {
        return syn::Error::new_spanned(
            &setup_fn.sig,
            "setup function must return a Context<C, P> or (Context<C, P>, B)",
        )
        .to_compile_error();
    };

    let Some((c_ty, p_ty, board)) = setup_types(return_type) else {
        return syn::Error::new_spanned(
            return_type,
            "setup function return type must be Context<C, P> or (Context<C, P>, B)",
        )
        .to_compile_error();
    };
    let b_ty = board.as_ref().map_or_else(
        || quote!(::control_rs_ets::board::NoBoard),
        |b| quote!(#b),
    );
    let entry = if board.is_some() {
        quote!(::control_rs_macros::ets_entrypoint!(#setup_name, board);)
    } else {
        quote!(::control_rs_macros::ets_entrypoint!(#setup_name);)
    };

    quote! {
        #setup_fn

        static ETS_SERVER: ::control_rs_ets::cell::StateCell<*mut ::control_rs_ets::Server<'static, #c_ty, #p_ty, #b_ty>> =
            ::control_rs_ets::cell::StateCell::new(::core::ptr::null_mut());

        #entry
        ::control_rs_macros::ets_panic!();
        ::control_rs_macros::ets_exception!();
    }
}

/// Attribute macro for setting up ETS entrypoint.
///
/// Annotates the hardware setup function, generates the standard main entrypoint,
/// and sets up the server event loop and QEMU-compatible panic handler.
#[proc_macro_attribute]
pub fn ets_setup(_attr: TokenStream, item: TokenStream) -> TokenStream {
    TokenStream::from(ets_setup_impl(item.into()))
}

/// Expand `ets_entrypoint!(setup_fn)`.
fn ets_entrypoint_impl(
    input: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let parsed = syn::parse::Parser::parse2(
        syn::punctuated::Punctuated::<syn::Ident, syn::Token![,]>::parse_terminated,
        input,
    );
    let idents = match parsed {
        Ok(i) => i,
        Err(e) => return e.to_compile_error(),
    };
    let mut idents = idents.into_iter();
    let Some(setup_name) = idents.next() else {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "expected identifier",
        )
        .to_compile_error();
    };
    let with_board = idents.next().is_some();
    let make_context = if with_board {
        quote! { let (context, board) = #setup_name(); }
    } else {
        quote! {
            let context = #setup_name();
            let board = ::control_rs_ets::board::NoBoard;
        }
    };
    quote! {
        #[cfg(target_os = "none")]
        unsafe extern "Rust" {
            static __ets_test_suites_start: u8;
            static __ets_test_suites_end: u8;
        }

        // ==================== Unified Entry Point ====================
        #[cfg(target_os = "none")]
        #[cfg_attr(target_arch = "arm", ::cortex_m_rt::entry)]
        #[cfg_attr(any(target_arch = "riscv32", target_arch = "riscv64"), ::riscv_rt::entry)]
        fn main() -> ! {
            let start = unsafe {
                &__ets_test_suites_start as *const u8 as *const &::control_rs_ets::SuiteDescriptor
            };
            let end = unsafe {
                &__ets_test_suites_end as *const u8 as *const &::control_rs_ets::SuiteDescriptor
            };

            let suites = unsafe { ::control_rs_ets::util::get_suites(start, end) };

            #make_context
            let mut server = ::control_rs_ets::Server::with_board(context, suites, board);
            ETS_SERVER.set(&mut server as *mut _);

            let _ = server.run();
            ETS_SERVER.set(::core::ptr::null_mut());
            server.exit();
        }
    }
}

/// Helper macro to define the unified ETS entrypoint `main`.
#[proc_macro]
pub fn ets_entrypoint(input: TokenStream) -> TokenStream {
    TokenStream::from(ets_entrypoint_impl(input.into()))
}

/// Expand `ets_panic!()`.
fn ets_panic_impl() -> proc_macro2::TokenStream {
    quote! {
        // ==================== Unified Panic Handler ====================
        #[cfg(target_os = "none")]
        #[panic_handler]
        fn panic(info: &::core::panic::PanicInfo) -> ! {
            let mut msg_buf = [0u8; 128];
            let pos = {
                let mut writer = ::control_rs_ets::util::FailureBufWriter { buf: &mut msg_buf, pos: 0 };
                let _ = ::core::fmt::write(&mut writer, format_args!("{}", info.message()));
                writer.pos
            };
            let msg = ::core::str::from_utf8(&msg_buf[..pos]).unwrap_or("panic occurred");

            let file = info.location().map_or("unknown", |l| l.file());
            let line = info.location().map_or(0, |l| l.line());

            let server_ptr = ETS_SERVER.get();
            unsafe {
                if !server_ptr.is_null() {
                    let server = &mut *server_ptr;
                    let comms_ok = server.context.comms_lock.try_lock();
                    ::control_rs_ets::util::handle_failure(
                        &mut server.context,
                        msg,
                        file,
                        line,
                        comms_ok,
                    );
                } else {
                    loop {
                        ::core::hint::spin_loop();
                    }
                }
            }
        }
    }
}

/// Helper macro to define the target ETS panic handler.
#[proc_macro]
pub fn ets_panic(input: TokenStream) -> TokenStream {
    let _ = input;
    TokenStream::from(ets_panic_impl())
}

/// Expand `ets_exception!()`.
#[allow(clippy::too_many_lines)]
fn ets_exception_impl() -> proc_macro2::TokenStream {
    quote! {
        // ==================== Unified Exception Handler Implementation ====================
        #[cfg(all(target_os = "none", target_arch = "arm"))]
        #[::cortex_m_rt::exception]
        unsafe fn HardFault(ef: &::cortex_m_rt::ExceptionFrame) -> ! {
            let mut msg_buf = [0u8; 128];
            let pos = {
                let mut writer = ::control_rs_ets::util::FailureBufWriter { buf: &mut msg_buf, pos: 0 };
                let _ = ::core::fmt::write(
                    &mut writer,
                    format_args!("HardFault at pc=0x{:08x}, lr=0x{:08x}", ef.pc() as usize, ef.lr() as usize),
                );
                writer.pos
            };
            let msg = ::core::str::from_utf8(&msg_buf[..pos]).unwrap_or("exception occurred");

            let server_ptr = ETS_SERVER.get();
            if !server_ptr.is_null() {
                let server = unsafe { &mut *server_ptr };
                let comms_ok = server.context.comms_lock.try_lock();
                ::control_rs_ets::util::handle_exception(
                    &mut server.context,
                    msg,
                    comms_ok,
                );
            } else {
                loop {
                    ::core::hint::spin_loop();
                }
            }
        }

        #[cfg(all(target_os = "none", any(target_arch = "riscv32", target_arch = "riscv64")))]
        #[unsafe(no_mangle)]
        unsafe fn ExceptionHandler(_ef: &mut ::riscv_rt::TrapFrame) -> ! {
            let mut msg_buf = [0u8; 128];
            let pos = {
                let mut writer = ::control_rs_ets::util::FailureBufWriter { buf: &mut msg_buf, pos: 0 };
                let _ = ::core::fmt::write(
                    &mut writer,
                    format_args!(
                        "Exception mcause=0x{:08x}, mepc=0x{:08x}",
                        ::riscv::register::mcause::read().bits(),
                        ::riscv::register::mepc::read()
                    ),
                );
                writer.pos
            };
            let msg = ::core::str::from_utf8(&msg_buf[..pos]).unwrap_or("exception occurred");

            let server_ptr = ETS_SERVER.get();
            if !server_ptr.is_null() {
                let server = unsafe { &mut *server_ptr };
                let comms_ok = server.context.comms_lock.try_lock();
                ::control_rs_ets::util::handle_exception(
                    &mut server.context,
                    msg,
                    comms_ok,
                );
            } else {
                loop {
                    ::core::hint::spin_loop();
                }
            }
        }
    }
}

/// Helper macro to define the target ETS trap/exception handlers.
#[proc_macro]
#[allow(clippy::too_many_lines)]
pub fn ets_exception(input: TokenStream) -> TokenStream {
    let _ = input;
    TokenStream::from(ets_exception_impl())
}

#[cfg(test)]
mod tests {
    use super::{
        ets_entrypoint_impl, ets_exception_impl, ets_panic_impl,
        ets_setup_impl, ets_suite_impl, extract_context_generics,
        extract_doc_string, generate_suite_descriptors, get_param_constructor,
        packet_derive_impl, process_static_param, process_test_fn,
    };
    use syn::{ItemFn, ItemStatic, parse_quote};

    #[test]
    fn extract_doc_string_strips_and_truncates() {
        let item: ItemFn = parse_quote! {
            #[doc = " hello"]
            #[doc = "world"]
            fn sample() {}
        };
        assert_eq!(extract_doc_string(&item.attrs), "hello\nworld");

        let long: ItemFn = parse_quote! {
            #[doc = " aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]
            fn long_doc() {}
        };
        let extracted = extract_doc_string(&long.attrs);
        assert!(extracted.ends_with("..."));
        assert!(extracted.chars().count() <= 160);
    }

    #[test]
    fn extract_doc_string_keeps_exactly_160_chars() {
        let doc = "b".repeat(160);
        let item: ItemFn = parse_quote! {
            #[doc = #doc]
            fn boundary() {}
        };
        assert_eq!(extract_doc_string(&item.attrs), doc);
    }

    #[test]
    fn qualified_self_types_are_not_params() {
        let qualified: syn::Type = parse_quote!(<Wrapper as Trait>::u8);
        assert_eq!(get_param_constructor(&qualified), None);
    }

    #[test]
    fn param_constructors_cover_supported_types() {
        let u8_ty: syn::Type = parse_quote!(u8);
        let u16_ty: syn::Type = parse_quote!(u16);
        let u32_ty: syn::Type = parse_quote!(u32);
        let u64_ty: syn::Type = parse_quote!(u64);
        let i8_ty: syn::Type = parse_quote!(i8);
        let i32_ty: syn::Type = parse_quote!(i32);
        let bool_ty: syn::Type = parse_quote!(bool);
        let f32_ty: syn::Type = parse_quote!(f32);
        let other: syn::Type = parse_quote!(String);
        let tuple: syn::Type = parse_quote!((u8, u8));
        assert_eq!(get_param_constructor(&u8_ty), Some("u8"));
        assert_eq!(get_param_constructor(&u16_ty), Some("u16"));
        assert_eq!(get_param_constructor(&u32_ty), Some("u32"));
        assert_eq!(get_param_constructor(&u64_ty), None);
        assert_eq!(get_param_constructor(&i8_ty), None);
        assert_eq!(get_param_constructor(&i32_ty), Some("i32"));
        assert_eq!(get_param_constructor(&bool_ty), Some("bool"));
        assert_eq!(get_param_constructor(&f32_ty), Some("f32"));
        assert_eq!(get_param_constructor(&other), None);
        assert_eq!(get_param_constructor(&tuple), None);
    }

    #[test]
    fn process_test_fn_skips_helpers_and_keeps_docs() {
        let helper: ItemFn = parse_quote! {
            fn _hidden() {}
        };
        assert!(process_test_fn(&helper).is_none());
        let test_fn: ItemFn = parse_quote! {
            /// a case
            fn visible() {}
        };
        let (name, doc) = process_test_fn(&test_fn).unwrap();
        assert_eq!(name.to_string(), "visible");
        assert_eq!(doc, "a case");
    }

    /// Regression: helpers with arguments used to be registered as ETS tests
    /// (`fn()`), which failed QEMU compile (E0308) on PR #47.
    #[test]
    fn process_test_fn_skips_parameterized_helpers() {
        let helper: ItemFn = parse_quote! {
            fn inf_norm_from_identity(m: &u8) -> f64 {
                0.0
            }
        };
        assert!(process_test_fn(&helper).is_none());
        let generic: ItemFn = parse_quote! {
            fn assert_inv_identity_roundtrip<const N: usize>(a: &u8) {}
        };
        assert!(process_test_fn(&generic).is_none());
        let returning: ItemFn = parse_quote! {
            fn scale() -> f32 {
                1.0
            }
        };
        assert!(process_test_fn(&returning).is_none());
    }

    #[test]
    fn process_static_param_rewrites_supported_types() {
        let mut item: ItemStatic = parse_quote! {
            /// gain
            static GAIN: u32 = 3;
        };
        let ident = process_static_param(&mut item).unwrap();
        assert_eq!(ident.to_string(), "GAIN");
        let mut skip: ItemStatic = parse_quote! {
            static OTHER: f64 = 1.0;
        };
        assert!(process_static_param(&mut skip).is_none());
    }

    #[test]
    fn extract_context_generics_parses_context_path() {
        let ty: syn::Type = parse_quote!(Context<u8, bool>);
        let (c, p) = extract_context_generics(&ty).unwrap();
        assert_eq!(quote::quote!(#c).to_string(), "u8");
        assert_eq!(quote::quote!(#p).to_string(), "bool");
        let bad: syn::Type = parse_quote!(u32);
        assert!(extract_context_generics(&bad).is_none());
        let not_ctx: syn::Type = parse_quote!(Server<u8, bool>);
        assert!(extract_context_generics(&not_ctx).is_none());
        let missing_args: syn::Type = parse_quote!(Context);
        assert!(extract_context_generics(&missing_args).is_none());
        let one_arg: syn::Type = parse_quote!(Context<u8>);
        assert!(extract_context_generics(&one_arg).is_none());
        let reference: syn::Type = parse_quote!(&u8);
        assert!(extract_context_generics(&reference).is_none());
        let lifetime_first: syn::Type = parse_quote!(Context<'static, u8>);
        assert!(extract_context_generics(&lifetime_first).is_none());
        let lifetime_second: syn::Type = parse_quote!(Context<u8, 'static>);
        assert!(extract_context_generics(&lifetime_second).is_none());
    }

    #[test]
    fn generate_suite_descriptors_emits_wrappers_and_registry() {
        let foo = syn::Ident::new("foo", proc_macro2::Span::call_site());
        let gain = syn::Ident::new("GAIN", proc_macro2::Span::call_site());
        let items = generate_suite_descriptors(
            "suite",
            "docs",
            (&[(foo, "a test".to_string())], None),
            &[gain],
        );
        // One case wrapper, then the four registry items.
        assert_eq!(items.len(), 5);
    }

    #[test]
    fn ets_suite_impl_expands_inline_module_and_rejects_external() {
        let expanded = ets_suite_impl(quote::quote! {
            /// suite docs
            mod sample {
                /// gain
                static GAIN: u32 = 3;
                fn case_a() {}
                fn inf_norm_from_identity(m: &u8) -> f64 {
                    0.0
                }
                const SKIP: u8 = 0;
            }
        });
        let text = expanded.to_string();
        assert!(text.contains("SUITE_DESCRIPTOR"));
        assert!(text.contains("case_a"));
        assert!(
            !text.contains("__ets_case_inf_norm_from_identity"),
            "parameterized helpers must not be ETS executables: {text}"
        );

        let err = ets_suite_impl(quote::quote! {
            mod external;
        });
        assert!(err.to_string().contains("inline modules"));
        assert!(
            ets_suite_impl(quote::quote! { 123 })
                .to_string()
                .contains("expected")
        );
    }

    #[test]
    fn lifecycle_impl_expands_to_slot_and_wrappers() {
        let text = ets_suite_impl(quote::quote! {
            mod task_suite {
                /// a task
                impl LifecycleTask for Loop {}
            }
        })
        .to_string();
        assert!(text.contains("TaskSlot < Loop >"));
        for wrapper in [
            "__ets_task_setup",
            "__ets_task_step",
            "__ets_task_reset",
            "__ets_task_teardown",
            "__ets_task_safe_state",
        ] {
            assert!(text.contains(wrapper), "{wrapper}");
        }
        assert!(text.contains("task_descriptor"));
        let two = ets_suite_impl(quote::quote! {
            mod two {
                impl LifecycleTask for A {}
                impl LifecycleTask for B {}
            }
        })
        .to_string();
        assert!(two.contains("at most one lifecycle task per suite"));
    }

    #[test]
    fn suite_statics_must_be_supported_params() {
        let text = ets_suite_impl(quote::quote! {
            mod bad {
                static RATE: f64 = 1.0;
            }
        })
        .to_string();
        assert!(text.contains("compile_error"));
    }

    #[test]
    fn param_bounds_are_emitted() {
        let mut item: ItemStatic = parse_quote! {
            #[param(min = 1, max = 9)]
            static GAIN: u8 = 3;
        };
        let _ = process_static_param(&mut item).unwrap();
        let text = quote::quote!(#item).to_string();
        assert!(text.contains("bounded"));
        assert!(text.contains("Param :: U8 (1)"));
        assert!(!text.contains("# [param"));
    }

    #[test]
    fn packet_derive_emits_info_and_codec() {
        let text = packet_derive_impl(quote::quote! {
            struct Sample { x: f32 }
        })
        .to_string();
        assert!(text.contains("Packet for Sample"));
        assert!(text.contains("postcard_decode"));
        assert!(text.contains("PayloadEncoding :: Postcard"));
        assert!(
            packet_derive_impl(quote::quote!(1))
                .to_string()
                .contains("expected")
        );
    }

    #[test]
    fn setup_accepts_a_board_binding() {
        let text = ets_setup_impl(quote::quote! {
            fn setup() -> (Context<u8, bool>, Timer) {
                unimplemented!()
            }
        })
        .to_string();
        assert!(text.contains("Timer"));
        assert!(text.contains("board"));
        let entry =
            ets_entrypoint_impl(quote::quote!(setup, board)).to_string();
        assert!(entry.contains("let (context , board) = setup ()"));
        let three = ets_setup_impl(quote::quote! {
            fn setup() -> (Context<u8, bool>, u8, u8) { unimplemented!() }
        });
        assert!(three.to_string().contains("compile_error"));
    }

    #[test]
    fn ets_setup_and_helper_macros_expand() {
        let setup = ets_setup_impl(quote::quote! {
            fn setup() -> Context<u8, bool> {
                unimplemented!()
            }
        });
        let setup_text = setup.to_string();
        assert!(setup_text.contains("ets_entrypoint"));
        assert!(setup_text.contains("ets_panic"));

        let entry = ets_entrypoint_impl(quote::quote!(setup));
        assert!(entry.to_string().contains("fn main"));
        assert!(
            ets_entrypoint_impl(quote::quote!(123))
                .to_string()
                .contains("expected identifier")
        );

        assert!(ets_panic_impl().to_string().contains("panic_handler"));
        assert!(ets_exception_impl().to_string().contains("HardFault"));
        assert!(
            ets_setup_impl(quote::quote! { 0 })
                .to_string()
                .contains("expected")
        );
        let non_ctx = ets_setup_impl(quote::quote! {
            fn setup() -> u32 { 0 }
        });
        assert!(non_ctx.to_string().contains("compile_error"));
        let no_ret = ets_setup_impl(quote::quote! {
            fn setup() {}
        });
        assert!(no_ret.to_string().contains("compile_error"));
    }
}
