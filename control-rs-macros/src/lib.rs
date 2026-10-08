//! Procedural macros for `control-rs` Embedded Test Server (ETS) firmware.
//!
//! | Macro | Generates |
//! |:--|:--|
//! | `#[ets_suite]` | Suite descriptor and case registration in `.ets_test_suites`; a `#[setup]`, `#[step]`, `#[reset]`, `#[teardown]` lifecycle case in `.ets_loops` |
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

/// Names of the lifecycle markers, in declaration order.
const LOOP_MARKERS: [&str; 4] = ["setup", "step", "reset", "teardown"];

/// The three items generated for a loop: step wrapper, descriptor, section pointer.
type LoopItems = [Item; 3];

/// The function identifier of each marker, in `LOOP_MARKERS` order.
type MarkerSlots = [Option<syn::Ident>; 4];

/// Input and output packet types of a typed step.
type PacketTypes = (syn::Type, syn::Type);

/// Type alias for a test function's identifier and its description.
type TestFnInfo = (syn::Ident, String);

/// A lifecycle marker removed from a function: its slot and the attribute.
type TakenMarker = (usize, syn::Attribute);

/// The functions marked as the lifecycle case of a suite.
#[derive(Default)]
struct LoopFns {
    /// Slots in `LOOP_MARKERS` order.
    idents: MarkerSlots,
    /// Description of the loop: the step function's doc comment.
    doc: String,
    /// Host link timeout from `#[step(link_timeout_ms = N)]`.
    link_timeout_ms: u32,
    /// Input and output packet types of the typed step.
    packets: Option<PacketTypes>,
}

/// What `#[ets_suite]` collects from the items of a suite module.
struct SuiteParts {
    errors: Option<syn::Error>,
    loop_fns: LoopFns,
    settings: Vec<syn::Ident>,
    tests: Vec<TestFnInfo>,
}

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

/// Checks if a given `syn::Type` is a supported primitive setting.
/// Returns the corresponding Atomic wrapper type name if matched.
fn get_atomic_wrapper_name(ty: &syn::Type) -> Option<&'static str> {
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
        "u8" => Some("AtomicU8Setting"),
        "u16" => Some("AtomicU16Setting"),
        "u32" => Some("AtomicU32Setting"),
        "u64" => Some("AtomicU64Setting"),
        "i8" => Some("AtomicI8Setting"),
        "i32" => Some("AtomicI32Setting"),
        "bool" => Some("AtomicBoolSetting"),
        "f32" => Some("AtomicF32Setting"),
        _ => None,
    }
}

/// If the static item matches a supported atomic setting type,
/// mutates it into the corresponding `AtomicSetting` static definition and returns its identifier.
fn process_static_setting(item_static: &mut ItemStatic) -> Option<syn::Ident> {
    // 1. Search phase: Delegate to the helper function
    let atomic_type_str_option = get_atomic_wrapper_name(&item_static.ty);

    // 2. Mutation phase: If matched, clone required fields and overwrite
    if let Some(atomic_type_str) = atomic_type_str_option {
        let vis = item_static.vis.clone();
        let name_ident = item_static.ident.clone();
        let init_expr = item_static.expr.clone();
        let attrs = item_static.attrs.clone();

        let setting_doc = extract_doc_string(&attrs);
        let atomic_type_ident = format_ident!("{}", atomic_type_str);

        let new_static: ItemStatic = parse_quote! {
            #(#attrs)*
            #vis static #name_ident: ::control_rs_ets::settings::#atomic_type_ident =
                ::control_rs_ets::settings::#atomic_type_ident::new(stringify!(#name_ident), #setting_doc, #init_expr);
        };

        *item_static = new_static;

        return Some(name_ident);
    }

    None
}

/// If the function is a test executable (`fn()` and does not start with `_`),
/// extracts its identifier and doc comments.
///
/// Parameterized functions are helpers: `ExecDescriptor::test_fn` is `fn()`, so
/// registering them fails QEMU/`no_std` compilation with E0308.
fn process_test_fn(item_fn: &ItemFn) -> Option<TestFnInfo> {
    let fn_name = &item_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    if fn_name_str.starts_with('_') || !item_fn.sig.inputs.is_empty() {
        None
    } else {
        let test_doc = extract_doc_string(&item_fn.attrs);
        Some((fn_name.clone(), test_doc))
    }
}

/// Removes the lifecycle marker from `item_fn` and returns its slot in
/// `LOOP_MARKERS` with the attribute.
fn take_loop_marker(item_fn: &mut ItemFn) -> Option<TakenMarker> {
    let (slot, pos) =
        item_fn.attrs.iter().enumerate().find_map(|(pos, a)| {
            LOOP_MARKERS
                .iter()
                .position(|m| a.path().is_ident(m))
                .map(|slot| (slot, pos))
        })?;
    Some((slot, item_fn.attrs.remove(pos)))
}

/// Adds `error` to the combined diagnostics.
fn push_error(errors: &mut Option<syn::Error>, error: syn::Error) {
    match errors.as_mut() {
        Some(all) => all.combine(error),
        None => *errors = Some(error),
    }
}

/// Normalizes a token stream for signature comparison.
fn squash(tokens: &impl quote::ToTokens) -> String {
    quote!(#tokens).to_string().replace(' ', "")
}

/// Checks `fn() -> Result<(), &'static str>`.
fn check_result_signature(item_fn: &ItemFn) -> syn::Result<()> {
    let ret = match &item_fn.sig.output {
        syn::ReturnType::Type(_, ty) => squash(ty),
        syn::ReturnType::Default => String::new(),
    };
    if item_fn.sig.inputs.is_empty() && ret == "Result<(),&'staticstr>" {
        Ok(())
    } else {
        Err(syn::Error::new_spanned(
            &item_fn.sig,
            "expected `fn() -> Result<(), &'static str>`",
        ))
    }
}

/// The last type argument of the last path segment named `name`.
fn last_type_arg(ty: &syn::Type, name: &str) -> Option<syn::Type> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    let segment = path.path.segments.last()?;
    if segment.ident != name {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    args.args.iter().rev().find_map(|a| match a {
        syn::GenericArgument::Type(t) => Some(t.clone()),
        _ => None,
    })
}

/// Checks `fn(&LoopContext<'_, I>) -> LoopStatus<O>` and returns `(I, O)`.
fn check_step_signature(item_fn: &ItemFn) -> syn::Result<PacketTypes> {
    let error = || {
        syn::Error::new_spanned(
            &item_fn.sig,
            "expected `fn(&LoopContext<'_, I>) -> LoopStatus<O>`",
        )
    };
    let mut inputs = item_fn.sig.inputs.iter();
    let (Some(syn::FnArg::Typed(arg)), None) = (inputs.next(), inputs.next())
    else {
        return Err(error());
    };
    let syn::Type::Reference(reference) = arg.ty.as_ref() else {
        return Err(error());
    };
    let input =
        last_type_arg(&reference.elem, "LoopContext").ok_or_else(error)?;
    let syn::ReturnType::Type(_, ret) = &item_fn.sig.output else {
        return Err(error());
    };
    let output = last_type_arg(ret, "LoopStatus").ok_or_else(error)?;
    Ok((input, output))
}

/// Parses the arguments of `#[step]`: only `link_timeout_ms = N`, with `N`
/// zero or at least 500.
fn parse_step_args(attr: &syn::Attribute) -> syn::Result<u32> {
    let syn::Meta::List(list) = &attr.meta else {
        return Ok(0);
    };
    let assign: syn::MetaNameValue = list.parse_args()?;
    let literal = match &assign.value {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(int),
            ..
        }) if assign.path.is_ident("link_timeout_ms") => int,
        _ => {
            return Err(syn::Error::new_spanned(
                &assign,
                "expected `link_timeout_ms = N`",
            ));
        }
    };
    let value: u32 = literal.base10_parse()?;
    if (1..500).contains(&value) {
        return Err(syn::Error::new_spanned(
            literal,
            "link_timeout_ms must be 0 or at least 500",
        ));
    }
    Ok(value)
}

/// Records a marked function in `loop_fns`, or fails on a repeat or a
/// malformed signature.
fn record_loop_fn(
    loop_fns: &mut LoopFns,
    suite_name: &str,
    (slot, attr): &TakenMarker,
    item_fn: &ItemFn,
) -> syn::Result<()> {
    let slot = *slot;
    let marker = LOOP_MARKERS.get(slot).copied().unwrap_or("");
    if let Some(first) = loop_fns.idents.get(slot).and_then(Option::as_ref) {
        return Err(syn::Error::new_spanned(
            &item_fn.sig.ident,
            format!(
                "suite `{suite_name}` already has a `#[{marker}]` function: `{first}`"
            ),
        ));
    }
    if marker == "step" {
        loop_fns.packets = Some(check_step_signature(item_fn)?);
        loop_fns.link_timeout_ms = parse_step_args(attr)?;
        loop_fns.doc = extract_doc_string(&item_fn.attrs);
    } else {
        check_result_signature(item_fn)?;
    }
    if let Some(entry) = loop_fns.idents.get_mut(slot) {
        *entry = Some(item_fn.sig.ident.clone());
    }
    Ok(())
}

/// Generates the loop step wrapper, descriptor and section pointer.
fn generate_loop_descriptors(loop_fns: &LoopFns) -> Option<LoopItems> {
    let [Some(setup), Some(step), Some(reset), Some(teardown)] =
        &loop_fns.idents
    else {
        return None;
    };
    let (input, output) = loop_fns.packets.as_ref()?;
    let doc = &loop_fns.doc;
    let timeout = loop_fns.link_timeout_ms;
    Some([
        parse_quote! {
            fn __ets_loop_step(
                io: &mut ::control_rs_ets::LoopIo<'_>,
            ) -> ::control_rs_ets::LoopOutcome {
                ::control_rs_ets::loop_step::<#input, #output>(io, #step)
            }
        },
        parse_quote! {
            static LOOP_DESCRIPTOR: ::control_rs_ets::LoopDescriptor = ::control_rs_ets::LoopDescriptor {
                suite: &SUITE_DESCRIPTOR,
                name: stringify!(#step),
                description: #doc,
                input_type: stringify!(#input),
                output_type: stringify!(#output),
                setup: #setup,
                step: __ets_loop_step,
                reset: #reset,
                teardown: #teardown,
                link_timeout_ms: #timeout,
            };
        },
        parse_quote! {
            /// Pointer to the loop descriptor, linked into the ETS loops section.
            #[cfg_attr(target_vendor = "apple", unsafe(link_section = "__DATA,__ets_loops"))]
            #[cfg_attr(not(target_vendor = "apple"), unsafe(link_section = ".ets_loops"))]
            #[used]
            pub static LOOP_DESCRIPTOR_PTR: &::control_rs_ets::LoopDescriptor = &LOOP_DESCRIPTOR;
        },
    ])
}

/// Generates the static descriptor items to append to the module.
fn generate_suite_descriptors(
    suite_name: &str,
    suite_doc: &str,
    tests: &[TestFnInfo],
    settings: &[syn::Ident],
) -> [Item; 4] {
    let test_descriptors = tests.iter().map(|(t, doc)| {
        quote! {
            ::control_rs_ets::ExecDescriptor {
                name: stringify!(#t),
                description: #doc,
                test_fn: #t,
            }
        }
    });

    let setting_ptrs = settings.iter().map(|s| {
        quote! {
            &#s
        }
    });

    [
        parse_quote! {
            static EXECUTABLES: &[::control_rs_ets::ExecDescriptor] = &[
                #(#test_descriptors),*
            ];
        },
        parse_quote! {
            static SETTINGS: &[&dyn ::control_rs_ets::Setting] = &[
                #(#setting_ptrs),*
            ];
        },
        parse_quote! {
            static SUITE_DESCRIPTOR: ::control_rs_ets::SuiteDescriptor = ::control_rs_ets::SuiteDescriptor {
                name: #suite_name,
                description: #suite_doc,
                executables: EXECUTABLES,
                settings: SETTINGS,
            };
        },
        parse_quote! {
            /// Pointer to the suite descriptor, linked into the ETS test suites section.
            #[cfg_attr(target_vendor = "apple", unsafe(link_section = "__DATA,__ets_suites"))]
            #[cfg_attr(not(target_vendor = "apple"), unsafe(link_section = ".ets_test_suites"))]
            #[used]
            pub static SUITE_DESCRIPTOR_PTR: &::control_rs_ets::SuiteDescriptor = &SUITE_DESCRIPTOR;
        },
    ]
}

/// Walks the module items: settings, cases and the lifecycle markers.
fn collect_parts(suite_name: &str, items: &mut [Item]) -> SuiteParts {
    let mut parts = SuiteParts {
        errors: None,
        loop_fns: LoopFns::default(),
        settings: Vec::new(),
        tests: Vec::new(),
    };
    for inner_item in items {
        match inner_item {
            Item::Static(item_static) => {
                if let Some(setting) = process_static_setting(item_static) {
                    parts.settings.push(setting);
                }
            }
            Item::Fn(item_fn) => {
                if let Some(marker) = take_loop_marker(item_fn) {
                    if let Err(e) = record_loop_fn(
                        &mut parts.loop_fns,
                        suite_name,
                        &marker,
                        item_fn,
                    ) {
                        push_error(&mut parts.errors, e);
                    }
                } else if let Some(test) = process_test_fn(item_fn) {
                    parts.tests.push(test);
                }
            }
            _ => {}
        }
    }
    parts
}

/// The error for a lifecycle case that lacks some of its four markers.
fn missing_marker_error(
    suite_name: &str,
    loop_fns: &LoopFns,
) -> Option<syn::Error> {
    let missing: Vec<String> = LOOP_MARKERS
        .iter()
        .zip(&loop_fns.idents)
        .filter(|(_, ident)| ident.is_none())
        .map(|(m, _)| format!("`#[{m}]`"))
        .collect();
    let any_marked = loop_fns.idents.iter().any(Option::is_some);
    (any_marked && !missing.is_empty()).then(|| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            format!(
                "suite `{suite_name}` defines a lifecycle case but is missing {}",
                missing.join(", ")
            ),
        )
    })
}

/// Appends the descriptors to the items of a suite module; returns the
/// diagnostics for a malformed lifecycle case.
fn expand_items(
    suite_name: &str,
    suite_doc: &str,
    items: &mut Vec<Item>,
) -> Option<syn::Error> {
    let mut parts = collect_parts(suite_name, items);
    if parts.errors.is_none()
        && let Some(e) = missing_marker_error(suite_name, &parts.loop_fns)
    {
        push_error(&mut parts.errors, e);
    }

    items.extend(generate_suite_descriptors(
        suite_name,
        suite_doc,
        &parts.tests,
        &parts.settings,
    ));
    if parts.errors.is_none()
        && let Some(loop_code) = generate_loop_descriptors(&parts.loop_fns)
    {
        items.extend(loop_code);
    }
    parts.errors
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

    let errors = item_mod
        .content
        .as_mut()
        .and_then(|(_, items)| expand_items(&suite_name, &suite_doc, items));

    let compile_errors = errors.map(|e| e.to_compile_error());
    quote! {
        #item_mod
        #compile_errors
    }
}

/// Attribute macro for declaring a ETS test suite.
///
/// Converts statics to atomic settings and registers functions as test executables.
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

/// Expand `#[ets_setup]` for a function returning `Context<C, P>`.
fn ets_setup_impl(item: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let setup_fn: ItemFn = match syn::parse2(item) {
        Ok(f) => f,
        Err(e) => return e.to_compile_error(),
    };
    let setup_name = &setup_fn.sig.ident;

    let syn::ReturnType::Type(_, return_type) = &setup_fn.sig.output else {
        return syn::Error::new_spanned(
            &setup_fn.sig,
            "setup function must return a Context<C, P>",
        )
        .to_compile_error();
    };

    let Some((c_ty, p_ty)) = extract_context_generics(return_type) else {
        return syn::Error::new_spanned(
            return_type,
            "setup function return type must be Context<C, P>",
        )
        .to_compile_error();
    };

    quote! {
        #setup_fn

        static ETS_SERVER: ::core::sync::atomic::AtomicPtr<::control_rs_ets::Server<'static, #c_ty, #p_ty>> =
            ::core::sync::atomic::AtomicPtr::new(::core::ptr::null_mut());

        ::control_rs_macros::ets_entrypoint!(#setup_name);
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
    let setup_name: syn::Ident = match syn::parse2(input) {
        Ok(i) => i,
        Err(e) => return e.to_compile_error(),
    };
    quote! {
        #[cfg(target_os = "none")]
        unsafe extern "Rust" {
            static __ets_test_suites_start: u8;
            static __ets_test_suites_end: u8;
            static __ets_loops_start: u8;
            static __ets_loops_end: u8;
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

            let loops_start = unsafe {
                &__ets_loops_start as *const u8 as *const &::control_rs_ets::LoopDescriptor
            };
            let loops_end = unsafe {
                &__ets_loops_end as *const u8 as *const &::control_rs_ets::LoopDescriptor
            };
            let loops = unsafe { ::control_rs_ets::util::get_loops(loops_start, loops_end) };

            let context = #setup_name();
            let mut server = ::control_rs_ets::Server::new(context, suites).with_loops(loops);
            ETS_SERVER.store(&mut server as *mut _, ::core::sync::atomic::Ordering::Release);

            let _ = server.run();
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

            let server_ptr = ETS_SERVER.load(::core::sync::atomic::Ordering::Acquire);
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

            let server_ptr = ETS_SERVER.load(::core::sync::atomic::Ordering::Acquire);
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

            let server_ptr = ETS_SERVER.load(::core::sync::atomic::Ordering::Acquire);
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
        extract_doc_string, generate_suite_descriptors,
        get_atomic_wrapper_name, process_static_setting, process_test_fn,
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
    fn qualified_self_types_are_not_settings() {
        let qualified: syn::Type = parse_quote!(<Wrapper as Trait>::u8);
        assert_eq!(get_atomic_wrapper_name(&qualified), None);
    }

    #[test]
    fn atomic_wrapper_names_cover_supported_types() {
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
        assert_eq!(get_atomic_wrapper_name(&u8_ty), Some("AtomicU8Setting"));
        assert_eq!(get_atomic_wrapper_name(&u16_ty), Some("AtomicU16Setting"));
        assert_eq!(get_atomic_wrapper_name(&u32_ty), Some("AtomicU32Setting"));
        assert_eq!(get_atomic_wrapper_name(&u64_ty), Some("AtomicU64Setting"));
        assert_eq!(get_atomic_wrapper_name(&i8_ty), Some("AtomicI8Setting"));
        assert_eq!(get_atomic_wrapper_name(&i32_ty), Some("AtomicI32Setting"));
        assert_eq!(
            get_atomic_wrapper_name(&bool_ty),
            Some("AtomicBoolSetting")
        );
        assert_eq!(get_atomic_wrapper_name(&f32_ty), Some("AtomicF32Setting"));
        assert_eq!(get_atomic_wrapper_name(&other), None);
        assert_eq!(get_atomic_wrapper_name(&tuple), None);
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
    }

    #[test]
    fn process_static_setting_rewrites_supported_types() {
        let mut item: ItemStatic = parse_quote! {
            /// gain
            static GAIN: u32 = 3;
        };
        let ident = process_static_setting(&mut item).unwrap();
        assert_eq!(ident.to_string(), "GAIN");
        let mut skip: ItemStatic = parse_quote! {
            static OTHER: f64 = 1.0;
        };
        assert!(process_static_setting(&mut skip).is_none());
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
    fn generate_suite_descriptors_emits_four_items() {
        let foo = syn::Ident::new("foo", proc_macro2::Span::call_site());
        let gain = syn::Ident::new("GAIN", proc_macro2::Span::call_site());
        let items = generate_suite_descriptors(
            "suite",
            "docs",
            &[(foo, "a test".to_string())],
            &[gain],
        );
        assert_eq!(items.len(), 4);
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
            !text.contains("test_fn : inf_norm_from_identity")
                && !text.contains("test_fn: inf_norm_from_identity"),
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

    /// The expansion of `suite` as a whitespace-free string.
    fn expand(suite: proc_macro2::TokenStream) -> String {
        ets_suite_impl(suite).to_string().replace(' ', "")
    }

    #[test]
    fn loop_expands_descriptor() {
        let out = expand(quote::quote! {
            /// Motor suite.
            mod motor {
                static KP: u32 = 5;
                fn case_a() {}
                fn case_b() {}
                #[setup]
                fn setup() -> Result<(), &'static str> { Ok(()) }
                /// Speed loop.
                #[step(link_timeout_ms = 1000)]
                fn speed_loop(ctx: &LoopContext<'_, f32>) -> LoopStatus<u8> { todo!() }
                #[reset]
                fn reset() -> Result<(), &'static str> { Ok(()) }
                #[teardown]
                fn teardown() -> Result<(), &'static str> { Ok(()) }
            }
        });
        assert_eq!(out.matches("ExecDescriptor{").count(), 2);
        assert!(out.contains("LoopDescriptor{suite:&SUITE_DESCRIPTOR"));
        assert!(out.contains("name:stringify!(speed_loop)"));
        assert!(out.contains("description:\"Speedloop.\""));
        assert!(out.contains("input_type:stringify!(f32)"));
        assert!(out.contains("output_type:stringify!(u8)"));
        assert!(out.contains(
            "setup:setup,step:__ets_loop_step,reset:reset,teardown:teardown"
        ));
        assert!(out.contains("link_timeout_ms:1000u32"));
        assert!(out.contains("loop_step::<f32,u8>(io,speed_loop)"));
        assert!(out.contains("link_section=\".ets_loops\""));
        assert!(!out.contains("compile_error"));

        let entry = ets_entrypoint_impl(quote::quote!(setup))
            .to_string()
            .replace(' ', "");
        assert!(entry.contains(".with_loops(loops)"));

        let suite_only = expand(quote::quote! {
            mod plain { fn case_a() {} }
        });
        assert!(!suite_only.contains("LoopDescriptor"));
    }

    #[test]
    fn loop_diagnostics_name_the_defect() {
        let missing = expand(quote::quote! {
            mod motor {
                #[setup]
                fn setup() -> Result<(), &'static str> { Ok(()) }
                #[step]
                fn s(ctx: &LoopContext<'_, ()>) -> LoopStatus<()> { todo!() }
            }
        });
        assert!(missing.contains(
            "suite`motor`definesalifecyclecasebutismissing`#[reset]`,`#[teardown]`"
        ));

        let bad_setup = expand(quote::quote! {
            mod motor {
                #[setup]
                fn setup() -> u8 { 0 }
                #[step]
                fn s(ctx: &LoopContext<'_, ()>) -> LoopStatus<()> { todo!() }
                #[reset]
                fn reset() -> Result<(), &'static str> { Ok(()) }
                #[teardown]
                fn teardown() -> Result<(), &'static str> { Ok(()) }
            }
        });
        assert!(bad_setup.contains("expected`fn()->Result<(),&'staticstr>`"));

        let bad_step = expand(quote::quote! {
            mod motor {
                #[setup]
                fn setup() -> Result<(), &'static str> { Ok(()) }
                #[step]
                fn s() {}
                #[reset]
                fn reset() -> Result<(), &'static str> { Ok(()) }
                #[teardown]
                fn teardown() -> Result<(), &'static str> { Ok(()) }
            }
        });
        assert!(
            bad_step
                .contains("expected`fn(&LoopContext<'_,I>)->LoopStatus<O>`")
        );

        let bad_timeout = expand(quote::quote! {
            mod motor {
                #[setup]
                fn setup() -> Result<(), &'static str> { Ok(()) }
                #[step(link_timeout_ms = 499)]
                fn s(ctx: &LoopContext<'_, ()>) -> LoopStatus<()> { todo!() }
                #[reset]
                fn reset() -> Result<(), &'static str> { Ok(()) }
                #[teardown]
                fn teardown() -> Result<(), &'static str> { Ok(()) }
            }
        });
        assert!(bad_timeout.contains("link_timeout_msmustbe0oratleast500"));
    }

    #[test]
    fn second_loop_in_suite_rejected() {
        for marker in ["setup", "step", "reset", "teardown"] {
            let marker = quote::format_ident!("{}", marker);
            let (sig_a, sig_b) = if marker == "step" {
                (
                    quote::quote!(
                        fn first(ctx: &LoopContext<'_, ()>) -> LoopStatus<()> {
                            todo!()
                        }
                    ),
                    quote::quote!(
                        fn second(ctx: &LoopContext<'_, ()>) -> LoopStatus<()> {
                            todo!()
                        }
                    ),
                )
            } else {
                (
                    quote::quote!(
                        fn first() -> Result<(), &'static str> {
                            Ok(())
                        }
                    ),
                    quote::quote!(
                        fn second() -> Result<(), &'static str> {
                            Ok(())
                        }
                    ),
                )
            };
            let out = expand(quote::quote! {
                mod motor {
                    #[#marker] #sig_a
                    #[#marker] #sig_b
                }
            });
            assert!(
                out.contains(&format!(
                    "suite`motor`alreadyhasa`#[{marker}]`function:`first`"
                )),
                "{marker}: {out}"
            );
        }
    }

    #[test]
    fn loop_step_wrapper_output_overflow() {
        use control_rs_ets::{LoopContext, LoopIo, LoopRunState, LoopStatus};
        fn big(_: &LoopContext<'_, ()>) -> LoopStatus<u64> {
            LoopStatus::Running(u64::MAX)
        }
        let mut out = [0u8; 4];
        let mut io = LoopIo {
            input: None,
            input_seq: None,
            output: &mut out,
            output_len: 0,
            step: 0,
        };
        let outcome = control_rs_ets::loop_step::<(), u64>(&mut io, big);
        assert_eq!(outcome.status, LoopRunState::Error);
        assert_eq!(outcome.message, Some("output overflow"));
        assert!(expand(quote::quote! {
            mod m {
                #[setup] fn a() -> Result<(), &'static str> { Ok(()) }
                #[step] fn s(ctx: &LoopContext<'_, ()>) -> LoopStatus<u64> { todo!() }
                #[reset] fn b() -> Result<(), &'static str> { Ok(()) }
                #[teardown] fn c() -> Result<(), &'static str> { Ok(()) }
            }
        })
        .contains("loop_step::<(),u64>(io,s)"));
    }

    #[test]
    fn loop_step_wrapper_input_decode() {
        use control_rs_ets::{LoopContext, LoopIo, LoopRunState, LoopStatus};
        use core::sync::atomic::{AtomicBool, Ordering};
        static CALLED: AtomicBool = AtomicBool::new(false);
        fn spy(_: &LoopContext<'_, f32>) -> LoopStatus<()> {
            CALLED.store(true, Ordering::SeqCst);
            LoopStatus::Running(())
        }
        let mut out = [0u8; 4];
        let mut io = LoopIo {
            input: Some(&[]),
            input_seq: Some(0),
            output: &mut out,
            output_len: 0,
            step: 0,
        };
        let outcome = control_rs_ets::loop_step::<f32, ()>(&mut io, spy);
        assert_eq!(outcome.status, LoopRunState::Error);
        assert_eq!(outcome.message, Some("input decode"));
        assert!(!CALLED.load(Ordering::SeqCst));
    }
}
