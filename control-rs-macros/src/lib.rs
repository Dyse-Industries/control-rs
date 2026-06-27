//! Procedural macros for the control-rs HIL testing framework.
//! Provides attributes like `#[hil_suite]` and `#[hil_setup]` to declare HIL test suites and setup functions.

#![allow(
    unused_extern_crates,
    clippy::uninlined_format_args,
    clippy::missing_panics_doc,
    clippy::panic,
    clippy::expect_used,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants
)]

extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Item, ItemFn, ItemMod, ItemStatic, Type, parse_macro_input, parse_quote,
};

/// Type alias for a test function's identifier and its description.
type TestFnInfo = (syn::Ident, String);

/// Checks if the given type matches the target type name.
fn is_type_name(ty: &Type, name: &str) -> bool {
    if let Type::Path(type_path) = ty {
        type_path
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == name)
    } else {
        false
    }
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
        format!("{}...", truncated)
    } else {
        full_doc.to_string()
    }
}

/// Checks if a given `syn::Type` is a supported primitive setting.
/// Returns the corresponding Atomic wrapper type name if matched.
fn get_atomic_wrapper_name(ty: &syn::Type) -> Option<&'static str> {
    let supported_types = [
        ("u8", "AtomicU8Setting"),
        ("u16", "AtomicU16Setting"),
        ("u32", "AtomicU32Setting"),
        ("u64", "AtomicU64Setting"),
        ("i8", "AtomicI8Setting"),
        ("i32", "AtomicI32Setting"),
        ("bool", "AtomicBoolSetting"),
        ("f32", "AtomicF32Setting"),
    ];

    for (prim_type, atomic_type_str) in supported_types {
        if is_type_name(ty, prim_type) {
            return Some(atomic_type_str);
        }
    }

    None
}

/// If the static item matches a supported atomic setting type (u32 or u8),
/// mutates it into the corresponding `AtomicSetting` static definition and returns its identifier.
fn process_static_setting(item_static: &mut ItemStatic) -> Option<syn::Ident> {
    // 1. Search phase: Delegate to the helper function
    let atomic_type_str_option = get_atomic_wrapper_name(&item_static.ty);

    // 2. Mutation phase: If matched, clone what we need and overwrite
    if let Some(atomic_type_str) = atomic_type_str_option {
        let name_ident = item_static.ident.clone();
        let init_expr = item_static.expr.clone();
        let attrs = item_static.attrs.clone();

        let setting_doc = extract_doc_string(&attrs);
        let atomic_type_ident = format_ident!("{}", atomic_type_str);

        let new_static: ItemStatic = parse_quote! {
            #(#attrs)*
            pub static #name_ident: ::control_rs_hil::settings::#atomic_type_ident =
                ::control_rs_hil::settings::#atomic_type_ident::new(stringify!(#name_ident), #setting_doc, #init_expr);
        };

        *item_static = new_static;

        return Some(name_ident);
    }

    None
}

/// If the function is a test executable (does not start with `_`),
/// extracts its identifier and doc comments.
fn process_test_fn(item_fn: &ItemFn) -> Option<TestFnInfo> {
    let fn_name = &item_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    if fn_name_str.starts_with('_') {
        None
    } else {
        let test_doc = extract_doc_string(&item_fn.attrs);
        Some((fn_name.clone(), test_doc))
    }
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
            ::control_rs_hil::ExecDescriptor {
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
            static EXECUTABLES: &[::control_rs_hil::ExecDescriptor] = &[
                #(#test_descriptors),*
            ];
        },
        parse_quote! {
            static SETTINGS: &[&dyn ::control_rs_hil::Setting] = &[
                #(#setting_ptrs),*
            ];
        },
        parse_quote! {
            static SUITE_DESCRIPTOR: ::control_rs_hil::SuiteDescriptor = ::control_rs_hil::SuiteDescriptor {
                name: #suite_name,
                description: #suite_doc,
                executables: EXECUTABLES,
                settings: SETTINGS,
            };
        },
        parse_quote! {
            /// Pointer to the suite descriptor, linked into the HIL test suites section.
            #[unsafe(link_section = ".hil_test_suites")]
            #[used]
            pub static SUITE_DESCRIPTOR_PTR: &::control_rs_hil::SuiteDescriptor = &SUITE_DESCRIPTOR;
        },
    ]
}

/// Attribute macro for declaring a HIL test suite.
///
/// Converts statics to atomic settings and registers functions as test executables.
#[proc_macro_attribute]
pub fn hil_suite(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut item_mod = parse_macro_input!(item as ItemMod);
    let suite_name = item_mod.ident.to_string();
    let suite_doc = extract_doc_string(&item_mod.attrs);

    let mut tests = Vec::new();
    let mut settings = Vec::new();

    if let Some((_, ref mut items)) = item_mod.content {
        for inner_item in items.iter_mut() {
            match inner_item {
                Item::Static(item_static) => {
                    if let Some(setting) = process_static_setting(item_static) {
                        settings.push(setting);
                    }
                }
                Item::Fn(item_fn) => {
                    if let Some(test) = process_test_fn(item_fn) {
                        tests.push(test);
                    }
                }
                _ => {}
            }
        }

        let suite_desc_code = generate_suite_descriptors(
            &suite_name,
            &suite_doc,
            &tests,
            &settings,
        );
        items.extend(suite_desc_code);
    }

    let expanded = quote! {
        #item_mod
    };

    TokenStream::from(expanded)
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

/// Attribute macro for setting up the HIL server entrypoint.
///
/// Annotates the hardware setup function, generates the standard main entrypoint,
/// and sets up the server event loop and QEMU-compatible panic handler.
#[proc_macro_attribute]
pub fn hil_setup(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let setup_fn = parse_macro_input!(item as ItemFn);
    let setup_name = &setup_fn.sig.ident;

    let return_type = match &setup_fn.sig.output {
        syn::ReturnType::Type(_, ty) => ty,
        _ => panic!("setup function must return a Context"),
    };

    let (c_ty, p_ty) = extract_context_generics(return_type)
        .expect("setup function return type must be Context<C, P>");

    let expanded = quote! {
        #setup_fn

        static mut HIL_SERVER: Option<::control_rs_hil::Server<'static, #c_ty, #p_ty>> = None;

        ::control_rs_macros::hil_entrypoint!(#setup_name);
        ::control_rs_macros::hil_panic!();
        ::control_rs_macros::hil_exception!();
    };

    TokenStream::from(expanded)
}

/// Helper macro to define the unified HIL entrypoint `main`.
#[proc_macro]
pub fn hil_entrypoint(input: TokenStream) -> TokenStream {
    let setup_name = parse_macro_input!(input as syn::Ident);
    let expanded = quote! {
        #[cfg(target_os = "none")]
        unsafe extern "Rust" {
            static __hil_test_suites_start: u8;
            static __hil_test_suites_end: u8;
        }

        // ==================== Unified Entry Point ====================
        #[cfg(target_os = "none")]
        #[cfg_attr(target_arch = "arm", ::cortex_m_rt::entry)]
        #[cfg_attr(target_arch = "riscv32", ::riscv_rt::entry)]
        fn main() -> ! {
            let start = unsafe {
                &__hil_test_suites_start as *const u8 as *const &::control_rs_hil::SuiteDescriptor
            };
            let end = unsafe {
                &__hil_test_suites_end as *const u8 as *const &::control_rs_hil::SuiteDescriptor
            };

            let suites = unsafe { ::control_rs_hil::util::get_suites(start, end) };

            let context = #setup_name();
            unsafe {
                HIL_SERVER = Some(::control_rs_hil::Server::new(context, suites));
            }

            let server = unsafe { HIL_SERVER.as_mut().unwrap() };
            let _ = server.run();
            server.exit();
        }
    };
    TokenStream::from(expanded)
}

/// Helper macro to define the target HIL panic handler.
#[proc_macro]
pub fn hil_panic(input: TokenStream) -> TokenStream {
    let _ = input;
    let expanded = quote! {
        // ==================== Unified Panic Handler ====================
        #[cfg(target_os = "none")]
        #[panic_handler]
        fn panic(info: &::core::panic::PanicInfo) -> ! {
            let mut msg_buf = [0u8; 128];
            let pos = {
                let mut writer = ::control_rs_hil::util::FailureBufWriter { buf: &mut msg_buf, pos: 0 };
                let _ = ::core::fmt::write(&mut writer, format_args!("{}", info.message()));
                writer.pos
            };
            let msg = ::core::str::from_utf8(&msg_buf[..pos]).unwrap_or("panic occurred");

            let file = info.location().map_or("unknown", |l| l.file());
            let line = info.location().map_or(0, |l| l.line());

            unsafe {
                if let Some(server) = HIL_SERVER.as_mut() {
                    let comms_ok = server.context.comms_lock.try_lock();
                    ::control_rs_hil::util::handle_failure(
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
    };
    TokenStream::from(expanded)
}

/// Helper macro to define the target HIL trap/exception handlers.
#[proc_macro]
pub fn hil_exception(input: TokenStream) -> TokenStream {
    let _ = input;
    let expanded = quote! {
        // ==================== Unified Exception Handler Implementation ====================
        #[cfg(all(target_os = "none", target_arch = "arm"))]
        #[::cortex_m_rt::exception]
        unsafe fn HardFault(ef: &::cortex_m_rt::ExceptionFrame) -> ! {
            let mut msg_buf = [0u8; 128];
            let pos = {
                let mut writer = ::control_rs_hil::util::FailureBufWriter { buf: &mut msg_buf, pos: 0 };
                let _ = ::core::fmt::write(
                    &mut writer,
                    format_args!("HardFault at pc=0x{:08x}, lr=0x{:08x}", ef.pc() as usize, ef.lr() as usize),
                );
                writer.pos
            };
            let msg = ::core::str::from_utf8(&msg_buf[..pos]).unwrap_or("exception occurred");

            if let Some(server) = HIL_SERVER.as_mut() {
                let comms_ok = server.context.comms_lock.try_lock();
                ::control_rs_hil::util::handle_exception(
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

        #[cfg(all(target_os = "none", target_arch = "riscv32"))]
        #[unsafe(no_mangle)]
        unsafe fn ExceptionHandler(_ef: &mut ::riscv_rt::TrapFrame) -> ! {
            let mut msg_buf = [0u8; 128];
            let pos = {
                let mut writer = ::control_rs_hil::util::FailureBufWriter { buf: &mut msg_buf, pos: 0 };
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

            if let Some(server) = HIL_SERVER.as_mut() {
                let comms_ok = server.context.comms_lock.try_lock();
                ::control_rs_hil::util::handle_exception(
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
    };
    TokenStream::from(expanded)
}
