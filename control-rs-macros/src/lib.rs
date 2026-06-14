extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    Item, ItemFn, ItemMod, ItemStatic, Type, parse_macro_input, parse_quote,
};

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
        for item in items.iter_mut() {
            match item {
                Item::Static(item_static) => {
                    let name_ident = &item_static.ident;
                    let init_expr = &item_static.expr;
                    let attrs = &item_static.attrs;
                    let setting_doc = extract_doc_string(attrs);

                    if is_type_name(&item_static.ty, "u32") {
                        settings.push(name_ident.clone());
                        let new_static: ItemStatic = parse_quote! {
                            #(#attrs)*
                            pub static #name_ident: ::control_rs_hil::settings::AtomicU32Setting =
                                ::control_rs_hil::settings::AtomicU32Setting::new(stringify!(#name_ident), #setting_doc, #init_expr);
                        };
                        *item_static = new_static;
                    } else if is_type_name(&item_static.ty, "u8") {
                        settings.push(name_ident.clone());
                        let new_static: ItemStatic = parse_quote! {
                            #(#attrs)*
                            pub static #name_ident: ::control_rs_hil::settings::AtomicU8Setting =
                                ::control_rs_hil::settings::AtomicU8Setting::new(stringify!(#name_ident), #setting_doc, #init_expr);
                        };
                        *item_static = new_static;
                    }
                }
                Item::Fn(item_fn) => {
                    let fn_name = &item_fn.sig.ident;
                    let fn_name_str = fn_name.to_string();
                    if !fn_name_str.starts_with('_') {
                        let test_doc = extract_doc_string(&item_fn.attrs);
                        tests.push((fn_name.clone(), test_doc));
                    }
                }
                _ => {}
            }
        }

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

        let suite_desc_code: [Item; 4] = [
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
        ];

        items.extend(suite_desc_code);
    }

    let expanded = quote! {
        #item_mod
    };

    TokenStream::from(expanded)
}

/// Attribute macro for setting up the HIL server entrypoint.
///
/// Annotates the hardware setup function, generates the standard main entrypoint,
/// and sets up the server event loop and QEMU-compatible panic handler.
#[proc_macro_attribute]
pub fn hil_setup(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let setup_fn = parse_macro_input!(item as ItemFn);
    let setup_name = &setup_fn.sig.ident;

    let expanded = quote! {
        #setup_fn

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

            let len = (end as usize - start as usize) / ::core::mem::size_of::<&::control_rs_hil::SuiteDescriptor>();
            let suites = unsafe { ::core::slice::from_raw_parts(start, len) };

            let context = #setup_name();

            let mut server = ::control_rs_hil::Server::new_with_executor(context.comms, context.timer, context.executor, suites);
            let _ = server.run();

            #[cfg(target_arch = "arm")]
            ::cortex_m_semihosting::debug::exit(::cortex_m_semihosting::debug::EXIT_SUCCESS);

            #[cfg(target_arch = "riscv32")]
            ::semihosting::process::exit(0);

            #[cfg(not(target_arch = "riscv32"))]
            loop {}
        }

        // ==================== Failure Formatting Helper ====================
        #[cfg(target_os = "none")]
        struct FailureBufWriter<'a> {
            buf: &'a mut [u8],
            pos: usize,
        }

        #[cfg(target_os = "none")]
        impl<'a> ::core::fmt::Write for FailureBufWriter<'a> {
            fn write_str(&mut self, s: &str) -> ::core::fmt::Result {
                let bytes = s.as_bytes();
                let len = bytes.len();
                if self.pos + len > self.buf.len() {
                    return Err(::core::fmt::Error);
                }
                self.buf[self.pos..self.pos + len].copy_from_slice(bytes);
                self.pos += len;
                Ok(())
            }
        }

        // ==================== Unified Failure Implementation ====================
        #[cfg(target_os = "none")]
        unsafe fn handle_failure(msg: &str, file: &str, line: u32) -> ! {
            #[cfg(target_arch = "arm")]
            ::cortex_m::interrupt::disable();

            #[cfg(target_arch = "riscv32")]
            ::riscv::interrupt::disable();

            let suite = ::control_rs_hil::server::CURRENT_SUITE.load(::core::sync::atomic::Ordering::SeqCst);
            let test = ::control_rs_hil::server::CURRENT_TEST.load(::core::sync::atomic::Ordering::SeqCst);

            if let (Some(sender), ptr) = (
                ::control_rs_hil::server::PANIC_TELEMETRY_SENDER,
                ::control_rs_hil::server::ACTIVE_COMMS_PTR,
            ) {
                if !ptr.is_null() {
                    if suite >= 0 && test >= 0 {
                        sender(
                            ptr,
                            &::control_rs_hil::comms::Telemetry::TestStateChange {
                                suite_id: suite as u16,
                                test_id: test as u16,
                                state: ::control_rs_hil::comms::TestState::Failed,
                            },
                        );
                    }
                    sender(
                        ptr,
                        &::control_rs_hil::comms::Telemetry::TargetPanic {
                            message: msg,
                            file,
                            line,
                        },
                    );
                }
            }

            // Wait for OkToReset command from host
            loop {
                if let (Some(poller), ptr) = (
                    ::control_rs_hil::server::PANIC_CMD_POLLER,
                    ::control_rs_hil::server::ACTIVE_COMMS_PTR,
                ) {
                    if !ptr.is_null() {
                        if let Some(::control_rs_hil::comms::Command::OkToReset) = poller(ptr) {
                            break;
                        }
                    }
                }

                if let (Some(flusher), ptr) = (
                    ::control_rs_hil::server::PANIC_COMMS_FLUSHER,
                    ::control_rs_hil::server::ACTIVE_COMMS_PTR,
                ) {
                    if !ptr.is_null() {
                        flusher(ptr);
                    } else {
                        ::core::hint::spin_loop();
                    }
                } else {
                    ::core::hint::spin_loop();
                }

                // Small delay to prevent pegging the CPU too hard
                for _ in 0..1000 {
                    ::core::hint::spin_loop();
                }
            }

            #[cfg(target_arch = "arm")]
            ::cortex_m::peripheral::SCB::sys_reset();

            #[cfg(target_arch = "riscv32")]
            ::semihosting::process::exit(1);

            #[cfg(not(any(target_arch = "arm", target_arch = "riscv32")))]
            loop {}
        }

        // ==================== Unified Panic Handler ====================
        #[cfg(target_os = "none")]
        #[panic_handler]
        fn panic(info: &::core::panic::PanicInfo) -> ! {
            let mut msg_buf = [0u8; 128];
            let pos = {
                let mut writer = FailureBufWriter { buf: &mut msg_buf, pos: 0 };
                let _ = ::core::fmt::write(&mut writer, format_args!("{}", info.message()));
                writer.pos
            };
            let msg = ::core::str::from_utf8(&msg_buf[..pos]).unwrap_or("panic occurred");

            let file = info.location().map_or("unknown", |l| l.file());
            let line = info.location().map_or(0, |l| l.line());

            unsafe {
                handle_failure(msg, file, line);
            }
        }

        // ==================== Unified Exception Handler Implementation ====================
        #[cfg(target_os = "none")]
        unsafe fn handle_exception(pc: usize, lr_or_cause: usize) -> ! {
            let mut msg_buf = [0u8; 128];
            let pos = {
                let mut writer = FailureBufWriter { buf: &mut msg_buf, pos: 0 };
                #[cfg(target_arch = "arm")]
                let _ = ::core::fmt::write(&mut writer, format_args!("HardFault at pc=0x{:08x}, lr=0x{:08x}", pc, lr_or_cause));
                #[cfg(target_arch = "riscv32")]
                let _ = ::core::fmt::write(&mut writer, format_args!("Exception mcause=0x{:08x}, mepc=0x{:08x}", lr_or_cause, pc));
                writer.pos
            };
            let msg = ::core::str::from_utf8(&msg_buf[..pos]).unwrap_or("exception occurred");

            handle_failure(msg, "exception_handler", 0);
        }

        #[cfg(all(target_os = "none", target_arch = "arm"))]
        #[::cortex_m_rt::exception]
        unsafe fn HardFault(ef: &::cortex_m_rt::ExceptionFrame) -> ! {
            handle_exception(ef.pc() as usize, ef.lr() as usize);
        }

        #[cfg(all(target_os = "none", target_arch = "riscv32"))]
        #[unsafe(no_mangle)]
        unsafe fn ExceptionHandler(_ef: &mut ::riscv_rt::TrapFrame) -> ! {
            handle_exception(::riscv::register::mepc::read(), ::riscv::register::mcause::read().bits());
        }
    };

    TokenStream::from(expanded)
}
