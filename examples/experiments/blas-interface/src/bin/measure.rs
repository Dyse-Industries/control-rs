//! Disassembly-based codegen measurement for the `blas_interface` gemv
//! variants, for any target triple given on the command line.
//!
//! Builds the crate's `staticlib` for the target at opt-level 3 (`release`
//! profile default), `codegen-units=1`, `panic=abort`, extracts the
//! crate's own object code from the resulting `.a` archive, disassembles
//! each `#[no_mangle]` `gemv_*` symbol, and reports:
//!
//! - `instrs`: total instruction count in the symbol's body.
//! - `branches+calls`: count of jump (conditional or unconditional) and
//!   call instructions, using the target ISA's own mnemonics.
//! - `panic paths`: count of *conditional* jumps whose target block —
//!   followed through any chain of unconditional jumps — reaches a call
//!   relocated against a `core::panicking::*` symbol (a bounds-check or
//!   unwind-abort trap) before any other branch or return. This is the
//!   "opaque boundary" premise from `blas_interface`'s crate doc: these
//!   functions have no in-crate caller, so LLVM cannot see how the
//!   argument structs are constructed and cannot fold a variant's bounds
//!   checks away.
//!
//! Only compiles and disassembles — never executes the target artifact, so
//! bare-metal (`-none-`) triples work here without hardware or an emulator.
//!
//! Requires `rustup component add llvm-tools` (for a `llvm-objdump`/
//! `llvm-ar`/`llvm-nm` with broad target support — the system `objdump` on
//! macOS, for example, lacks a RISC-V backend) and, for cross targets, the
//! Rust target itself (`rustup target add <triple>`).
//!
//! Run with: `cargo run --bin measure -- [TARGET_TRIPLE]`

use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_TARGET: &str = "x86_64-apple-darwin";

/// Instruction-set family, inferred from the target triple. Determines
/// which mnemonics count as jumps/calls/returns during disassembly.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Isa {
    X86_64,
    RiscV,
    Arm,
}

impl Isa {
    fn from_triple(triple: &str) -> Result<Self, Box<dyn Error>> {
        if triple.starts_with("x86_64") {
            Ok(Self::X86_64)
        } else if triple.starts_with("riscv32") || triple.starts_with("riscv64") {
            Ok(Self::RiscV)
        } else if triple.starts_with("thumb") || triple.starts_with("arm") {
            Ok(Self::Arm)
        } else {
            Err(format!(
                "unrecognized architecture for target triple {triple:?} \
                 (expected x86_64-*, riscv32*-*, riscv64*-*, thumb*-*, or arm*-*)"
            )
            .into())
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::X86_64 => "x86_64",
            Self::RiscV => "risc-v",
            Self::Arm => "arm/thumb",
        }
    }
}

struct Variant {
    symbol: &'static str,
    description: &'static str,
}

// Ordered to match the lettered variants in src/lib.rs and the table in
// README.md. H sits next to B (same slice buffer, array x/y). C8/C16 and
// D8/D16 sit next to C/D so size is the only changing axis.
const VARIANTS: &[Variant] = &[
    Variant {
        symbol: "gemv_dyn",
        description: "[A] runtime lda/order, slice buf/x/y -- y = A*x",
    },
    Variant {
        symbol: "gemv_const_4",
        description: "[B] assoc consts, slice buf/x/y -- y = A*x",
    },
    Variant {
        symbol: "gemv_const_xy_4",
        description: "[H] assoc consts, slice buf, [f32; 4] x/y -- y = A*x",
    },
    Variant {
        symbol: "gemv_arr_4",
        description: "[C] assoc consts, [f32; 16] 4x4 -- y = A*x",
    },
    Variant {
        symbol: "gemv_arr_8",
        description: "[C8] assoc consts, [f32; 64] 8x8 -- y = A*x",
    },
    Variant {
        symbol: "gemv_arr_16",
        description: "[C16] assoc consts, [f32; 256] 16x16 -- y = A*x",
    },
    Variant {
        symbol: "gemv_ptr_4",
        description: "[D] assoc consts, raw ptr 4x4 -- y = A*x",
    },
    Variant {
        symbol: "gemv_ptr_8",
        description: "[D8] assoc consts, raw ptr 8x8 -- y = A*x",
    },
    Variant {
        symbol: "gemv_ptr_16",
        description: "[D16] assoc consts, raw ptr 16x16 -- y = A*x",
    },
    Variant {
        symbol: "gemv_ptr_ab_4",
        description: "[E] assoc consts, raw ptr 4x4 -- y = alpha*A*x + beta*y",
    },
    Variant {
        symbol: "gemv_checked_4",
        description: "[G] assoc consts, explicit i/j check, [f32; 4] x/y -- y = A*x",
    },
    Variant {
        symbol: "gemv_storage_trait_4",
        description: "[I] Design A: generic storage via trait method, [f32; 4] x/y",
    },
    Variant {
        symbol: "gemv_generic_fn_4",
        description: "[J] Design C: generic storage via generic function directly",
    },
    Variant {
        symbol: "gemv_nested_4",
        description: "[K] Design B: proposed nested arrays (statically sized)",
    },
];

struct Insn {
    addr: u64,
    mnemonic: String,
    /// Jump target as a raw address, when the disassembler prints one
    /// directly (RISC-V/ARM: `0x186 <gemv_dyn+0x186>`).
    target_addr: Option<u64>,
    /// Jump target as a local pseudo-label, when the disassembler doesn't
    /// print a raw address (x86/Mach-O: ` <L10>`).
    target_label: Option<String>,
}

struct Reloc {
    addr: u64,
    symbol: String,
}

struct Disassembly {
    isa: Isa,
    insns: Vec<Insn>,
    labels: HashMap<String, usize>,
    relocs: Vec<Reloc>,
}

struct Metrics {
    instrs: usize,
    branches_calls: usize,
    panic_paths: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(());
    }
    let target = args
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_TARGET.to_string());
    let isa = Isa::from_triple(&target)?;

    let sysroot = rustc_sysroot()?;
    let tools_dir = find_llvm_tools_dir(&sysroot)?;
    let ar = tools_dir.join("llvm-ar");
    let objdump = tools_dir.join("llvm-objdump");
    let nm = tools_dir.join("llvm-nm");

    let crate_root = locate_crate_root()?;
    let flags = build_release_staticlib(&crate_root, &target)?;
    let archive = find_staticlib(&crate_root, &target)?;

    let scratch = crate_root
        .join("target")
        .join(&target)
        .join("release")
        .join("measure-extract");
    let members = extract_own_members(&ar, &archive, &scratch)?;

    println!("Measured");
    println!(
        "{target} ({}), opt-level={}, codegen-units={}, panic={}, f32 GEMV",
        isa.label(),
        flags.opt_level,
        flags.codegen_units,
        flags.panic
    );
    println!("behind a staticlib boundary with no in-crate caller, so LLVM cannot see");
    println!("construction:\n");
    if flags.opt_level == "0" {
        println!("WARNING: opt-level=0 also disables bounds-check elimination and");
        println!("constant folding. gemv_const/gemv_const_xy/gemv_arr/");
        println!("gemv_ptr/gemv_ptr_ab/gemv_checked carry #[inline(always)] so");
        println!("the #[no_mangle] wrappers still get the real algorithm body");
        println!("here (LLVM's AlwaysInliner runs regardless of opt-level)");
        println!("rather than a bare trampoline -- but that body is the");
        println!("*unoptimized* one: checks/folds an opt-level=3 build");
        println!("removes are still present, so these counts are not");
        println!("comparable to an opt-level=3 build (see README.md).\n");
    }
    println!(
        "{:<66} {:>7} {:>16} {:>12}",
        "variant", "instrs", "branches+calls", "panic paths"
    );
    for variant in VARIANTS {
        let symbol = symbol_for(variant.symbol, &target);
        let object = find_symbol_object(&nm, &members, &symbol)?;
        let asm = disassemble(&objdump, object, &symbol, isa, &target)?;
        let disassembly = parse(&asm, isa);
        let metrics = measure(&disassembly);
        println!(
            "{:<66} {:>7} {:>16} {:>12}",
            variant.description, metrics.instrs, metrics.branches_calls, metrics.panic_paths
        );
    }
    Ok(())
}

fn print_usage() {
    println!("Usage: measure [TARGET_TRIPLE]");
    println!();
    println!("Cross-compiles the blas_interface staticlib for TARGET_TRIPLE (default:");
    println!("{DEFAULT_TARGET}) and disassembles each gemv_* symbol to report");
    println!("instruction / branch+call / panic-path counts. Compiles and disassembles");
    println!("only -- never executes the target artifact.");
    println!();
    println!("Examples:");
    println!("  measure");
    println!("  measure riscv32imac-unknown-none-elf");
    println!("  measure riscv32imafc-unknown-none-elf");
    println!("  measure riscv64gc-unknown-none-elf");
    println!("  measure thumbv7em-none-eabi");
    println!("  measure thumbv7em-none-eabihf");
    println!("  measure x86_64-unknown-linux-gnu");
}

fn rustc_sysroot() -> Result<PathBuf, Box<dyn Error>> {
    let output = Command::new("rustc")
        .arg("--print")
        .arg("sysroot")
        .output()?;
    if !output.status.success() {
        return Err("rustc --print sysroot failed".into());
    }
    Ok(PathBuf::from(String::from_utf8(output.stdout)?.trim()))
}

/// Locates the `llvm-{ar,objdump,nm}` bundled with the active toolchain
/// (`rustup component add llvm-tools`), which — unlike e.g. macOS's system
/// `objdump` — support every architecture rustc itself can target.
fn find_llvm_tools_dir(sysroot: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let rustlib = sysroot.join("lib").join("rustlib");
    for entry in fs::read_dir(&rustlib)? {
        let bin = entry?.path().join("bin");
        if bin.join("llvm-objdump").is_file() {
            return Ok(bin);
        }
    }
    Err(format!(
        "no llvm-objdump found under {}; run `rustup component add llvm-tools`",
        rustlib.display()
    )
    .into())
}

fn locate_crate_root() -> Result<PathBuf, Box<dyn Error>> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let output = Command::new("cargo")
        .args(["locate-project", "--workspace", "--message-format", "plain"])
        .current_dir(manifest_dir)
        .output()?;
    if !output.status.success() {
        return Err("cargo locate-project failed to find the crate root".into());
    }
    let manifest_path = String::from_utf8(output.stdout)?;
    let root = PathBuf::from(manifest_path.trim())
        .parent()
        .ok_or("crate manifest path has no parent directory")?
        .to_path_buf();
    Ok(root)
}

/// The `-C` codegen flags rustc actually received for the `blas_interface`
/// staticlib build, read off cargo's own verbose build log rather than
/// assumed -- so the header this prints always matches what was really
/// built, whether opt-level/codegen-units came from `Cargo.toml`'s
/// `[profile.release]` or this function's own `--config` overrides.
struct BuildFlags {
    opt_level: String,
    codegen_units: String,
    panic: String,
}

fn build_release_staticlib(crate_root: &Path, target: &str) -> Result<BuildFlags, Box<dyn Error>> {
    let output = Command::new("cargo")
        .args([
            "build",
            "-v",
            "--release",
            "--lib",
            "--no-default-features",
            "--target",
            target,
            "--config",
            "profile.release.codegen-units=1",
            "--config",
            "profile.release.panic=\"abort\"",
        ])
        .current_dir(crate_root)
        // An outer `cargo run` may set CARGO_TARGET_DIR (CI, rust-analyzer,
        // this sandbox). The staticlib lookup below is under crate_root/target,
        // so drop the inherited dir and write there.
        .env_remove("CARGO_TARGET_DIR")
        .output()?;
    if !output.status.success() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        return Err(format!("cargo build --release --lib --target {target} failed").into());
    }
    // cargo -v prints each "Running `rustc ...`" invocation to stderr.
    parse_build_flags(&String::from_utf8_lossy(&output.stderr))
}

/// Extracts the `-C opt-level`/`-C codegen-units`/`-C panic` values from the
/// `rustc` invocation line for this crate in a `cargo build -v` log.
fn parse_build_flags(verbose_log: &str) -> Result<BuildFlags, Box<dyn Error>> {
    let line = verbose_log
        .lines()
        .find(|l| l.contains("--crate-name blas_interface") && l.contains("--crate-type staticlib"))
        .ok_or(
            "could not find the blas_interface rustc invocation in `cargo \
             build -v` output",
        )?;
    Ok(BuildFlags {
        // Absent means rustc's own default (0), not "unset" -- cargo only
        // emits `-C opt-level` when the resolved profile value is nonzero.
        opt_level: extract_c_value(line, "opt-level").unwrap_or_else(|| "0".to_string()),
        codegen_units: extract_c_value(line, "codegen-units")
            .ok_or("no -C codegen-units in rustc invocation")?,
        panic: extract_c_value(line, "panic").ok_or("no -C panic in rustc invocation")?,
    })
}

/// Extracts the value of a `-C key=value` flag from a shell-quoted `rustc`
/// invocation line, e.g. `extract_c_value(line, "opt-level")` on a line
/// containing `-C opt-level=3` returns `Some("3")`.
fn extract_c_value(line: &str, key: &str) -> Option<String> {
    let needle = format!("-C {key}=");
    let start = line.find(&needle)?.checked_add(needle.len())?;
    let rest = line.get(start..)?;
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    rest.get(..end).map(str::to_string)
}

fn find_staticlib(crate_root: &Path, target: &str) -> Result<PathBuf, Box<dyn Error>> {
    let dir = crate_root.join("target").join(target).join("release");
    for candidate in ["libblas_interface.a", "blas_interface.lib"] {
        let path = dir.join(candidate);
        if path.is_file() {
            return Ok(path);
        }
    }
    Err(format!("no staticlib found under {}", dir.display()).into())
}

/// Extracts the archive members that are this crate's own compiled code
/// (as opposed to bundled dependency objects like `core`/`compiler_builtins`)
/// into `scratch`. There's normally exactly one with `codegen-units=1`, but
/// this doesn't assume that — every matching member is extracted and later
/// searched for the requested symbol.
fn extract_own_members(
    ar: &Path,
    archive: &Path,
    scratch: &Path,
) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let list = Command::new(ar).arg("t").arg(archive).output()?;
    if !list.status.success() {
        return Err(format!("{} t {} failed", ar.display(), archive.display()).into());
    }
    let listing = String::from_utf8(list.stdout)?;
    let members: Vec<&str> = listing
        .lines()
        .filter(|line| line.starts_with("blas_interface-"))
        .collect();
    if members.is_empty() {
        return Err(format!(
            "no blas_interface object member found in {}",
            archive.display()
        )
        .into());
    }

    let _ = fs::remove_dir_all(scratch);
    fs::create_dir_all(scratch)?;
    let mut paths = Vec::new();
    for member in members {
        let status = Command::new(ar)
            .arg("x")
            .arg(archive)
            .arg(member)
            .current_dir(scratch)
            .status()?;
        if !status.success() {
            return Err(format!("{} x {archive:?} {member} failed", ar.display()).into());
        }
        paths.push(scratch.join(member));
    }
    Ok(paths)
}

fn find_symbol_object<'a>(
    nm: &Path,
    members: &'a [PathBuf],
    symbol: &str,
) -> Result<&'a Path, Box<dyn Error>> {
    for member in members {
        let output = Command::new(nm).arg(member).output()?;
        if !output.status.success() {
            continue;
        }
        let text = String::from_utf8_lossy(&output.stdout);
        if text
            .lines()
            .any(|line| line.split_whitespace().last() == Some(symbol))
        {
            return Ok(member);
        }
    }
    Err(format!("symbol {symbol} not found in any extracted object").into())
}

/// Mach-O (Apple targets) prefixes C symbols with `_`; ELF (Linux, RISC-V,
/// ARM/Thumb) and Windows x86-64 do not. Known, accepted gap: 32-bit
/// `i686-pc-windows-*` also underscore-prefixes; out of scope here.
fn symbol_for(base: &str, target: &str) -> String {
    if target.contains("apple") {
        format!("_{base}")
    } else {
        base.to_string()
    }
}

fn disassemble(
    objdump: &Path,
    object: &Path,
    symbol: &str,
    isa: Isa,
    target: &str,
) -> Result<String, Box<dyn Error>> {
    let mut cmd = Command::new(objdump);
    cmd.arg("-d")
        .arg("-r")
        .arg("--symbolize-operands")
        .arg("--no-show-raw-insn");
    if target.contains("apple") {
        // Mach-O: every function shares one __TEXT,__text section, and
        // --disassemble-symbols correctly bounds the scoped output using
        // the object's own function-boundary metadata there.
        cmd.arg(format!("--disassemble-symbols={symbol}"));
    } else {
        // ELF (the default function-sections layout puts every function
        // in its own .text.<symbol> section): scope by section instead of
        // --disassemble-symbols, which can stop early on RISC-V when a
        // local PC-relative anchor symbol (.LpcrelHiN, used to address
        // shared f32 constants) sits inside the function body -- even
        // though that code is still within the symbol's own recorded
        // size, i.e. still part of it.
        cmd.arg(format!("--section=.text.{symbol}"));
    }
    if isa == Isa::X86_64 {
        cmd.arg("--x86-asm-syntax=att");
    }
    cmd.arg(object);
    let output = cmd.output()?;
    if !output.status.success() {
        return Err(format!("objdump failed for symbol {symbol}").into());
    }
    Ok(String::from_utf8(output.stdout)?)
}

/// Extracts a jump/call operand's target: either a raw address (RISC-V/ARM
/// print `0x186 <gemv_dyn+0x186>` directly) or, failing that, a bracketed
/// local pseudo-label (x86/Mach-O prints just ` <L10>`, no address).
fn extract_target(operand: &str) -> (Option<u64>, Option<String>) {
    let operand = operand.trim();
    // RISC-V/ARM conditional branches with a register operand print
    // e.g. "s4, 0x18a <gemv_dyn+0x18a>" -- the address isn't at the front
    // of the field, so search for "0x" anywhere before the "<...>"
    // annotation (which itself embeds the same hex value as a
    // symbol-relative offset, not useful to search into).
    let head = operand.split('<').next().unwrap_or(operand);
    if let Some(pos) = head.find("0x") {
        let hex = &head[pos + 2..];
        let hex_len = hex.chars().take_while(char::is_ascii_hexdigit).count();
        if hex_len > 0 {
            if let Ok(addr) = u64::from_str_radix(&hex[..hex_len], 16) {
                return (Some(addr), None);
            }
        }
    }
    if let Some(start) = operand.find('<') {
        if let Some(end) = operand[start..].find('>') {
            return (None, Some(operand[start + 1..start + end].to_string()));
        }
    }
    (None, None)
}

fn parse(asm: &str, isa: Isa) -> Disassembly {
    let mut insns = Vec::new();
    let mut labels = HashMap::new();
    let mut relocs = Vec::new();

    for line in asm.lines() {
        let trimmed = line.trim();

        // Local label definition, e.g. "<L2>:" (only emitted for Mach-O).
        if let Some(label) = trimmed.strip_prefix('<').and_then(|s| s.strip_suffix(">:")) {
            labels.insert(label.to_string(), insns.len());
            continue;
        }

        let Some((addr_field, rest)) = trimmed.split_once(':') else {
            continue;
        };
        if addr_field.is_empty() || !addr_field.chars().all(|c| c.is_ascii_hexdigit()) {
            continue;
        }
        let Ok(addr) = u64::from_str_radix(addr_field, 16) else {
            continue;
        };
        let rest = rest.trim_start();

        // Relocation record: "  RELOC_TYPE\tsymbol" -- the type token is a
        // conventionally uppercase constant on every format seen here
        // (X86_64_RELOC_BRANCH, R_RISCV_CALL_PLT, R_ARM_THM_CALL, ...),
        // unlike any real mnemonic, which is always lowercase.
        if rest.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
            if let Some(symbol) = rest
                .split('\t')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .last()
            {
                relocs.push(Reloc {
                    addr,
                    symbol: symbol.to_string(),
                });
            }
            continue;
        }

        let mut fields = rest.split('\t').map(str::trim).filter(|s| !s.is_empty());
        let Some(mnemonic) = fields.next() else {
            continue;
        };
        let (target_addr, target_label) = fields.next().map_or((None, None), extract_target);
        insns.push(Insn {
            addr,
            mnemonic: mnemonic.to_string(),
            target_addr,
            target_label,
        });
    }

    Disassembly {
        isa,
        insns,
        labels,
        relocs,
    }
}

fn mnemonic_base(mnemonic: &str) -> &str {
    mnemonic.split('.').next().unwrap_or(mnemonic)
}

fn is_unconditional_jump(mnemonic: &str, isa: Isa) -> bool {
    let base = mnemonic_base(mnemonic);
    match isa {
        Isa::X86_64 => base == "jmp",
        Isa::RiscV => base == "j" || base == "jr",
        Isa::Arm => base == "b",
    }
}

fn is_conditional_jump(mnemonic: &str, isa: Isa) -> bool {
    let base = mnemonic_base(mnemonic);
    match isa {
        Isa::X86_64 => base.starts_with('j') && base != "jmp",
        Isa::RiscV => matches!(
            base,
            "beq"
                | "bne"
                | "blt"
                | "bge"
                | "bltu"
                | "bgeu"
                | "beqz"
                | "bnez"
                | "blez"
                | "bgez"
                | "bltz"
                | "bgtz"
        ),
        Isa::Arm => {
            (base.starts_with('b') && !matches!(base, "b" | "bl" | "blx" | "bx"))
                || base == "cbz"
                || base == "cbnz"
        }
    }
}

fn is_jump(mnemonic: &str, isa: Isa) -> bool {
    is_conditional_jump(mnemonic, isa) || is_unconditional_jump(mnemonic, isa)
}

fn is_call(mnemonic: &str, isa: Isa) -> bool {
    let base = mnemonic_base(mnemonic);
    match isa {
        Isa::X86_64 => base.starts_with("call"),
        Isa::RiscV => base == "jal" || base == "jalr" || base == "call",
        Isa::Arm => base == "bl" || base == "blx",
    }
}

/// Not exhaustive (e.g. ARM's `pop {..., pc}` is only matched via the
/// `pop`-prefix heuristic, not by checking for `pc` in the register list)
/// — `resolves_to_panic`'s own step budget is the backstop against a
/// missed return wandering into unrelated code.
fn is_return(mnemonic: &str, isa: Isa) -> bool {
    let base = mnemonic_base(mnemonic);
    match isa {
        Isa::X86_64 => base.starts_with("ret"),
        Isa::RiscV => base == "ret",
        Isa::Arm => base == "bx" || base.starts_with("pop"),
    }
}

fn resolve_jump_target(insn: &Insn, disassembly: &Disassembly) -> Option<usize> {
    if let Some(addr) = insn.target_addr {
        return disassembly.insns.iter().position(|i| i.addr == addr);
    }
    if let Some(label) = &insn.target_label {
        return disassembly.labels.get(label).copied();
    }
    None
}

/// Whether the block starting at `disassembly.insns[start]` leads, through
/// zero or more unconditional jumps, to a call whose relocation targets a
/// `core::panicking::*` symbol — checked by scanning every relocation
/// address within the block's own address span, not by assuming a fixed
/// byte offset between a call instruction and its relocation (that offset
/// differs by ISA: x86's `call` carries its own relocation, RISC-V's
/// external calls split into a relocated `auipc` followed by a plain
/// `jalr`, ARM's `bl` carries its own relocation directly).
fn resolves_to_panic(start: usize, disassembly: &Disassembly, depth: u32) -> bool {
    const MAX_CHAIN: u32 = 8;
    const MAX_BLOCK_LEN: usize = 40;
    if depth >= MAX_CHAIN {
        return false;
    }

    let isa = disassembly.isa;
    let Some(block_start_addr) = disassembly.insns.get(start).map(|i| i.addr) else {
        return false;
    };

    let mut idx = start;
    let mut steps = 0;
    while idx < disassembly.insns.len() && steps < MAX_BLOCK_LEN {
        let insn = &disassembly.insns[idx];
        if is_call(&insn.mnemonic, isa) {
            let end_addr = disassembly
                .insns
                .get(idx + 1)
                .map_or(insn.addr.saturating_add(16), |next| next.addr);
            return disassembly.relocs.iter().any(|r| {
                r.addr >= block_start_addr && r.addr < end_addr && r.symbol.contains("panicking")
            });
        }
        if is_unconditional_jump(&insn.mnemonic, isa) {
            return match resolve_jump_target(insn, disassembly) {
                Some(next) => resolves_to_panic(next, disassembly, depth.saturating_add(1)),
                None => false,
            };
        }
        if is_conditional_jump(&insn.mnemonic, isa) || is_return(&insn.mnemonic, isa) {
            return false;
        }
        idx += 1;
        steps += 1;
    }
    false
}

fn measure(disassembly: &Disassembly) -> Metrics {
    let isa = disassembly.isa;
    let instrs = disassembly.insns.len();
    let branches_calls = disassembly
        .insns
        .iter()
        .filter(|insn| is_jump(&insn.mnemonic, isa) || is_call(&insn.mnemonic, isa))
        .count();
    let panic_paths = disassembly
        .insns
        .iter()
        .filter(|insn| is_conditional_jump(&insn.mnemonic, isa))
        .filter(|insn| {
            resolve_jump_target(insn, disassembly)
                .is_some_and(|start| resolves_to_panic(start, disassembly, 0))
        })
        .count();
    Metrics {
        instrs,
        branches_calls,
        panic_paths,
    }
}
