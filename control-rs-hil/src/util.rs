//! Utility functions for HIL testing and tracking metrics.
#![allow(dead_code)]

#[cfg(all(target_os = "none", feature = "stack-paint"))]
#[inline(always)]
pub fn get_sp() -> usize {
    let sp: usize;
    unsafe {
        core::arch::asm!("mov {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags));
    }
    sp
}

/// Paints the stack region with a sentinel byte sequence (0xCDCD_CDCD) from the bottom of the stack
/// up to the current stack pointer.
///
/// # Safety
/// This function directly manipulates the stack memory region and must only be called in a controlled,
/// single-threaded context during the test setup phase.
#[cfg(all(target_os = "none", feature = "stack-paint"))]
pub unsafe fn paint_stack() {
    unsafe extern "C" {
        static mut _stack_end: u32;
    }

    let stack_end_ptr = core::ptr::addr_of!(_stack_end) as usize;
    let sp = get_sp();

    if sp > stack_end_ptr {
        let mut ptr = stack_end_ptr as *mut u32;
        let limit_ptr = sp as *mut u32;

        while ptr < limit_ptr {
            unsafe {
                core::ptr::write_volatile(ptr, 0xCDCD_CDCD);
                ptr = ptr.add(1);
            }
        }
    }
}

/// Scans the stack from bottom-up (lowest address upwards) to find the first byte
/// that is not the sentinel value, and returns the stack high-water mark in bytes.
///
/// # Safety
/// This function reads raw stack memory and relies on valid linker symbol addresses.
#[cfg(all(target_os = "none", feature = "stack-paint"))]
pub unsafe fn scan_stack(sp: usize) -> u32 {
    unsafe extern "C" {
        static mut _stack_end: u32;
    }

    let stack_end_ptr = core::ptr::addr_of!(_stack_end) as usize;

    let mut ptr = stack_end_ptr as *const u32;
    let limit_ptr = sp as *const u32;

    while ptr < limit_ptr {
        let is_sentinel =
            unsafe { core::ptr::read_volatile(ptr) == 0xCDCD_CDCD };
        if !is_sentinel {
            break;
        }
        unsafe {
            ptr = ptr.add(1);
        }
    }

    let lowest_address = ptr as usize;
    if lowest_address < sp {
        (sp - lowest_address) as u32
    } else {
        0
    }
}
