# Terminal User Interface

**Implementation Order:** 7
**Estimated Time:** 4 days

![Date Badge](https://img.shields.io/badge/Date-May_24,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Needs%20Review-yellow)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

## 1. Context & Objective

The final component of this stack is the host console menu, a dedicated
interactive tool to drive the HIL firmware with a rich GUI (like the TUI shown
above). This acts as a runner to upload the firmware, start execution, and
dynamically display the available options, tests, and real-time metrics,
providing a vast improvement in user experience for exploring and running
on-target tests.

## 2. Architectural Overview

```bash
===============================================================================
 TARGET: Teensy 4.1 (i.MX RT1062) | CLOCK: 600 MHz | FPU: HARD (eabihf)
 LINK:   probe-rs (RTT) via DAPLink                | SPEED: 2000 kHz
===============================================================================
 [ RUNNING ] control_rs::math
-------------------------------------------------------------------------------
 NAME                                     CYCLES      TIME       VARIANCE
 ▼ math::storage
   ├─ contiguous_storage_alloc            1,204       2.00µs     ± 0.1%
   └─ noncontiguous_storage_dma           3,410       5.68µs     ± 0.4%

 ▼ math::subprograms::level3
   ├─ gemm_10x10_f32 (soft-float)         84,500      140.8µs    [CACHED]
   ├─ gemm_10x10_f32 (hard-float)        [ RUN... ]   ---        ---
   └─ gemm_50x50_f32 (hard-float)         PENDING     ---        ---

 ▼ math::edge_cases
   ├─ underflow_and_precision_loss        42          0.07µs     ± 0.0%
   └─ floating_point_epsilon_bounds       58          0.09µs     ± 0.0%
-------------------------------------------------------------------------------
 [ RTT LOGS ] (Autoscroll: ON)
 > [INFO] Host connected. Target halted.
 > [INFO] Discovered 24 targets via procedural macro test registry.
 > [PASS] storage::contiguous_storage_alloc
 > [PASS] storage::noncontiguous_storage_dma
 > [EXEC] subprograms::level3::gemm_10x10_f32 (hard-float)...
===============================================================================
 (f)ilter | (r)un all | (s)top | (c)lear cache | (q)uit
```