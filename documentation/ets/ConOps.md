# Concept of Operations (ConOps): Embedded Test Server (ETS)

---

## 1. System Topology

```mermaid
flowchart LR
    subgraph HOST ["Host Environment (Workstation or CI Runner)"]
        CONSOLE["Interactive Terminal (control-rs-tui)"]
        RUNNER["Headless Runner (control-rs-ci)"]
    end

    subgraph TARGET ["Target System (Silicon or QEMU)"]
        ETS[["Embedded Test Server (ETS)"]]
        UT["Unit Tests"]
        LOOP["Control Loop"]
        GAINS["Live Parameters"]
        ETS --- UT
        ETS --- LOOP
        ETS --- GAINS
    end

    subgraph PLANT ["Physical Plant / Hardware Interface"]
        HW["Sensors · Actuators · Power Electronics"]
    end

    HOST <-->|" Commands, Settings & Stream Data "| ETS
    ETS <==>|" Actuation & Sensor Ingestion "| PLANT
```

---

## 2. Interface Allocations

```mermaid
flowchart TD
    subgraph CLIENTS ["Operational Interfaces"]
        TUI_BOX["Interactive Dashboard (control-rs-tui)"]
        CI_BOX["Headless Automation (control-rs-ci)"]
    end

    subgraph SYSTEM_LAYERS ["Target Verification Scope"]
        MATH_LAYER["Layer 1: Numerics & Math Core<br/>(Fixed-Point, Complex, Realization)"]
        CONTROL_LAYER["Layer 2: Discrete Control<br/>(PID, State-Space Estimators)"]
        RT_LAYER["Layer 3: Real-Time Execution<br/>(Periodic Tasks, Parameter Groups)"]
        HW_LAYER["Layer 4: Target Architecture<br/>(Interrupts, Cycle Counters, Stack RAM)"]
    end

    TUI_BOX -->|" Live Tuning & Telemetry "| CONTROL_LAYER
    TUI_BOX -->|" Step Profiling "| RT_LAYER
    TUI_BOX -->|" Stack & Clock Inspection "| HW_LAYER
    CI_BOX -->|" Deterministic Assertions "| MATH_LAYER
    CI_BOX -->|" Boundary & Execution Gates "| CONTROL_LAYER
```

---

## 3. Operational Scenarios

| #     | Scenario                 | Configuration                                        | Interface Boundary                                      | Operational Objective                            |
|:------|:-------------------------|:-----------------------------------------------------|:--------------------------------------------------------|:-------------------------------------------------|
| **1** | **Unit Test**            | Host Native                                          | Test selection → Pass/fail verdict                      | Verify mathematical algorithms and realization   |
| **2** | **Emulator**             | Host + QEMU                                          | Semihosting link commands → Summary telemetry           | Pre-silicon execution on target ISA targets      |
| **3** | **Device Under Test**    | Physical Silicon (USB CDC / UART)                    | Benchmark command → Cycle count, duration, stack peak   | Verify execution bounds on target hardware       |
| **4** | **Hardware in the Loop** | Modified DUT (Plant Sim) or Unmodified DUT (ETS Rig) | State injection ↔ Actuator commands & step telemetry    | Validate closed-loop timing and safe state fault |
| **5** | **Data Acquisition**     | Silicon with Plant Peripherals                       | Stimulus pattern → High-rate telemetry stream           | Stream high-rate state for model identification  |
| **6** | **Live Tuning**          | Interactive Session                                  | Parameter mutation command ↔ Verification telemetry     | Update controller gains during live execution    |

### 3.1 Algorithm & Unit Test Execution

```mermaid
flowchart LR
    HOST["Host Environment<br/>(Automated Runner)"]

    subgraph TARGET ["Target System"]
        ETS[["Embedded Test Server<br/>Dispatch & Exception Isolation"]]
        ALG["Algorithms Under Test<br/>Stimulus → Output Assertion"]
    end

    HOST -->|" 1. Select & Dispatch Test "| ETS
    ETS -->|" 2. Execute Case "| ALG
    ETS -->|" 3. Verdict & Failure Record "| HOST
```

### 3.2 Target Emulation (QEMU)

```mermaid
flowchart LR
    subgraph SERVER ["Host Execution (CI / Development Container)"]
        RUNNER["Headless Runner (control-rs-ci)<br/>1. Spawns QEMU · 4. Collects Artifacts"]

        subgraph SIM ["Virtual Machine"]
            VCHIP[["Firmware Image + ETS<br/>2. Emulates Core & Peripherals<br/>3. Runs Suites & Terminates"]]
        end

        RUNNER <-->|" Semihosting Link "| VCHIP
        VCHIP -.->|" Execution Results & Exit Code "| RUNNER
    end
```

### 3.3 Hardware Benchmarking (Silicon DUT)

```mermaid
flowchart LR
    HOST["Host Session<br/>Architecture & Clock Profiling"]

    subgraph MCU ["Physical Microcontroller (Device Under Test)"]
        S1["1. Bracket Entry<br/>Mask Interrupts & Paint Stack"]
        S2[["2. Test Execution<br/>Run on Target Silicon Core"]]
        S3["3. Bracket Exit<br/>Sample Cycles, SysTick & Peak Stack"]
        S1 --> S2 --> S3
    end

    HOST -->|" Dispatch Benchmark "| S1
    S3 -->|" Metrics Telemetry Frame "| HOST
```

### 3.4 Hardware-in-the-Loop Simulation (Closed-Loop Control)

Closed-loop verification separates into two distinct configurations depending
on whether the device under test (DUT) integrates ETS or runs unmodified
production firmware:

#### 3.4.1 Modified DUT (Internal Plant Simulation)

```mermaid
flowchart LR
    HOST["Host Environment<br/>(Plant Simulation & Step Orchestrator)"]

    subgraph MCU ["Target Microcontroller (Modified DUT)"]
        ETS[["Embedded Test Server (ETS)<br/>State Injection & Telemetry Streaming"]]
        CTRL["Discrete Controller<br/>Step Computation & Timing Guard"]
    end

    HOST -->|" 1. Step Command & Injected Plant State "| ETS
    ETS -->|" 2. State Input "| CTRL
    CTRL -->|" 3. Actuator Output "| ETS
    ETS -->|" 4. Control Response Telemetry "| HOST
```

#### 3.4.2 Unmodified DUT (External ETS Rig)

```mermaid
flowchart LR
    HOST["Host Environment<br/>(Test Orchestration & Telemetry Capture)"]

    subgraph RIG ["Test Rig Microcontroller (ETS Harness)"]
        ETS[["Embedded Test Server (ETS)<br/>Signal Generation & Sensor Sampling"]]
    end

    subgraph DUT ["Device Under Test (Unmodified Target)"]
        CTRL["Production Firmware<br/>(Closed-Loop Controller Execution)"]
    end

    HOST -->|" 1. Scenario Command (Host Link) "| ETS
    ETS ==>|" 2. Electrical Stimulus & Sensor Emulation "| CTRL
    CTRL ==>|" 3. Physical Actuation & Control Signals "| ETS
    ETS -->|" 4. Telemetry Frame Stream (Host Link) "| HOST
```

### 3.5 High-Rate Telemetry & Data Acquisition

```mermaid
flowchart LR
    PLANT["Physical Sensors & Plant<br/>Current, Velocity, Position"]

    subgraph MCU ["Target Microcontroller + ETS"]
        DAQ[["Data Acquisition Runtime<br/>Sample Ingestion & Frame Streaming"]]
    end

    HOST["Host Session<br/>Dataset Recording & Analysis"]
    PLANT -->|" Sensor Ingestion "| DAQ
    DAQ -->|" Telemetry Stream "| HOST
```

### 3.6 Live Parameter Tuning

```mermaid
flowchart LR
    HOST["Host Client (control-rs-tui)<br/>Live Gain Mutation"]

    subgraph MCU ["Running Controller Target"]
        V["1. Validate Bounds<br/>Verify Type & Limit Invariants"]
        U[["2. Atomically Apply<br/>Update Parameter Slot"]]
        C["3. Control Execution<br/>Subsequent Cycles Consume Gain"]
        V --> U --> C
    end

    HOST -->|" Set Parameter Request "| V
    C -->|" Telemetry Response "| HOST
```

