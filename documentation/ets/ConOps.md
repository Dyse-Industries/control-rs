# Concept of Operations (ConOps): Embedded Test Server (ETS)

A visual guide to testing, benchmarking, data acquisition, and live tuning on
virtual and physical microcontrollers.

---

## 1. What is ETS?

1. **It Lives on the Chip:** Built directly into the microcontroller firmware so
   the chip can test and monitor itself.
2. **It Connects to Your Computer:** Talks to developer laptops or automated
   cloud servers over a cable or virtual connection.
3. **It Tests, Records & Tunes Live:** Runs tests, measures speed and memory,
   logs sensor data, and adjusts live parameters.

```mermaid
flowchart LR
    subgraph HOST ["Host Computer (Engineer Laptop or Cloud Server)"]
        CONSOLE["Interactive Console<br/>(For Humans)"]
        RUNNER["Automated Runner<br/>(For CI Bots)"]
    end

    subgraph TARGET ["Target Microcontroller (Physical Chip or Emulator)"]
        ETS[["Embedded Test Server (ETS)"]]
        UT["Run Tests"]
        LOOP["Control Plant"]
        GAINS["Tune Live"]
        ETS --- UT
        ETS --- LOOP
        ETS --- GAINS
    end

    subgraph PLANT ["Physical Machine (When Connected)"]
        HW["Motors · Power Electronics<br/>Sensors · Encoders"]
    end

    HOST <-->|" Commands & Tuning /<br/>Results, Timing & Data "| ETS
    ETS <==>|" Drive Signals /<br/>Sensor Readings "| PLANT
```

---

## 2. Who Uses ETS and Why

Different team members use ETS to test different parts of the system and answer
different questions:

```mermaid
flowchart TD
    subgraph PERSONAS ["Who is Testing?"]
        P1["Control Systems Engineer<br/><b>Focus:</b> Algorithm Stability & Math"]
        P2["Firmware Engineer<br/><b>Focus:</b> Chip Speed & Memory Limits"]
        P3["QA & Test Engineer<br/><b>Focus:</b> Test Coverage & Parameter Sweeps"]
        P4["DevOps & CI Automation<br/><b>Focus:</b> Automated Regression Prevention"]
    end

    subgraph ETS_FRONTENDS ["Which Interface Do They Use?"]
        TUI_BOX["Visual Dashboard (control-rs-tui)<br/>Interactive Terminal Console"]
        CI_BOX["Headless Command Line (control-rs-ci)<br/>Automated Quality Gate Runner"]
    end

    subgraph SYSTEM_LAYERS ["What Part of the Code Gets Tested?"]
        MATH_LAYER["Layer 1: Math & Numerics<br/>(Fixed-Point, Complex Numbers, Matrices)"]
        CONTROL_LAYER["Layer 2: Control Algorithms<br/>(PID Controllers, State-Space Models)"]
        RT_LAYER["Layer 3: Real-Time Tasks<br/>(Periodic Tasks, Parameter Tables)"]
        HW_LAYER["Layer 4: Target Silicon & Drivers<br/>(Interrupts, Cycle Counters, Stack RAM)"]
    end

    P1 -->|" tunes & inspects "| TUI_BOX
    P2 -->|" profiles & debugs "| TUI_BOX
    P3 -->|" sweeps & asserts "| TUI_BOX
    P3 -->|" batches tests "| CI_BOX
    P4 -->|" automates gates "| CI_BOX
    TUI_BOX --> CONTROL_LAYER
    TUI_BOX --> RT_LAYER
    TUI_BOX --> HW_LAYER
    CI_BOX --> MATH_LAYER
    CI_BOX --> CONTROL_LAYER
```

---

## 3. Six Ways ETS is Used

| #     | Use Case                 | Setup                                                | What Crosses the Boundary                               | Operational Goal                                 |
|:------|:-------------------------|:-----------------------------------------------------|:--------------------------------------------------------|:-------------------------------------------------|
| **1** | **Unit Test**            | Host + Cargo                                         | Test selection → Pass/fail verdict                      | Verify math and edge cases                       |
| **2** | **Emulator**             | Host + Qemu + Cargo                                  | Virtual link commands → Test summary                    | Catch bugs without physical hardware             |
| **3** | **Device Under Test**    | Physical board plugged into host                     | Benchmark request → Clock cycles, time & RAM            | Prove code fits chip speed and memory limits     |
| **4** | **Hardware in the Loop** | Modified DUT (sim plant) or Unmodified DUT (ETS rig) | Sensor feedback / stimulus ↔ Control commands & reports | Verify closed-loop stability, timing & safe stop |
| **5** | **Data Acquisition**     | Chip wired to physical sensors & hardware            | Test stimulus & sensor stream → Recorded dataset        | Characterize hardware and build plant models     |
| **6** | **Hand-Tuning**          | Console attached to a running controller             | Live parameter tweak ↔ Immediate response change        | Dial in smooth performance on the bench          |

### 3.1 Math & Algorithm Logic

```mermaid
flowchart LR
    HOST["Host Computer<br/>(Engineer or Automated Runner)"]

    subgraph TARGET ["Target Microcontroller"]
        ETS[["Embedded Test Server<br/>Runs Test & Catches Faults"]]
        ALG["Equations Under Test<br/>Known Inputs → Expected Outputs"]
    end

    HOST -->|" 1. Select & Run Test "| ETS
    ETS -->|" 2. run "| ALG
    ETS -->|" 3. Pass / Fail Verdict "| HOST
```

### 3.2 Virtual Chip on a Host

```mermaid
flowchart LR
    subgraph SERVER ["Laptop or Cloud Server (No Physical Board Attached)"]
        RUNNER["Automated Test Runner<br/>1. Starts Simulator · 4. Collects Summary"]

        subgraph SIM ["Chip Emulator"]
            VCHIP[["Simulated Microcontroller + ETS<br/>2. Emulates Processor & Memory<br/>3. Runs Tests & Exits When Done"]]
        end

        RUNNER <-->|" Virtual Link "| VCHIP
        VCHIP -.->|" Test Report & Exit "| RUNNER
    end
```

### 3.3 Benchmarking Real Silicon

```mermaid
flowchart LR
    HOST["Host Console<br/>Chip Identity · Speed · Memory"]

    subgraph MCU ["Physical Microcontroller (Device Under Test)"]
        S1["1. Prepare<br/>Isolate Run & Mark Memory"]
        S2[["2. Execute<br/>Run on Physical Processor"]]
        S3["3. Measure<br/>Clock Cycles, Time & Peak Memory"]
        S1 --> S2 --> S3
    end

    HOST -->|" Run Benchmark "| S1
    S3 -->|" Speed & Memory Report "| HOST
```

### 3.4 Hardware in the Loop (Closed-Loop Control)

Closed-loop verification separates into two distinct configurations depending
on whether the device under test (DUT) is modified with ETS or runs unmodified
production firmware:

#### 3.4.1 Modified DUT

```mermaid
flowchart LR
    HOST["Host Computer<br/>(Simulated Plant Physics & Test Runner)"]

    subgraph MCU ["Target Microcontroller (Modified DUT)"]
        ETS[["Embedded Test Server (ETS)<br/>Dispatches Test · Injects State · Reports"]]
        CTRL["Active Controller<br/>Computes Control · Monitors Timing<br/>Safe Shutdown on Stop or Fault"]
    end

    HOST -->|" 1. Test Dispatch & Simulated State "| ETS
    ETS -->|" 2. State Input "| CTRL
    CTRL -->|" 3. Control Output "| ETS
    ETS -->|" 4. Control Signals & Telemetry "| HOST
```

#### 3.4.2 Unmodified DUT

```mermaid
flowchart LR
    HOST["Host Computer<br/>(Test Runner & Telemetry Console)"]

    subgraph RIG ["Physical Test Rig (Runs ETS)"]
        ETS[["Embedded Test Server (ETS)<br/>Test Dispatch · Signal Drive/Measure · Reporting"]]
    end

    subgraph DUT ["Target Microcontroller (Unmodified DUT)"]
        CTRL["Production Controller Firmware<br/>(Unmodified Target · No ETS)"]
    end

    HOST -->|" 1. Test Dispatch (Host Link) "| ETS
    ETS ==>|" 2. Sensor Drive & Electrical Stimulus "| CTRL
    CTRL ==>|" 3. Drive Output & Control Signals "| ETS
    ETS -->|" 4. Test Reporting & Telemetry (Host Link) "| HOST
```

### 3.5 Data Acquisition & Experimentation

```mermaid
flowchart LR
    PLANT["Physical Hardware & Sensors<br/>Motion · Current · Position"]

    subgraph MCU ["Target Microcontroller + ETS"]
        DAQ[["Data Acquisition<br/>Applies Test Stimulus &<br/>Streams Labeled Measurements"]]
    end

    HOST["Host Computer<br/>Recorded Dataset →<br/>System Modeling & Plots"]
    PLANT -->|" Raw Sensor Readings "| DAQ
    DAQ -->|" Continuous Measurement Stream "| HOST
```

### 3.6 Hand-Tuning Parameter Adjustments

```mermaid
flowchart LR
    OP["Engineer at Desk<br/>Nudges Gains or Setpoint<br/>(No Reflashing)"]

    subgraph MCU ["Running Microcontroller"]
        V["1. Validate<br/>Check Safe Min / Max Limits"]
        U[["2. Update<br/>Apply Without Stopping Loop"]]
        C["3. Control<br/>Machine Uses New Gains Live"]
        V --> U --> C
    end

    OP -->|" New Parameter Value "| V
    C -->|" Updated Live Response "| OP
```

