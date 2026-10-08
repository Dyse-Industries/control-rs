// Loop-run tests, included into `server::tests` so their paths are
// `server::tests::test_loop_*`.

/// Counts of a run: setup, steps, reset and teardown calls.
fn calls() -> [usize; 4] {
    let c = counts();
    [c.setup, c.steps, c.reset, c.teardown]
}

#[test]
fn test_loop_setup_once_then_steps() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[
        (RUN, None, &[1]),
        (RUN, None, &[2]),
        (RUN, None, &[3]),
        (LoopRunState::Pass, None, &[]),
    ];
    let _guard = begin(Config::new(SCRIPT));
    let (res, server) =
        run_loops(LOOPS_PLAIN, Run::new(std::vec![start(0, false)]));
    assert_eq!(res, Err("Exit loop"));
    assert_eq!(calls(), [1, 4, 0, 1], "setup, steps, reset, teardown");
    assert_eq!(counts().setup_before_first_step, 1);
    assert_eq!(final_state(&server), Some((LoopRunState::Pass, None)));
}

#[test]
fn test_loop_status_continues_or_ends() {
    const WARNS: &[lifecycle_support::Scripted] = &[
        (LoopRunState::Warn, Some("careful"), &[4]),
        (LoopRunState::Pass, None, &[]),
    ];
    for terminal in [
        LoopRunState::Pass,
        LoopRunState::Fail,
        LoopRunState::Error,
    ] {
        let script: &'static [lifecycle_support::Scripted] = std::boxed::Box::leak(
            std::vec![
                (RUN, None, &[1][..]),
                (terminal, Some("end"), &[][..]),
                (RUN, None, &[9][..]),
            ]
            .into_boxed_slice(),
        );
        let _guard = begin(Config::new(script));
        let (_, server) =
            run_loops(LOOPS_PLAIN, Run::new(std::vec![start(0, false)]));
        let c = counts();
        assert_eq!(c.steps, 2, "{terminal:?}: no step after a terminal status");
        assert_eq!(c.teardown, 1);
        assert_eq!(
            final_state(&server),
            Some((terminal, Some("end".to_string())))
        );
    }

    // `Warn` continues and carries its packet.
    let _guard = begin(Config::new(WARNS));
    let (_, server) =
        run_loops(LOOPS_PLAIN, Run::new(std::vec![start(0, false)]));
    assert_eq!(counts().steps, 2);
    let sample = sent(&server).into_iter().find_map(|t| match t {
        Telemetry::LoopSample { seq, payload, .. } => {
            Some((seq, payload.to_vec()))
        }
        _ => None,
    });
    assert_eq!(sample, Some((0, std::vec![4])));
    assert!(
        states(&server)
            .contains(&(LoopRunState::Warn, Some("careful".to_string())))
    );
}

#[test]
fn test_loop_state_sent_on_change() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[
        (RUN, None, &[1]),
        (RUN, None, &[2]),
        (LoopRunState::Warn, Some("w"), &[3]),
        (LoopRunState::Warn, Some("w"), &[4]),
        (LoopRunState::Warn, Some("x"), &[5]),
        (RUN, None, &[6]),
        (LoopRunState::Pass, None, &[]),
    ];
    let _guard = begin(Config::new(SCRIPT));
    let (_, server) =
        run_loops(LOOPS_PLAIN, Run::new(std::vec![start(0, false)]));
    assert_eq!(
        states(&server),
        [
            (LoopRunState::Running, None),
            (LoopRunState::Warn, Some("w".to_string())),
            (LoopRunState::Warn, Some("x".to_string())),
            (LoopRunState::Running, None),
            (LoopRunState::Pass, None),
        ],
        "a changed packet alone sends no state frame"
    );
}

#[test]
fn test_loop_stop_now_at_boundary() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[(RUN, None, &[1])];
    let _guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![start(0, false), Command::Heartbeat, stop()];
    let (_, server) = run_loops(LOOPS_PLAIN, Run::new(cmds));
    let c = counts();
    assert_eq!(c.steps, 2, "no step after the boundary that polled StopNow");
    assert_eq!(c.teardown, 1);
    assert_eq!(final_state(&server), Some((LoopRunState::Aborted, None)));
}

#[test]
fn test_loop_teardown_once_per_end_path() {
    const FOREVER: &[lifecycle_support::Scripted] = &[(RUN, None, &[1])];
    const PASS: &[lifecycle_support::Scripted] =
        &[(LoopRunState::Pass, None, &[])];
    let slow = |commands| Run {
        tick_ns: 200_000_000,
        ..Run::new(commands)
    };
    let paths = [
        (
            "terminal status",
            Config::new(PASS),
            LOOPS_PLAIN,
            Run::new(std::vec![start(0, false)]),
            LoopRunState::Pass,
        ),
        (
            "stop",
            Config::new(FOREVER),
            LOOPS_PLAIN,
            Run::new(std::vec![start(0, false), stop()]),
            LoopRunState::Aborted,
        ),
        (
            "step bound",
            Config::new(FOREVER),
            LOOPS_PLAIN,
            Run::new(std::vec![start(2, false)]),
            LoopRunState::Bounded,
        ),
        (
            "setup error",
            Config {
                setup_err: Some("no setup"),
                ..Config::new(FOREVER)
            },
            LOOPS_PLAIN,
            Run::new(std::vec![start(0, false)]),
            LoopRunState::Error,
        ),
        (
            "link timeout",
            Config::new(FOREVER),
            LOOPS_TIMEOUT,
            slow(std::vec![start(0, false)]),
            LoopRunState::TimedOut,
        ),
    ];
    for (path, config, loops, run, expect) in paths {
        let _guard = begin(config);
        let (_, server) = run_loops(loops, run);
        assert_eq!(counts().teardown, 1, "{path}: teardown called once");
        assert_eq!(final_state(&server).map(|s| s.0), Some(expect), "{path}");
        let reports = count_frames(&server, |t| {
            matches!(t, Telemetry::TeardownReport { .. })
        });
        assert_eq!(reports, 1, "{path}: one TeardownReport");
        if path == "setup error" {
            assert_eq!(counts().steps, 0, "a failed setup runs no step");
        }
    }
}

#[test]
fn test_teardown_report_independent_of_verdict() {
    const PASS: &[lifecycle_support::Scripted] =
        &[(LoopRunState::Pass, None, &[])];
    let _guard = begin(Config {
        teardown_err: Some("teardown failed"),
        ..Config::new(PASS)
    });
    let (_, server) =
        run_loops(LOOPS_PLAIN, Run::new(std::vec![start(0, false)]));
    let report = sent(&server).into_iter().find_map(|t| match t {
        Telemetry::TeardownReport { ok, message, .. } => {
            Some((ok, message.map(str::to_string)))
        }
        _ => None,
    });
    assert_eq!(report, Some((false, Some("teardown failed".to_string()))));
    assert_eq!(final_state(&server), Some((LoopRunState::Pass, None)));
}

#[test]
fn test_loop_set_setting_between_steps() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[(RUN, None, &[1])];
    let _setting = lock_setting();
    let _ = TEST_U8_SETTING.set(SettingValue::U8(42));
    let guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![start(0, false), set_u8(7), stop()];
    let (_, server) = run_loops(LOOPS_PLAIN, Run::new(cmds));
    assert_eq!(counts().reset, 1, "reset is called once");
    assert_eq!(counts().reset_saw_setting, 7, "reset sees the new value");
    let confirmed = count_frames(&server, |t| {
        matches!(
            t,
            Telemetry::SettingInfo {
                value: SettingValue::U8(7),
                ..
            }
        )
    });
    assert_eq!(confirmed, 1, "the update is confirmed by SettingInfo");
    drop(guard);

    // A reset `Err` ends the run with `Error` and teardown.
    let _guard = begin(Config {
        reset_err: Some("reset failed"),
        ..Config::new(SCRIPT)
    });
    let cmds = std::vec![start(0, false), set_u8(8)];
    let (_, server) = run_loops(LOOPS_PLAIN, Run::new(cmds));
    assert_eq!(
        final_state(&server),
        Some((LoopRunState::Error, Some("reset failed".to_string())))
    );
    assert_eq!(counts().teardown, 1);
    let _ = TEST_U8_SETTING.set(SettingValue::U8(42));
}

#[test]
fn test_loop_output_per_step() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[
        (RUN, None, &[10]),
        (RUN, None, &[]),
        (RUN, None, &[12]),
        (LoopRunState::Pass, None, &[]),
    ];
    let _guard = begin(Config::new(SCRIPT));
    let (_, server) =
        run_loops(LOOPS_PLAIN, Run::new(std::vec![start(0, false)]));
    let samples = samples(&server);
    assert_eq!(
        samples,
        [(0, std::vec![10]), (2, std::vec![12])],
        "an empty output yields no sample"
    );
}

#[test]
fn test_loop_link_timeout_tears_down() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[(RUN, None, &[1])];
    let slow = || Run {
        tick_ns: 200_000_000,
        ..Run::new(std::vec![start(0, false)])
    };
    let guard = begin(Config::new(SCRIPT));
    let (_, server) = run_loops(LOOPS_TIMEOUT, slow());
    assert_eq!(final_state(&server), Some((LoopRunState::TimedOut, None)));
    assert_eq!(counts().teardown, 1);
    drop(guard);

    // Without a declared timeout the run keeps going until the link ends.
    let _guard = begin(Config::new(SCRIPT));
    let (res, server) = run_loops(LOOPS_PLAIN, slow());
    assert_eq!(res, Err("Exit loop"));
    assert_eq!(final_state(&server), Some((LoopRunState::Running, None)));
    assert!(counts().steps > 10);
}

#[test]
fn test_loop_stats_reported() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[
        (RUN, None, &[1]),
        (RUN, None, &[2]),
        (RUN, None, &[3]),
        (LoopRunState::Pass, None, &[]),
    ];
    let _guard = begin(Config::new(SCRIPT));
    let (_, server) =
        run_loops(LOOPS_PLAIN, Run::new(std::vec![start(0, false)]));
    let reported = sent(&server).into_iter().find_map(|t| match t {
        Telemetry::LoopStats { steps, time_us, .. } => Some((steps, time_us)),
        _ => None,
    });
    assert_eq!(counts().steps, 4);
    // One clock read at setup entry and one at teardown entry, 1 ms apart.
    assert_eq!(reported, Some((4, 1_000)));
}

#[test]
fn test_loop_max_steps_bounds_run() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[(RUN, None, &[1])];
    let _guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![
        start(3, false),
        Command::Heartbeat,
        Command::Heartbeat,
        Command::Heartbeat,
        Command::Heartbeat,
    ];
    let (_, server) = run_loops(LOOPS_PLAIN, Run::new(cmds));
    assert_eq!(counts().steps, 3, "exactly n steps");
    assert_eq!(final_state(&server), Some((LoopRunState::Bounded, None)));

    let events = &server.context.comms.events;
    let first = events
        .iter()
        .position(|e| *e == Event::State(LoopRunState::Running))
        .unwrap();
    let last = events
        .iter()
        .rposition(|e| *e == Event::State(LoopRunState::Bounded))
        .unwrap();
    let polls = events
        .get(first..last)
        .unwrap()
        .iter()
        .filter(|e| **e == Event::Poll)
        .count();
    assert_eq!(polls, 2, "no command is polled after step n");
}

#[test]
fn test_loop_input_newest_wins() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[
        (RUN, None, &[]),
        (RUN, None, &[]),
        (RUN, None, &[]),
        (RUN, None, &[]),
        (LoopRunState::Pass, None, &[]),
    ];
    let _guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![
        start(0, false),
        input(5, &[1]),
        input(3, &[9]),
        input(5, &[2]),
    ];
    let (_, server) = run_loops(LOOPS_PLAIN, Run::new(cmds));
    let seen = seen_inputs();
    assert_eq!(seen.first(), Some(&(None, None)), "no input before the first");
    assert_eq!(seen.get(1), Some(&(Some(std::vec![1]), Some(5))));
    assert_eq!(
        seen.get(2),
        Some(&(Some(std::vec![1]), Some(5))),
        "an older seq is ignored"
    );
    assert_eq!(seen.get(3), Some(&(Some(std::vec![2]), Some(5))));
    assert_eq!(log_count(&server), 1, "the stale input is logged");
}

#[test]
fn test_loop_lockstep_waits_for_input() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[(RUN, None, &[])];
    let guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![
        start(3, true),
        input(0, &[10]),
        input(1, &[11]),
        input(2, &[12]),
    ];
    let (_, server) = run_loops(LOOPS_PLAIN, Run::new(cmds));
    assert_eq!(
        seen_inputs(),
        [
            (Some(std::vec![10]), Some(0)),
            (Some(std::vec![11]), Some(1)),
            (Some(std::vec![12]), Some(2)),
        ],
        "step k sees input k"
    );
    let samples = server
        .context
        .comms
        .events
        .iter()
        .filter(|e| matches!(e, Event::Sample(_)))
        .count();
    assert_eq!(samples, 3, "every step sends its output, empty included");
    drop(guard);

    // An input above the awaited index ends the run.
    let _guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![start(0, true), input(2, &[1])];
    let (_, server) = run_loops(LOOPS_PLAIN, Run::new(cmds));
    assert_eq!(counts().steps, 0);
    assert_eq!(
        final_state(&server),
        Some((
            LoopRunState::Error,
            Some("input sequence gap".to_string())
        ))
    );
}

#[test]
fn test_loop_lockstep_wait_ends() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[(RUN, None, &[])];
    let guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![start(0, true), stop()];
    let (_, server) = run_loops(LOOPS_PLAIN, Run::new(cmds));
    assert_eq!(final_state(&server), Some((LoopRunState::Aborted, None)));
    assert_eq!(calls(), [1, 0, 0, 1]);
    drop(guard);

    let _guard = begin(Config::new(SCRIPT));
    let run = Run {
        tick_ns: 200_000_000,
        ..Run::new(std::vec![start(0, true)])
    };
    let (_, server) = run_loops(LOOPS_TIMEOUT, run);
    assert_eq!(final_state(&server), Some((LoopRunState::TimedOut, None)));
    assert_eq!(calls(), [1, 0, 0, 1]);
}

#[test]
fn test_loop_never_masks_interrupts() {
    const SCRIPT: &[lifecycle_support::Scripted] =
        &[(RUN, None, &[1]), (LoopRunState::Pass, None, &[])];
    let _setting = lock_setting();
    let _guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![start(0, false), Command::Heartbeat, set_u8(1)];
    let (_, server) = run_loops(LOOPS_TIMEOUT, Run::new(cmds));
    assert_eq!(
        server
            .context
            .cpu_utils
            .masked
            .load(std::sync::atomic::Ordering::SeqCst),
        0
    );
    let _ = TEST_U8_SETTING.set(SettingValue::U8(42));
}

#[test]
fn test_loop_boundary_work_bounded() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[
        (RUN, None, &[1]),
        (RUN, None, &[2]),
        (RUN, None, &[3]),
        (LoopRunState::Pass, None, &[]),
    ];
    let _guard = begin(Config::new(SCRIPT));
    let (_, server) =
        run_loops(LOOPS_PLAIN, Run::new(std::vec![start(0, false)]));
    let events = &server.context.comms.events;
    let sample_at: std::vec::Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|(_, e)| matches!(e, Event::Sample(_)))
        .map(|(i, _)| i)
        .collect();
    assert_eq!(sample_at.len(), 3);
    for pair in sample_at.windows(2) {
        let (from, to) = (pair.first().unwrap(), pair.get(1).unwrap());
        let between = events.get(from.saturating_add(1)..*to).unwrap();
        let count = |pick| count_events(between, pick);
        assert!(count(|e| *e == Event::Poll) <= 1, "at most one poll");
        assert!(count(|e| *e == Event::Flush) <= 1, "at most one flush");
        assert!(count(|e| matches!(e, Event::State(_))) <= 1, "one state");
    }
}

#[test]
fn test_loop_rejects_commands_during_run() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[(RUN, None, &[1])];
    let guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![
        start(0, false),
        Command::ListSuites,
        Command::RunExecutable {
            suite_id: 0,
            test_id: 0,
        },
        start(0, false),
        Command::SetSetting {
            suite_id: 5,
            setting_id: 0,
            value: SettingValue::U8(1),
        },
        stop(),
    ];
    let (_, server) = run_loops(LOOPS_PLAIN, Run::new(cmds));
    assert_eq!(log_count(&server), 4, "four rejections are logged");
    assert_eq!(counts().setup, 1, "the second StartLoop did not start");
    let case_states = count_frames(&server, |t| {
        matches!(t, Telemetry::TestStateChange { .. })
    });
    assert_eq!(case_states, 0, "RunExecutable did not run a case");
    assert_eq!(final_state(&server), Some((LoopRunState::Aborted, None)));
    drop(guard);

    // Outside a run, addressing a loop as a case and a case as a loop is logged.
    let _guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![
        Command::RunExecutable {
            suite_id: 0,
            test_id: 1,
        },
        Command::StartLoop {
            suite_id: 0,
            test_id: 0,
            max_steps: 0,
            lockstep: false,
        },
    ];
    let (_, server) = run_loops(LOOPS_PLAIN, Run::new(cmds));
    assert_eq!(log_count(&server), 2);
    assert_eq!(states(&server), []);
}

#[test]
fn test_loop_discovery() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[(RUN, None, &[1])];
    let guard = begin(Config::new(SCRIPT));
    let (_, server) =
        run_loops(LOOPS_PLAIN, Run::new(std::vec![Command::ListSuites]));
    let frames = sent(&server);
    let after_info: std::vec::Vec<&Telemetry<'_>> = frames
        .iter()
        .skip_while(|t| !matches!(t, Telemetry::SuiteInfo { .. }))
        .collect();
    assert!(matches!(
        after_info.first(),
        Some(Telemetry::SuiteInfo { .. })
    ));
    assert!(matches!(after_info.get(1), Some(Telemetry::TestInfo { .. })));
    assert!(matches!(
        after_info.get(2),
        Some(Telemetry::SettingInfo { .. })
    ));
    assert!(matches!(
        after_info.get(3),
        Some(Telemetry::LifecycleSuite {
            suite_id: 0,
            loop_count: 1
        })
    ));
    assert!(matches!(
        after_info.get(4),
        Some(Telemetry::LoopInfo {
            suite_id: 0,
            test_id: 1,
            name: "scripted",
            ..
        })
    ));
    assert!(matches!(
        after_info.get(5),
        Some(Telemetry::DiscoveryComplete)
    ));
    drop(guard);

    // A suite without a loop sends no extra frame.
    let _guard = begin(Config::new(SCRIPT));
    let (_, server) = run_loops(&[], Run::new(std::vec![Command::ListSuites]));
    let loop_frames = count_frames(&server, |t| {
        matches!(
            t,
            Telemetry::LifecycleSuite { .. } | Telemetry::LoopInfo { .. }
        )
    });
    assert_eq!(loop_frames, 0);
}

#[test]
fn test_second_loop_for_suite_skipped() {
    const SCRIPT: &[lifecycle_support::Scripted] =
        &[(LoopRunState::Pass, None, &[])];
    let _guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![Command::ListSuites, start(0, false)];
    let (_, server) = run_loops(LOOPS_TWINS, Run::new(cmds));
    let lifecycle = count_frames(&server, |t| {
        matches!(t, Telemetry::LifecycleSuite { .. })
    });
    assert_eq!(lifecycle, 1, "one lifecycle record per suite");
    assert_eq!(log_count(&server), 1, "the second loop is logged once");
    let c: Counts = counts();
    assert_eq!(c.setup, 1, "the first loop runs");
    assert_eq!(c.twin_setup, 0, "the second loop never runs");
}
