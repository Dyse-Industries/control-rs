// Task-run tests, included into `server::tests` so their paths are
// `server::tests::test_task_*`.

/// Counts of a run: setup, steps, reset and teardown calls.
fn calls() -> [usize; 4] {
    let c = counts();
    [c.setup, c.steps, c.reset, c.teardown]
}

#[test]
fn test_task_setup_once_then_steps() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[
        (RUN, None, &[1]),
        (RUN, None, &[2]),
        (RUN, None, &[3]),
        (TaskRunState::Pass, None, &[]),
    ];
    let _guard = begin(Config::new(SCRIPT));
    let (res, server) =
        run_tasks(TASKS_PLAIN, Run::new(std::vec![start(0, false)]));
    assert_eq!(res, Err("Exit loop"));
    assert_eq!(calls(), [1, 4, 0, 1], "setup, steps, reset, teardown");
    assert_eq!(counts().setup_before_first_step, 1);
    assert_eq!(final_state(&server), Some((TaskRunState::Pass, None)));
}

#[test]
fn test_task_status_continues_or_ends() {
    const WARNS: &[lifecycle_support::Scripted] = &[
        (TaskRunState::Warn, Some("careful"), &[4]),
        (TaskRunState::Pass, None, &[]),
    ];
    for terminal in [
        TaskRunState::Pass,
        TaskRunState::Fail,
        TaskRunState::Error,
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
            run_tasks(TASKS_PLAIN, Run::new(std::vec![start(0, false)]));
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
        run_tasks(TASKS_PLAIN, Run::new(std::vec![start(0, false)]));
    assert_eq!(counts().steps, 2);
    let sample = sent(&server).into_iter().find_map(|t| match t {
        Telemetry::TaskSample { seq, payload, .. } => {
            Some((seq, payload.to_vec()))
        }
        _ => None,
    });
    assert_eq!(sample, Some((0, std::vec![4])));
    assert!(
        states(&server)
            .contains(&(TaskRunState::Warn, Some("careful".to_string())))
    );
}

#[test]
fn test_task_state_sent_on_change() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[
        (RUN, None, &[1]),
        (RUN, None, &[2]),
        (TaskRunState::Warn, Some("w"), &[3]),
        (TaskRunState::Warn, Some("w"), &[4]),
        (TaskRunState::Warn, Some("x"), &[5]),
        (RUN, None, &[6]),
        (TaskRunState::Pass, None, &[]),
    ];
    let _guard = begin(Config::new(SCRIPT));
    let (_, server) =
        run_tasks(TASKS_PLAIN, Run::new(std::vec![start(0, false)]));
    assert_eq!(
        states(&server),
        [
            (TaskRunState::Running, None),
            (TaskRunState::Warn, Some("w".to_string())),
            (TaskRunState::Warn, Some("x".to_string())),
            (TaskRunState::Running, None),
            (TaskRunState::Pass, None),
        ],
        "a changed packet alone sends no state frame"
    );
}

#[test]
fn test_task_stop_now_at_boundary() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[(RUN, None, &[1])];
    let _guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![start(0, false), Command::Heartbeat, stop()];
    let (_, server) = run_tasks(TASKS_PLAIN, Run::new(cmds));
    let c = counts();
    assert_eq!(c.steps, 2, "no step after the boundary that polled StopNow");
    assert_eq!(c.teardown, 1);
    assert_eq!(final_state(&server), Some((TaskRunState::Aborted, None)));
}

#[test]
fn test_task_teardown_once_per_end_path() {
    const FOREVER: &[lifecycle_support::Scripted] = &[(RUN, None, &[1])];
    const PASS: &[lifecycle_support::Scripted] =
        &[(TaskRunState::Pass, None, &[])];
    let slow = |commands| Run {
        tick_ns: 200_000_000,
        ..Run::new(commands)
    };
    let paths = [
        (
            "terminal status",
            Config::new(PASS),
            TASKS_PLAIN,
            Run::new(std::vec![start(0, false)]),
            TaskRunState::Pass,
        ),
        (
            "stop",
            Config::new(FOREVER),
            TASKS_PLAIN,
            Run::new(std::vec![start(0, false), stop()]),
            TaskRunState::Aborted,
        ),
        (
            "step bound",
            Config::new(FOREVER),
            TASKS_PLAIN,
            Run::new(std::vec![start(2, false)]),
            TaskRunState::Bounded,
        ),
        (
            "setup error",
            Config {
                setup_err: Some("no setup"),
                ..Config::new(FOREVER)
            },
            TASKS_PLAIN,
            Run::new(std::vec![start(0, false)]),
            TaskRunState::Error,
        ),
        (
            "link timeout",
            Config::new(FOREVER),
            TASKS_TIMEOUT,
            slow(std::vec![start(0, false)]),
            TaskRunState::TimedOut,
        ),
    ];
    for (path, config, tasks, run, expect) in paths {
        let _guard = begin(config);
        let (_, server) = run_tasks(tasks, run);
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
fn test_task_stop_keeps_verdict_when_flush_fails() {
    const FOREVER: &[lifecycle_support::Scripted] = &[(RUN, None, &[1])];
    let _guard = begin(Config::new(FOREVER));
    // StopNow records `Aborted`, then the same boundary flushes. A dead link
    // there must not replace that verdict with `link lost`.
    let (res, server) = run_tasks(
        TASKS_PLAIN,
        Run {
            fail_flush_when_drained: true,
            ..Run::new(std::vec![start(0, false), stop()])
        },
    );
    assert_eq!(res, Err("flush failed"));
    assert_eq!(counts().teardown, 1);
    assert_eq!(final_state(&server), Some((TaskRunState::Aborted, None)));
}

#[test]
fn test_task_link_loss_tears_down() {
    const FOREVER: &[lifecycle_support::Scripted] = &[(RUN, None, &[1])];
    let _guard = begin(Config::new(FOREVER));
    let (res, server) = run_tasks(
        TASKS_PLAIN,
        Run {
            tick_ns: 200_000_000,
            ..Run::new(std::vec![start(0, false)])
        },
    );
    assert_eq!(res, Err("Exit loop"));
    assert_eq!(counts().teardown, 1);
    assert_eq!(
        final_state(&server),
        Some((TaskRunState::Error, Some("link lost".to_string())))
    );
}

#[test]
fn test_teardown_report_independent_of_verdict() {
    const PASS: &[lifecycle_support::Scripted] =
        &[(TaskRunState::Pass, None, &[])];
    let _guard = begin(Config {
        teardown_err: Some("teardown failed"),
        ..Config::new(PASS)
    });
    let (_, server) =
        run_tasks(TASKS_PLAIN, Run::new(std::vec![start(0, false)]));
    let report = sent(&server).into_iter().find_map(|t| match t {
        Telemetry::TeardownReport { ok, message, .. } => {
            Some((ok, message.map(str::to_string)))
        }
        _ => None,
    });
    assert_eq!(report, Some((false, Some("teardown failed".to_string()))));
    assert_eq!(final_state(&server), Some((TaskRunState::Pass, None)));
}

#[test]
fn test_task_set_setting_between_steps() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[(RUN, None, &[1])];
    let _setting = lock_setting();
    let _ = TEST_U8_SETTING.set(SettingValue::U8(42));
    let guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![start(0, false), set_u8(7), stop()];
    let (_, server) = run_tasks(TASKS_PLAIN, Run::new(cmds));
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
    let (_, server) = run_tasks(TASKS_PLAIN, Run::new(cmds));
    assert_eq!(
        final_state(&server),
        Some((TaskRunState::Error, Some("reset failed".to_string())))
    );
    assert_eq!(counts().teardown, 1);
    let _ = TEST_U8_SETTING.set(SettingValue::U8(42));
}

#[test]
fn test_task_output_per_step() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[
        (RUN, None, &[10]),
        (RUN, None, &[]),
        (RUN, None, &[12]),
        (TaskRunState::Pass, None, &[]),
    ];
    let _guard = begin(Config::new(SCRIPT));
    let (_, server) =
        run_tasks(TASKS_PLAIN, Run::new(std::vec![start(0, false)]));
    let samples = samples(&server);
    assert_eq!(
        samples,
        [(0, std::vec![10]), (2, std::vec![12])],
        "an empty output yields no sample"
    );
}

#[test]
fn test_task_link_timeout_tears_down() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[(RUN, None, &[1])];
    let slow = || Run {
        tick_ns: 200_000_000,
        ..Run::new(std::vec![start(0, false)])
    };
    let guard = begin(Config::new(SCRIPT));
    let (_, server) = run_tasks(TASKS_TIMEOUT, slow());
    assert_eq!(final_state(&server), Some((TaskRunState::TimedOut, None)));
    assert_eq!(counts().teardown, 1);
    drop(guard);

    // Without a declared timeout the run keeps going until the link ends;
    // FR-5 still requires teardown once setup has run.
    let _guard = begin(Config::new(SCRIPT));
    let (res, server) = run_tasks(TASKS_PLAIN, slow());
    assert_eq!(res, Err("Exit loop"));
    assert_eq!(
        final_state(&server),
        Some((TaskRunState::Error, Some("link lost".to_string())))
    );
    assert_eq!(counts().teardown, 1);
    assert!(counts().steps > 10);
}

#[test]
fn test_task_stats_reported() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[
        (RUN, None, &[1]),
        (RUN, None, &[2]),
        (RUN, None, &[3]),
        (TaskRunState::Pass, None, &[]),
    ];
    let _guard = begin(Config::new(SCRIPT));
    let (_, server) =
        run_tasks(TASKS_PLAIN, Run::new(std::vec![start(0, false)]));
    let reported = sent(&server).into_iter().find_map(|t| match t {
        Telemetry::TaskStats { steps, time_us, .. } => Some((steps, time_us)),
        _ => None,
    });
    assert_eq!(counts().steps, 4);
    // One clock read at setup entry and one at teardown entry, 1 ms apart.
    assert_eq!(reported, Some((4, 1_000)));
}

#[test]
fn test_task_max_steps_bounds_run() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[(RUN, None, &[1])];
    let _guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![
        start(3, false),
        Command::Heartbeat,
        Command::Heartbeat,
        Command::Heartbeat,
        Command::Heartbeat,
    ];
    let (_, server) = run_tasks(TASKS_PLAIN, Run::new(cmds));
    assert_eq!(counts().steps, 3, "exactly n steps");
    assert_eq!(final_state(&server), Some((TaskRunState::Bounded, None)));

    let events = &server.context.comms.events;
    let first = events
        .iter()
        .position(|e| *e == Event::State(TaskRunState::Running))
        .unwrap();
    let last = events
        .iter()
        .rposition(|e| *e == Event::State(TaskRunState::Bounded))
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
fn test_task_input_newest_wins() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[
        (RUN, None, &[]),
        (RUN, None, &[]),
        (RUN, None, &[]),
        (RUN, None, &[]),
        (TaskRunState::Pass, None, &[]),
    ];
    let _guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![
        start(0, false),
        input(5, &[1]),
        input(3, &[9]),
        input(5, &[2]),
    ];
    let (_, server) = run_tasks(TASKS_PLAIN, Run::new(cmds));
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
fn test_task_lockstep_waits_for_input() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[(RUN, None, &[])];
    let guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![
        start(3, true),
        input(0, &[10]),
        input(1, &[11]),
        input(2, &[12]),
    ];
    let (_, server) = run_tasks(TASKS_PLAIN, Run::new(cmds));
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
    let (_, server) = run_tasks(TASKS_PLAIN, Run::new(cmds));
    assert_eq!(counts().steps, 0);
    assert_eq!(
        final_state(&server),
        Some((
            TaskRunState::Error,
            Some("input sequence gap".to_string())
        ))
    );
}

#[test]
fn test_task_lockstep_wait_ends() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[(RUN, None, &[])];
    let guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![start(0, true), stop()];
    let (_, server) = run_tasks(TASKS_PLAIN, Run::new(cmds));
    assert_eq!(final_state(&server), Some((TaskRunState::Aborted, None)));
    assert_eq!(calls(), [1, 0, 0, 1]);
    drop(guard);

    let _guard = begin(Config::new(SCRIPT));
    let run = Run {
        tick_ns: 200_000_000,
        ..Run::new(std::vec![start(0, true)])
    };
    let (_, server) = run_tasks(TASKS_TIMEOUT, run);
    assert_eq!(final_state(&server), Some((TaskRunState::TimedOut, None)));
    assert_eq!(calls(), [1, 0, 0, 1]);
}

#[test]
fn test_task_never_masks_interrupts() {
    const SCRIPT: &[lifecycle_support::Scripted] =
        &[(RUN, None, &[1]), (TaskRunState::Pass, None, &[])];
    let _setting = lock_setting();
    let _guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![start(0, false), Command::Heartbeat, set_u8(1)];
    let (_, server) = run_tasks(TASKS_TIMEOUT, Run::new(cmds));
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
fn test_task_boundary_work_bounded() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[
        (RUN, None, &[1]),
        (RUN, None, &[2]),
        (RUN, None, &[3]),
        (TaskRunState::Pass, None, &[]),
    ];
    let _guard = begin(Config::new(SCRIPT));
    let (_, server) =
        run_tasks(TASKS_PLAIN, Run::new(std::vec![start(0, false)]));
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
fn test_task_rejects_commands_during_run() {
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
    let (_, server) = run_tasks(TASKS_PLAIN, Run::new(cmds));
    assert_eq!(log_count(&server), 4, "four rejections are logged");
    assert_eq!(counts().setup, 1, "the second StartTask did not start");
    let case_states = count_frames(&server, |t| {
        matches!(t, Telemetry::TestStateChange { .. })
    });
    assert_eq!(case_states, 0, "RunExecutable did not run a case");
    assert_eq!(final_state(&server), Some((TaskRunState::Aborted, None)));
    drop(guard);

    // Outside a run, addressing a task as a case and a case as a task is logged.
    let _guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![
        Command::RunExecutable {
            suite_id: 0,
            test_id: 1,
        },
        Command::StartTask {
            suite_id: 0,
            test_id: 0,
            max_steps: 0,
            lockstep: false,
        },
    ];
    let (_, server) = run_tasks(TASKS_PLAIN, Run::new(cmds));
    assert_eq!(log_count(&server), 2);
    assert_eq!(states(&server), []);
}

#[test]
fn test_task_discovery() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[(RUN, None, &[1])];
    let guard = begin(Config::new(SCRIPT));
    let (_, server) =
        run_tasks(TASKS_PLAIN, Run::new(std::vec![Command::ListSuites]));
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
            task_count: 1
        })
    ));
    assert!(matches!(
        after_info.get(4),
        Some(Telemetry::TaskInfo {
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

    // A suite without a task sends no extra frame.
    let _guard = begin(Config::new(SCRIPT));
    let (_, server) = run_tasks(&[], Run::new(std::vec![Command::ListSuites]));
    let task_frames = count_frames(&server, |t| {
        matches!(
            t,
            Telemetry::LifecycleSuite { .. } | Telemetry::TaskInfo { .. }
        )
    });
    assert_eq!(task_frames, 0);
}

#[test]
fn test_second_task_for_suite_skipped() {
    const SCRIPT: &[lifecycle_support::Scripted] =
        &[(TaskRunState::Pass, None, &[])];
    let _guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![Command::ListSuites, start(0, false)];
    let (_, server) = run_tasks(TASKS_TWINS, Run::new(cmds));
    let lifecycle = count_frames(&server, |t| {
        matches!(t, Telemetry::LifecycleSuite { .. })
    });
    assert_eq!(lifecycle, 1, "one lifecycle record per suite");
    assert_eq!(log_count(&server), 1, "the second task is logged once");
    let c: Counts = counts();
    assert_eq!(c.setup, 1, "the first task runs");
    assert_eq!(c.twin_setup, 0, "the second task never runs");
}

#[test]
fn test_task_link_deadline_is_strict() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[(RUN, None, &[1])];
    let _guard = begin(Config::new(SCRIPT));
    // Clock reads at start, then at each boundary: 0, 250, 500, 750 ms with a
    // 500 ms timeout. The deadline itself is still within the timeout.
    let run = Run {
        tick_ns: 250_000_000,
        ..Run::new(std::vec![start(0, false)])
    };
    let (_, server) = run_tasks(TASKS_TIMEOUT, run);
    assert_eq!(final_state(&server), Some((TaskRunState::TimedOut, None)));
    assert_eq!(counts().steps, 3, "timed out after, not at, the deadline");
}

#[test]
fn test_task_stop_rejects_other_ids() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[(RUN, None, &[1])];
    let _guard = begin(Config::new(SCRIPT));
    let wrong_test = Command::StopNow {
        suite_id: 0,
        test_id: 9,
    };
    let wrong_suite = Command::StopNow {
        suite_id: 5,
        test_id: 1,
    };
    let cmds = std::vec![start(0, false), wrong_test, wrong_suite, stop()];
    let (_, server) = run_tasks(TASKS_PLAIN, Run::new(cmds));
    assert_eq!(counts().steps, 3, "only the matching StopNow stops the run");
    assert_eq!(log_count(&server), 2, "both mismatches are logged");
    assert_eq!(final_state(&server), Some((TaskRunState::Aborted, None)));
}

#[test]
fn test_task_lockstep_ignores_stale_input() {
    const SCRIPT: &[lifecycle_support::Scripted] = &[(RUN, None, &[])];
    let _guard = begin(Config::new(SCRIPT));
    let cmds = std::vec![
        start(3, true),
        input(0, &[10]),
        input(0, &[99]),
        input(1, &[11]),
        input(2, &[12]),
    ];
    let (_, server) = run_tasks(TASKS_PLAIN, Run::new(cmds));
    assert_eq!(
        seen_inputs(),
        [
            (Some(std::vec![10]), Some(0)),
            (Some(std::vec![11]), Some(1)),
            (Some(std::vec![12]), Some(2)),
        ],
        "the stale input never reaches a step"
    );
    assert_eq!(log_count(&server), 1, "the stale input is logged");
}

#[test]
fn test_task_message_truncates_at_a_char_boundary() {
    let ascii = "a".repeat(300);
    let wide = std::format!("a{}", "é".repeat(200));
    let lengths = within_deadline(move || {
        [
            truncate_message(&ascii).len(),
            truncate_message(&wide).len(),
            truncate_message("short").len(),
        ]
    });
    assert_eq!(lengths, [MAX_MESSAGE_SIZE, 255, 5]);
}
