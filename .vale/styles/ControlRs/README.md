# ControlRs Vale Style Package

Prose, perspective, and technical style rules for `control-rs` safety-critical control software.

## Rules

| Rule | Level | Description |
|:---|:---|:---|
| `Perspective` | `error` | Enforces objective third-person for concepts and imperative for procedures (bans *we*, *I*, *our*, *let's*, etc.). |
| `AntiMeta` | `error` | Enforces elimination of draft backstory, commit commentary, and pipeline meta-narrative. |
| `DefensiveFiller` | `error` | Flags conversational padding and defensive justifications (*obviously*, *simply*, *clearly*, etc.). |
| `Tense` | `error` | Flags future-tense auxiliaries in favor of present-tense descriptions. |
| `Wordiness` | `error` | Flags bloated phrases (*in order to*, *prior to*, *due to the fact that*, etc.). |
| `Spelling` | `error` | Spell check against the `ControlRs` vocabulary; skips `snake_case` identifiers and subscript symbols (`K_p`, `T_s`) in every scope, including source comments. Replaces `Vale.Spelling`. |

## Export / Packaging

This package is self-contained. To export for other projects:
1. Copy or archive the `ControlRs/` directory.
2. In the target project's `.vale.ini`, place it under `StylesPath` and include `ControlRs` in `BasedOnStyles`.
