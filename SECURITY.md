# Security policy

## Reporting a vulnerability

Please report security problems privately by email to **support@openbeam.world** with
"V2E security" in the subject. Do not open a public issue for a vulnerability.

Include what you found, how to reproduce it, and which version or commit you used. We will
acknowledge the report, keep you informed while we work on a fix, and credit you in the release
notes unless you ask us not to.

## Scope notes

- Connectors talk to engines and cloud sessions with credentials that you supply. V2E never
  writes credentials into run directories, call ledgers or logs; if you find a credential in
  any of those, that is a security bug.
- Optional perception adapters download model weights only when you ask them to. Report any
  download that happens without an explicit request.
