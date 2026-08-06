# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for potential security vulnerabilities.

Instead, use GitHub's **Private Vulnerability Reporting / Security Advisories** for this repository if enabled. If it is
not available, open an issue with minimal details and request a private follow-up from maintainers.

## Supported versions

Security fixes are applied to the default branch and the latest tagged minor
release when one exists. The current `0.2.x` source line is alpha software;
older untagged snapshots are not supported.

## Running generated experiments

Treat generated code and downloaded research artifacts as untrusted. Prefer
the isolated executor, keep network access disabled unless inputs must be
fetched, pin container images by digest, and never mount credential or personal
data directories into an experiment workspace. A valid ARA records provenance;
it is not proof that its code or inputs are safe.
