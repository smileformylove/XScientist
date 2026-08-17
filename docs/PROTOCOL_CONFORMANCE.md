# Protocol producer conformance

External agents and platforms can validate their JSON producer independently of
XScientist's orchestration runtime.

```bash
xscientist conformance init ./xscientist-conformance
xscientist conformance check ./xscientist-conformance
```

The generated kit contains a canonical known-good Research Object and a
known-bad object whose content was changed after content addressing. Passing
means the good object validates and the bad object is rejected.

Validate one artifact against any packaged schema:

```bash
xscientist conformance check object.json --schema research_object
```

Validation uses the packaged Draft 2020-12 registry offline, including local
resolution for `$ref`. Research Objects receive both JSON Schema validation and
canonical identity/content-hash validation.

Conformance proves wire-format compatibility. It does not grant independent
scientific authority or promote a claim.
