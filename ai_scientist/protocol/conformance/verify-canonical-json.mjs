#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(process.argv[2] || resolve(here, "canonical-json-v1.json"));

function validateString(value) {
  for (let index = 0; index < value.length; index += 1) {
    const unit = value.charCodeAt(index);
    if (unit >= 0xd800 && unit <= 0xdbff) {
      const next = value.charCodeAt(index + 1);
      if (!(next >= 0xdc00 && next <= 0xdfff)) throw new Error("lone high surrogate");
      index += 1;
    } else if (unit >= 0xdc00 && unit <= 0xdfff) {
      throw new Error("lone low surrogate");
    }
  }
  return value;
}

function canonical(value) {
  if (value === null) return "null";
  if (value === true) return "true";
  if (value === false) return "false";
  if (typeof value === "string") return JSON.stringify(validateString(value));
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("non-finite number");
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (typeof value === "object") {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(validateString(key))}:${canonical(value[key])}`)
      .join(",")}}`;
  }
  throw new Error(`unsupported value: ${typeof value}`);
}

const fixture = JSON.parse(await readFile(fixturePath, "utf8"));
let failures = 0;
for (const testCase of fixture.cases) {
  const actual = canonical(testCase.value);
  const digest = `sha256:${createHash("sha256").update(actual, "utf8").digest("hex")}`;
  if (actual !== testCase.canonical || digest !== testCase.sha256) {
    failures += 1;
    process.stderr.write(`${testCase.name}: canonical JSON conformance failed\n`);
  }
}
if (failures) process.exit(1);
process.stdout.write(
  JSON.stringify({ profile: fixture.profile, cases: fixture.cases.length, ok: true }) + "\n",
);
