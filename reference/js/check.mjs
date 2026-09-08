// Recompute every line of a rankscope goldens file with the JavaScript kernel.
//   node reference/js/check.mjs tests/goldens.txt          report mismatches, exit 1 if any
//   node reference/js/check.mjs tests/goldens.txt --emit   print the file with JS answers (for `rankscope goldens --check`)

import { readFileSync } from "node:fs";
import * as rc from "./rankscope.mjs";

const TOLERANCE = 1e-9;

const num = (v) => (v === Infinity ? "inf" : v === -Infinity ? "-inf" : Number(v.toPrecision(12)).toString());
const parseNum = (t) => (t === "inf" ? Infinity : t === "-inf" ? -Infinity : Number(t));
const parseList = (t) => (t === "-" ? [] : t.split(",").map(parseNum));

function compute(kind, args) {
  switch (kind) {
    case "wilson": { const [lo, hi] = rc.wilson(Number(args[0]), Number(args[1])); return `${num(lo)} ${num(hi)}`; }
    case "rule_of_three": return num(rc.ruleOfThree(Number(args[0])));
    case "negatives_for": return String(rc.negativesFor(Number(args[0])));
    case "null_floor": { const o = rc.nullFloor(parseList(args[1]), Number(args[0])); return `${o.n} ${num(o.mean)} ${num(o.std)} ${num(o.floor)}`; }
    case "conformal": return num(rc.conformalQuantile(parseList(args[1]), Number(args[0])));
    case "hit": return num(rc.hitRate(parseList(args[1]), Number(args[0])));
    case "mrr": return num(rc.mrr(parseList(args[0])));
    case "recall": { const [rel, ranking] = args[1].split("|"); return num(rc.recallAt(ranking.split(","), rel === "-" ? [] : rel.split(","), Number(args[0]))); }
    case "ndcg": {
      const [gradesText, ranking] = args[1].split("|");
      const grades = Object.fromEntries(gradesText.split(",").map((p) => { const [d, g] = p.split(":"); return [d, Number(g)]; }));
      return num(rc.ndcgAt(ranking.split(","), grades, Number(args[0])));
    }
    case "rrf": return rc.rrf(args[1].split(";").map((l) => l.split(",")), Number(args[0])).map((h) => `${h.doc}:${num(h.score)}`).join(",");
    case "rank": {
      const pairs = args[1].split(",").map((p) => p.split(":"));
      const groupOf = (doc) => (pairs.find(([, d]) => d === doc) ?? [doc])[0];
      return String(rc.rankOf(pairs.map(([, d]) => ({ doc: d })), args[2].split("+"), args[0], groupOf));
    }
    default: throw new Error(`unknown golden kind ${kind}`);
  }
}

function close(expected, actual) {
  if (expected === actual) return true;
  const split = (s) => s.replace(/ /g, ",").split(",");
  const left = split(expected), right = split(actual);
  if (left.length !== right.length) return false;
  return left.every((part, i) => {
    const [la, va] = part.includes(":") ? part.split(":") : ["", part];
    const [lb, vb] = right[i].includes(":") ? right[i].split(":") : ["", right[i]];
    const a = parseNum(va), b = parseNum(vb);
    if (la !== lb) return false;
    if (!Number.isFinite(a) || !Number.isFinite(b)) return a === b;
    return Math.abs(a - b) <= TOLERANCE;
  });
}

const path = process.argv[2];
if (!path) { console.error("usage: node check.mjs GOLDENS [--emit]"); process.exit(2); }
const emit = process.argv.includes("--emit");
const lines = readFileSync(path, "utf8").split("\n");
if (lines[0].trim() !== "rankscope-goldens 1") { console.error("not a rankscope goldens file"); process.exit(2); }
let checked = 0, failed = 0;
const out = [lines[0]];
for (const line of lines.slice(1)) {
  if (!line.trim() || !line.includes(" -> ")) continue;
  const [left, expected] = line.split(" -> ");
  const [kind, ...args] = left.split(" ");
  const actual = compute(kind, args);
  out.push(`${left} -> ${actual}`);
  checked += 1;
  if (!close(expected.trim(), actual)) { failed += 1; if (!emit) console.log(`MISMATCH ${line}\n    javascript: ${actual}`); }
}
if (emit) { console.log(out.join("\n")); }
else { console.log(`${checked} golden cases, ${failed} mismatching`); process.exit(failed ? 1 : 0); }
