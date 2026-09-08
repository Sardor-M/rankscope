// The rankscope arithmetic in JavaScript. Kept in step with the Python package by tests/goldens.txt.

export function mean(values) {
  if (values.length === 0) return 0;
  let total = 0;
  for (const v of values) total += v;
  return total / values.length;
}

export function sampleStd(values) {
  const n = values.length;
  if (n < 2) return 0;
  const centre = mean(values);
  let total = 0;
  for (const v of values) total += (v - centre) * (v - centre);
  return Math.sqrt(total / (n - 1));
}

export function wilson(successes, n, z = 1.96) {
  if (n <= 0) return [0, 0];
  const p = successes / n;
  const denominator = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / denominator;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denominator;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

export function ruleOfThree(n) {
  return n > 0 ? 3 / n : 1;
}

export function negativesFor(bound, z = 1.96) {
  let n = 1;
  while (wilson(0, n, z)[1] > bound) n += 1;
  return n;
}

export function nullFloor(scores, sigmas = 3) {
  if (scores.length === 0) return { n: 0, mean: 0, std: 0, floor: Infinity };
  const centre = mean(scores);
  const spread = sampleStd(scores);
  return { n: scores.length, mean: centre, std: spread, floor: centre + sigmas * spread };
}

export function conformalIndex(n, alpha) {
  return Math.ceil((n + 1) * (1 - alpha));
}

export function conformalMinN(alpha) {
  let n = 1;
  while (conformalIndex(n, alpha) > n) n += 1;
  return n;
}

export function conformalQuantile(nonconformity, alpha) {
  const values = [...nonconformity].sort((a, b) => a - b);
  const n = values.length;
  const index = conformalIndex(n, alpha);
  if (n === 0 || index > n) return Infinity;
  return values[index - 1];
}

export function hitRate(ranks, k) {
  if (ranks.length === 0) return 0;
  return ranks.filter((r) => r > 0 && r <= k).length / ranks.length;
}

export function mrr(ranks) {
  if (ranks.length === 0) return 0;
  let total = 0;
  for (const r of ranks) if (r > 0) total += 1 / r;
  return total / ranks.length;
}

export function recallAt(docs, relevant, k) {
  const wanted = new Set(relevant);
  if (wanted.size === 0) return 0;
  const found = docs.slice(0, k).filter((d) => wanted.has(d)).length;
  return found / wanted.size;
}

export function ndcgAt(docs, grades, k) {
  let dcg = 0;
  docs.slice(0, k).forEach((doc, i) => {
    const gain = 2 ** (grades[doc] ?? 0) - 1;
    if (gain) dcg += gain / Math.log2(i + 2);
  });
  const ideal = Object.values(grades).filter((g) => g > 0).sort((a, b) => b - a).slice(0, k);
  let idcg = 0;
  ideal.forEach((g, i) => { idcg += (2 ** g - 1) / Math.log2(i + 2); });
  return idcg ? dcg / idcg : 0;
}

export function rrf(lanes, k = 60) {
  const scores = new Map();
  for (const hits of lanes) {
    hits.forEach((doc, i) => scores.set(doc, (scores.get(doc) ?? 0) + 1 / (k + i + 1)));
  }
  return [...scores.entries()]
    .sort((a, b) => (b[1] !== a[1] ? b[1] - a[1] : a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .map(([doc, score]) => ({ doc, score }));
}

export function rankOf(hits, relevant, level = "doc", groupOf = null) {
  let wanted = new Set(relevant);
  if (level === "group") wanted = new Set([...wanted].map(groupOf));
  for (let i = 0; i < hits.length; i += 1) {
    const key = level === "doc" ? hits[i].doc : groupOf(hits[i].doc);
    if (wanted.has(key)) return i + 1;
  }
  return 0;
}
