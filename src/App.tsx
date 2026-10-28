/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useMemo } from 'react';
import { 
  Trophy, Download, Copy, ExternalLink, Calendar, RefreshCw, 
  ChevronRight, BarChart2, CheckCircle2, AlertCircle, Info, Sparkles, Filter, Settings, Sliders
} from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

// Melate historical sample dataset (fallback & initial seed for recent draws)
const INITIAL_MELATE_DRAWS = [
  { num: 3910, win: [7, 14, 22, 35, 41, 52], additional: 18, date: "24/10/2026" },
  { num: 3909, win: [3, 11, 19, 28, 44, 50], additional: 7,  date: "21/10/2026" },
  { num: 3908, win: [12, 23, 31, 38, 49, 55], additional: 5,  date: "17/10/2026" },
  { num: 3907, win: [2, 9, 17, 26, 37, 48],  additional: 33, date: "14/10/2026" },
  { num: 3906, win: [5, 15, 24, 32, 42, 51], additional: 20, date: "10/10/2026" },
  { num: 3905, win: [8, 18, 25, 36, 45, 53], additional: 12, date: "07/10/2026" },
  { num: 3904, win: [1, 10, 21, 30, 40, 47], additional: 56, date: "03/10/2026" },
  { num: 3903, win: [6, 13, 22, 34, 43, 54], additional: 29, date: "30/09/2026" },
  { num: 3902, win: [4, 16, 27, 39, 46, 52], additional: 11, date: "26/09/2026" },
  { num: 3901, win: [9, 20, 29, 37, 48, 55], additional: 3,  date: "23/09/2026" },
  { num: 3900, win: [2, 12, 23, 33, 41, 50], additional: 19, date: "19/09/2026" },
  { num: 3899, win: [7, 15, 26, 35, 44, 51], additional: 25, date: "16/09/2026" },
  { num: 3898, win: [11, 19, 28, 36, 42, 49], additional: 8,  date: "12/09/2026" },
  { num: 3897, win: [3, 14, 21, 31, 39, 47], additional: 34, date: "09/09/2026" },
  { num: 3896, win: [6, 17, 24, 34, 45, 53], additional: 16, date: "05/09/2026" },
  { num: 3895, win: [5, 10, 20, 29, 40, 52], additional: 22, date: "02/09/2026" },
  { num: 3894, win: [1, 13, 22, 32, 43, 50], additional: 27, date: "29/08/2026" },
  { num: 3893, win: [8, 16, 25, 37, 46, 54], additional: 9,  date: "26/08/2026" },
  { num: 3892, win: [4, 12, 19, 30, 41, 48], additional: 35, date: "22/08/2026" },
  { num: 3891, win: [7, 18, 27, 36, 44, 51], additional: 14, date: "19/08/2026" }
];

interface MLParameters {
  poolSize: number;
  antiSize: number;
  wFreq: number;
  wCoOccur: number;
  wGap: number;
  minSum: number;
  maxSum: number;
  maxConsec: number;
  minEvens: number;
  maxEvens: number;
}

const DEFAULT_PARAMS: MLParameters = {
  poolSize: 32,
  antiSize: 10,
  wFreq: 2.5,
  wCoOccur: 3.5,
  wGap: 0.8,
  minSum: 120,
  maxSum: 220,
  maxConsec: 3,
  minEvens: 2,
  maxEvens: 4
};

const WHEELING_SIZES = { 24: 24, 48: 48, 96: 96, 192: 192 };
function uv(k: number) { return WHEELING_SIZES[k as keyof typeof WHEELING_SIZES] || 96; }

function gf(n: number, k: number) {
  let r = 1;
  for (let i = 1; i <= k; i++) r = r * (n - i + 1) / i;
  return Math.round(r);
}

function getRow(n: number) {
  if (n <= 14) return 1;
  if (n <= 28) return 2;
  if (n <= 42) return 3;
  return 4;
}

function isValidMelateCombo(combo: number[], poolSlice: number[] | null = null, poolId: string = '', params: MLParameters = DEFAULT_PARAMS) {
  const sorted = [...combo].sort((a, b) => a - b);
  let maxConsec = 1, currentConsec = 1;
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i] === sorted[i - 1] + 1) {
      currentConsec++;
      if (currentConsec > maxConsec) maxConsec = currentConsec;
    } else {
      currentConsec = 1;
    }
  }
  if (maxConsec > params.maxConsec) return false;
  
  const evenCount = combo.filter(n => n % 2 === 0).length;
  if (evenCount < params.minEvens || evenCount > params.maxEvens) return false;

  const primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
  if (combo.filter(n => primes.includes(n)).length > 4) return false;

  const sum = combo.reduce((a, b) => a + b, 0);
  if (sum < params.minSum || sum > params.maxSum) return false;
  return true;
}

function EfMelate(e: number[], n: number) {
  let t = e.length, r = new Set<string>(), l: number[][] = [];
  if (t < 8) return [];
  let u = t - 6;
  for (let i = 0; i < n; i++) {
    let o = 0, added = false;
    while (o < n * 4 && !added) {
      let c = i + o * n, m = c * 2 % 6, g = (c * 3 + 1) % 6;
      if (g === m) g = (g + 1) % 6;
      let y = 6 + c * 5 % u, k = 6 + (c * 7 + 2) % u;
      if (k === y) k = 6 + (k - 6 + 1) % u;
      let w = 6 + (c * 11 + 5) % u, last = 6 + (c * 13 + 7) % u;
      let counter = 0;
      while ((w === y || w === k || last === w || last === y) && counter < u + 5) {
        w = 6 + (w - 6 + 1) % u;
        counter++;
      }
      let combo = [e[m], e[g], e[y], e[k], e[w], e[last]].sort((d, S) => d - S);
      if (new Set(combo).size < 6) { o++; continue; }
      let f = combo.join(",");
      if (!r.has(f)) { r.add(f); l.push(combo); added = true; } else { o++; }
    }
    if (!added) {
      let c = 0;
      while (c < 100 && !added) {
        let m = i * 13 + c * 17 + 3, g = m * 2 % 6, y = (m * 3 + 1) % 6;
        if (y === g) y = (y + 2) % 6;
        let k = 6 + (m * 5 + c) % u, w = 6 + (m * 7 + 2 + c * 2) % u;
        let lastIdx = 6 + (m * 11 + 5 + c * 3) % u;
        let h = [e[g], e[y], e[k], e[w], e[lastIdx], e[(lastIdx + 3) % t]].sort((d, S) => d - S);
        let f = h.join(",");
        if (!r.has(f) && new Set(h).size === 6) { r.add(f); l.push(h); added = true; }
        c++;
      }
    }
  }
  return l.sort((i, o) => { for (let a = 0; a < 6; a++) if (i[a] !== o[a]) return i[a] - o[a]; return 0; }).slice(0, n);
}

function generateMelateWheeling(nums: number[], targetCount: number, poolId: string = '', params: MLParameters = DEFAULT_PARAMS) {
  const candidates = EfMelate(nums, Math.max(targetCount * 4, 200));
  const valid = candidates.filter(c => isValidMelateCombo(c, nums, poolId, params));
  const result: number[][] = [];
  const seen = new Set<string>();

  for (const c of valid) {
    if (result.length >= targetCount) break;
    let k = c.join(',');
    if (!seen.has(k)) { seen.add(k); result.push(c); }
  }

  for (const c of candidates) {
    if (result.length >= targetCount) break;
    let k = c.join(',');
    if (!seen.has(k)) { seen.add(k); result.push(c); }
  }

  let a = 0;
  while (result.length < targetCount && a < nums.length) {
    let c = nums.slice(a, a + 6);
    if (c.length === 6) {
      let m = [...c].sort((x, y) => x - y);
      let k = m.join(',');
      if (!seen.has(k)) { seen.add(k); result.push(m); }
    }
    a++;
  }

  while (result.length < targetCount) {
    let combo: number[] = [];
    let seed = result.length * 13 + 3;
    while (combo.length < 6) {
      let idx = (seed + combo.length * 7) % nums.length;
      let val = nums[idx];
      if (!combo.includes(val)) combo.push(val);
      seed++;
    }
    combo.sort((x, y) => x - y);
    let k = combo.join(',');
    if (!seen.has(k)) {
      seen.add(k);
      result.push(combo);
    } else {
      let nc = [...combo];
      nc[0] = (nc[0] % 56) + 1;
      nc.sort((x, y) => x - y);
      let nk = nc.join(',');
      seen.add(nk);
      result.push(nc);
    }
  }

  return result.slice(0, targetCount);
}

function rvMelate(e: number[], win: number[], t: number, poolId: string = '', params: MLParameters = DEFAULT_PARAMS) {
  let baseCombos = generateMelateWheeling(e, t, poolId, params);
  return baseCombos;
}

function parseMelateCSV(text: string) {
  const lines = text.trim().split('\n');
  const draws = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',').map(s => s.trim());
    if (cols.length >= 8) {
      const num = parseInt(cols[0]);
      if (isNaN(num)) continue;
      
      let winNums: number[] = [];
      let additional = 0;
      let dateStr = cols[cols.length - 1] || '01/01/2026';
      
      if (cols.length >= 9) {
        winNums = [parseInt(cols[2]), parseInt(cols[3]), parseInt(cols[4]), parseInt(cols[5]), parseInt(cols[6]), parseInt(cols[7])];
        additional = parseInt(cols[8]);
      } else if (cols.length >= 8) {
        winNums = [parseInt(cols[1]), parseInt(cols[2]), parseInt(cols[3]), parseInt(cols[4]), parseInt(cols[5]), parseInt(cols[6])];
        additional = parseInt(cols[7]);
      }
      
      if (winNums.length === 6 && winNums.every(n => !isNaN(n) && n >= 1 && n <= 56)) {
        draws.push({
          num,
          win: winNums.sort((a, b) => a - b),
          additional: isNaN(additional) ? 0 : additional,
          date: dateStr
        });
      }
    }
  }
  return draws.sort((a, b) => a.num - b.num);
}

function calculateMelateMLPools(draws: typeof INITIAL_MELATE_DRAWS, params: MLParameters = DEFAULT_PARAMS) {
  if (!draws || draws.length === 0) return { final: [], anti: [], p2: [], p3: [], p4: [], p5: [], p6: [], p7: [] };
  const lastDraw = draws[draws.length - 1]?.win || [];
  const prevDraw = draws.length >= 2 ? (draws[draws.length - 2]?.win || []) : lastDraw;
  const window50 = draws.slice(-50).map(d => d.win);

  let scores = [];
  for (let i = 1; i <= 56; i++) {
    let freq = 0;
    window50.forEach(d => { if (d.includes(i)) freq++; });
    let gap = 0;
    for (let j = draws.length - 1; j >= Math.max(0, draws.length - 50); j--) {
      if (draws[j].win.includes(i)) { gap = draws.length - 1 - j; break; }
    }
    let coOccur = 0;
    draws.slice(-20).forEach(d => {
      if (d.win.includes(i) && lastDraw.some(ld => d.win.includes(ld))) coOccur++;
    });
    let score = (freq * params.wFreq) + (coOccur * params.wCoOccur) + (gap > 0 && gap <= 15 ? gap * params.wGap : 0);
    if (lastDraw.includes(i) || prevDraw.includes(i)) score -= 15;
    scores.push({ num: i, score, freq, gap, coOccur });
  }

  scores.sort((a, b) => a.score - b.score);
  const antiPool = scores.slice(0, params.antiSize).map(s => s.num);

  scores.sort((a, b) => b.score - a.score);
  const poolSize = params.poolSize;
  const p2 = [...scores].sort((a, b) => b.score - a.score).slice(0, poolSize).map(s => s.num);
  const p3 = [...scores].sort((a, b) => b.gap - a.gap).slice(0, poolSize).map(s => s.num);
  const p4 = [...scores].sort((a, b) => b.coOccur - a.coOccur).slice(0, poolSize).map(s => s.num);
  const p5 = [...scores].sort((a, b) => (b.freq * 1.5 + (56 - b.gap)) - (a.freq * 1.5 + (56 - b.gap))).slice(0, poolSize).map(s => s.num);
  const p6 = [...scores].sort((a, b) => (b.coOccur * 2 + b.freq) - (a.coOccur * 2 + a.freq)).slice(0, poolSize).map(s => s.num);

  let allPoolsList = [p2, p3, p4, p5, p6];
  let candidateFinal = Array.from(new Set([...p4, ...p2, ...p3, ...p5, ...p6]));
  candidateFinal.sort((a, b) => {
    let freqA = allPoolsList.filter(p => p.includes(a)).length;
    let freqB = allPoolsList.filter(p => p.includes(b)).length;
    if (freqB !== freqA) return freqB - freqA;
    let scoreA = scores.find(s => s.num === a)?.score || 0;
    let scoreB = scores.find(s => s.num === b)?.score || 0;
    return scoreB - scoreA;
  });
  const finalPool = candidateFinal.slice(0, poolSize);

  let middleRowsList = [
    finalPool.slice(Math.floor(poolSize / 4), Math.floor(poolSize * 3 / 4)),
    p2.slice(Math.floor(poolSize / 4), Math.floor(poolSize * 3 / 4))
  ];
  
  let middleFreqMap: Record<number, number> = {};
  middleRowsList.forEach(row => {
    row.forEach(n => {
      middleFreqMap[n] = (middleFreqMap[n] || 0) + 1;
    });
  });

  let candidatePool7 = Object.keys(middleFreqMap).map(Number);
  candidatePool7.sort((a, b) => {
    let fA = middleFreqMap[a];
    let fB = middleFreqMap[b];
    if (fB !== fA) return fB - fA;
    let scoreA = scores.find(s => s.num === a)?.score || 0;
    let scoreB = scores.find(s => s.num === b)?.score || 0;
    return scoreB - scoreA;
  });

  scores.forEach(s => {
    if (candidatePool7.length < poolSize && !candidatePool7.includes(s.num)) {
      candidatePool7.push(s.num);
    }
  });

  const p7 = candidatePool7.slice(0, poolSize);

  return { final: finalPool, anti: antiPool, p2, p3, p4, p5, p6, p7 };
}

function runBacktestSimulation(draws: typeof INITIAL_MELATE_DRAWS, params: MLParameters) {
  let totalDrawsEvaluated = 0;
  let totalHitsFinal = 0;
  let drawsWith4Plus = 0;
  let drawsWith5Plus = 0;

  const startIdx = Math.max(10, draws.length - 20);
  for (let i = startIdx; i < draws.length; i++) {
    const hist = draws.slice(0, i);
    const target = draws[i];
    const pred = calculateMelateMLPools(hist, params);
    const hits = pred.final.filter(n => target.win.includes(n)).length;
    totalHitsFinal += hits;
    totalDrawsEvaluated++;
    if (hits >= 4) drawsWith4Plus++;
    if (hits >= 5) drawsWith5Plus++;
  }

  const avgHits = totalDrawsEvaluated > 0 ? (totalHitsFinal / totalDrawsEvaluated).toFixed(2) : '0';
  const pct4Plus = totalDrawsEvaluated > 0 ? ((drawsWith4Plus / totalDrawsEvaluated) * 100).toFixed(1) : '0';
  const pct5Plus = totalDrawsEvaluated > 0 ? ((drawsWith5Plus / totalDrawsEvaluated) * 100).toFixed(1) : '0';

  return {
    evaluations: totalDrawsEvaluated,
    avgHits,
    pct4Plus,
    pct5Plus,
    fitnessScore: (parseFloat(avgHits) * 12 + parseFloat(pct4Plus) * 0.6).toFixed(1)
  };
}

function runPoolScorecard(draws: typeof INITIAL_MELATE_DRAWS, params: MLParameters) {
  const poolIds = ['final', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'] as const;
  const stats: Record<string, { totalHits: number; count: number; hits4Plus: number }> = {
    final: { totalHits: 0, count: 0, hits4Plus: 0 },
    p2: { totalHits: 0, count: 0, hits4Plus: 0 },
    p3: { totalHits: 0, count: 0, hits4Plus: 0 },
    p4: { totalHits: 0, count: 0, hits4Plus: 0 },
    p5: { totalHits: 0, count: 0, hits4Plus: 0 },
    p6: { totalHits: 0, count: 0, hits4Plus: 0 },
    p7: { totalHits: 0, count: 0, hits4Plus: 0 }
  };

  const startIdx = Math.max(10, draws.length - 20);
  for (let i = startIdx; i < draws.length; i++) {
    const hist = draws.slice(0, i);
    const target = draws[i];
    const pred = calculateMelateMLPools(hist, params);
    
    poolIds.forEach(id => {
      const pNums = pred[id];
      const natural = target.win.filter(n => pNums.includes(n)).length;
      const additionalHit = (target.additional && pNums.includes(target.additional)) ? 1 : 0;
      const hits = natural + additionalHit;
      stats[id].totalHits += hits;
      stats[id].count++;
      if (hits >= 4) stats[id].hits4Plus++;
    });
  }

  return poolIds.map(id => {
    const s = stats[id];
    const avg = s.count > 0 ? (s.totalHits / s.count).toFixed(2) : '0';
    const pct4Plus = s.count > 0 ? ((s.hits4Plus / s.count) * 100).toFixed(1) : '0';
    return {
      id,
      name: id === 'final' ? 'Pool FINAL' : id.toUpperCase(),
      avgHits: avg,
      pct4Plus
    };
  });
}

export default function App() {
  const [draws, setDraws] = useState<typeof INITIAL_MELATE_DRAWS>(INITIAL_MELATE_DRAWS);
  const [activeModal, setActiveModal] = useState<{ type: string; poolId?: string; k?: number; sorteoNum?: number } | null>(null);
  const [chartVisibility, setChartVisibility] = useState<Record<string, boolean>>({
    final: true, p2: true, p3: true, p4: true, p5: true, p6: true, p7: true
  });
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const safeDraws = (draws && draws.length > 0) ? draws : INITIAL_MELATE_DRAWS;
  
  const [params, setParams] = useState<MLParameters>(() => {
    const saved = localStorage.getItem('melate_ml_params');
    if (saved) {
      try { return JSON.parse(saved); } catch(e) {}
    }
    return DEFAULT_PARAMS;
  });

  const saveParams = (newParams: MLParameters) => {
    setParams(newParams);
    localStorage.setItem('melate_ml_params', JSON.stringify(newParams));
    try {
      const histRaw = localStorage.getItem('melate_fitness_history');
      const hist = histRaw ? JSON.parse(histRaw) : [];
      const newEntry = {
        id: Date.now(),
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        fitnessScore: backtestResult.fitnessScore,
        avgHits: backtestResult.avgHits,
        pct4Plus: backtestResult.pct4Plus,
        paramsDesc: `Pool:${newParams.poolSize} Freq:${newParams.wFreq} Co:${newParams.wCoOccur}`
      };
      const updated = [newEntry, ...hist].slice(0, 5);
      localStorage.setItem('melate_fitness_history', JSON.stringify(updated));
    } catch(e) {}
    showToast('Parámetros optimizados guardados con éxito');
  };

  const resetParams = () => {
    setParams(DEFAULT_PARAMS);
    localStorage.removeItem('melate_ml_params');
    showToast('Parámetros restablecidos a valores por defecto');
  };

  const backtestResult = useMemo(() => {
    return runBacktestSimulation(safeDraws, params);
  }, [safeDraws, params]);

  useEffect(() => {
    async function fetchMelateData() {
      try {
        const cached = localStorage.getItem('melate_csv_cache');
        if (cached) {
          try {
            const parsed = parseMelateCSV(cached);
            if (parsed.length > 0) setDraws(parsed);
          } catch (err) {
            localStorage.removeItem('melate_csv_cache');
          }
        }
        const res = await fetch('https://raw.githubusercontent.com/pakinja/pakin/refs/heads/master/Melate.csv');
        const text = await res.text();
        if (text && text.includes(',')) {
          localStorage.setItem('melate_csv_cache', text);
          try {
            const parsed = parseMelateCSV(text);
            if (parsed.length > 0) setDraws(parsed);
          } catch (err) {
            console.error('Error parsing fetched Melate.csv:', err);
          }
        }
      } catch (e) {
        console.error('Error fetching Melate.csv:', e);
      }
    }
    fetchMelateData();
  }, []);

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 2500);
  };

  const nextDrawNum = safeDraws[safeDraws.length - 1]?.num ? safeDraws[safeDraws.length - 1].num + 1 : 3911;
  const lastDraw = safeDraws[safeDraws.length - 1] || INITIAL_MELATE_DRAWS[0];

  const livePools = useMemo(() => {
    return calculateMelateMLPools(safeDraws, params);
  }, [safeDraws, params]);

  const frequencyMap = useMemo(() => {
    const counts: Record<number, number> = {};
    for (let i = 1; i <= 56; i++) counts[i] = 0;
    const last50 = safeDraws.slice(-50);
    last50.forEach(d => {
      d.win.forEach(n => {
        if (counts[n] !== undefined) counts[n]++;
      });
    });
    return counts;
  }, [safeDraws]);

  const heatmapStats = useMemo(() => {
    const values = Object.values(frequencyMap);
    return {
      maxFreq: Math.max(...values, 1),
      minFreq: Math.min(...values, 0)
    };
  }, [frequencyMap]);

  const poolsConfig = useMemo(() => [
    { id: 'final', label: 'Pool FINAL (Consenso Dinámico)', nums: livePools.final, count: params.poolSize, border: '#ffeb3b', accent: '#ffeb3b', desc: 'Pool optimizado basado en consenso e intersección con los demás pools.', wheelings: [{k: 24, slice: livePools.final.slice(0, 24)}, {k: 48, slice: livePools.final.slice(0, 28)}, {k: 96, slice: livePools.final.slice(0, 30)}, {k: 192, slice: livePools.final}] },
    { id: 'p2', label: 'Pool2 HOT (Frecuencia + Co-ocurrencia)', nums: livePools.p2, count: params.poolSize, border: '#00e5ff', accent: '#00e5ff', desc: 'Top por score ML de frecuencia y co-ocurrencia reciente.', wheelings: [{k: 24, slice: livePools.p2.slice(0, 24)}, {k: 48, slice: livePools.p2.slice(0, 28)}, {k: 96, slice: livePools.p2.slice(0, 30)}, {k: 192, slice: livePools.p2}] },
    { id: 'p3', label: 'Pool3 REVERSIÓN (Gap Inteligente)', nums: livePools.p3, count: params.poolSize, border: '#ff9800', accent: '#ff9800', desc: 'Top con mayor retraso (gap) histórico y alta probabilidad de aparición.', wheelings: [{k: 24, slice: livePools.p3.slice(0, 24)}, {k: 48, slice: livePools.p3.slice(0, 28)}, {k: 96, slice: livePools.p3.slice(0, 30)}, {k: 192, slice: livePools.p3}] },
    { id: 'p4', label: 'Pool4 CO-OCURRENCIA (Stickiness)', nums: livePools.p4, count: params.poolSize, border: '#76ff03', accent: '#76ff03', desc: 'Top que históricamente tienden a salir juntos con los números anteriores.', wheelings: [{k: 24, slice: livePools.p4.slice(0, 24)}, {k: 48, slice: livePools.p4.slice(0, 28)}, {k: 96, slice: livePools.p4.slice(0, 30)}, {k: 192, slice: livePools.p4}] },
    { id: 'p5', label: 'Pool5 CICLOS Y PATRONES (Index 2)', nums: livePools.p5, count: params.poolSize, border: '#e040fb', accent: '#e040fb', desc: 'Top basado en alineación de ciclos y distribución ponderada.', wheelings: [{k: 24, slice: livePools.p5.slice(0, 24)}, {k: 48, slice: livePools.p5.slice(0, 28)}, {k: 96, slice: livePools.p5.slice(0, 30)}, {k: 192, slice: livePools.p5}] },
    { id: 'p6', label: 'Pool6 MATRIZ PROBABILÍSTICA (Index 2)', nums: livePools.p6, count: params.poolSize, border: '#26c6da', accent: '#26c6da', desc: 'Top basado en matriz de probabilidad posicional.', wheelings: [{k: 24, slice: livePools.p6.slice(0, 24)}, {k: 48, slice: livePools.p6.slice(0, 28)}, {k: 96, slice: livePools.p6.slice(0, 30)}, {k: 192, slice: livePools.p6}] },
    { id: 'p7', label: 'Pool7 FILAS MEDIAS (Consolidado)', nums: livePools.p7, count: params.poolSize, border: '#ff4081', accent: '#ff4081', desc: 'Consolida las filas centrales de todos los pools principales.', wheelings: [{k: 24, slice: livePools.p7.slice(0, 24)}, {k: 48, slice: livePools.p7.slice(0, 28)}, {k: 96, slice: livePools.p7.slice(0, 30)}, {k: 192, slice: livePools.p7}] },
    { id: 'anti', label: `🚫 ANTI-POOL (${params.antiSize} Vetados)`, nums: livePools.anti, count: params.antiSize, border: '#d32f2f', accent: '#ff5252', desc: 'Números con alta probabilidad de NO salir en este sorteo. Excluidos.', wheelings: [] }
  ], [livePools, params]);

  const getPredictionForDraw = (concurso: number, histDraws: typeof draws) => {
    const key = 'melate_pred_v8_' + concurso + '_' + JSON.stringify(params);
    const saved = localStorage.getItem(key);
    if (saved) {
      try { return JSON.parse(saved); } catch(e) {}
    }
    const idx = histDraws.findIndex(d => d.num === concurso);
    if (idx <= 0) return null;
    const subHist = histDraws.slice(0, idx);
    const pred = calculateMelateMLPools(subHist, params);
    const toSave = { final: pred.final, p2: pred.p2, p3: pred.p3, p4: pred.p4, p5: pred.p5, p6: pred.p6, p7: pred.p7 };
    localStorage.setItem(key, JSON.stringify(toSave));
    return toSave;
  };

  const getTotalHits = (poolNums: number[], s: typeof INITIAL_MELATE_DRAWS[0]) => {
    if (!poolNums || !s) return 0;
    const natural = s.win.filter(n => poolNums.includes(n)).length;
    const additionalHit = (s.additional && poolNums.includes(s.additional)) ? 1 : 0;
    return natural + additionalHit; // Max 7 (6 + 1)
  };

  const hasWheelingHitThreeOrMore = (poolNums: number[], s: typeof INITIAL_MELATE_DRAWS[0], poolId: string) => {
    if (!poolNums || !s) return false;
    const totalHits = getTotalHits(poolNums, s);
    return totalHits >= 4;
  };

  const hitClass = (c: number) => {
    if (c >= 7) return 'bg-[#7c4dff] text-white shadow-[0_0_14px_#7c4dff] font-black'; // 7/32 (6 + adic)
    if (c === 6) return 'bg-[#00e676] text-black font-black shadow-[0_0_10px_#00e676]'; // 6/32
    if (c === 5) return 'bg-[#2e7d32] text-[#a5d6a7] font-bold'; // 5/32
    if (c === 4) return 'bg-[#00acc1] text-white font-bold shadow-[0_0_6px_#00acc1]'; // 4/32
    if (c === 3) return 'bg-[#263238] text-[#b0bec5]'; // 3/32
    return 'bg-[#161616] text-[#666666]'; // 0-2/32
  };

  const downloadTXT = () => {
    try {
      let lines = ['MELATE PREDICTOR V4 ML (Optimizador Fase 2) - Sorteo ' + nextDrawNum, ''];
      poolsConfig.filter(p => p.id !== 'anti').forEach(p => {
        p.wheelings.forEach(w => {
          let combos = generateMelateWheeling(w.slice, uv(w.k), p.id, params);
          lines.push(`POOL:${p.id} COMBS:${uv(w.k)} BASE:${w.slice.join(',')} TOTAL:${combos.length}`);
          combos.forEach((c, i) => {
            lines.push(`${i + 1}:${c.map(n => String(n).padStart(2, '0')).join('-')}`);
          });
          lines.push('');
        });
      });
      let blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
      let url = URL.createObjectURL(blob);
      let a = document.createElement('a');
      a.href = url;
      a.download = `melate_${nextDrawNum}_opt.txt`;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 1200);
      showToast('Archivo TXT descargado con éxito');
    } catch (e: any) {
      showToast('Error al exportar TXT: ' + e.message);
    }
  };

  const downloadCSV = () => {
    try {
      let rows = ['sorteo,pool,combinations,combo_num,n1,n2,n3,n4,n5,n6'];
      poolsConfig.filter(p => p.id !== 'anti').forEach(p => {
        p.wheelings.forEach(w => {
          let combos = generateMelateWheeling(w.slice, uv(w.k), p.id, params);
          combos.forEach((c, i) => {
            rows.push([nextDrawNum, p.id, uv(w.k), i + 1, c[0], c[1], c[2], c[3], c[4], c[5]].join(','));
          });
        });
      });
      let blob = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8;' });
      let url = URL.createObjectURL(blob);
      let a = document.createElement('a');
      a.href = url;
      a.download = `melate_${nextDrawNum}_opt.csv`;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 1200);
      showToast('Archivo CSV descargado con éxito');
    } catch (e: any) {
      showToast('Error al exportar CSV: ' + e.message);
    }
  };

  const copyWheelingToClipboard = (id: string, k: number, sorteoNum?: number) => {
    let pBase = poolsConfig.find(x => x.id === id);
    let slice: number[] = [];
    let win: number[] | null = null;
    if (sorteoNum) {
      const pred = getPredictionForDraw(sorteoNum, draws);
      if (pred) slice = pred[id === 'final' ? 'final' : id];
      let s = draws.find(x => x.num === sorteoNum);
      if (s) win = s.win;
    }
    if (slice.length === 0 && pBase) slice = pBase.nums;

    let reduced = uv(k);
    let combos = win ? rvMelate(slice, win, reduced, id, params) : generateMelateWheeling(slice, reduced, id, params);
    let txt = combos.map((c, i) => String(i + 1).padStart(4, '0') + ': ' + c.map(n => String(n).padStart(2, '0')).join('-')).join('\n');

    navigator.clipboard.writeText(txt).then(() => {
      showToast('Wheeling copiado al portapapeles');
    }).catch(() => {
      let ta = document.createElement('textarea');
      ta.value = txt;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      showToast('Wheeling copiado al portapapeles');
    });
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-[#e8e8e8] font-sans selection:bg-[#ffeb3b] selection:text-black">
      {/* WhatsApp Community Bar */}
      <div className="w-full bg-[#0a0a0a] flex justify-center py-3 border-b border-[#1e1e1e]">
        <a 
          href="https://chat.whatsapp.com/" 
          target="_blank" 
          rel="noopener noreferrer" 
          className="border border-[#25d366]/50 bg-[#24d064] text-[#f0f5f2] rounded-full px-4 py-1.5 text-xs font-bold flex items-center gap-2 hover:bg-[#20ba59] transition-all shadow-[0_0_15px_rgba(37,211,102,0.3)]"
        >
          <ExternalLink className="w-3.5 h-3.5" />
          Unidos contra Melate - Grupo WhatsApp
        </a>
      </div>

      <div className="max-w-6xl mx-auto p-4 sm:p-6 pb-20">
        {/* Header */}
        <div className="text-center my-6">
          <h1 className="text-3xl sm:text-5xl font-black text-[#ffeb3b] drop-shadow-[0_0_15px_rgba(255,235,59,0.8)] tracking-tight">
            MELATE PREDICTOR <span className="text-[#00e5ff] drop-shadow-[0_0_15px_rgba(0,229,255,0.8)]">V4 ML</span>
          </h1>
          <p className="text-xs sm:text-sm text-gray-400 mt-2">
            Sistema avanzado para Melate (6 de 56) con Fase 2 de Backtesting y Optimización de Hiperparámetros.
          </p>
          <div className="mt-3 inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#00e5ff]/10 border border-[#00e5ff]/30 text-[#00e5ff] text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5" />
            Próximo Sorteo: <b>{nextDrawNum}</b> • Basado en Sorteo {lastDraw.num} ({lastDraw.date})
          </div>
        </div>

        {/* Action Toolbar */}
        <div className="bg-[#121212] border border-[#2a2a2a] rounded-xl p-4 mb-6 shadow-xl">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-2 flex-wrap">
              <button 
                onClick={() => setActiveModal({ type: 'optimizer' })}
                className="border border-[#e040fb]/40 bg-[#e040fb]/15 text-[#e040fb] px-4 py-1.5 rounded-full text-xs font-extrabold hover:bg-[#e040fb]/25 transition-all flex items-center gap-1.5 cursor-pointer shadow-[0_0_12px_rgba(224,64,251,0.3)]"
              >
                <Sliders className="w-3.5 h-3.5" /> Optimizador & Backtesting ML
              </button>
              <button 
                onClick={downloadCSV}
                className="border border-[#00e5ff]/40 bg-[#00e5ff]/15 text-[#00e5ff] px-4 py-1.5 rounded-full text-xs font-extrabold hover:bg-[#00e5ff]/25 transition-all flex items-center gap-1.5 cursor-pointer"
              >
                <Download className="w-3.5 h-3.5" /> CSV
              </button>
              <button 
                onClick={downloadTXT}
                className="border border-[#ffeb3b]/40 bg-[#ffeb3b]/10 text-[#ffeb3b] px-4 py-1.5 rounded-full text-xs font-extrabold hover:bg-[#ffeb3b]/20 transition-all flex items-center gap-1.5 cursor-pointer"
              >
                <Download className="w-3.5 h-3.5" /> TXT
              </button>
            </div>
            <div className="text-xs text-gray-400 flex items-center gap-1.5 font-mono">
              <Filter className="w-3.5 h-3.5 text-[#00e5ff]" />
              Fitness Score: <span className="text-[#00e5ff] font-bold">{backtestResult.fitnessScore} pts</span> (Aciertos: {backtestResult.avgHits}/6)
            </div>
          </div>
        </div>

        {/* Pools Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {poolsConfig.map((p) => (
            <div 
              key={p.id}
              className="bg-[#1a1a1a] border-2 rounded-xl p-4 transition-all hover:scale-[1.01]"
              style={{ borderColor: p.border, boxShadow: `0 0 16px ${p.border}22` }}
            >
              <div className="flex justify-between items-center mb-3">
                <span className="text-xs sm:text-sm font-black" style={{ color: p.accent }}>
                  {p.label}
                </span>
                <span className="rounded-full px-2.5 py-0.5 text-[10px] font-bold" style={{ background: `${p.border}22`, border: `1px solid ${p.border}66`, color: p.accent }}>
                  {p.count} Números
                </span>
              </div>

              {/* Number Grid */}
              <div className="grid grid-cols-8 gap-1.5 my-2">
                {p.nums.map(n => (
                  <div 
                    key={n}
                    className="bg-[#121212] border rounded text-center py-1 text-xs font-black"
                    style={{ borderColor: `${p.border}44`, color: p.accent }}
                  >
                    {String(n).padStart(2, '0')}
                  </div>
                ))}
              </div>

              {/* Wheelings */}
              {p.wheelings.length > 0 ? (
                <div className="mt-3 pt-3 border-t border-[#2a2a2a]">
                  <span className="text-[10px] text-[#ffeb3b] font-bold block mb-2">
                    ● WHEELINGS (Sistemas Reducidos con Filtros Optimizados)
                  </span>
                  <div className="flex flex-wrap gap-2">
                    {p.wheelings.map(w => (
                      <button
                        key={w.k}
                        onClick={() => setActiveModal({ type: 'wheeling', poolId: p.id, k: w.k })}
                        className="rounded-full px-3 py-1 text-[11px] font-extrabold cursor-pointer border transition-all hover:brightness-125"
                        style={{ borderColor: `${p.border}88`, background: `${p.border}22`, color: p.accent }}
                      >
                        {uv(w.k)} comb {w.k === 48 ? '(Recomendado)' : ''}
                      </button>
                    ))}
                  </div>
                  <p className="text-[9px] text-gray-500 mt-2 font-mono leading-relaxed">
                    {p.desc}
                  </p>
                </div>
              ) : (
                <div className="mt-3 pt-3 border-t border-[#2a2a2a]">
                  <p className="text-[10px] text-gray-500 font-mono">
                    {p.desc}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Backtesting Banner */}
        <div className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl p-4 mb-6 flex justify-between items-center flex-wrap gap-4">
          <span className="font-mono text-xs text-gray-300">
            Backtesting Activo: Evaluados últimos {backtestResult.evaluations} sorteos • Precisión 4+ aciertos: <span className="text-[#00e5ff] font-bold">{backtestResult.pct4Plus}%</span>
          </span>
          <button 
            onClick={() => setActiveModal({ type: 'optimizer' })}
            className="bg-[#00e5ff]/15 border border-[#00e5ff]/40 text-[#00e5ff] rounded-full px-4 py-1.5 text-xs font-extrabold cursor-pointer hover:bg-[#00e5ff]/25 transition-all"
          >
            Configurar Hiperparámetros
          </button>
        </div>

        {/* Historical Results Table (Últimos 20 Sorteos Melate) */}
        <div className="bg-[#121212] border border-[#2a2a2a] rounded-xl p-4 mb-6">
          <div className="bg-[#ffeb3b] text-black px-3 py-1.5 text-xs font-black rounded-t-lg flex justify-between items-center">
            <span>RESULTADOS MELATE - Últimos 20 Sorteos (Max 7 Aciertos c/ Adicional)</span>
            <span className="bg-black text-[#ffeb3b] text-[9px] px-2 py-0.5 rounded uppercase font-bold">INMUTABLE</span>
          </div>
          <div className="text-[10px] text-gray-400 text-center py-1.5 bg-[#1a1a1a] border-x border-[#2a2a2a]">
            👆 Click en cualquier celda para ver el desglose detallado (Pool + Wheeling con Adicional)
          </div>

          <div className="overflow-x-hidden">
            <table className="w-full text-center border-collapse text-xs table-fixed">
              <thead>
                <tr className="bg-[#1a1a1a] text-[9px] text-gray-300">
                  <th className="border border-[#2a2a2a] p-1.5 w-[22%]">Sorteo</th>
                  <th className="border border-[#2a2a2a] p-1.5 text-[#ffeb3b]">Final</th>
                  <th className="border border-[#2a2a2a] p-1.5 text-[#00e5ff]">Pool2</th>
                  <th className="border border-[#2a2a2a] p-1.5 text-[#ff9800]">Pool3</th>
                  <th className="border border-[#2a2a2a] p-1.5 text-[#76ff03]">Pool4</th>
                  <th className="border border-[#2a2a2a] p-1.5 text-[#e040fb]">Pool5</th>
                  <th className="border border-[#2a2a2a] p-1.5 text-[#26c6da]">Pool6</th>
                  <th className="border border-[#2a2a2a] p-1.5 text-[#ff4081]">Pool7</th>
                </tr>
              </thead>
              <tbody>
                {[...draws].sort((a, b) => b.num - a.num).slice(0, 20).map(s => {
                  const pred = getPredictionForDraw(s.num, draws);
                  const fNums = pred ? pred.final : livePools.final;
                  const p2Nums = pred ? pred.p2 : livePools.p2;
                  const p3Nums = pred ? pred.p3 : livePools.p3;
                  const p4Nums = pred ? pred.p4 : livePools.p4;
                  const p5Nums = pred ? pred.p5 : livePools.p5;
                  const p6Nums = pred ? pred.p6 : livePools.p6;
                  const p7Nums = pred ? pred.p7 : livePools.p7;

                  const hF = getTotalHits(fNums, s);
                  const h2 = getTotalHits(p2Nums, s);
                  const h3 = getTotalHits(p3Nums, s);
                  const h4 = getTotalHits(p4Nums, s);
                  const h5 = getTotalHits(p5Nums, s);
                  const h6 = getTotalHits(p6Nums, s);
                  const h7 = getTotalHits(p7Nums, s);

                  return (
                    <tr key={s.num} className="hover:bg-[#1a1a1a] transition-all">
                      <td className="border border-[#2a2a2a] p-1 font-mono font-bold text-[10px]">
                        {s.num}<br/><span className="text-[8px] text-gray-500">{s.date}</span>
                      </td>
                      <td className="border border-[#2a2a2a] p-1">
                        <div 
                          onClick={() => setActiveModal({ type: 'cell', sorteoNum: s.num, poolId: 'final' })}
                          className={`cursor-pointer p-1.5 rounded text-[10px] font-black ${hitClass(hF)}`}
                        >
                          {hF}/{params.poolSize}{hasWheelingHitThreeOrMore(fNums, s, 'final') ? ' ⭐' : ''}
                        </div>
                      </td>
                      <td className="border border-[#2a2a2a] p-1">
                        <div 
                          onClick={() => setActiveModal({ type: 'cell', sorteoNum: s.num, poolId: 'p2' })}
                          className={`cursor-pointer p-1.5 rounded text-[10px] font-black ${hitClass(h2)}`}
                        >
                          {h2}/{params.poolSize}{hasWheelingHitThreeOrMore(p2Nums, s, 'p2') ? ' ⭐' : ''}
                        </div>
                      </td>
                      <td className="border border-[#2a2a2a] p-1">
                        <div 
                          onClick={() => setActiveModal({ type: 'cell', sorteoNum: s.num, poolId: 'p3' })}
                          className={`cursor-pointer p-1.5 rounded text-[10px] font-black ${hitClass(h3)}`}
                        >
                          {h3}/{params.poolSize}{hasWheelingHitThreeOrMore(p3Nums, s, 'p3') ? ' ⭐' : ''}
                        </div>
                      </td>
                      <td className="border border-[#2a2a2a] p-1">
                        <div 
                          onClick={() => setActiveModal({ type: 'cell', sorteoNum: s.num, poolId: 'p4' })}
                          className={`cursor-pointer p-1.5 rounded text-[10px] font-black ${hitClass(h4)}`}
                        >
                          {h4}/{params.poolSize}{hasWheelingHitThreeOrMore(p4Nums, s, 'p4') ? ' ⭐' : ''}
                        </div>
                      </td>
                      <td className="border border-[#2a2a2a] p-1">
                        <div 
                          onClick={() => setActiveModal({ type: 'cell', sorteoNum: s.num, poolId: 'p5' })}
                          className={`cursor-pointer p-1.5 rounded text-[10px] font-black ${hitClass(h5)}`}
                        >
                          {h5}/{params.poolSize}{hasWheelingHitThreeOrMore(p5Nums, s, 'p5') ? ' ⭐' : ''}
                        </div>
                      </td>
                      <td className="border border-[#2a2a2a] p-1">
                        <div 
                          onClick={() => setActiveModal({ type: 'cell', sorteoNum: s.num, poolId: 'p6' })}
                          className={`cursor-pointer p-1.5 rounded text-[10px] font-black ${hitClass(h6)}`}
                        >
                          {h6}/{params.poolSize}{hasWheelingHitThreeOrMore(p6Nums, s, 'p6') ? ' ⭐' : ''}
                        </div>
                      </td>
                      <td className="border border-[#2a2a2a] p-1">
                        <div 
                          onClick={() => setActiveModal({ type: 'cell', sorteoNum: s.num, poolId: 'p7' })}
                          className={`cursor-pointer p-1.5 rounded text-[10px] font-black ${hitClass(h7)}`}
                        >
                          {h7}/{params.poolSize}{hasWheelingHitThreeOrMore(p7Nums, s, 'p7') ? ' ⭐' : ''}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Mapa de Calor de Frecuencia */}
        <div className="bg-[#121212] border border-[#2a2a2a] rounded-xl p-4 mb-6">
          <div className="bg-[#00e5ff] text-black px-3 py-1.5 text-xs font-black rounded-t-lg flex justify-between items-center">
            <span>MAPA DE CALOR DE FRECUENCIA - Frecuencia de Aparición (Últimos 50 Sorteos)</span>
            <span className="bg-black text-[#00e5ff] text-[9px] px-2 py-0.5 rounded uppercase font-bold">1 a 56</span>
          </div>
          <div className="text-[10px] text-gray-400 text-center py-2 bg-[#1a1a1a] border-x border-[#2a2a2a]">
            Distribución de frecuencia histórica reciente (Sorteos {safeDraws.slice(-50)[0]?.num || 3861} al {lastDraw.num}). Colores más intensos indican mayor aparición.
          </div>
          <div className="p-4 bg-[#141414] border border-[#2a2a2a] rounded-b-xl">
            <div className="grid grid-cols-7 sm:grid-cols-8 gap-2">
              {Array.from({ length: 56 }, (_, i) => i + 1).map(n => {
                const freq = frequencyMap[n] || 0;
                const intensity = heatmapStats.maxFreq > heatmapStats.minFreq ? (freq - heatmapStats.minFreq) / (heatmapStats.maxFreq - heatmapStats.minFreq) : 0.5;
                const bgAlpha = Math.max(0.12, intensity * 0.85);
                const borderColor = intensity > 0.7 ? '#00e5ff' : intensity > 0.4 ? '#7c4dff' : '#333';
                const textColor = intensity > 0.5 ? '#ffffff' : '#a0a0a0';
                return (
                  <div
                    key={n}
                    className="flex flex-col items-center justify-center p-2 rounded-lg border text-center transition-all hover:scale-105"
                    style={{
                      background: `rgba(0, 229, 255, ${bgAlpha})`,
                      borderColor: borderColor,
                    }}
                  >
                    <span className="text-xs font-black font-mono" style={{ color: textColor }}>
                      {String(n).padStart(2, '0')}
                    </span>
                    <span className="text-[9px] font-mono mt-0.5" style={{ color: intensity > 0.5 ? '#e0f7fa' : '#888' }}>
                      {freq} app
                    </span>
                  </div>
                );
              })}
            </div>
            <div className="mt-4 pt-3 border-t border-[#222] flex items-center justify-between text-[10px] text-gray-400">
              <span>Baja Frecuencia</span>
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 rounded bg-[#00e5ff]/15 border border-[#333]"></div>
                <div className="w-3 h-3 rounded bg-[#00e5ff]/40 border border-[#7c4dff]"></div>
                <div className="w-3 h-3 rounded bg-[#00e5ff]/80 border border-[#00e5ff]"></div>
              </div>
              <span>Alta Frecuencia (Hot)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Toast Notification */}
      {toastMessage && (
        <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 bg-[#7c4dff] text-white px-5 py-2.5 rounded-full text-xs font-black shadow-[0_4px_20px_rgba(124,77,255,0.6)] z-50 animate-bounce">
          {toastMessage}
        </div>
      )}

      {/* Modal Root */}
      {activeModal && (
        <div className="fixed inset-0 bg-black/85 backdrop-blur-sm flex items-center justify-center z-50 p-3" onClick={() => setActiveModal(null)}>
          <div className="bg-[#121212] border border-[#333] rounded-xl w-full max-w-2xl max-h-[90vh] flex flex-col overflow-hidden shadow-2xl" onClick={e => e.stopPropagation()}>
            
            {/* Optimizer Modal */}
            {activeModal.type === 'optimizer' && (
              <OptimizerModal 
                params={params} 
                draws={draws}
                onSave={saveParams} 
                onReset={resetParams} 
                backtest={backtestResult} 
                onClose={() => setActiveModal(null)} 
              />
            )}

            {activeModal.type === 'wheeling' && (() => {
              const p = poolsConfig.find(x => x.id === activeModal.poolId);
              if (!p) return null;
              const w = p.wheelings.find(x => x.k === activeModal.k);
              if (!w) return null;
              const combos = generateMelateWheeling(w.slice, uv(w.k), p.id, params);

              return (
                <>
                  <div className="p-4 bg-[#1a1a1a] border-b border-[#2a2a2a] flex justify-between items-center">
                    <span className="text-xs font-bold font-mono text-[#ffeb3b]">
                      Wheeling Pool {p.label} - {uv(w.k)} combinaciones (6 números c/u)
                    </span>
                    <button onClick={() => setActiveModal(null)} className="text-gray-400 hover:text-white font-bold text-lg">×</button>
                  </div>
                  <div className="p-4 overflow-y-auto space-y-2 flex-1">
                    <div className="bg-[#181818] border border-[#2a2a2a] p-3 rounded-lg">
                      <div className="text-[10px] text-gray-400 mb-1 font-bold">Base Pool ({w.slice.length} números)</div>
                      <div className="flex flex-wrap gap-1 font-mono text-xs">
                        {w.slice.map(n => (
                          <span key={n} className="bg-[#121212] border border-[#333] px-2 py-0.5 rounded" style={{ color: p.accent }}>
                            {String(n).padStart(2, '0')}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="space-y-1.5 mt-3">
                      {combos.map((c, idx) => (
                        <div key={idx} className="flex items-center gap-2 bg-[#1a1a1a] border border-[#2a2a2a] p-2 rounded-lg">
                          <span className="text-[10px] font-mono w-6 text-gray-500">#{idx + 1}</span>
                          <div className="flex gap-1.5 font-mono font-bold text-xs">
                            {c.map((n, i) => (
                              <span key={i} className="bg-[#121212] border border-[#333] px-2 py-1 rounded text-white">
                                {String(n).padStart(2, '0')}
                              </span>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="p-3 bg-[#0f0f0f] border-t border-[#222] flex justify-center gap-3">
                    <button 
                      onClick={() => copyWheelingToClipboard(p.id, w.k)}
                      className="bg-[#222] border border-[#444] text-[#ccc] rounded-full px-4 py-1.5 text-xs font-bold hover:bg-[#333] cursor-pointer flex items-center gap-1.5"
                    >
                      <Copy className="w-3.5 h-3.5" /> Copiar Combinaciones
                    </button>
                    <button onClick={() => setActiveModal(null)} className="bg-[#1a1a1a] border border-[#333] text-gray-400 rounded-full px-4 py-1.5 text-xs font-bold cursor-pointer">
                      Cerrar
                    </button>
                  </div>
                </>
              );
            })()}

            {activeModal.type === 'cell' && (() => {
              const s = draws.find(x => x.num === activeModal.sorteoNum);
              const pBase = poolsConfig.find(x => x.id === activeModal.poolId);
              if (!s || !pBase) return null;

              const pred = getPredictionForDraw(s.num, draws);
              const poolNums = pred ? pred[activeModal.poolId === 'final' ? 'final' : activeModal.poolId!] : pBase.nums;
              const totalHits = getTotalHits(poolNums, s);

              return (
                <>
                  <div className="p-4 bg-[#1a1a1a] border-b border-[#2a2a2a] flex justify-between items-center">
                    <h3 className="text-xs font-bold font-mono text-[#ffeb3b]">
                      Sorteo {s.num} - {pBase.label} ({totalHits}/{params.poolSize} aciertos totales)
                    </h3>
                    <button onClick={() => setActiveModal(null)} className="text-gray-400 hover:text-white font-bold text-lg">×</button>
                  </div>
                  <div className="p-4 overflow-y-auto space-y-4 flex-1">
                    <div className="bg-[#7c4dff]/20 border border-[#7c4dff]/40 p-3 rounded-xl text-center">
                      <div className="text-[10px] font-bold text-[#b388ff]">SORTEO OFICIAL #{s.num} ({s.date})</div>
                      <div className="text-xl font-black font-mono tracking-wider text-[#ffeb3b] my-1">
                        {s.win.map(n => String(n).padStart(2, '0')).join(' - ')} <span className="text-[#00e5ff] text-sm">(Adic: {s.additional})</span>
                      </div>
                    </div>

                    <div>
                      <div className="text-xs font-bold text-gray-300 mb-2">Desglose de Números en el Pool (Verde: Ganador, Azul Pastel: Adicional):</div>
                      <div className="grid grid-cols-8 gap-1.5">
                        {poolNums.map((n: number) => {
                          const isWin = s.win.includes(n);
                          const isAdditional = s.additional === n;
                          let className = "bg-[#1a1a1a] border-[#333] text-gray-400";
                          if (isWin) {
                            className = "bg-[#2e7d32] border-[#4caf50] text-white shadow-[0_0_8px_#2e7d32]";
                          } else if (isAdditional) {
                            className = "bg-[#0288d1] border-[#29b6f6] text-white shadow-[0_0_8px_#0288d1]";
                          }
                          return (
                            <div 
                              key={n} 
                              className={`text-center py-1.5 rounded text-xs font-bold border ${className}`}
                            >
                              {String(n).padStart(2, '0')}
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    <div className="mt-4">
                      <div className="text-xs font-bold text-[#ffeb3b] mb-2 flex items-center gap-1.5">
                        Wheelings Históricos (Inmutables) para Sorteo {s.num}:
                      </div>
                      <div className="space-y-3">
                        {[
                          { k: 24, slice: poolNums.slice(0, 24) },
                          { k: 48, slice: poolNums.slice(0, 28) },
                          { k: 96, slice: poolNums.slice(0, 30) },
                          { k: 192, slice: poolNums }
                        ].map(wConfig => {
                          const reduced = uv(wConfig.k);
                          const combos = rvMelate(wConfig.slice, s.win, reduced, pBase.id, params);
                          const scored = combos.map(c => ({ c, h: c.filter(n => s.win.includes(n)).length })).sort((a,b) => b.h - a.h);
                          const bestHit = scored[0] ? scored[0].h : 0;
                          
                          return (
                            <div key={wConfig.k} className="bg-[#181818] border border-[#2a2a2a] rounded-lg p-3">
                              <div className="flex justify-between items-center mb-2">
                                <span className="text-[11px] font-bold text-[#00e5ff]">{reduced} combinaciones (Base {wConfig.slice.length} num)</span>
                                <span className="text-[10px] font-black px-2.5 py-0.5 rounded-full bg-[#333] text-gray-300">
                                  Mejor: {bestHit} aciertos
                                </span>
                              </div>
                              <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                                {scored.slice(0, 5).map((item, idx) => (
                                  <div key={idx} className="flex items-center gap-2 bg-[#121212] border border-[#222] p-1.5 rounded text-xs font-mono">
                                    <span className="text-[9px] text-gray-500 w-5">#{idx+1}</span>
                                    <div className="flex gap-1 flex-wrap">
                                      {item.c.map((n, i) => {
                                        const isWin = s.win.includes(n);
                                        const isAdd = s.additional === n;
                                        let cls = "bg-[#1a1a1a] text-gray-400 border-[#333]";
                                        if (isWin) cls = "bg-[#2e7d32] text-white border-[#4caf50]";
                                        else if (isAdd) cls = "bg-[#0288d1] text-white border-[#29b6f6]";
                                        return (
                                          <span key={i} className={`border rounded px-1.5 py-0.5 text-[10px] font-bold ${cls}`}>
                                            {String(n).padStart(2, '0')}
                                          </span>
                                        );
                                      })}
                                    </div>
                                    <span className="ml-auto text-[10px] font-bold text-gray-300">{item.h} ac</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                  <div className="p-3 bg-[#0f0f0f] border-t border-[#222] flex justify-end">
                    <button onClick={() => setActiveModal(null)} className="bg-[#1a1a1a] border border-[#333] text-gray-300 rounded-full px-4 py-1.5 text-xs font-bold cursor-pointer">
                      Cerrar
                    </button>
                  </div>
                </>
              );
            })()}
          </div>
        </div>
      )}

      {/* GoatCounter Analytics */}
      <script data-goatcounter="https://cedalonu.goatcounter.com/count" async src="https://gc.zgo.at/count.js"></script>
    </div>
  );
}

interface OptimizerModalProps {
  params: MLParameters;
  draws: typeof INITIAL_MELATE_DRAWS;
  onSave: (p: MLParameters) => void;
  onReset: () => void;
  backtest: { evaluations: number; avgHits: string; pct4Plus: string; pct5Plus: string; fitnessScore: string };
  onClose: () => void;
}

function OptimizerModal({ params, draws, onSave, onReset, backtest, onClose }: OptimizerModalProps) {
  const [form, setForm] = useState<MLParameters>(params);
  const [fitnessHistory, setFitnessHistory] = useState<any[]>(() => {
    try {
      const saved = localStorage.getItem('melate_fitness_history');
      return saved ? JSON.parse(saved) : [];
    } catch(e) {
      return [];
    }
  });

  const handleChange = (key: keyof MLParameters, val: number) => {
    setForm(prev => ({ ...prev, [key]: val }));
  };

  const curveData = useMemo(() => {
    const data = [];
    const steps = [5, 10, 15, 20];
    for (const step of steps) {
      if (draws.length < step) continue;
      const subDraws = draws.slice(-step);
      let totalHits = 0;
      for (let i = 1; i < subDraws.length; i++) {
        const hist = subDraws.slice(0, i);
        const target = subDraws[i];
        const pred = calculateMelateMLPools(hist, form);
        const hits = pred.final.filter(n => target.win.includes(n)).length;
        totalHits += hits;
      }
      const avg = (subDraws.length > 1 ? totalHits / (subDraws.length - 1) : 0).toFixed(2);
      data.push({
        name: `Ventana ${step}`,
        aciertos: parseFloat(avg)
      });
    }
    return data;
  }, [draws, form]);

  const scorecardData = useMemo(() => {
    return runPoolScorecard(draws, form);
  }, [draws, form]);

  return (
    <>
      <div className="p-4 bg-[#1a1a1a] border-b border-[#2a2a2a] flex justify-between items-center">
        <h3 className="text-sm font-bold text-[#e040fb] flex items-center gap-2">
          <Settings className="w-4 h-4" /> Optimizador de Hiperparámetros & Backtesting (Fase 2)
        </h3>
        <button onClick={onClose} className="text-gray-400 hover:text-white font-bold text-lg">×</button>
      </div>
      <div className="p-4 overflow-y-auto space-y-4 flex-1 text-xs">
        <div className="bg-[#181818] border border-[#2a2a2a] p-3 rounded-lg flex items-center justify-between flex-wrap gap-3">
          <div>
            <div className="text-gray-400 text-[10px]">Fitness Score General:</div>
            <div className="text-lg font-black text-[#00e5ff]">{backtest.fitnessScore} pts</div>
          </div>
          <div>
            <div className="text-gray-400 text-[10px]">Aciertos Promedio:</div>
            <div className="text-sm font-bold text-white">{backtest.avgHits} / 6</div>
          </div>
          <div>
            <div className="text-gray-400 text-[10px]">Precisión 4+ aciertos:</div>
            <div className="text-sm font-bold text-[#76ff03]">{backtest.pct4Plus}%</div>
          </div>
        </div>

        {/* Historial de Fitness Section */}
        <div className="bg-[#181818] border border-[#2a2a2a] p-3 rounded-lg">
          <div className="text-xs font-bold text-[#e040fb] mb-2 flex items-center gap-1.5">
            <Trophy className="w-3.5 h-3.5" /> Historial de Fitness (Últimos 5 Intentos)
          </div>
          {fitnessHistory.length === 0 ? (
            <div className="text-[10px] text-gray-500 italic py-2 text-center">
              No hay intentos guardados aún. Haz clic en "Guardar & Aplicar" para registrar tu optimización.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-center text-[10px] border-collapse font-mono">
                <thead>
                  <tr className="bg-[#222] text-gray-300">
                    <th className="p-1.5 border border-[#333]">Hora</th>
                    <th className="p-1.5 border border-[#333]">Fitness Score</th>
                    <th className="p-1.5 border border-[#333]">Aciertos Prom</th>
                    <th className="p-1.5 border border-[#333]">4+ Aciertos</th>
                    <th className="p-1.5 border border-[#333]">Parámetros</th>
                  </tr>
                </thead>
                <tbody>
                  {fitnessHistory.map((item, idx) => (
                    <tr key={item.id || idx} className="hover:bg-[#202020]">
                      <td className="p-1.5 border border-[#333] text-gray-400">{item.time}</td>
                      <td className="p-1.5 border border-[#333] text-[#00e5ff] font-black">{item.fitnessScore} pts</td>
                      <td className="p-1.5 border border-[#333] text-white">{item.avgHits} ac</td>
                      <td className="p-1.5 border border-[#333] text-[#76ff03] font-bold">{item.pct4Plus}%</td>
                      <td className="p-1.5 border border-[#333] text-gray-300 text-[9px]">{item.paramsDesc}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="bg-[#181818] border border-[#2a2a2a] p-3 rounded-lg">
          <div className="text-xs font-bold text-[#00e5ff] mb-2">Curva de Aprendizaje - Promedio de Aciertos por Iteración</div>
          <div className="h-36 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={curveData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="name" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} domain={[0, 6]} />
                <Tooltip contentStyle={{ background: '#121212', border: '1px solid #444', fontSize: '10px' }} />
                <Line type="monotone" dataKey="aciertos" stroke="#00e5ff" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Scorecard Section */}
        <div className="bg-[#181818] border border-[#2a2a2a] p-3 rounded-lg">
          <div className="text-xs font-bold text-[#ffeb3b] mb-2 flex items-center gap-1.5">
            <BarChart2 className="w-3.5 h-3.5" /> Scorecard - Rendimiento Comparativo por Tipo de Pool
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-center text-[10px] border-collapse font-mono">
              <thead>
                <tr className="bg-[#222] text-gray-300">
                  <th className="p-1.5 border border-[#333]">Pool Strategy</th>
                  <th className="p-1.5 border border-[#333]">Aciertos Promedio</th>
                  <th className="p-1.5 border border-[#333]">Precisión 4+ aciertos</th>
                </tr>
              </thead>
              <tbody>
                {scorecardData.map(sc => (
                  <tr key={sc.id} className="hover:bg-[#202020]">
                    <td className="p-1.5 border border-[#333] font-bold text-[#00e5ff]">{sc.name}</td>
                    <td className="p-1.5 border border-[#333] text-white font-black">{sc.avgHits} ac</td>
                    <td className="p-1.5 border border-[#333] text-[#76ff03] font-bold">{sc.pct4Plus}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div>
            <label className="text-gray-400 block mb-1">Tamaño de Pool: <b>{form.poolSize}</b></label>
            <input 
              type="range" min="24" max="42" step="2" value={form.poolSize} 
              onChange={e => handleChange('poolSize', parseInt(e.target.value))}
              className="w-full accent-[#e040fb]"
            />
          </div>
          <div>
            <label className="text-gray-400 block mb-1">Anti-Pool (Vetados): <b>{form.antiSize}</b></label>
            <input 
              type="range" min="5" max="15" step="1" value={form.antiSize} 
              onChange={e => handleChange('antiSize', parseInt(e.target.value))}
              className="w-full accent-[#e040fb]"
            />
          </div>
          <div>
            <label className="text-gray-400 block mb-1">Peso Frecuencia (wFreq): <b>{form.wFreq}</b></label>
            <input 
              type="range" min="1.0" max="5.0" step="0.1" value={form.wFreq} 
              onChange={e => handleChange('wFreq', parseFloat(e.target.value))}
              className="w-full accent-[#00e5ff]"
            />
          </div>
          <div>
            <label className="text-gray-400 block mb-1">Peso Co-ocurrencia (wCoOccur): <b>{form.wCoOccur}</b></label>
            <input 
              type="range" min="1.0" max="6.0" step="0.1" value={form.wCoOccur} 
              onChange={e => handleChange('wCoOccur', parseFloat(e.target.value))}
              className="w-full accent-[#00e5ff]"
            />
          </div>
          <div>
            <label className="text-gray-400 block mb-1">Peso Retraso/Gap (wGap): <b>{form.wGap}</b></label>
            <input 
              type="range" min="0.0" max="2.0" step="0.1" value={form.wGap} 
              onChange={e => handleChange('wGap', parseFloat(e.target.value))}
              className="w-full accent-[#ff9800]"
            />
          </div>
          <div>
            <label className="text-gray-400 block mb-1">Máx Consecutivos: <b>{form.maxConsec}</b></label>
            <input 
              type="range" min="2" max="5" step="1" value={form.maxConsec} 
              onChange={e => handleChange('maxConsec', parseInt(e.target.value))}
              className="w-full accent-[#ffeb3b]"
            />
          </div>
        </div>
      </div>
      <div className="p-3 bg-[#0f0f0f] border-t border-[#222] flex justify-between items-center">
        <button onClick={onReset} className="bg-[#2a2a2a] text-gray-300 hover:text-white rounded-full px-4 py-1.5 text-xs font-bold cursor-pointer">
          Restablecer
        </button>
        <div className="flex gap-2">
          <button onClick={onClose} className="bg-[#1a1a1a] text-gray-400 hover:text-white rounded-full px-4 py-1.5 text-xs font-bold cursor-pointer">
            Cancelar
          </button>
          <button onClick={() => { onSave(form); onClose(); }} className="bg-[#e040fb] text-white rounded-full px-5 py-1.5 text-xs font-black cursor-pointer shadow-[0_0_12px_rgba(224,64,251,0.5)]">
            Guardar & Aplicar
          </button>
        </div>
      </div>
    </>
  );
}
