#!/usr/bin/env python3
"""
shadow_diff.py — Quantify TRAIN-vs-LIVE feature divergence for the XGB bot.

This is the analysis half of the shadow harness (see shadow_capture.py). The
documented #1 driver of this bot's shadow->live degradation is feature
divergence: tick features approximated from REST trades (vs S3 quote ticks in
training), features the live engine never computes (now surfaced as NaN by the
engine, i.e. median-fallback), and silent value drift on same-named features.

Two alignment modes (a live capture taken NOW does NOT share wall-clock minutes
with a historical training parquet, so a plain timestamp join would be empty):

  --align dist  (DEFAULT) — distribution comparison, no timestamp overlap needed.
      Per feature: median-fallback rate (NaN in capture), center shift (in train
      IQRs), scale drift, out-of-distribution rate. Catches the dominant failure
      modes (engine-never-computes, scale/shift, OOD). Misses a subtle per-minute
      drift when the two distributions happen to match.

  --align minute — per-minute matched on (coin, ts_min). Catches per-minute value
      AND sign divergence. REQUIRES a training parquet that COVERS the capture
      window: build one with data/build_features_hl_xgb.py over the capture days
      and pass it via --train_parquet. If windows don't overlap it errors loudly
      with instructions (it does NOT silently produce an empty report).

Both report, per feature, a [0,1] severity WEIGHTED by each model's XGBoost gain
importance, so the headline is "what fraction of a model's predictive mass sits
on features that diverge live." Verdict bands: OK / CAUTION / DO-NOT-DEPLOY, and
INSUFFICIENT-DATA when there aren't enough live samples.

USAGE
  # Default: works against the existing training parquet + a fresh capture.
  python3 autonomous_agent/shadow_diff.py \
      --capture 'data/shadow_capture/shadow_capture_SOL_*.parquet' \
      --models_dir models/live_v8 --train_dir data/artifacts_xgb \
      --out data/shadow_capture/shadow_diff_report

  # Gold-standard, after rebuilding a training parquet over the capture days:
  python3 autonomous_agent/shadow_diff.py --align minute \
      --capture '...SOL_*.parquet' --train_parquet data/artifacts_xgb/<covering>.parquet

  python3 autonomous_agent/shadow_diff.py --self_test
"""
import argparse
import glob
import json
import math
import os
import sys

HFT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if HFT_DIR not in sys.path:
    sys.path.insert(0, HFT_DIR)

COIN_TO_ASSET = {"BTC": "btc_usd", "ETH": "eth_usd", "SOL": "sol_usd"}
LEADLAG_PREFIXES = ("bn_", "cb_")
NON_FEATURE_COLS = {"ts_unix", "ts_min", "coin", "snapshot_missing"}

# Heuristic severity bands (gain-weighted). Calibrate against real shadow->live
# PnL once captures exist; until then these are starting points, not law.
SEV_OK, SEV_CAUTION = 0.10, 0.25
MIN_N_BOTH = 30      # min comparable minutes before value-drift terms are trusted
MIN_ALIGNED = 60     # min aligned minutes before a per-model verdict is trustworthy


# ---------------------------------------------------------------------------
# Per-feature metrics
# ---------------------------------------------------------------------------

def feature_metrics(live, train):
    """live, train: aligned pandas Series (same index). live may contain NaN
    (engine median-fallback). Returns a metrics dict."""
    import numpy as np
    n = int(len(live))
    if n == 0:
        return dict(n=0, n_both=0, pct_fallback=1.0, corr=float("nan"),
                    mean_abs_diff=float("nan"), rel_diff=float("nan"),
                    bias=float("nan"), frac_sign_flip=float("nan"))
    live_nan = live.isna()
    pct_fallback = float(live_nan.mean())
    both = (~live_nan) & (~train.isna())
    n_both = int(both.sum())
    if n_both == 0:
        return dict(n=n, n_both=0, pct_fallback=pct_fallback, corr=float("nan"),
                    mean_abs_diff=float("nan"), rel_diff=float("nan"),
                    bias=float("nan"), frac_sign_flip=float("nan"))
    lv = live[both].astype(float).to_numpy()
    tv = train[both].astype(float).to_numpy()
    diff = np.abs(lv - tv)
    mean_abs_diff = float(np.mean(diff))
    rel_diff = float(np.median(diff / (np.abs(tv) + 1e-9)))
    bias = float(np.mean(lv - tv))
    if np.std(lv) < 1e-12 or np.std(tv) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(lv, tv)[0, 1])
    nonzero = (lv != 0) | (tv != 0)
    frac_sign_flip = float(np.mean((np.sign(lv) != np.sign(tv)) & nonzero)) if nonzero.any() else 0.0
    return dict(n=n, n_both=n_both, pct_fallback=pct_fallback, corr=corr,
                mean_abs_diff=mean_abs_diff, rel_diff=rel_diff, bias=bias,
                frac_sign_flip=frac_sign_flip)


def severity(m, min_n=MIN_N_BOTH):
    """[0,1] divergence severity as a convex blend:

        severity = pct_fallback * 1.0 + (1 - pct_fallback) * value_severity

    so a feature the engine NEVER computes scores 1.0 (maximal, per the
    docstring's intent), and value drift on the computed minutes raises the
    rest. value_severity includes rel_diff (MAGNITUDE/scale/shift drift) because
    correlation alone is invariant to exactly those transforms — a 100x-scaled
    or additively-biased feature has perfect corr yet is catastrophic for a tree
    model (it routes through entirely different splits), so it must NOT read safe.
    Too few comparable minutes -> value terms are floored so low-n can't pass."""
    pf = m["pct_fallback"]
    nb = m["n_both"]
    if nb == 0:
        # No comparable minutes: pure fallback is maximal; otherwise we cannot
        # verify value fidelity at all, so refuse to call it safe.
        return 1.0 if pf >= 0.999 else max(pf, 0.5)
    fs = 0.0 if (m["frac_sign_flip"] is None or math.isnan(m["frac_sign_flip"])) else m["frac_sign_flip"]
    corr = m["corr"]
    corr_term = 0.0 if (corr is None or math.isnan(corr)) else (1.0 - max(corr, 0.0))
    rd = m["rel_diff"]
    drift_term = 0.0 if (rd is None or math.isnan(rd)) else min(1.0, float(rd))
    val_sev = min(1.0, 0.55 * drift_term + 0.30 * fs + 0.15 * corr_term)
    if nb < min_n and pf < 1.0:
        # too few comparable minutes to trust the value terms; don't allow "safe"
        val_sev = max(val_sev, 0.5)
    return min(1.0, pf + (1.0 - pf) * val_sev)


def metrics_with_leadlag(feat, live_idxed, train_idxed, train_lag1):
    """Compute metrics at lag0; for bn_*/cb_* also at lag1 (train shifted +1min
    to meet live at T), keeping whichever lag correlates better."""
    if feat not in train_idxed.columns:
        return None  # training doesn't have this feature; can't compare
    live = live_idxed[feat] if feat in live_idxed.columns else _all_nan_like(live_idxed)
    m0 = feature_metrics(live.reindex(train_idxed.index), train_idxed[feat])
    chosen, lag = m0, 0
    if feat.startswith(LEADLAG_PREFIXES) and feat in train_lag1.columns:
        common = live_idxed.index.intersection(train_lag1.index)
        if len(common) > 0:
            m1 = feature_metrics(
                (live if feat in live_idxed.columns else _all_nan_like(live_idxed)).reindex(common),
                train_lag1[feat].reindex(common),
            )
            c0 = -1.0 if math.isnan(m0["corr"]) else m0["corr"]
            c1 = -1.0 if math.isnan(m1["corr"]) else m1["corr"]
            if c1 > c0:
                chosen, lag = m1, 1
    chosen = dict(chosen)
    chosen["leadlag_lag"] = lag
    return chosen


def _all_nan_like(df):
    import numpy as np
    import pandas as pd
    return pd.Series(np.nan, index=df.index)


# ---------------------------------------------------------------------------
# Distribution metrics (NO timestamp overlap required)
#
# The minute-matched path above needs live capture and training to share
# wall-clock minutes. A live capture taken NOW vs a historical training parquet
# never overlaps, so the default comparison is distribution-level: it asks "are
# the live feature DISTRIBUTIONS in the same place/shape as training, and how
# often does the engine fall back to medians?" This catches the dominant failure
# modes (engine-never-computes, scale/shift drift, out-of-distribution live
# values) without needing a same-window training parquet. It does NOT catch a
# subtle per-minute value/sign drift when the two distributions happen to match
# — use --align minute (with a training parquet covering the capture window) for
# that gold-standard check.
# ---------------------------------------------------------------------------

def feature_dist_metrics(live, train_col):
    """Distribution-level divergence for ONE feature. `live` is the capture
    Series (NaN = engine fallback); `train_col` is the training reference."""
    import numpy as np
    n = int(len(live))
    pct_fallback = float(live.isna().mean()) if n else 1.0
    lv = live.dropna().astype(float).to_numpy()
    tv = train_col.dropna().astype(float).to_numpy()
    n_live = int(lv.size)
    if n_live == 0 or tv.size < 10:
        return dict(n_live=n_live, pct_fallback=pct_fallback,
                    center_shift=float("nan"), scale_ratio=float("nan"),
                    ood_rate=float("nan"))
    t_p01, t_p25, t_med, t_p75, t_p99 = np.percentile(tv, [1, 25, 50, 75, 99])
    l_p25, l_med, l_p75 = np.percentile(lv, [25, 50, 75])
    t_iqr = t_p75 - t_p25
    l_iqr = l_p75 - l_p25
    denom = abs(t_iqr) + 1e-9
    center_shift = float(abs(l_med - t_med) / denom)          # in train IQRs
    scale_ratio = float((abs(l_iqr) + 1e-12) / denom)         # ~1 if same spread
    ood_rate = float(np.mean((lv < t_p01) | (lv > t_p99)))    # baseline ~0.02
    return dict(n_live=n_live, pct_fallback=pct_fallback,
                center_shift=center_shift, scale_ratio=scale_ratio,
                ood_rate=ood_rate)


def dist_severity(m, min_live=MIN_N_BOTH):
    """[0,1] distribution severity, same convex shape as the minute severity:
    pf*1 + (1-pf)*value_severity, where value_severity blends center shift (in
    train IQRs), scale drift (decades), and out-of-distribution rate. A feature
    scaled/shifted (perfect corr, but catastrophic for a tree model) scores high
    here via center_shift + scale + OOD."""
    pf = m["pct_fallback"]
    nl = m.get("n_live", 0)
    cs, sr, ood = m["center_shift"], m["scale_ratio"], m["ood_rate"]
    if nl == 0 or cs is None or (isinstance(cs, float) and math.isnan(cs)):
        return 1.0 if pf >= 0.999 else max(pf, 0.5)
    center_term = min(1.0, cs / 2.0)                          # >=2 IQR shift = max
    scale_term = min(1.0, abs(math.log10(sr + 1e-12)))        # 1 decade off = max
    ood_term = min(1.0, max(0.0, (ood - 0.02) / 0.30))        # >=32% OOD = max
    val_sev = min(1.0, 0.40 * center_term + 0.35 * scale_term + 0.25 * ood_term)
    if nl < min_live and pf < 1.0:
        val_sev = max(val_sev, 0.5)   # too few live samples to call it safe
    return min(1.0, pf + (1.0 - pf) * val_sev)


# ---------------------------------------------------------------------------
# Model gain importance
# ---------------------------------------------------------------------------

def load_models_gain(models_dir):
    """Return {model_name: {"features": [...], "gain": {feat: norm_gain}}} for
    every models_dir/{asset}/{cfg}/ with model_*.json + features.json."""
    import numpy as np
    import xgboost as xgb
    out = {}
    feat_paths = sorted(glob.glob(os.path.join(models_dir, "*", "*", "features.json")))
    for fp in feat_paths:
        d = os.path.dirname(fp)
        name = os.path.relpath(d, models_dir).replace(os.sep, "/")
        with open(fp) as fh:
            feats = json.load(fh)
        gains = {}
        n_models = 0
        for i in range(3):
            mp = os.path.join(d, f"model_{i}.json")
            if not os.path.exists(mp):
                continue
            b = xgb.Booster()
            b.load_model(mp)
            imp = b.get_score(importance_type="gain")  # {feat: gain}, omits unused feats
            for k, v in imp.items():
                gains[k] = gains.get(k, 0.0) + float(v)
            n_models += 1
        if n_models:
            for k in gains:
                gains[k] /= n_models
        total = sum(gains.values()) or 1.0
        gains = {k: v / total for k, v in gains.items()}  # normalize to sum 1
        out[name] = {"features": feats, "gain": gains}
    if not out:
        raise SystemExit(f"No models found under {models_dir}")
    return out


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def find_train_parquet(coin, train_dir, template):
    asset = COIN_TO_ASSET.get(coin.upper())
    if not asset:
        return None
    matches = sorted(glob.glob(os.path.join(train_dir, template.format(asset=asset))))
    return matches[-1] if matches else None


def run_diff(capture_glob, models_dir, train_dir, template, out_prefix,
             align="dist", train_parquet=None,
             min_n_both=MIN_N_BOTH, min_aligned=MIN_ALIGNED):
    import pandas as pd

    cap_paths = sorted(glob.glob(capture_glob))
    if not cap_paths:
        raise SystemExit(f"No capture parquet matched: {capture_glob}")
    cap = pd.concat([pd.read_parquet(p) for p in cap_paths], ignore_index=True)
    cap["ts_min"] = pd.to_datetime(cap["ts_min"], utc=True)
    coins = sorted(cap["coin"].astype(str).str.upper().unique())
    cap_lo, cap_hi = cap["ts_min"].min(), cap["ts_min"].max()
    print(f"[shadow_diff] capture: {len(cap)} rows, coins={coins}, "
          f"window {cap_lo} .. {cap_hi}, align={align}")

    models = load_models_gain(models_dir)
    feat_gain_max = {}
    for mname, mv in models.items():
        for f, g in mv["gain"].items():
            feat_gain_max[f] = max(feat_gain_max.get(f, 0.0), g)
    eval_feats = sorted(set().union(*[set(mv["features"]) for mv in models.values()]))

    rows = []                # per (coin, feature)
    coin_samples = {}        # count the per-model verdict trusts (overlap mins / live rows)
    for coin in coins:
        tp = train_parquet or find_train_parquet(coin, train_dir, template)
        if not tp or not os.path.exists(tp):
            print(f"[shadow_diff] WARN no training parquet for {coin}; skipping")
            continue
        tr = pd.read_parquet(tp)
        tr["ts_min"] = pd.to_datetime(tr["ts_min"], utc=True)
        tr = tr.drop_duplicates("ts_min", keep="last").set_index("ts_min").sort_index()
        tr_lo, tr_hi = tr.index.min(), tr.index.max()
        cc = cap[cap["coin"].astype(str).str.upper() == coin].copy()
        cc = cc.drop_duplicates("ts_min", keep="last").set_index("ts_min").sort_index()

        if align == "minute":
            common = cc.index.intersection(tr.index)
            coin_samples[coin] = len(common)
            print(f"[shadow_diff] {coin}: train={os.path.basename(tp)} "
                  f"train_window {tr_lo} .. {tr_hi} overlap_minutes={len(common)}")
            if len(common) == 0:
                # Loud, actionable: do NOT silently produce an empty report.
                print(f"[shadow_diff] ERROR {coin}: --align minute needs a training parquet that "
                      f"COVERS the capture window ({cap_lo} .. {cap_hi}), but this training data "
                      f"ends {tr_hi}. Rebuild a training parquet over the capture days "
                      f"(data/build_features_hl_xgb.py) and pass it via --train_parquet, OR use "
                      f"--align dist (distribution comparison, no overlap needed).")
                continue
            live_idxed = cc.loc[common]
            train_idxed = tr.loc[common]
            train_lag1 = tr.copy()
            train_lag1.index = train_lag1.index + pd.Timedelta(minutes=1)
            for feat in eval_feats:
                m = metrics_with_leadlag(feat, live_idxed, train_idxed, train_lag1)
                if m is None:
                    continue
                m.update(coin=coin, feature=feat, mode="minute",
                         gain_max=feat_gain_max.get(feat, 0.0), sev=severity(m, min_n_both))
                rows.append(m)
        else:  # align == "dist"  (default)
            coin_samples[coin] = len(cc)
            print(f"[shadow_diff] {coin}: train={os.path.basename(tp)} "
                  f"train_window {tr_lo} .. {tr_hi} live_rows={len(cc)} (distribution comparison)")
            for feat in eval_feats:
                if feat not in tr.columns:
                    continue   # training doesn't have it; can't form a reference distribution
                live_col = cc[feat] if feat in cc.columns else _all_nan_like(cc)
                m = feature_dist_metrics(live_col, tr[feat])
                m.update(coin=coin, feature=feat, mode="dist",
                         gain_max=feat_gain_max.get(feat, 0.0), sev=dist_severity(m, min_n_both))
                rows.append(m)

    if not rows:
        raise SystemExit("[shadow_diff] no comparable (coin, feature) rows produced — check that "
                         "capture and training share coins, and (for --align minute) that the "
                         "training parquet covers the capture window")

    df = pd.DataFrame(rows)
    df["gain_weighted_sev"] = df["gain_max"] * df["sev"]
    df = df.sort_values("gain_weighted_sev", ascending=False).reset_index(drop=True)

    summaries = []
    for mname, mv in models.items():
        coin = _model_coin(mname)
        sub = df[(df["coin"] == coin) & (df["feature"].isin(mv["gain"]))]
        gw_fallback = float(sum(mv["gain"][f] * _lookup(sub, f, "pct_fallback", 1.0) for f in mv["gain"]))
        # An un-comparable gain-bearing feature defaults to MAX severity (1.0):
        # a feature we cannot even compare is not safe.
        gw_sev = float(sum(mv["gain"][f] * _lookup(sub, f, "sev", 1.0) for f in mv["gain"]))
        n_samp = coin_samples.get(coin, 0)
        if n_samp < min_aligned:
            verdict = "INSUFFICIENT-DATA"
        elif gw_sev < SEV_OK:
            # dist mode is blind to per-minute decorrelation (same distribution,
            # broken correspondence) — the exact v3/v5/v7 failure mode. So its
            # best verdict is NEVER a clean deploy-clearance; it only means "no
            # gross distributional drift / fallback". A real GO needs --align
            # minute against a training parquet covering the capture window.
            verdict = "OK-DIST-ONLY" if align == "dist" else "OK"
        elif gw_sev < SEV_CAUTION:
            verdict = "CAUTION"
        else:
            verdict = "DO-NOT-DEPLOY"
        top = sub.sort_values("gain_weighted_sev", ascending=False).head(5)
        summaries.append(dict(model=mname, coin=coin, n_samples=n_samp, align=align,
                              gain_weighted_fallback_pct=round(100 * gw_fallback, 2),
                              gain_weighted_severity=round(gw_sev, 4), verdict=verdict,
                              top_diverging=list(top["feature"])))

    _write_report(df, summaries, out_prefix, align)
    return df, summaries


def _model_coin(model_name):
    # model_name like "sol/short_1m_tp0" -> SOL
    asset = model_name.split("/")[0].upper()
    return asset if asset in COIN_TO_ASSET else asset


def _lookup(sub, feat, col, default):
    hit = sub[sub["feature"] == feat]
    if hit.empty:
        return default
    v = hit.iloc[0][col]
    return default if (v is None or (isinstance(v, float) and math.isnan(v))) else v


def _write_report(df, summaries, out_prefix, align):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    csv_path = out_prefix + ".csv"
    md_path = out_prefix + ".md"
    df.to_csv(csv_path, index=False)

    lines = ["# Shadow-diff report (train vs live feature divergence)", ""]
    if align == "dist":
        lines.append("> **WARNING — dist mode CANNOT clear a model for deploy.** It is blind to "
                     "per-minute decorrelation: a feature can share training's distribution yet be "
                     "completely wrong minute-to-minute (same dist, broken correspondence) — the "
                     "EXACT failure mode behind v3/v5/v7 live losses — and still score `OK-DIST-ONLY`. "
                     "Use dist only as a pre-screen for gross drift / median-fallback. A real GO "
                     "requires `--align minute` against a training parquet that covers the capture window.")
    else:
        lines.append("Alignment mode: **minute** (per-minute matched). Catches per-minute value "
                     "AND sign divergence; requires a training parquet overlapping the capture "
                     "window.")
    lines.append("")
    lines.append(f"Gain-weighted severity bands: OK < {SEV_OK} <= CAUTION < {SEV_CAUTION} <= "
                 "DO-NOT-DEPLOY. Heuristic — calibrate against real shadow->live PnL.")
    lines.append("")
    lines.append("## Per-model verdict")
    lines.append("")
    lines.append("| model | coin | samples | gain-wtd fallback % | gain-wtd severity | verdict | top diverging (high gain) |")
    lines.append("|---|---|---:|---:|---:|---|---|")
    for s in summaries:
        lines.append(f"| {s['model']} | {s['coin']} | {s['n_samples']} | {s['gain_weighted_fallback_pct']} | "
                     f"{s['gain_weighted_severity']} | {s['verdict']} | {', '.join(s['top_diverging'])} |")
    lines.append("")
    lines.append("## Top 25 features by gain-weighted severity")
    lines.append("")
    if align == "dist":
        lines.append("| coin | feature | gain | n_live | fallback% | center_shift(IQR) | scale_ratio | ood% | severity |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in df.head(25).iterrows():
            lines.append(
                f"| {r['coin']} | {r['feature']} | {r['gain_max']:.4f} | {int(r.get('n_live', 0))} | "
                f"{100*r['pct_fallback']:.1f} | {_fmt(r.get('center_shift'))} | {_fmt(r.get('scale_ratio'))} | "
                f"{_fmt(100*_nan(r.get('ood_rate')))} | {r['sev']:.3f} |")
    else:
        lines.append("| coin | feature | gain | n_both | fallback% | corr | mean_abs_diff | rel_diff | sign_flip% | lag | severity |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in df.head(25).iterrows():
            lines.append(
                f"| {r['coin']} | {r['feature']} | {r['gain_max']:.4f} | {int(r.get('n_both', 0))} | "
                f"{100*r['pct_fallback']:.1f} | {_fmt(r.get('corr'))} | {_fmt(r.get('mean_abs_diff'))} | "
                f"{_fmt(r.get('rel_diff'))} | {_fmt(100*_nan(r.get('frac_sign_flip')))} | "
                f"{int(r.get('leadlag_lag', 0))} | {r['sev']:.3f} |")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[shadow_diff] wrote {csv_path} and {md_path}")
    print("\n".join(f"  {s['model']}: {s['verdict']} "
                    f"(gain-wtd severity {s['gain_weighted_severity']}, "
                    f"fallback {s['gain_weighted_fallback_pct']}%)" for s in summaries))


def _nan(v):
    return float("nan") if v is None else v


def _fmt(v):
    try:
        if v is None or math.isnan(v):
            return "nan"
    except TypeError:
        return "nan"
    return f"{v:.3f}"


# ---------------------------------------------------------------------------
# Self test (no real data needed)
# ---------------------------------------------------------------------------

def self_test():
    import numpy as np
    import pandas as pd
    print("[self_test] exercising metric math on synthetic series...")
    idx = pd.date_range("2026-01-01", periods=200, freq="min", tz="UTC")
    rng = np.random.default_rng(0)
    train = pd.Series(rng.normal(0, 1, len(idx)), index=idx)

    # 1) identical -> severity ~0
    m = feature_metrics(train.copy(), train)
    assert m["pct_fallback"] == 0.0 and (m["corr"] > 0.99), m
    assert severity(m) < 0.05, severity(m)

    # 2) 30% median-fallback (NaN) -> pct_fallback ~0.3, severity ~0.3
    live = train.copy(); live.iloc[:60] = np.nan
    m = feature_metrics(live, train)
    assert abs(m["pct_fallback"] - 0.30) < 0.02, m
    assert abs(severity(m) - 0.30) < 0.05, severity(m)

    # 3) sign-flipped -> high sign-flip + negative corr -> high severity
    m = feature_metrics(-train, train)
    assert m["frac_sign_flip"] > 0.9 and m["corr"] < -0.9, m
    assert severity(m) > 0.6, severity(m)

    # 4) VALUE DRIFT (the false-OK fix): live = 100*train + 50. Correlation is
    #    invariant to scale/shift (corr ~1) but severity MUST flag it.
    drifted = 100.0 * train + 50.0
    m = feature_metrics(drifted, train)
    assert m["corr"] > 0.99, m
    assert severity(m) > 0.4, ("value drift not penalized", severity(m))

    # 5) 100% fallback -> severity 1.0 (maximal, not capped at 0.5)
    m = feature_metrics(pd.Series(np.nan, index=idx), train)
    assert m["pct_fallback"] == 1.0 and severity(m) == 1.0, severity(m)

    # 6) too few comparable minutes -> cannot read "safe" even if identical
    short_idx = idx[:20]
    m = feature_metrics(train.loc[short_idx].copy(), train.loc[short_idx])
    assert severity(m, min_n=30) >= 0.5, severity(m, min_n=30)

    # 7) gain normalization sums to ~1
    g = {"a": 2.0, "b": 2.0}
    tot = sum(g.values()); g = {k: v/tot for k, v in g.items()}
    assert abs(sum(g.values()) - 1.0) < 1e-9

    # 8) DISTRIBUTION mode (the join-fix): no timestamp overlap needed.
    md = feature_dist_metrics(train.copy(), train)            # identical dist
    assert dist_severity(md) < 0.10, ("dist identical", dist_severity(md), md)
    md = feature_dist_metrics(100.0 * train + 50.0, train)    # scale+shift drift
    assert dist_severity(md) > 0.6, ("dist value drift not caught", dist_severity(md), md)
    md = feature_dist_metrics(pd.Series(np.nan, index=idx), train)  # full fallback
    assert dist_severity(md) == 1.0, ("dist full fallback", dist_severity(md))
    lf = train.copy(); lf.iloc[:140] = np.nan                  # 70% fallback
    md = feature_dist_metrics(lf, train)
    assert abs(md["pct_fallback"] - 0.70) < 0.02 and dist_severity(md) > 0.65, ("dist fallback", md)
    md_low = feature_dist_metrics(train.iloc[:10].copy(), train)  # too few live samples
    assert dist_severity(md_low, min_live=30) >= 0.5, ("dist low-n floor", dist_severity(md_low, 30))
    print("[self_test] OK — minute metrics (fallback/sign-flip/value-drift/full-fallback/low-n) "
          "AND distribution metrics (identical/drift/fallback/low-n) all behave.")


def main():
    ap = argparse.ArgumentParser(description="Train-vs-live feature divergence analysis")
    ap.add_argument("--capture", help="glob for shadow_capture parquet(s)")
    ap.add_argument("--models_dir", default=os.path.join(HFT_DIR, "models", "live_v8"))
    ap.add_argument("--train_dir", default=os.path.join(HFT_DIR, "data", "artifacts_xgb"))
    ap.add_argument("--train_template", default="xgb_features_hyperliquid_{asset}_*.parquet",
                    help="glob template per asset (the most recent match is used)")
    ap.add_argument("--out", default=os.path.join(HFT_DIR, "data", "shadow_capture", "shadow_diff_report"),
                    help="output prefix (.csv + .md are written)")
    ap.add_argument("--align", choices=["dist", "minute"], default="dist",
                    help="dist = distribution comparison (default; no timestamp overlap needed). "
                         "minute = per-minute matched (needs a training parquet covering the capture window)")
    ap.add_argument("--train_parquet", default=None,
                    help="explicit training parquet to compare against (overrides auto-pick); "
                         "in practice required for --align minute over a fresh capture window")
    ap.add_argument("--min_aligned", type=int, default=MIN_ALIGNED,
                    help="min samples (overlap minutes in 'minute' mode, live rows in 'dist' mode) "
                         "before a per-model verdict is trusted (else INSUFFICIENT-DATA)")
    ap.add_argument("--min_n_both", type=int, default=MIN_N_BOTH,
                    help="min comparable samples before a feature's value-drift terms are trusted")
    ap.add_argument("--self_test", action="store_true", help="run synthetic metric checks and exit")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.capture:
        ap.error("--capture is required (or use --self_test)")
    run_diff(args.capture, args.models_dir, args.train_dir, args.train_template, args.out,
             align=args.align, train_parquet=args.train_parquet,
             min_n_both=args.min_n_both, min_aligned=args.min_aligned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
