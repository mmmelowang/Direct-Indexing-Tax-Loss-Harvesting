"""Load point-in-time S&P 500 data from WRDS (CRSP).

    python -m tlh.data.wrds_loader --start 1999-01-01 --end 2025-12-31 --out data/crsp_sp500.pkl

Uses the new-format (CIZ) CRSP tables, which run through 2025-12-31; the old
(SIZ) tables stop at 2024-12-31. The old delisting table is read only for its
numeric delisting codes, which decide the Shumway (1997) fill.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd

from .market import MarketData

TABLES_CIZ = {
    "sp500_members": "crsp.dsp500list_v2",        # permno, mbrstartdt, mbrenddt
    "daily": "crsp.dsf_v2",                       # permno, dlycaldt, dlyret, dlyretx, dlyprc, dlycap, shrout, siccd, ticker, dlydelflg
    "delist": "crsp_a_stock.stkdelists",          # permno, delistingdt, delret, delactiontype, delreasontype, delretmisstype
    "index": "crsp.dsp500_v2",                    # caldt, vwretd
    "legacy_delist": "crsp.dsedelist",            # numeric delisting codes (through 2024) for the Shumway rule
}

def sic_to_sector(sic) -> int:
    """Coarse 10-sector map from SIC codes (Fama-French-10 style)."""
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return 9
    if 100 <= s <= 999 or 2000 <= s <= 2399 or 2700 <= s <= 2749 or 2770 <= s <= 2799 \
            or 3100 <= s <= 3199 or 3940 <= s <= 3989:
        return 0  # consumer non-durables
    if 2500 <= s <= 2519 or 3630 <= s <= 3659 or 3710 <= s <= 3711 or 3714 <= s <= 3716 \
            or 3750 <= s <= 3751 or 3792 <= s <= 3792 or 3900 <= s <= 3939 or 3990 <= s <= 3999:
        return 1  # consumer durables
    if 2520 <= s <= 2589 or 2600 <= s <= 2699 or 2750 <= s <= 2769 or 3000 <= s <= 3099 \
            or 3200 <= s <= 3569 or 3580 <= s <= 3629 or 3700 <= s <= 3709 or 3712 <= s <= 3713 \
            or 3715 <= s <= 3715 or 3717 <= s <= 3749 or 3752 <= s <= 3791 or 3793 <= s <= 3799 \
            or 3830 <= s <= 3839 or 3860 <= s <= 3899:
        return 2  # manufacturing
    if 1200 <= s <= 1399 or 2900 <= s <= 2999:
        return 3  # energy
    if 3570 <= s <= 3579 or 3660 <= s <= 3692 or 3694 <= s <= 3699 or 3810 <= s <= 3829 \
            or 7370 <= s <= 7379:
        return 4  # business equipment / tech
    if 4800 <= s <= 4899:
        return 5  # telecom
    if 5000 <= s <= 5999 or 7200 <= s <= 7299 or 7600 <= s <= 7699:
        return 6  # shops
    if 2830 <= s <= 2839 or 3693 <= s <= 3693 or 3840 <= s <= 3859 or 8000 <= s <= 8099:
        return 7  # healthcare
    if 4900 <= s <= 4949:
        return 8  # utilities
    if 6000 <= s <= 6999:
        return 10  # finance
    return 9      # other


def download(start: str, end: str):
    """Download the CIZ tables with standardized column names: (members, daily, delistings, index)."""
    import wrds
    T = TABLES_CIZ
    db = wrds.Connection()
    mem = db.raw_sql(f"""
        select permno, indno, mbrstartdt as start, mbrenddt as ending from {T['sp500_members']}
        where mbrenddt >= '{start}' and mbrstartdt <= '{end}'
    """)
    print("S&P 500 list: index numbers found:", sorted(mem.indno.dropna().unique().tolist()))
    permnos = tuple(int(p) for p in mem.permno.unique())
    daily = db.raw_sql(f"""
        select permno, dlycaldt as date, dlyret as ret, dlyretx as retx, dlyprc as prc,
               dlycap as mcap, shrout, siccd, ticker, dlydelflg
        from {T['daily']}
        where permno in {permnos} and dlycaldt between '{start}' and '{end}'
    """)
    delist = db.raw_sql(f"""
        select permno, delistingdt as dlstdt, delret as dlret, delactiontype, delreasontype, delretmisstype
        from {T['delist']}
        where permno in {permnos} and delistingdt between '{start}' and '{end}'
    """)
    try:   # numeric legacy codes, only used to decide the Shumway fill
        codes = db.raw_sql(f"""
            select permno, dlstdt, dlstcd from {T['legacy_delist']}
            where permno in {permnos} and dlstdt between '{start}' and '{end}'
        """)
        delist = delist.merge(codes[["permno", "dlstcd"]].drop_duplicates("permno"), on="permno", how="left")
    except Exception as e:
        print("legacy delisting codes not available:", str(e)[:100])
    idx = db.raw_sql(f"""
        select caldt, vwretd, sprtrn from {T['index']}
        where caldt between '{start}' and '{end}'
    """)
    db.close()
    return mem, daily, delist, idx



def _num(s: pd.Series) -> np.ndarray:
    """Numeric column -> float64 numpy array; WRDS nullable values (<NA>) become NaN."""
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)


# Shumway (1997): performance-related delistings (codes 500, 520-584) with a
# missing delisting return get -30%.
SHUMWAY_CODES = {500, *range(520, 585)}
SHUMWAY_RET = -0.30


def build_market(mem, daily, delist, idx) -> MarketData:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["permno"] = _num(daily["permno"]).astype(int)
    for c in ("ret", "retx", "prc", "shrout"):
        daily[c] = _num(daily[c])
    mem = mem.copy()
    mem["permno"] = _num(mem["permno"]).astype(int)
    mem["start"], mem["ending"] = pd.to_datetime(mem["start"]), pd.to_datetime(mem["ending"])
    delist = delist.copy()
    delist["dlret"] = _num(delist["dlret"])

    # CIZ: delisting returns may come as extra daily rows flagged dlydelflg == 'Y'.
    # Move them into ``delist`` (merged into the last trading day below) so they are
    # never double counted with the separate delisting table.
    if "dlydelflg" in daily:
        is_del = daily["dlydelflg"].astype(str).str.upper().eq("Y").to_numpy()
        print(f"daily rows flagged as delisting returns: {int(is_del.sum())}")
        if is_del.any():
            rows = daily.loc[is_del, ["permno", "date", "ret"]].rename(columns={"date": "dlstdt", "ret": "dlret"})
            daily = daily.loc[~is_del]
            # keep reason codes from the delisting table for the Shumway rule
            extra = [c for c in ("dlstcd",) if c in delist]
            rows = rows.merge(delist[["permno"] + extra].drop_duplicates("permno"), on="permno", how="left")
            delist = rows
    idx = idx.copy()
    idx["caldt"] = pd.to_datetime(idx["caldt"])
    idx["vwretd"] = _num(idx["vwretd"])
    # delisting returns: report missing ones and apply the Shumway (1997) fill
    delist["dlret"] = _num(delist["dlret"])
    miss = np.isnan(delist["dlret"].to_numpy())
    fill = np.zeros(len(delist), bool)
    if "dlstcd" in delist:
        fill = miss & np.isin(_num(delist["dlstcd"]), list(SHUMWAY_CODES))
        delist.loc[fill, "dlret"] = SHUMWAY_RET
    print(f"delisting returns: {len(delist)} total, {int(miss.sum())} missing, "
          f"{int(fill.sum())} filled with {SHUMWAY_RET:.0%} (Shumway), rest treated as 0")

    # market cap: CIZ provides dlycap; otherwise |price| x shares. Units do not matter (weights are normalized).
    cap_alt = np.abs(daily["prc"]) * daily["shrout"]
    if "mcap" in daily:
        cap = pd.Series(_num(daily["mcap"]), index=daily.index)
        both = cap.notna() & cap_alt.notna() & (cap_alt > 0)
        scale = float(np.median(cap[both] / cap_alt[both])) if both.any() else 1.0
        print(f"market cap: dlycap missing on {int(cap.isna().sum())} rows; fallback |prc|*shrout scaled by {scale:.4g}")
        daily["mcap"] = cap.fillna(cap_alt * scale)
    else:
        daily["mcap"] = cap_alt
    ret = daily.pivot(index="date", columns="permno", values="ret").sort_index()
    retx = daily.pivot(index="date", columns="permno", values="retx").reindex(ret.index)
    mcap = daily.pivot(index="date", columns="permno", values="mcap").reindex(ret.index)
    dates, ids = ret.index, ret.columns.values
    R, RX, MC = ret.values.astype(float), retx.values.astype(float), mcap.values.astype(float)

    # merge delisting returns into the last available trading day
    col = {p: j for j, p in enumerate(ids)}
    valid = ~np.isnan(R)
    for _, row in delist.dropna(subset=["dlret"]).iterrows():
        j = col.get(int(row.permno))
        if j is None or not valid[:, j].any():
            continue
        t = np.where(valid[:, j])[0][-1]
        if dates[t] > pd.Timestamp(row.dlstdt) + pd.Timedelta(days=10):
            continue   # stock kept trading well after this record: not a real exit
        R[t, j] = (1 + R[t, j]) * (1 + row.dlret) - 1

    # missing retx where ret exists -> assume no dividend that day
    RX = np.where(np.isnan(RX) & ~np.isnan(R), R, RX)

    # point-in-time membership
    inidx = np.zeros(R.shape, dtype=bool)
    for _, row in mem.iterrows():
        j = col.get(int(row.permno))
        if j is None:
            continue
        inidx[:, j] |= (dates >= row.start) & (dates <= row.ending)

    # sector and ticker: most recent values
    src = daily.assign(siccd=_num(daily["siccd"])).sort_values("date")
    last_sic = src.dropna(subset=["siccd"]).groupby("permno").siccd.last()
    last_tic = src.dropna(subset=["ticker"]).groupby("permno").ticker.last()
    sector = np.array([sic_to_sector(last_sic.get(p)) for p in ids])
    tick = np.array([str(last_tic.get(p, p)) for p in ids])

    bench = idx.set_index("caldt").vwretd.reindex(dates).values.astype(float)
    return MarketData(dates, ids, R, RX, MC, inidx, sector, tick, bench)


def validate(md: MarketData) -> None:
    """Sanity checks printed after loading."""
    n_members = md.in_index.sum(axis=1)
    print(f"days {md.T}, assets {md.N}, {md.dates[0].date()} .. {md.dates[-1].date()}")
    print(f"index members per day: min {n_members.min()}, median {int(np.median(n_members))}, max {n_members.max()}  (expect ~495-505)")
    print(f"delisted assets in sample: {int(md.delists.sum())}")
    print("sector counts:", dict(zip(*np.unique(md.sector, return_counts=True))))
    rb = md.bench_returns()[1:]
    if md.bench_ret_official is not None:
        ro = md.bench_ret_official[1:]
        ok = ~np.isnan(ro)
        d = rb[ok] - ro[ok]
        yrs = ok.sum() / 252
        print(f"reconstructed vs CRSP S&P 500 (vwretd): daily corr {np.corrcoef(rb[ok], ro[ok])[0, 1]:.5f}, "
              f"tracking diff stdev (ann.) {d.std() * np.sqrt(252):.4%}, "
              f"ann. return {np.prod(1 + rb[ok]) ** (1 / yrs) - 1:.2%} vs {np.prod(1 + ro[ok]) ** (1 / yrs) - 1:.2%}")
    big = np.nanmax(np.abs(md.ret))
    print(f"largest absolute daily return in data: {big:.2f}")


def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1999-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", default="data/crsp_sp500.pkl")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    print("downloading from WRDS (a few minutes)...")
    raw = download(a.start, a.end)
    with open(a.out.replace(".pkl", "_raw.pkl"), "wb") as f:
        pickle.dump(raw, f)
    md = build_market(*raw)
    with open(a.out, "wb") as f:
        pickle.dump(md, f)
    print(f"saved {md.T} days x {md.N} assets to {a.out}")
    validate(md)


if __name__ == "__main__":
    main()
