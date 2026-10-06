# -*- coding: utf-8 -*-
"""血統レーダー v2 の R3b: DL挑戦モデル v3(条件エンコーダ → レースの要求ベクトル)(2026-10-06)。

事前登録 RADAR_V2_PREREG_2026_10_06.md §8・追補3(3-5)・追補4。実行は専用 venv: .venv_dl/Scripts/python.exe
- 入力(レースの条件、すべて発走前に分かる量): 競馬場・芝ダ・log距離・内外回り・馬場状態・クラス序数・新馬戦・頭数(完走馬の最大馬番)・
  予測ペース(同じコース(内外回り込み)のその日より前のレースのペースの平均)・出走馬の過去の前半位置の平均と過去走のある馬の割合。
  §8 の「ペース区分」は当該レースの実際のラップから作った値で結果の情報が入るため使わない(追補3)。
- 出力: 9本の血統の値(脚質・キレ・芝・ダ・距離帯4つ・新馬、z = 値/τ、時点表 Y−1年末まで、本人除外)に掛ける要求 req(x)。
- 効用 u = β1·S(父の総合力) + Σ_j req_j(x)·z_j。損失は上位3頭の Plackett–Luce(radar_v2_eval.PL と同じ定義)+ L2(MLP の重み)。
- 構造: 埋め込み(競馬場3・内外回り2・馬場2)+数値8 → 32 → 32 → 9(ReLU)、約1.9千パラメータ。
- 学習: Y年の評価には 2014〜Y−1年のレース。L2 {1e-4, 1e-3} × エポック {10, 30} は 2019・2020年の検証(学習は各前年まで)だけで選ぶ。seed 3本の効用の平均。
- 対照(すべて毎年 Y−1年までで推定し直す): C0 父の総合力のみ / C1 v2の形(S+F、係数2) / C2 v2の基準で軸ごとの係数(S+5軸) / C3 条件に線形な要求(隠れ層なし、同じ入力)。
- 副次(5)の主比較(追補4で固定): v3 対 C2 の1レースあたり ΔLL(2021〜2026年)、95%CI下限>0(日付×競馬場のブロック)。

使い方:
  --build   特徴量表を作る(radar_v2/dl_table.parquet、Git管理外)
  --smoke   動作確認(f=0 で C0 と一致、1エポックの所要時間)。2020年までのみ
  --tune    L2・エポック数の選択(2019・2020年の検証のみ)→ radar_v2/r3b_tune.json
  --final   2021〜2026年の評価(1回のみ、結果があれば拒否)→ radar_v2/radar_v2_dl_eval.json
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import radar_v2_common as C  # noqa: E402
import radar_v2_aptitude as A  # noqa: E402
import radar_v2_eval as E  # noqa: E402

TABLE = C.OUT_DIR / "dl_table.parquet"
TUNE_OUT = C.OUT_DIR / "r3b_tune.json"
FINAL_OUT = C.OUT_DIR / "radar_v2_dl_eval.json"
TRAIN_START = 2014
SEEDS = [20261006, 20261007, 20261008]
GRID = [(l2, ep) for l2 in (1e-4, 1e-3) for ep in (10, 30)]
LR, BATCH = 3e-3, 256
ZCOLS = ["z_front", "z_kire", "z_s|芝", "z_s|ダ"] + [f"z_b|{b}" for b in E.BUCKETS] + ["z_debut"]
CATS = {"racecourse": ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"],
        "turn_type": ["単一", "内回り", "外回り", "内外", "unknown"], "going": ["良", "稍重", "重", "不良"]}
EMB = {"racecourse": 3, "turn_type": 2, "going": 2}
NUMS = ["is_dirt", "logd", "class_n", "debut", "n_start", "prior_pace", "field_front", "field_hist"]
COMPS = ["F_front", "F_kire", "F_surf", "F_db", "F_deb"]
log = E.log
torch.set_num_threads(8)


# ------------------------------------------------------------------ 特徴量表
def race_inputs(runs: pd.DataFrame) -> pd.DataFrame:
    """レース単位の入力(発走前に分かる量のみ)。"""
    r = runs.sort_values(["date", "race_id"])
    races = r.drop_duplicates("race_id").set_index("race_id")[["date", "year", "racecourse", "surface", "logd", "turn_type",
                                                               "going", "class_ord", "is_debut_race", "course_key_io", "pace"]].copy()
    races["n_start"] = r.groupby("race_id")["umaban_num"].max()
    # 予測ペース: 同じコースキーで日付が厳密に前のレースのペースの平均(当日のレースは含めない)
    d = races.groupby(["course_key_io", "date"])["pace"].agg(["sum", "count"]).sort_index()
    cs = d.groupby(level=0).cumsum() - d
    races = races.join(cs.rename(columns={"sum": "pp_sum", "count": "pp_n"}), on=["course_key_io", "date"])
    g = races.groupby("date")["pace"].agg(["sum", "count"]).sort_index()
    ga = g.cumsum() - g
    races = races.join(ga.rename(columns={"sum": "pa_sum", "count": "pa_n"}), on="date")
    races["prior_pace"] = (races["pp_sum"] / races["pp_n"].replace(0, np.nan)).fillna(races["pa_sum"] / races["pa_n"].replace(0, np.nan))
    # 出走馬の過去の前半位置(その日より前の自身の走の front_raw の平均)の、出走馬平均と過去走のある馬の割合
    h = r.sort_values(["horse_id", "date"])[["race_id", "horse_id", "front_raw"]].copy()
    x, n = h["front_raw"].fillna(0), h["front_raw"].notna().astype(float)
    h["pf"] = (x.groupby(h["horse_id"]).cumsum() - x) / (n.groupby(h["horse_id"]).cumsum() - n).replace(0, np.nan)
    races["field_front"] = h.groupby("race_id")["pf"].mean()
    races["field_hist"] = h.groupby("race_id")["pf"].apply(lambda s: s.notna().mean())
    races["is_dirt"] = (races["surface"] == "ダ").astype(float)
    races["class_n"] = races["class_ord"] / 7.0
    races["debut"] = races["is_debut_race"].astype(float)
    return races.drop(columns=["pace", "pp_sum", "pp_n", "pa_sum", "pa_n"])   # 当該レースの実際のペースは持たない


def components(D: pd.DataFrame) -> pd.DataFrame:
    rel = E.rel
    D["F_front"] = rel("front") * D["q_front"].fillna(0) * D["z_front"].fillna(0)
    D["F_kire"] = rel("kire") * D["q_kire"].fillna(0) * D["z_kire"].fillna(0)
    D["F_surf"] = D["r_surface"] * D["z_surface"].fillna(0)
    D["F_db"] = D["r_dbucket"] * D["z_dbucket"].fillna(0)
    D["F_deb"] = rel("debut") * D["is_debut_race"].astype(float) * D["z_debut"].fillna(0)
    assert np.allclose(D[COMPS].sum(axis=1), D["F"])
    return D


def build():
    t0 = time.time()
    runs = pd.read_parquet(A.RUNS)
    runs = A.add_tiers(runs, A.fixed_thresholds(runs))
    runs["umaban_num"] = pd.to_numeric(runs["umaban"], errors="coerce")
    req = pd.read_parquet(C.OUT_DIR / "race_req.parquet")
    races = race_inputs(runs)
    keep = ["race_id", "horse_id", "date", "year", "pos", "umaban_num", "field", "S", "F", "ped_S_horse_id"] + ZCOLS + COMPS
    frames = []
    for Y in range(TRAIN_START, 2027):
        f = components(E.features_for_year(runs, req, Y))
        frames.append(f[keep])
        log("features", Y, len(f))
    D = pd.concat(frames, ignore_index=True)
    D = D[D["field"] >= 5].join(races.drop(columns=["date", "year"]), on="race_id")
    D["block"] = D["date"].dt.strftime("%Y%m%d") + "_" + D["race_id"].str[4:6]
    # 確定オッズ(市場を入れた比較用)
    od = []
    for y in range(TRAIN_START, 2027):
        for p in sorted(Path(C.RESULTS_DIR / str(y)).glob("*.csv")):
            od.append(pd.read_csv(p, dtype=str, usecols=["race_id", "horse_id", "odds_final"]))
    od = pd.concat(od).drop_duplicates(["race_id", "horse_id"])
    od["odds"] = pd.to_numeric(od["odds_final"], errors="coerce")
    D = D.merge(od[["race_id", "horse_id", "odds"]], on=["race_id", "horse_id"], how="left")
    D["logp"] = np.log((1 / D["odds"]) / D.groupby("race_id")["odds"].transform(lambda s: (1 / s).sum()))
    D.to_parquet(TABLE, index=False)
    log("wrote", TABLE, len(D), "rows", D["race_id"].nunique(), "races", round(time.time() - t0), "sec")


# ------------------------------------------------------------------ テンソル化
class Batch:
    """レース×馬の詰め物テンソル(順序は E.PL と同じ: race_id, pos, umaban_num)。"""

    def __init__(self, df: pd.DataFrame, stats: dict):
        d = df.sort_values(["race_id", "pos", "umaban_num"])
        d = d[d.groupby("race_id")["race_id"].transform("size") >= 3]
        self.race_ids, idx = np.unique(d["race_id"].to_numpy(), return_index=True)
        counts = np.diff(np.append(idx, len(d)))
        R, N = len(counts), int(counts.max())
        pos = np.arange(len(d)) - np.repeat(idx, counts)
        ri = np.repeat(np.arange(R), counts)
        Z = np.zeros((R, N, len(ZCOLS)), np.float32)
        Z[ri, pos] = d[ZCOLS].fillna(0).to_numpy(np.float32)
        S = np.zeros((R, N), np.float32)
        S[ri, pos] = d["S"].fillna(0).to_numpy(np.float32)
        M = np.zeros((R, N), bool)
        M[ri, pos] = True
        first = d.drop_duplicates("race_id").set_index("race_id").loc[self.race_ids]
        num = first[NUMS].astype(float).copy()
        for c in NUMS:
            num[c] = ((num[c] - stats[c][0]) / stats[c][1]).fillna(0.0)
        self.num = torch.tensor(num.to_numpy(np.float32))
        self.cat = {k: torch.tensor(first[k].map({v: i for i, v in enumerate(vs)}).fillna(len(vs) - 1).astype(int).to_numpy())
                    for k, vs in CATS.items()}
        self.Z, self.S, self.M = torch.tensor(Z), torch.tensor(S), torch.tensor(M)
        self.blocks = first["block"].to_numpy()

    def take(self, i):
        return {"num": self.num[i], "cat": {k: v[i] for k, v in self.cat.items()}, "Z": self.Z[i], "S": self.S[i], "M": self.M[i]}

    def all(self):
        return self.take(slice(None))


def norm_stats(df: pd.DataFrame) -> dict:
    first = df.drop_duplicates("race_id")
    return {c: (float(first[c].mean()), float(first[c].std() or 1.0)) for c in NUMS}


def pl_ll(u: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """上位3頭の PL 対数尤度(レースごと)。行は着順で並んでいる。"""
    u = u.masked_fill(~M, float("-inf"))
    ll = torch.zeros(u.shape[0])
    for k in range(3):
        ll = ll + u[:, k] - torch.logsumexp(u[:, k:], dim=1)
    return ll


class Encoder(torch.nn.Module):
    def __init__(self, hidden: bool = True):
        super().__init__()
        self.emb = torch.nn.ModuleDict({k: torch.nn.Embedding(len(CATS[k]), EMB[k]) for k in CATS})
        d_in = len(NUMS) + sum(EMB.values())
        out = len(ZCOLS)
        self.net = (torch.nn.Sequential(torch.nn.Linear(d_in, 32), torch.nn.ReLU(), torch.nn.Linear(32, 32), torch.nn.ReLU(),
                                        torch.nn.Linear(32, out)) if hidden else torch.nn.Linear(d_in, out))
        self.b1 = torch.nn.Parameter(torch.tensor(0.3))

    def req(self, b):
        x = torch.cat([b["num"]] + [self.emb[k](b["cat"][k]) for k in CATS], dim=1)
        return self.net(x)

    def forward(self, b):
        return self.b1 * b["S"] + torch.einsum("rj,rnj->rn", self.req(b), b["Z"])

    def l2(self):
        return sum((p ** 2).sum() for n, p in self.named_parameters() if n != "b1")


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def train(tr: Batch, l2: float, epochs: int, seed: int, hidden=True) -> Encoder:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Encoder(hidden)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    R = len(tr.race_ids)
    for _ in range(epochs):
        perm = rng.permutation(R)
        for s in range(0, R, BATCH):
            b = tr.take(torch.tensor(perm[s:s + BATCH]))
            loss = -pl_ll(m(b), b["M"]).mean() + l2 * m.l2()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


def ens_u(models, b) -> torch.Tensor:
    with torch.no_grad():
        return torch.stack([m(b) for m in models]).mean(0)


# ------------------------------------------------------------------ 対照(scipy の PL、毎年推定し直す)
def pl_fit_eval(fit_df, test_df, feats):
    beta = E.PL(fit_df, feats).fit()
    t = E.PL(test_df, feats)
    return t.race_ll(beta), t.race_ids, beta


def fold(D, Y):
    tr = D[D["year"].between(TRAIN_START, Y - 1)]
    te = D[D["year"] == Y]
    return tr, te


# ------------------------------------------------------------------ 実行モード
def smoke(D):
    D = D[D["year"] <= 2020]
    assert D["year"].max() <= 2020
    tr, te = fold(D, 2020)
    st = norm_stats(tr)
    btr, bte = Batch(tr, st), Batch(te, st)
    m = Encoder(True)
    log("params v3", n_params(m), "C3", n_params(Encoder(False)))
    # f=0(最後の層を0)なら C0(u = β1·S)の尤度と一致
    with torch.no_grad():
        m.net[-1].weight.zero_()
        m.net[-1].bias.zero_()
        m.b1.fill_(0.3)
        ll_t = pl_ll(m(bte.all()), bte.M).numpy()
    ll_e = E.PL(te, ["S"]).race_ll(np.array([0.3]))
    assert np.array_equal(bte.race_ids, E.PL(te, ["S"]).race_ids) and np.allclose(ll_t, ll_e, atol=1e-4), (ll_t[:3], ll_e[:3])
    log("f=0 matches C0: OK", float(np.abs(ll_t - ll_e).max()))
    t0 = time.time()
    train(btr, 1e-4, 1, SEEDS[0])
    log("1 epoch sec", round(time.time() - t0, 1), "train races", len(btr.race_ids))


def tune(D):
    D = D[D["year"] <= 2020]
    assert D["year"].max() <= 2020
    res = {"grid": [], "note": "検証は2019・2020年(学習は2014〜各前年)。2021年以降は読まない。"}
    for l2, ep in GRID:
        rows = {}
        for Y in (2019, 2020):
            tr, te = fold(D, Y)
            st = norm_stats(tr)
            btr, bte = Batch(tr, st), Batch(te, st)
            ms = [train(btr, l2, ep, s) for s in SEEDS]
            ll = pl_ll(ens_u(ms, bte.all()), bte.M).numpy()
            c0, _, _ = pl_fit_eval(tr, te, ["S"])
            rows[Y] = {"ll": float(ll.mean()), "dll_vs_C0": float(ll.mean() - c0.mean())}
            log("tune", l2, ep, Y, rows[Y])
        res["grid"].append({"l2": l2, "epochs": ep, "by_year": rows, "mean_ll": float(np.mean([r["ll"] for r in rows.values()]))})
    best = max(res["grid"], key=lambda g: g["mean_ll"])
    res["selected"] = {"l2": best["l2"], "epochs": best["epochs"]}
    TUNE_OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log("selected", res["selected"])


def final(D, years=range(2021, 2027), out=None):
    out = out or FINAL_OUT
    if out == FINAL_OUT and FINAL_OUT.exists():
        sys.exit(f"{FINAL_OUT.name} は既にあります(2021〜2026年の (5) は1回だけ)。")
    sel = json.loads(TUNE_OUT.read_text(encoding="utf-8"))["selected"]
    t0 = time.time()
    per = {k: [] for k in ["C0", "C1", "C2", "C3", "v3", "C2_mkt0", "C2_mkt", "v3_mkt"]}
    betas = {}
    for Y in years:
        tr, te = fold(D, Y)
        st = norm_stats(tr)
        btr, bte = Batch(tr, st), Batch(te, st)
        c0, r0, _ = pl_fit_eval(tr, te, ["S"])
        assert np.array_equal(r0, bte.race_ids)
        c1, _, b1 = pl_fit_eval(tr, te, ["S", "F"])
        c2, _, b2 = pl_fit_eval(tr, te, ["S"] + COMPS)
        m3 = [train(btr, sel["l2"], sel["epochs"], s, hidden=False) for s in SEEDS]
        mv = [train(btr, sel["l2"], sel["epochs"], s, hidden=True) for s in SEEDS]
        c3 = pl_ll(ens_u(m3, bte.all()), bte.M).numpy()
        v3 = pl_ll(ens_u(mv, bte.all()), bte.M).numpy()
        # 市場を入れた比較: 学習年の v3 の適合度 G = u − β1·S を固定し、[logp, S] → +G を学習年で推定
        def g_score(models, df):
            b = Batch(df, st)
            u = ens_u(models, b.all()).numpy()
            b1m = float(np.mean([m.b1.item() for m in models]))
            g = u - b1m * b.S.numpy()
            out = pd.Series(np.nan, index=df.index)
            dd = df.sort_values(["race_id", "pos", "umaban_num"])
            dd = dd[dd.groupby("race_id")["race_id"].transform("size") >= 3]
            idx = np.unique(dd["race_id"].to_numpy(), return_index=True)[1]
            cnt = np.diff(np.append(idx, len(dd)))
            pos = np.arange(len(dd)) - np.repeat(idx, cnt)
            out.loc[dd.index] = g[np.repeat(np.arange(len(cnt)), cnt), pos]
            return out
        trm, tem = tr[ok_race(tr)].copy(), te[ok_race(te)].copy()
        trm["G3"], tem["G3"] = g_score(mv, trm), g_score(mv, tem)
        trm["G2"], tem["G2"] = trm[COMPS].to_numpy() @ b2[1:], tem[COMPS].to_numpy() @ b2[1:]
        mk0, rm, _ = pl_fit_eval(trm, tem, ["logp", "S"])
        mk2, _, _ = pl_fit_eval(trm, tem, ["logp", "S", "G2"])
        mk3, _, _ = pl_fit_eval(trm, tem, ["logp", "S", "G3"])
        for k, v in [("C0", c0), ("C1", c1), ("C2", c2), ("C3", c3), ("v3", v3)]:
            per[k].append(pd.Series(v, index=bte.race_ids))
        for k, v in [("C2_mkt0", mk0), ("C2_mkt", mk2), ("v3_mkt", mk3)]:
            per[k].append(pd.Series(v, index=rm))
        betas[Y] = {"C1": [round(float(x), 4) for x in b1], "C2": [round(float(x), 4) for x in b2],
                    "v3_b1": round(float(np.mean([m.b1.item() for m in mv])), 4)}
        log("final", Y, {k: round(float(v[-1].mean() - per["C0"][-1].mean()), 5) for k, v in per.items() if k in ("C1", "C2", "C3", "v3")})
    P = {k: pd.concat(v) for k, v in per.items()}
    blk = D.drop_duplicates("race_id").set_index("race_id")["block"]

    def cmp(a, b, name):
        d = (P[a] - P[b]).dropna()
        ci, p, se = E.block_boot(d.to_numpy(), blk.loc[d.index].to_numpy(), with_p=True)
        yrs = pd.Series(d.index.str[:4], index=d.index)
        return {"name": name, "n_races": int(len(d)), "dll_per_race": round(float(d.mean()), 5), "ci95_block": ci,
                "p_one_sided": round(p, 5), "by_year": d.groupby(yrs).mean().round(5).to_dict()}
    res = {"selected_hparams": sel, "seeds": SEEDS, "train_start": TRAIN_START, "inputs": NUMS + list(CATS),
           "params": {"v3": n_params(Encoder(True)), "C3": n_params(Encoder(False))},
           "primary_5": cmp("v3", "C2", "(5) v3 対 C2(v2の基準+軸ごとの係数、毎年推定)"),
           "vs_C0": {k: cmp(k, "C0", f"{k} 対 C0(父の総合力のみ、毎年推定)") for k in ["C1", "C2", "C3", "v3"]},
           "v3_vs_C3": cmp("v3", "C3", "v3 対 C3(条件に線形な要求): 隠れ層の効果"),
           "C3_vs_C2": cmp("C3", "C2", "C3 対 C2: 条件から学んだ要求(線形)の効果"),
           "market": {"C2": cmp("C2_mkt", "C2_mkt0", "市場+S → +C2 の適合度"), "v3": cmp("v3_mkt", "C2_mkt0", "市場+S → +v3 の適合度")},
           "betas": betas}
    R4 = json.loads((C.OUT_DIR / "radar_v2_eval.json").read_text(encoding="utf-8"))
    fam = {k: v["p"] for k, v in R4["holm"].items()}
    fam["5"] = res["primary_5"]["p_one_sided"]
    res["years"] = list(years)
    res["holm_final"] = E.holm(fam)
    res["manifest"] = E.manifest() | {"torch": torch.__version__, "dl_table_sha256": E.file_sha([TABLE]),
                                       "tune_sha256": E.file_sha([TUNE_OUT])}
    res["elapsed_sec"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                         encoding="utf-8")
    E.append_ledger({"script": "radar_v2_dl_condition_encoder.py", "mode": "confirm (5)" if out == FINAL_OUT else "dry (2019-2020)",
                     "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "git_commit": res["manifest"]["git_commit"], "git_dirty_files": res["manifest"]["git_dirty_files"],
                     "output": out.name, "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest()})
    log("primary (5)", res["primary_5"]["dll_per_race"], res["primary_5"]["ci95_block"])


def ok_race(X):
    return X.groupby("race_id")["logp"].transform(lambda s: bool(np.isfinite(s).all()))


def main():
    if sys.platform == "win32":   # 長い学習の途中で PC がスリープしないように(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    if "--build" in sys.argv:
        build()
        return
    D = pd.read_parquet(TABLE)
    if "--smoke" in sys.argv:
        smoke(D)
    elif "--tune" in sys.argv:
        tune(D)
    elif "--final-dry" in sys.argv:   # 本番と同じ処理を2019・2020年(検証の期間)だけで通す配線確認
        D = D[D["year"] <= 2020]
        assert D["year"].max() <= 2020
        final(D, years=(2019, 2020), out=C.OUT_DIR / "r3b_final_dry.txt")
    elif "--final" in sys.argv:
        final(D)


if __name__ == "__main__":
    main()
