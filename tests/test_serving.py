import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud.data import PROCESSED_DIR, TARGET, add_entity_key, add_time_columns
from fraud.decision import load_costs
from fraud.features import add_past_features, add_row_features, feature_columns
from fraud.serving.app import create_app
from fraud.serving.bundle import BUNDLE_DIR, build_bundle, load_bundle
from fraud.serving.history import HistoryStore
from fraud.serving.scorer import Scorer


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    """A tiny but complete bundle: synthetic transactions, a small LightGBM, real bundle builder."""
    rng = np.random.default_rng(0)
    n = 4_000
    card = rng.integers(1000, 1100, n)
    amount = rng.exponential(100, n)
    fraud = (rng.random(n) < 0.05 + 0.2 * (card % 10 == 0)).astype(np.int8)
    raw = pd.DataFrame({
        "TransactionID": np.arange(n, dtype=np.int32),
        TARGET: fraud,
        "TransactionDT": np.sort(rng.integers(86_400, 60 * 86_400, n)).astype(np.int32),
        "TransactionAmt": (amount * (1 + fraud)).astype(np.float32),
        "ProductCD": pd.Categorical(rng.choice(["W", "C", "H"], n)),
        "card1": card.astype(np.int16),
        "addr1": rng.choice([100.0, 200.0, np.nan], n).astype(np.float32),
        "D1": rng.integers(0, 300, n).astype(np.float32),
        "V1": rng.normal(size=n).astype(np.float32),
    })
    raw = add_entity_key(add_time_columns(raw))
    df = add_past_features(add_row_features(raw))
    features = feature_columns(df)
    model = lgb.LGBMClassifier(n_estimators=30, num_leaves=8, verbose=-1).fit(df[features].iloc[:3000], df[TARGET].iloc[:3000])
    valid = df.iloc[3000:]
    out = build_bundle(tmp_path_factory.mktemp("bundle"), model.booster_, raw,
                       model.predict_proba(valid[features])[:, 1], valid[TARGET], valid["TransactionAmt"],
                       load_costs(), cap=0.02, version="test")
    return out


@pytest.fixture(scope="module")
def client(bundle_dir):
    with TestClient(create_app(bundle_dir)) as c:
        yield c


def tx(**overrides):
    base = {"TransactionID": 99_001, "TransactionDT": 61 * 86_400, "TransactionAmt": 250.0, "card1": 1010,
            "ProductCD": "C", "addr1": 100, "D1": 3, "V1": 0.5}
    return base | overrides


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_version"] == "test" and body["history_keys"] > 0


def test_score_returns_a_consistent_decision(client):
    r = client.post("/score", json=tx(), params={"record": False})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["decision"] in {"approve", "decline"}
    # decline exactly when p is above the transaction's own threshold
    assert (body["fraud_probability"] > body["decline_threshold"]) == (body["decision"] == "decline")
    assert len(body["reasons"]) <= 3
    assert all(reason["contribution"] > 0 and reason["description"] for reason in body["reasons"])


def test_explain_modes(client):
    never = client.post("/score", json=tx(), params={"record": False, "explain": "never"}).json()
    always = client.post("/score", json=tx(), params={"record": False, "explain": "always"}).json()
    auto = client.post("/score", json=tx(), params={"record": False}).json()
    assert never["explained"] is False and never["reasons"] == []
    assert always["explained"] is True and len(always["reasons"]) > 0
    assert always["fraud_probability"] == never["fraud_probability"]   # explaining never changes the score
    assert auto["explained"] == (auto["decision"] == "decline")
    assert client.post("/score", json=tx(), params={"explain": "sometimes"}).status_code == 422


def test_threshold_falls_with_amount(client):
    small = client.post("/score", json=tx(TransactionAmt=10.0), params={"record": False}).json()
    large = client.post("/score", json=tx(TransactionAmt=2000.0), params={"record": False}).json()
    assert large["decline_threshold"] < small["decline_threshold"]


def test_validation_and_unknown_values(client):
    assert client.post("/score", json={"TransactionID": 1, "TransactionDT": 0, "TransactionAmt": 5.0}).status_code == 422
    assert client.post("/score", json=tx(TransactionAmt=-1.0)).status_code == 422
    unknown = client.post("/score", json=tx(ProductCD="never-seen", some_new_field="x"), params={"record": False})
    assert unknown.status_code == 200


def test_reasons_are_exact_shap_contributions(bundle_dir):
    bundle = load_bundle(bundle_dir)
    scorer = Scorer(bundle, HistoryStore.load(bundle_dir))
    x = scorer.frame(tx())[bundle.features]
    raw = bundle.booster.predict(x)[0]
    contributions = bundle.booster.predict(x, pred_contrib=True)[0]
    assert contributions.sum() == pytest.approx(np.log(raw / (1 - raw)), abs=1e-6)


def test_weekly_cap_update_writes_a_new_penalty_and_keeps_history(bundle_dir, tmp_path):
    import json
    import shutil

    from fraud.decision import bayes_policy, expected_good_decline_rate
    from fraud.serving.cap import update_penalty

    copy = shutil.copytree(bundle_dir, tmp_path / "bundle")
    rng = np.random.default_rng(1)
    p, amount = rng.beta(0.5, 6, 20_000), rng.exponential(100, 20_000)
    entry = update_penalty(copy, p, amount, target=0.009)
    decision = json.loads((copy / "decision.json").read_text())
    assert decision["fp_penalty"] == entry["new"] and len(decision["penalty_history"]) == 1
    assert expected_good_decline_rate(bayes_policy(p, amount, decision["costs"], fp_penalty=entry["new"]), p) <= 0.009
    assert load_bundle(copy).fp_penalty == entry["new"]   # the API picks it up on (re)load


def test_recording_updates_the_card_history(bundle_dir):
    bundle = load_bundle(bundle_dir)
    scorer = Scorer(bundle, HistoryStore.load(bundle_dir))
    first, second = tx(TransactionID=1, card1=1055), tx(TransactionID=2, card1=1055, TransactionDT=61 * 86_400 + 600)
    before = scorer.frame(second)["card1_n_1h"].iat[0]
    scorer.score(first, record=True)
    after = scorer.frame(second)["card1_n_1h"].iat[0]
    assert after == before + 1


@pytest.mark.data
@pytest.mark.skipif(not (BUNDLE_DIR / "model.txt").exists() or not (PROCESSED_DIR / "scores_tuned.parquet").exists(),
                    reason="needs the real bundle and data")
def test_api_scores_match_the_offline_pipeline_on_real_data():
    """Serving parity: 200 real test-week transactions, online features from a history seeded before them."""
    raw = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    offline = pd.read_parquet(PROCESSED_DIR / "scores_tuned.parquet").query("split == 'test'").set_index("TransactionID")
    t0 = 140 * 86_400 + 43_200
    rows = raw[raw["TransactionDT"] >= t0].sort_values("TransactionDT", kind="stable").head(200)
    rows = rows[rows["TransactionID"].isin(offline.index)]
    scorer = Scorer(load_bundle(BUNDLE_DIR), HistoryStore.from_frame(raw[raw["TransactionDT"] < t0]))
    online = []
    for r in rows.to_dict("records"):
        online.append(scorer.score({k: (None if pd.isna(v) else v) for k, v in r.items()}, record=True)["raw_score"])
    np.testing.assert_allclose(online, offline.loc[rows["TransactionID"], "score"], atol=1e-5)
