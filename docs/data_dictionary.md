# Data dictionary: IEEE-CIS Fraud Detection (Vesta)

Real, anonymised e-commerce card transactions released by Vesta for the 2019 IEEE-CIS Kaggle competition. Every number below is reproduced by [`notebooks/00_data_overview.ipynb`](../notebooks/00_data_overview.ipynb).

## Files

| File | Rows | Columns | Size | Role |
|---|---|---|---|---|
| `train_transaction.csv` | 590,540 | 394 | 683 MB | Labelled transactions (`isFraud`). The main table. |
| `train_identity.csv` | 144,233 | 41 | 27 MB | Device and network data for some train transactions. |
| `test_transaction.csv` | 506,691 | 393 | 613 MB | The following ~6 months, **no labels**. Used here as an unlabelled "production" period. |
| `test_identity.csv` | 141,907 | 41 | 26 MB | Same as above; column names use `id-01` instead of `id_01` (fixed in `load_raw`). |
| `sample_submission.csv` | 506,691 | 2 | 6 MB | Kaggle template (all 0.5). Not used. |

- **Key:** `TransactionID`, unique in every file. Every identity row matches a transaction.
- **Identity coverage:** only **24.4%** of train (28.0% of test) transactions have an identity record, so identity is left-joined; the rest have all `id_*`, `DeviceType`, `DeviceInfo` missing.
- **Time:** `TransactionDT` is seconds from an unpublished reference, not a timestamp. Train covers days 1–183, test days 213–396, with a **30-day gap** between them. Both periods show a volume peak about one year apart (train weeks 0–3, test weeks 52–55), consistent with the community assumption that day 0 is around 2017-12-01 and the peaks are the holiday season.

## Columns

Meanings follow Vesta's official description; most names are masked on purpose.

| Family | Cols | Meaning | Real type | Missing (min / median / max) | Distinct values |
|---|---|---|---|---|---|
| `TransactionAmt` | 1 | Payment amount in USD | numeric | 0 | 20,902 |
| `ProductCD` | 1 | Product code (W, C, R, H, S) | categorical | 0 | 5 |
| `card1`–`card6` | 6 | Payment card: card type, category, issuing bank, country, … | **categorical** (card1–3, card5 are numeric codes; card1 has 13,553 values) | 0 / 0.3% / 1.5% | up to 13,553 |
| `addr1`, `addr2` | 2 | Billing region, billing country | **categorical** codes | 11.1% | 332 / 74 |
| `dist1`, `dist2` | 2 | Distances (e.g. between billing address, IP, phone area) | numeric | 59.7% / 93.6% | 2,651 / 1,751 |
| `P_emaildomain`, `R_emaildomain` | 2 | Purchaser / recipient email domain | categorical | 16.0% / 76.8% | ~60 |
| `C1`–`C14` | 14 | Counts (e.g. addresses linked to the card); meaning masked | count, extreme tails | ~0 | up to 1,657 |
| `D1`–`D15` | 15 | Time deltas in days (e.g. since previous transaction) | numeric | 0.2% / 52.5% / 93.4% | up to 12,353 |
| `M1`–`M9` | 9 | Match checks (e.g. name on card vs address); T/F, M4 is M0/M1/M2 | categorical | 28.7% / 47.7% / 59.3% | 2–3 |
| `V1`–`V339` | 339 | Vesta-engineered features: ranking, counting, entity relations | numeric | 0% / 47.3% / 86.1% | up to 37,367 |
| `id_01`–`id_38` | 38 | Network (IP, ISP, proxy) and digital signature (browser, OS) | 01–11 numeric, 12–38 categorical | 75.6% / 82.4% / 99.2% after the join | up to 115,655 |
| `DeviceType`, `DeviceInfo` | 2 | Desktop / mobile; device model | categorical | 76.2% / 79.9% after the join | 2 / 1,786 |

Notes on value ranges:
- `TransactionAmt`: $0.25 to $31,937; median $68.77, 99th percentile $1,104; skew 14.4 (0.49 after `log1p`); 51.8% are whole-dollar amounts.
- `C*`: mostly 0 or 1 with maxima in the thousands (C1: median 1, p99 164, max 4,685). C3 is 0 on 99.6% of rows.
- `D*`: up to ~880 days; D4, D6, D11, D12, D14, D15 contain a few negative values; D9 lies in [0, 0.96] (likely time of day).

## Label

`isFraud = 1` for **3.50%** of train transactions (20,663; 1 in 29), and 3.87% of dollar volume.

Vesta's definition: a transaction with a reported chargeback is fraud, **and so are later transactions linked to it by user account, email address or billing address**. A transaction is labelled legitimate if nothing is reported within **120 days**.

Consequences:
- **Labels take up to ~4 months to mature.** A model can only be evaluated on transactions old enough; the 4-week delay used in M5 is optimistic.
- **Labels are entity-linked.** Card holders with a fraud in the training period have a **49.2%** fraud rate in later transactions vs 2.7% for everyone else (M1), so a model can score well by recognising people rather than behaviour.

## Findings and where they are used

| Finding | Evidence | Used in |
|---|---|---|
| **Missingness carries signal.** | Fraud rate 11.8% when `addr1` is missing vs 2.5% when present; 7.9% vs 2.1% with vs without an identity record; 10.5% vs 2.5% when `D8` is present vs missing. | Keep NaN (LightGBM routes it); never impute without a missing flag (M2 logistic). |
| **339 V columns are 14 blocks** that are missing on exactly the same rows. | Exact NaN-mask match; the same 14 blocks in test. | M4: monitor null rates per block, not per column. Near-duplicates inside blocks explain why correlation-based screening hurt in M2. |
| **Numeric-coded categoricals.** | `card1` 13,553 codes (7,041 seen fewer than 5 times), `addr1`, `card2/3/5`. | M4: JS divergence on values, not binned PSI. |
| **Strong segments.** | ProductCD C: 11.6% of rows, 11.7% fraud (W: 2.0%). Credit 6.7% vs debit 2.4%. Mobile 10.2%, desktop 6.5%. `M4 = M2`: 11.4%. Fraud rate by amount is U-shaped. | M3: segment-level cost analysis; M4: segment mix drift. |
| **The test period already drifts.** | `id_31` (browser): 60.6% of non-missing test rows use a version never seen in train (chrome 70/71, mobile safari 12); `id_30` (OS) 24.2%; `DeviceInfo` 13.8%. The V35–V52 block drops from 28.6% to 15.2% missing; D13 from 89.5% to 75.6%. ProductCD C rises from 11.6% to 13.7%. `TransactionAmt` is stable. | M4: real drift on the unlabelled test period, next to injected drift. |
| **Label maturity is 120 days.** | Vesta's definition. | M5: test label delays beyond 4 weeks. |
