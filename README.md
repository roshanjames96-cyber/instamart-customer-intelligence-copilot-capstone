# Instamart Customer Intelligence & Support Copilot

## Overview

This repository implements a deterministic, fully synthetic customer intelligence and support copilot across data cleaning, customer segmentation and lapse scoring, privacy-aware support triage, policy retrieval, and a tool-using refund agent. It uses only the local mock language model in `mock_llm.py`; it makes no network calls and needs no API key. All money is in INR.

## Reproduce

Run these commands from the repository root with Python 3.11 or newer. The scripts rebuild the generated data, clean database, models, triage outputs, agent ledger and checkpoints.

1. `python -m venv .venv`
2. Windows PowerShell: `\.venv\Scripts\Activate.ps1` (macOS/Linux: `source .venv/bin/activate`)
3. `python -m pip install --upgrade pip`
4. `python -m pip install -r requirements.txt`
5. `python generate_data.py`
6. `python run_all.py`

`run_all.py` regenerates the same data as step 5 before running the Parts in dependency order. The standalone generator's final output line is `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. `validate.py` checks the cleaned database and prints exactly `ALL CHECKPOINTS PASSED` on success.

## Part A

### Raw audit

Pandas shapes and inferred types from the audit run:

| Raw table | Shape | Column types |
|---|---:|---|
| customers.csv | 1,200 × 5 | customer_id object; city object; signup_date object; age string; gender object |
| products.csv | 40 × 4 | product_id object; product_name object; category object; price_inr int64 |
| orders.csv | 13,084 × 9 | order_id object; customer_id object; product_id object; quantity int64; order_ts object; status object; discount_pct int64; order_amount object; delivery_minutes object |
| tickets.csv | 164 × 5 | ticket_id object; customer_id object; order_id object; text object; true_category object |

Missing-value audit (count; percentage of table rows). Tickets are read with `keep_default_na=False`, so intentionally empty text fields remain empty strings and do not appear as pandas NA.

| Table | Column | Missing count | Missing % |
|---|---|---:|---:|
| customers | customer_id / city / signup_date | 0 each | 0.000 each |
| customers | age | 61 | 5.083 |
| customers | gender | 30 | 2.500 |
| products | product_id / product_name / category / price_inr | 0 each | 0.000 each |
| orders | all 9 columns | 0 each | 0.000 each |
| tickets | all 5 columns | 0 each | 0.000 each |

| Audit measure | Result |
|---|---:|
| Raw orders | 13,084 |
| Exact duplicate rows | 119 |
| Pending rows | 656 |
| Distinct order IDs | 12,309 |
| Non-numeric order amounts | 120 |
| Negative delivery times | 106 |
| Blank delivery times | 2,541 |
| Blank customer ages | 61 |
| Ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct nonblank raw gender spellings | 10 |
| Blank genders | 30 |

### Cleaning, storage and validation

Customers: city strings were trimmed and title-cased (24 spellings become 6 cities); gender variants were normalized to `male`, `female`, or `unknown` (610 female, 560 male, 30 unknown). Ages outside 10–100 were set missing, then missing ages were filled with the median of the normalized city (61 blank plus 6 impossible values; final missing ages 0 and all ages 18–60). Signup dates are datetimes in the clean table.

Orders: `order_ts` was parsed as datetime; 119 exact duplicate rows were dropped; sorting by timestamp and keeping the latest row for each ID removed the 656 superseded pending records. The 120 unparseable amounts were converted to missing then recalculated from product price, quantity and discount. Negative delivery times were set missing (106 invalid raw values); missing completed-order delivery times were filled with the completed-order median for that product category. Cancelled orders keep a missing delivery time. The clean table has 12,309 orders: 11,110 completed and 1,199 cancelled; no pending orders; no missing amounts; no missing delivery time for completed orders; 1,199 missing delivery times for cancelled orders. Completed revenue is INR 2,998,668.80. The three clean CSVs and SQLite tables are written under `data/clean/` and `data/instamart.db`.

Validation asserts 9 reference checks spanning row counts, order statuses, null handling, completed revenue, customer values and SQLite table names; it prints `ALL CHECKPOINTS PASSED`.

### SQL and pandas analysis

| Query | Result |
|---|---|
| Q1 — completed orders above INR 500, grouped customers above INR 8,000 | 30 customers |
| Q2 — revenue by city × category | 30 cells; total INR 2,998,668.80; largest is Bengaluru / Electronics at INR 233,909.45 |
| Q3 — products priced above catalogue mean | 11 products |
| Q4 — customers without completed orders | 25 customers |
| Q5 — customers above mean customer spend | 550 customers; mean spend INR 2,552.06 |
| Q6 — city spend ranks | Bengaluru C0082 INR 12,856.20; Chennai C0930 INR 10,151.40; Delhi C0291 INR 13,742.60; Hyderabad C0743 INR 12,730.05; Mumbai C0693 INR 14,766.55; Pune C0426 INR 10,377.10. Eight rows have `RANK()` different from `ROW_NUMBER()` because equal spend values share a rank, while row numbers remain unique. |
| Q7 — daily revenue with running total | 183 days; final cumulative revenue INR 2,998,668.80 |

The pandas pivot table's grand total also equals INR 2,998,668.80. Completed order spend by city: Bengaluru INR 523,643.40 / mean INR 276.47 / 1,894 orders; Chennai INR 463,629.15 / INR 272.72 / 1,700; Delhi INR 502,829.80 / INR 243.62 / 2,064; Hyderabad INR 528,589.90 / INR 284.95 / 1,855; Mumbai INR 479,058.50 / INR 297.00 / 1,613; Pune INR 500,918.05 / INR 252.48 / 1,984.

Completed order amounts have mean INR 269.91, median INR 135.00 and skewness 4.61. Q1 is INR 64.80, Q3 INR 297.00, lower and upper 1.5×IQR fences are INR −283.50 and INR 645.30, with 0 observations below and 897 above. The negative lower fence is a statistical threshold, not a possible transaction value. This describes a synthetic six-city grocery and convenience order history whose spend is strongly right-skewed: a small number of high-ticket baskets pull the mean above the median, with 897 high outliers. Order amount has the largest absolute correlation with discount percentage (about −0.095), followed by quantity (about +0.087); this small correlation does not establish causation.

## Part B

### RFM and segments

The full-window RFM snapshot is 2025-10-01 00:00:00. It covers 1,175 customers: median recency 19 days (range 0–180), mean frequency 9.46 (maximum 33), mean monetary value INR 2,552.06 (maximum INR 14,766.55).

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.76 | 0.465 |
| 3 | 1,268.23 | 0.445 |
| 4 | 743.07 | 0.526 |
| 5 | 617.37 | 0.480 |
| 6 | 503.17 | 0.482 |
| 7 | 408.46 | 0.416 |
| 8 | 366.50 | 0.412 |

K=4 is selected: it has the strongest silhouette and a clear customer story with four actionable groups. Inertia always falls as K rises because more centroids cannot increase within-cluster squared distance; its decline alone does not show whether added segments are stable or useful.

| Segment ID | Name | Size | Mean recency (days) | Mean frequency | Mean spend (INR) | Recommended action |
|---:|---|---:|---:|---:|---:|---|
| 0 | Steady Essentials | 436 | 28.86 | 3.77 | 1,057.62 | Encourage routine replenishment with relevant bundles. |
| 1 | Premium Champions | 138 | 24.12 | 6.77 | 7,129.70 | Retain with priority support and high-value product offers. |
| 2 | Frequent Basket Builders | 419 | 8.77 | 19.40 | 3,410.02 | Reward frequent baskets and cross-sell adjacent essentials. |
| 3 | At Risk Regulars | 182 | 109.04 | 2.20 | 685.99 | Use a low-cost win-back message and monitor reactivation. |

Premium is segment 1, the segment with the highest mean monetary value. Cluster numbers are identifiers only and have no ordinal meaning. PCA explained variance ratios are 0.634 and 0.218 (total 0.852). `outputs/customer_segments.csv` has the required customer-level RFM and segment fields.

### Lapse prediction

The cohort contains 1,156 customers with a completed pre-cutoff order; 443 (38.32%) lapsed during September. The stratified split has 924 training and 232 test customers. The always-not-lapsed baseline has accuracy 61.6% and positive-class F1 0. Features are recency, frequency, monetary, mean discount, cancellation rate, mean completed delivery, favorite product category, city, age, gender and tenure. No order on or after 2025-09-01 enters model features, and segment labels are not features. Numeric columns are imputed/scaled and nominal columns are imputed/one-hot encoded with `drop='first'`, all in a Pipeline fitted only on training rows; an assertion confirms the feature matrix contains no missing values.

The fitted scaler's training mean for frequency is 8.115 orders, versus 8.029 across the full cohort; the training subset differs because the split samples a finite subset of customers.

| Model | Search grid | Best parameters | Best CV F1 | Test confusion matrix (true rows 0/1, predicted columns 0/1) | Precision | Recall | F1 |
|---|---|---|---:|---|---:|---:|---:|
| Logistic regression | C ∈ {0.01, 0.1, 1, 10, 100}; 5-fold | C=10 | 0.656 | [[104, 39], [32, 57]] | 0.594 | 0.640 | 0.616 |
| Random forest | depth ∈ {3, 5, 8}; leaf ∈ {5, 10, 20}; 5-fold / 45 fits | depth=8, leaf=10 | 0.683 | [[113, 30], [32, 57]] | 0.655 | 0.640 | 0.648 |

The random forest is selected: its test F1 is higher, with precision also higher at the same recall, so it misses no more September lapsers than the logistic model while making fewer false alarms. Missing a lapsing customer loses a chance to retain future spend; wrongly targeting a loyal customer wastes campaign budget and can erode trust. The forest's five largest importances are monetary (0.282), frequency (0.274), recency (0.118), average delivery (0.059) and cancellation rate (0.054). Logistic coefficients include frequency (negative, −1.589: more orders reduce predicted lapse), favorite Electronics (negative, −0.810 versus the dropped baseline category), and monetary (positive, +0.199 conditional on its other inputs). The positive monetary sign is a conditional model effect, not a claim that higher spending causes lapse.

Training-only oversampling duplicates positive training rows with replacement using seed 42 and leaves the test set unchanged. Logistic regression on balanced training data has precision 0.595, recall 0.876 and F1 0.709, versus 0.594 / 0.640 / 0.616 without oversampling. Recall moved most; this finds more potential lapsers at the cost of more false alarms. The serialized random forest reloads with exactly identical test predictions. The Premium test retention list contains 10 customers, 1 of whom actually lapsed.

## Part C

The triage pipeline first skips empty text and flags ticket text over 400 estimated tokens; it never sends those tickets to the model. It then redacts email, phone and order ID, in that order, and asserts the patterns find no personal data in the prompt. Across 164 tickets there are 55 email, 91 phone and 132 order ID substitutions (278 total); zero matches remain after redaction. Four empty tickets are skipped and four long tickets (734 estimated tokens each) are flagged. The prompt user content contains only the redacted ticket text.

The zero-shot and few-shot prompts are JSON-only `CLASSIFY` prompts; the few-shot prompt has fraud and billing examples. Every call retries `TransientLLMError` at most three times with exponential jitter (base 0.05 seconds), logging ticket, retry number and wait. The full runs recover all 19 transient errors; each retry in this fixed mock is retry 0 with a 0.045–0.0575 second wait. Invalid JSON is accepted only if regex word extraction finds exactly one allowed category; other malformed or disallowed categories are sent to human review.

| Prompt | Validated tickets | JSON | Keyword recovery | Invalid to human | Accuracy on validated | Total estimated tokens | Tokens / validated ticket |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero-shot | 147 | 135 | 12 | 9 | 85.7% (126/147) | 8,932 | 60.76 |
| Few-shot | 147 | 135 | 12 | 9 | 100% (147/147) | 14,392 | 97.90 |

Few-shot uses 5,460 additional tokens, about 37.14 more per classified ticket, and gains 14.3 percentage points of accuracy. That cost is justified for high-impact fraud and payment triage where a wrong queue can delay urgent action; a lower-risk high-volume queue could use zero-shot first and reserve human review for uncertainty. The zero-shot confusion table follows (true category rows, predicted category columns):

| True \ Predicted | billing | delivery | fraud | technical |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 11 | 10 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 23 | 0 | 0 | 0 |
| technical | 0 | 0 | 0 | 29 |

With the zero-shot budget demonstration, TOKEN_BUDGET=6,000, 51 eligible ticket calls were stopped as `budget_exhausted`; 6,023 tokens were recorded because the rule checks the running total before each next call. The final few-shot pipeline uses budget 100,000 and produces all 164 rows. Routing sends empty, long, invalid and budget-stopped tickets to NORMAL human review; validated fraud to HIGH security; other premium customers to HIGH premium desk; and other valid tickets to NORMAL support. Routes produced: 17 human review, 34 security team, 11 premium desk, 102 support queue. Outputs are rule-validated categories for eligible JSON/keyword responses; flagged and unprocessed tickets go to a human.

## Part D

### Policy retrieval and answers

The documents produce 15 chunks (refund 4, delivery 3, escalation 3, payment 3, privacy 2), using 40 words and a 10-word overlap. `TfidfVectorizer()` is fitted once on the chunk corpus; retrieval ranks cosine similarity. TF measures how often a term occurs in a chunk, while IDF gives more weight to terms that appear in fewer chunks. TF-IDF can miss semantic matches when a question and policy use different words for the same idea.

| Evaluation | Answerable expected source in top 3 | Expected fact substring | Unanswerable fallback |
|---|---:|---:|---:|
| 10-word overlap | 12/12 | 12/12 | 2/2 |
| Zero overlap | 12/12 | 10/12 | 2/2 |

Without overlap, “What delivery time makes a completed order eligible for a full refund?” and “How soon is money returned when a payment fails but is deducted?” lose the expected fact from the retrieved chunk context because each answer crosses the boundary between adjacent 40-word chunks. Answer generation uses exactly a context and question user message, returns the answer plus retrieved source names, and falls back with “I could not find” when no context supports an answer.

### Refund agent

Tools query the cleaned SQLite order table, retrieve policy text, calculate refunds and append issuance/escalation actions to `outputs/refund_ledger.jsonl`. The tool rules refund only completed orders delivered in more than 45 minutes; delivery in 45 minutes or less returns INR 0; refunds above INR 1,000 go to a human. Ledger timestamps use a fixed synthetic timestamp to keep generated artifacts reproducible. Every scenario request is treated as being inside the policy's 7-day window.

| Scenario | Status | Refund (INR) | Model calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 892 |
| S2 | REFUND | 135.00 | 5 | 892 |
| S3 | ESCALATE | 1,999.00 | 5 | 879 |
| S4 | DENIED | 0.00 | 4 | 601 |
| S5 | DENIED | 0.00 | 3 | 334 |
| S6 | NOT_FOUND | 0.00 | 2 | 70 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 2,994 |

The S1 run with token_limit=200 returns `FALLBACK_TOKEN_LIMIT` (3 calls); with time_limit_seconds=0 it returns `FALLBACK_TIMEOUT` before the first call (0 calls). S7 reaches all six calls with the default 6,000-token limit and ends at the iteration limit. In the simulated S1 crash, the first two tool observations are checkpointed before the crash. Resuming executes this complete sequence exactly once: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`; final result REFUND INR 128.25. Checkpoints are JSON files under `state/` with the messages, tokens, step count and tools already run.

ReAct calls the model at each decision point (S1 five calls, S5 three, S6 two), so it spends more round trips and typically more prompt tokens than asking once for a complete plan. In return it sees actual tool observations before choosing the next action: a NOT_FOUND response stops cleanly, and a cancelled order is denied before calculation. A plan-and-execute design can reduce calls for the ordinary refund case but must validate its plan and replan when an observation differs. For refunds I would keep ReAct with strict iteration, time and token caps because the cost of issuing an unsupported refund outweighs the extra local mock calls.

## Limitations and Next Steps

This is synthetic data and a deterministic offline mock; its accuracy and policies do not represent real customers or live operating rules. TF-IDF and keyword classification are lexical methods and can miss meaning expressed with different vocabulary. Before operational use, the organization would need approved current policies, access controls, audit review, and evaluation on representative, privacy-reviewed data. The retention list is illustrative and not a leakage-free evaluation because its Premium flag uses whole-window RFM including the September label period; the lapse model features themselves use only pre-cutoff data. A future evaluation should define Premium using information available at the cutoff, use time-based validation, assess calibration and fairness, and monitor outcomes after deployment.
