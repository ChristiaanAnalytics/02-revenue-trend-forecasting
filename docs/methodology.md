# Methodology

## 1. Data Foundation
- **Source tables:** `customers.csv`, `orders.csv`, `order_items.csv`, `payments.csv`, `returns.csv`, `refunds.csv`, `products.csv`.
- **Revenue definition:** Prefer **payments** table for a cash-received view; fall back to `order_items` totals if payments missing.
- **Snapshot date:** Latest order date in the dataset (Sept 2025).

## 2. RFM Segmentation
1. **Recency (R):** Days since each customer’s most recent order.
2. **Frequency (F):** Count of distinct orders.
3. **Monetary (M):** Total revenue generated.

### Scoring
- Quantile-based 1–5 scores per dimension (5 = best).
- Combined as `RFM_Score` string and mapped to business segments:
  - Champions
  - Loyal
  - New Customers
  - At-Risk Loyal
  - Hibernating
  - Others

## 3. Churn Labelling & Features
- **Churn definition:** No purchase in the last **90 days** as of snapshot date.
- **Features:** Orders count, total revenue, average order value, recency days, customer tenure.

## 4. Modelling
- **Baseline model:** Logistic Regression with balanced class weights.
- **Evaluation:** ROC-AUC, Average Precision, decile lift analysis.

## 5. Customer Lifetime Value (6-month horizon)
- Horizon CLV = recent average monetary value × estimated retention probability.
- Optional BG/NBD + Gamma-Gamma model (`lifetimes`) for more rigorous estimation.

## 6. Validation & Outputs
- 75/25 train-test split, stratified by churn label.
- All metrics, figures and final scoring exported to `outputs/` for reproducibility.