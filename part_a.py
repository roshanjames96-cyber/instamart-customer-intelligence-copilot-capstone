from pathlib import Path
import json, sqlite3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA, RAW, CLEAN, OUT = ROOT / "data", ROOT / "data/raw", ROOT / "data/clean", ROOT / "outputs"
OUT.mkdir(exist_ok=True); CLEAN.mkdir(parents=True, exist_ok=True)

def audit_frame(df):
    return {"shape": list(df.shape), "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "missing": {c: {"count": int(df[c].isna().sum()), "pct": round(float(df[c].isna().mean()*100), 3)} for c in df}}

def run():
    customers = pd.read_csv(RAW / "customers.csv", dtype={"age":"string"})
    products = pd.read_csv(RAW / "products.csv")
    orders_raw = pd.read_csv(RAW / "orders.csv", keep_default_na=False)
    tickets = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)
    numeric_amount = pd.to_numeric(orders_raw.order_amount, errors="coerce")
    numeric_delivery = pd.to_numeric(orders_raw.delivery_minutes.replace("", np.nan), errors="coerce")
    audit = {"customers": audit_frame(customers), "products": audit_frame(products),
             "orders": audit_frame(orders_raw), "tickets": audit_frame(tickets),
             "figures": {"orders_rows":len(orders_raw), "exact_duplicate_rows":int(orders_raw.duplicated().sum()),
             "pending_rows":int((orders_raw.status=="pending").sum()), "distinct_order_id":int(orders_raw.order_id.nunique()),
             "non_numeric_amounts":int(numeric_amount.isna().sum()), "negative_delivery_minutes":int((numeric_delivery<0).sum()),
             "blank_delivery_minutes":int(orders_raw.delivery_minutes.astype(str).str.strip().eq("").sum()),
             "blank_age":int(customers.age.fillna("").astype(str).str.strip().eq("").sum()),
             "age_outside_10_100":int((pd.to_numeric(customers.age,errors="coerce").notna() & ~pd.to_numeric(customers.age,errors="coerce").between(10,100)).sum()),
             "raw_city_spellings":int(customers.city.nunique()), "raw_gender_spellings_nonblank":int(customers.gender.fillna("").astype(str).str.strip().replace("",np.nan).nunique()),
             "blank_gender":int(customers.gender.fillna("").astype(str).str.strip().eq("").sum())}}
    (OUT/"audit_report.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")

    customers["city"] = customers.city.astype(str).str.strip().str.title()
    g = customers.gender.fillna("").astype(str).str.strip().str.lower()
    customers["gender"] = g.map({"male":"male","m":"male","female":"female","f":"female"}).fillna("unknown")
    customers["age"] = pd.to_numeric(customers.age.replace("",np.nan), errors="coerce").astype(float)
    customers.loc[~customers.age.between(10,100), "age"] = np.nan
    customers["age"] = customers.age.fillna(customers.groupby("city").age.transform("median")).round().astype(int)
    customers["signup_date"] = pd.to_datetime(customers.signup_date)

    products["price_inr"] = pd.to_numeric(products.price_inr)
    orders = orders_raw.copy()
    orders["order_ts"] = pd.to_datetime(orders.order_ts)
    orders = orders.drop_duplicates()
    orders = orders.sort_values(["order_ts"]).drop_duplicates("order_id", keep="last").copy()
    for col in ["quantity","discount_pct","order_amount","delivery_minutes"]:
        orders[col] = pd.to_numeric(orders[col].replace("",np.nan),errors="coerce")
    fallback_amount = pd.Series(orders.merge(products[["product_id","price_inr"]],on="product_id",how="left").eval("price_inr * quantity * (1 - discount_pct / 100)").round(2).values,index=orders.index)
    orders["order_amount"] = orders.order_amount.fillna(fallback_amount)
    orders.loc[orders.delivery_minutes < 0,"delivery_minutes"] = np.nan
    category = pd.Series(orders.merge(products[["product_id","category"]],on="product_id",how="left")["category"].values,index=orders.index)
    comp = orders.status.eq("completed")
    med_by_category = orders.loc[comp].assign(category=category[comp].values).groupby("category").delivery_minutes.median()
    missing_completed = comp & orders.delivery_minutes.isna()
    orders.loc[missing_completed,"delivery_minutes"] = category[missing_completed].map(med_by_category)
    orders["order_ts"] = orders.order_ts.dt.strftime("%Y-%m-%d %H:%M:%S")

    customers.to_csv(CLEAN/"customers.csv",index=False,date_format="%Y-%m-%d")
    products.to_csv(CLEAN/"products.csv",index=False)
    orders.to_csv(CLEAN/"orders.csv",index=False)
    db=sqlite3.connect(DATA/"instamart.db")
    customers.to_sql("customers",db,if_exists="replace",index=False)
    products.to_sql("products",db,if_exists="replace",index=False)
    orders.to_sql("orders",db,if_exists="replace",index=False)
    db.close()
    metrics = {"clean_customers_city_counts":customers.city.value_counts().to_dict(), "clean_gender_counts":customers.gender.value_counts().to_dict(),
       "clean_orders":len(orders),"status_counts":orders.status.value_counts().to_dict(),
       "completed_revenue":round(float(orders.loc[comp,"order_amount"].sum()),2),
       "completed_delivery_missing":int(orders.loc[comp,"delivery_minutes"].isna().sum()),
       "cancelled_delivery_missing":int(orders.loc[~comp,"delivery_minutes"].isna().sum()),
       "queries":{},"pandas":{}}
    db=sqlite3.connect(DATA/"instamart.db")
    for i,sql in enumerate((ROOT/"sql/queries.sql").read_text().split(";"),1):
        if sql.strip():
            frame=pd.read_sql_query(sql,db); metrics["queries"][f"Q{i}"]={"rows":len(frame),"head":frame.head(10).to_dict(orient="records")}
    db.close()
    pivot=pd.pivot_table(orders[orders.status.eq("completed")].merge(customers[["customer_id","city"]],on="customer_id").merge(products[["product_id","category"]],on="product_id"),index="city",columns="category",values="order_amount",aggfunc="sum",margins=True,fill_value=0)
    cityagg=orders[orders.status.eq("completed")].merge(customers[["customer_id","city"]],on="customer_id").groupby("city").order_amount.agg(["sum","mean","count"])
    amt=orders.loc[comp,"order_amount"]; q1,q3=amt.quantile([.25,.75]); iqr=q3-q1; corr=orders.loc[comp,["order_amount","discount_pct","quantity","delivery_minutes"]].corr()
    metrics["pandas"]={"pivot_grand_total":float(pivot.loc["All","All"]),"city_agg":cityagg.round(2).to_dict(orient="index"),
      "amount_stats":{"mean":amt.mean(),"median":amt.median(),"skew":amt.skew(),"q1":q1,"q3":q3,"lower_fence":q1-1.5*iqr,"upper_fence":q3+1.5*iqr,
        "outliers_below":int((amt<q1-1.5*iqr).sum()),"outliers_above":int((amt>q3+1.5*iqr).sum())},"correlation":corr.round(4).to_dict()}
    (OUT/"part_a_report.json").write_text(json.dumps(metrics,indent=2,default=str),encoding="utf-8")
    print("Part A complete")

if __name__=="__main__": run()
