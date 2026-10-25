from pathlib import Path
import sqlite3
import pandas as pd
R=Path(__file__).resolve().parent
o=pd.read_csv(R/'data/clean/orders.csv')
c=pd.read_csv(R/'data/clean/customers.csv')
assert len(o)==12309
assert o.order_id.nunique()==12309
assert o.status.value_counts().to_dict()=={'completed':11110,'cancelled':1199}
assert o.order_amount.notna().all()
assert o.loc[o.status.eq('completed'),'delivery_minutes'].notna().all()
assert o.loc[o.status.eq('cancelled'),'delivery_minutes'].isna().all()
assert round(o.loc[o.status.eq('completed'),'order_amount'].sum(),2)==2998668.80
assert c.city.nunique()==6 and c.gender.value_counts().to_dict()=={'female':610,'male':560,'unknown':30}
assert c.age.between(18,60).all()
db=sqlite3.connect(R/'data/instamart.db')
assert {r[0] for r in db.execute("select name from sqlite_master where type='table'")}=={'customers','products','orders'}
db.close()
print('ALL CHECKPOINTS PASSED')
