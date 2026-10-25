from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from joblib import dump, load

R=Path(__file__).resolve().parent; OUT=R/'outputs'; OUT.mkdir(exist_ok=True)
def run():
    cust=pd.read_csv(R/'data/clean/customers.csv',parse_dates=['signup_date'])
    prod=pd.read_csv(R/'data/clean/products.csv')
    orders=pd.read_csv(R/'data/clean/orders.csv',parse_dates=['order_ts'])
    comp=orders[orders.status.eq('completed')].copy()
    snap=pd.Timestamp('2025-10-01')
    rfm=comp.groupby('customer_id').agg(last_order_ts=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index()
    rfm['recency']=(snap-rfm.last_order_ts).dt.days
    cols=['recency','frequency','monetary']; scaler=StandardScaler(); Z=scaler.fit_transform(rfm[cols])
    km={}; models={}
    for k in range(2,9):
        m=KMeans(n_clusters=k,n_init=10,random_state=42).fit(Z); models[k]=m
        km[str(k)]={'inertia':m.inertia_,'silhouette':silhouette_score(Z,m.labels_)}
    chosen=models[4]; rfm['segment_id']=chosen.labels_
    prof=rfm.groupby('segment_id').agg(size=('customer_id','size'),mean_recency=('recency','mean'),mean_frequency=('frequency','mean'),mean_monetary=('monetary','mean')).round(2)
    premium_id=int(prof.mean_monetary.idxmax())
    names={i:('Premium Champions' if i==premium_id else n) for i,n in enumerate(['Steady Essentials','High Value Occasional','Frequent Basket Builders','At Risk Regulars'])}
    rfm['segment_name']=rfm.segment_id.map(names); rfm['is_premium']=rfm.segment_id.eq(premium_id)
    seg=rfm[['customer_id','recency','frequency','monetary','segment_id','segment_name','is_premium']]
    seg.to_csv(OUT/'customer_segments.csv',index=False)
    pca=PCA(n_components=2).fit(Z)

    cutoff=pd.Timestamp('2025-09-01'); sept_end=pd.Timestamp('2025-10-01')
    pre=orders[orders.order_ts<cutoff].copy(); pre_comp=pre[pre.status.eq('completed')]; post=comp[(comp.order_ts>=cutoff)&(comp.order_ts<sept_end)]
    available=pre_comp.customer_id.unique()
    f=pre_comp.groupby('customer_id').agg(last=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum'),avg_discount=('discount_pct','mean'),avg_delivery=('delivery_minutes','mean'))
    f['recency']=(cutoff-f['last']).dt.days
    all_final=pre[pre.status.isin(['completed','cancelled'])]
    cancel=(all_final.status.eq('cancelled')).groupby(all_final.customer_id).agg(['sum','count'])
    f['cancellation_rate']=cancel['sum']/cancel['count']
    fav=pre_comp.merge(prod[['product_id','category']],on='product_id').groupby('customer_id').category.agg(lambda x:x.value_counts().index[0]).rename('favourite_category')
    features=f.drop(columns='last').join(fav).reset_index().merge(cust,on='customer_id',how='left')
    features['tenure_days']=(cutoff-features.signup_date).dt.days
    features['lapsed']=~features.customer_id.isin(post.customer_id)
    features['lapsed']=features.lapsed.astype(int)
    numeric=['recency','frequency','monetary','avg_discount','cancellation_rate','avg_delivery','age','tenure_days']
    categorical=['favourite_category','city','gender']
    X=features[numeric+categorical]; y=features.lapsed
    assert not X.isna().any().any(), f'feature NA: {X.isna().sum().to_dict()}'
    Xtr,Xte,ytr,yte,idxtr,idxte=train_test_split(X,y,features.index,test_size=.2,stratify=y,random_state=42)
    preprocessor=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),
        ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(drop='first',handle_unknown='ignore'))]),categorical)])
    lr=Pipeline([('prep',preprocessor),('model',LogisticRegression(max_iter=1000))])
    rf=Pipeline([('prep',preprocessor),('model',RandomForestClassifier(n_estimators=100,random_state=42))])
    gl=GridSearchCV(lr,{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1',n_jobs=-1).fit(Xtr,ytr)
    gr=GridSearchCV(rf,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1',n_jobs=-1).fit(Xtr,ytr)
    def report(model):
        pred=model.predict(Xte)
        return {'confusion_matrix':confusion_matrix(yte,pred,labels=[0,1]).tolist(),'precision_lapsed':precision_score(yte,pred,zero_division=0),'recall_lapsed':recall_score(yte,pred,zero_division=0),'f1_lapsed':f1_score(yte,pred,zero_division=0),'accuracy':accuracy_score(yte,pred)}
    lrrep,rfrep=report(gl.best_estimator_),report(gr.best_estimator_)
    # Keep every training row and duplicate positive rows to balance the training split only.
    positive=np.flatnonzero(np.asarray(ytr)==1); n0=int((ytr==0).sum()); add=np.random.RandomState(42).choice(positive,size=n0-len(positive),replace=True)
    Xbal=pd.concat([Xtr.reset_index(drop=True),Xtr.iloc[add].reset_index(drop=True)],ignore_index=True)
    ybal=pd.concat([ytr.reset_index(drop=True),ytr.iloc[add].reset_index(drop=True)],ignore_index=True)
    balanced=Pipeline([('prep',preprocessor),('model',LogisticRegression(max_iter=1000,C=gl.best_params_['model__C']))]).fit(Xbal,ybal)
    balrep=report(balanced)
    selected=gr.best_estimator_; dump(selected,OUT/'lapse_model.joblib'); reloaded=load(OUT/'lapse_model.joblib')
    assert np.array_equal(selected.predict(Xte),reloaded.predict(Xte))
    test_features=features.loc[idxte].copy(); premium_ids=set(seg.loc[seg.is_premium,'customer_id'])
    pp=reloaded.predict_proba(Xte)[:,list(reloaded.named_steps['model'].classes_).index(1)]
    retain=pd.DataFrame({'customer_id':test_features.customer_id.values,'lapse_probability':pp,'actually_lapsed':yte.values})
    retain=retain[retain.customer_id.isin(premium_ids)].sort_values('lapse_probability',ascending=False).head(10)
    retain.to_csv(OUT/'retention_list.csv',index=False)
    namesfeat=reloaded.named_steps['prep'].get_feature_names_out()
    imp=dict(zip(namesfeat,reloaded.named_steps['model'].feature_importances_))
    lrmodel=gl.best_estimator_.named_steps['model']; lrnames=gl.best_estimator_.named_steps['prep'].get_feature_names_out()
    coef=dict(zip(lrnames,lrmodel.coef_[0]))
    fullfreq=float(features.frequency.mean()); trainfreq=float(gl.best_estimator_.named_steps['prep'].named_transformers_['num'].named_steps['scale'].mean_[numeric.index('frequency')])
    metrics={'rfm':{'customers':len(rfm),'recency_median':rfm.recency.median(),'recency_min':rfm.recency.min(),'recency_max':rfm.recency.max(),'frequency_mean':rfm.frequency.mean(),'frequency_max':rfm.frequency.max(),'monetary_mean':rfm.monetary.mean(),'monetary_max':rfm.monetary.max()},
      'kmeans':km,'chosen_k':4,'segments':prof.to_dict(orient='index'),'segment_names':names,'premium_segment_id':premium_id,'pca_variance_ratio':pca.explained_variance_ratio_.tolist(),
      'cohort':{'n':len(features),'lapsed':int(y.sum()),'lapsed_rate_pct':float(y.mean()*100),'train':len(Xtr),'test':len(Xte),'baseline_accuracy':accuracy_score(yte,np.zeros(len(yte),dtype=int)),'baseline_f1':0},
      'frequency_scaler_training_mean':trainfreq,'full_cohort_frequency_mean':fullfreq,
      'logistic':{'best_params':gl.best_params_,'best_cv_f1':gl.best_score_,'test':lrrep},
      'random_forest':{'best_params':gr.best_params_,'best_cv_f1':gr.best_score_,'test':rfrep,'top_features':sorted(imp.items(),key=lambda x:x[1],reverse=True)[:5]},
      'balanced_logistic':balrep,'logistic_selected_coefficients':sorted(coef.items(),key=lambda x:abs(x[1]),reverse=True)[:10],
      'reload_predictions_identical':True,'retention_rows':len(retain),'retention_true_lapsed':int(retain.actually_lapsed.sum()),'retention_list':retain.to_dict(orient='records')}
    (OUT/'part_b_report.json').write_text(json.dumps(metrics,indent=2,default=float),encoding='utf-8')
    print('Part B complete')
if __name__=='__main__': run()
