from pathlib import Path
import json, re, random, time
import pandas as pd
import mock_llm

R=Path(__file__).resolve().parent; OUT=R/'outputs'; OUT.mkdir(exist_ok=True)
ALLOWED={'billing','delivery','technical','fraud'}; BASE_DELAY=.05; MAX_INPUT_TOKENS=400
ZERO="""You are a CLASSIFY support-ticket classifier. Allowed categories: billing, delivery, technical, fraud. Return JSON only in the form {\"category\": \"...\", \"reason\": \"...\"}."""
FEW="""You are a CLASSIFY support-ticket classifier. Allowed categories: billing, delivery, technical, fraud. Return JSON only in the form {\"category\": \"...\", \"reason\": \"...\"}.
EXAMPLES:
Ticket: I was charged twice and need a refund. -> {\"category\":\"billing\",\"reason\":\"duplicate charge\"}
Ticket: There is an unauthorized transaction; I did not place it. -> {\"category\":\"fraud\",\"reason\":\"unauthorized transaction\"}"""
PATTERNS=[('email',re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')),
 ('phone',re.compile(r'\b[6-9]\d{9}\b')),('order_id',re.compile(r'ORD-\d{6}'))]
def redact(text):
    counts={}; result=text
    for name,pat in PATTERNS:
        result,n=pat.subn({'email':'[EMAIL]','phone':'[PHONE]','order_id':'[ORDER_ID]'}[name],result); counts[name]=n
    return result,counts
def validate(content):
    try:
        obj=json.loads(content)
        if isinstance(obj,dict) and obj.get('category') in ALLOWED: return obj['category'],'classified_json'
        return '', 'flagged_invalid_output'
    except (json.JSONDecodeError,TypeError):
        words=re.findall(r'[a-z]+',(content or '').lower()); found=[c for c in ALLOWED if c in words]
        return (found[0],'classified_keyword') if len(found)==1 else ('','flagged_invalid_output')
def run_triage(prompt_kind,budget=None):
    tickets=pd.read_csv(R/'data/raw/tickets.csv',keep_default_na=False)
    mock_llm.reset(); rng=random.Random(42); running=0; rows=[]; retry_lines=[]; cats=FEW if prompt_kind=='few' else ZERO
    redaction_counts={n:0 for n,_ in PATTERNS}; leftover=0
    for _,t in tickets.iterrows():
        raw=str(t.text); clean,counts=redact(raw)
        for n,v in counts.items(): redaction_counts[n]+=v
        for _,p in PATTERNS: leftover+=len(p.findall(clean))
        row={'ticket_id':t.ticket_id,'customer_id':t.customer_id,'status':'','category':'','tokens_used':0}
        if not raw.strip(): row['status']='skipped_empty'
        elif mock_llm.est_tokens(raw)>MAX_INPUT_TOKENS: row['status']='flagged_long'
        else:
            if budget is not None and running>=budget: row['status']='budget_exhausted'
            else:
                assert not any(p.search(clean) for _,p in PATTERNS), 'PII reached prompt'
                tries=0
                while True:
                    try:
                        resp=mock_llm.chat([{'role':'system','content':cats},{'role':'user','content':clean}]); break
                    except mock_llm.TransientLLMError:
                        wait=BASE_DELAY*(2**tries)*rng.uniform(.9,1.15)
                        retry_lines.append(f"{t.ticket_id} retry={tries} wait={wait:.5f}s")
                        time.sleep(wait); tries+=1
                        if tries>3: raise
                used=resp.usage.total_tokens; running+=used; row['tokens_used']=used
                category,status=validate(resp.choices[0].message.content)
                row.update(status=status,category=category)
        rows.append(row)
    d=pd.DataFrame(rows)
    return d, {'prompt':prompt_kind,'tickets':len(d),'classified':int(d.status.isin(['classified_json','classified_keyword']).sum()),
       'status_counts':d.status.value_counts().to_dict(),'tokens':running,'tokens_per_classified':running/max(1,int(d.status.isin(['classified_json','classified_keyword']).sum())),
       'accuracy_classified':float((d[d.status.isin(['classified_json','classified_keyword'])].merge(tickets[['ticket_id','true_category']],on='ticket_id').category==d[d.status.isin(['classified_json','classified_keyword'])].merge(tickets[['ticket_id','true_category']],on='ticket_id').true_category).mean()),
       'redactions':redaction_counts,'redactions_total':sum(redaction_counts.values()),'leftover_pii':leftover,'retry_log':retry_lines}
def run():
    zero,zr=run_triage('zero')
    budget,bm=run_triage('zero',6000)
    budget.to_csv(OUT/'triage_budget_demo.csv',index=False)
    few,fr=run_triage('few',100000)
    seg=pd.read_csv(OUT/'customer_segments.csv').set_index('customer_id')
    cats=pd.read_csv(R/'data/raw/tickets.csv',keep_default_na=False)[['ticket_id','customer_id','true_category']]
    joined=few.merge(cats,on=['ticket_id','customer_id']).drop(columns='true_category')
    def route(row):
        premium=False; segment=''
        if row.customer_id in seg.index: premium=bool(seg.loc[row.customer_id,'is_premium']); segment=seg.loc[row.customer_id,'segment_name']
        if row.status in {'skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted'}: return pd.Series([segment,'NORMAL','human_review'])
        if row.category=='fraud': return pd.Series([segment,'HIGH','security_team'])
        if premium: return pd.Series([segment,'HIGH','premium_desk'])
        return pd.Series([segment,'NORMAL','support_queue'])
    joined[['segment_name','priority','route']]=joined.apply(route,axis=1)
    joined[['ticket_id','customer_id','status','category','segment_name','priority','route','tokens_used']].to_csv(OUT/'triage_results.csv',index=False)
    # Confusion matrix includes only validated classifications.
    zerojoin=zero.merge(cats,on=['ticket_id','customer_id']); classified=zerojoin[zerojoin.status.isin(['classified_json','classified_keyword'])]
    confusion=pd.crosstab(classified.true_category,classified.category).reindex(index=sorted(ALLOWED),columns=sorted(ALLOWED),fill_value=0)
    retry_lines=fr['retry_log']
    (OUT/'triage_retry.log').write_text('\n'.join(retry_lines)+'\n',encoding='utf-8')
    report={'zero_shot':zr,'few_shot':fr,'budget_demo':bm,'zero_shot_confusion':confusion.to_dict(),
      'routing_counts':joined.route.value_counts().to_dict(),'final_rows':len(joined)}
    (OUT/'part_c_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Part C complete')
if __name__=='__main__': run()
