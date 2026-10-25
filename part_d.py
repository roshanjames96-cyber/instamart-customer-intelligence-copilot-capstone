from pathlib import Path
import json, re, time, sqlite3
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

R=Path(__file__).resolve().parent; OUT=R/'outputs'; STATE=R/'state'; POL=R/'data/policies'
OUT.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
def chunk_documents(overlap=10):
    chunks=[]; step=40-overlap
    for path in sorted(POL.glob('*.txt')):
        words=path.read_text(encoding='utf-8').split()
        for start in range(0,len(words),step):
            part=words[start:start+40]
            chunks.append({'source':path.name,'text':' '.join(part)})
            if start+40>=len(words): break
    return chunks
CHUNKS=chunk_documents(10)
VECTORIZER=TfidfVectorizer().fit([c['text'] for c in CHUNKS])
MATRIX=VECTORIZER.transform([c['text'] for c in CHUNKS])
def retrieve(query,k=3,chunks=None,vectorizer=None,matrix=None):
    cs=chunks or CHUNKS; v=vectorizer or VECTORIZER; m=matrix if matrix is not None else MATRIX
    q=v.transform([query]); scores=cosine_similarity(q,m).ravel(); order=np.argsort(-scores,kind='stable')[:k]
    return [{'source':cs[i]['source'],'score':float(scores[i]),'text':cs[i]['text']} for i in order]
def answer(question):
    found=retrieve(question,3)
    context='\n'.join(f"[{c['source']}] {c['text']}" for c in found)
    messages=[{'role':'system','content':'POLICY ASSISTANT: Answer only from the supplied context. If the context does not contain the answer, say you could not find the information.'},
       {'role':'user','content':f'CONTEXT:\n{context}\nQUESTION:\n{question}'}]
    resp=mock_llm.chat(messages)
    return {'answer':resp.choices[0].message.content,'sources':[c['source'] for c in found]}
def evaluate_policy():
    qa=pd.read_csv(R/'data/raw/policy_eval.csv',keep_default_na=False)
    rows=[]
    for _,q in qa.iterrows():
        res=answer(q.question); docs=[x['source'] for x in retrieve(q.question,3)]
        rows.append({'question':q.question,'expected_doc':q.expected_doc,'expected_fact':q.expected_fact,'sources':docs,'answer':res['answer'],
          'doc_hit':q.expected_doc in docs if q.expected_doc!='NONE' else None,
          'fact_hit':q.expected_fact in res['answer'] if q.expected_doc!='NONE' else None,
          'fallback':res['answer'].startswith('I could not find') if q.expected_doc=='NONE' else None})
    no=chunk_documents(0); nv=TfidfVectorizer().fit([c['text'] for c in no]); nm=nv.transform([c['text'] for c in no])
    zero=[]
    for _,q in qa.iterrows():
        docs=retrieve(q.question,3,no,nv,nm); context='\n'.join(f"[{c['source']}] {c['text']}" for c in docs)
        resp=mock_llm.chat([{'role':'system','content':'POLICY ASSISTANT: Answer only from the supplied context. Say you could not find it otherwise.'},{'role':'user','content':f'CONTEXT:\n{context}\nQUESTION:\n{q.question}'}]).choices[0].message.content
        zero.append({'question':q.question,'expected_doc':q.expected_doc,'expected_fact':q.expected_fact,'sources':[x['source'] for x in docs],'answer':resp,
          'doc_hit':q.expected_doc in [x['source'] for x in docs] if q.expected_doc!='NONE' else None,
          'fact_hit':q.expected_fact in resp if q.expected_doc!='NONE' else None,'fallback':resp.startswith('I could not find') if q.expected_doc=='NONE' else None})
    def summ(rows):
        ans=[x for x in rows if x['expected_doc']!='NONE']; un=[x for x in rows if x['expected_doc']=='NONE']
        return {'answerable':len(ans),'doc_hits':sum(x['doc_hit'] for x in ans),'fact_hits':sum(x['fact_hit'] for x in ans),'unanswerable':len(un),'fallbacks':sum(x['fallback'] for x in un)}
    return {'overlap10':summ(rows),'overlap0':summ(zero),'overlap10_rows':rows,'overlap0_rows':zero,'chunks':len(CHUNKS),'chunk_counts':pd.Series([c['source'] for c in CHUNKS]).value_counts().to_dict()}

TOOLS_SPEC=[
 {'name':'get_order','description':'Look up a cleaned order in the analytical database.','parameters':{'type':'object','properties':{'order_id':{'type':'string'}},'required':['order_id']}},
 {'name':'search_policy','description':'Retrieve the most relevant refund policy passages.','parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query']}},
 {'name':'calculate_refund','description':'Calculate the late-delivery refund for a completed order.','parameters':{'type':'object','properties':{'order_amount':{'type':'number'},'delivery_minutes':{'type':'number'}},'required':['order_amount','delivery_minutes']}},
 {'name':'issue_refund','description':'Record an automatically approved refund in the ledger.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}},
 {'name':'escalate_to_human','description':'Record a refund that requires human approval.','parameters':{'type':'object','properties':{'order_id':{'type':'string'},'refund_inr':{'type':'number'}},'required':['order_id','refund_inr']}}]
def get_order(order_id):
    db=sqlite3.connect(R/'data/instamart.db'); row=db.execute('SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?',(order_id,)).fetchone(); db.close()
    return {'error':'order not found'} if row is None else {'order_id':row[0],'status':row[1],'order_amount':float(row[2]),'delivery_minutes':float(row[3]) if row[3] is not None else None}
def search_policy(query): return {'results':retrieve(query,3)}
def calculate_refund(order_amount,delivery_minutes): return {'refund_inr':round(float(order_amount),2) if float(delivery_minutes)>45 else 0.0}
def ledger(order_id,refund_inr,name):
    row={'timestamp':'2025-10-01T00:00:00Z','tool_name':name,'order_id':order_id,'refund_inr':round(float(refund_inr),2)}
    with (OUT/'refund_ledger.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
    return {'refund_inr':row['refund_inr']}
def issue_refund(order_id,refund_inr):return ledger(order_id,refund_inr,'issue_refund')
def escalate_to_human(order_id,refund_inr):return ledger(order_id,refund_inr,'escalate_to_human')
TOOL_MAP={'get_order':get_order,'search_policy':search_policy,'calculate_refund':calculate_refund,'issue_refund':issue_refund,'escalate_to_human':escalate_to_human}
def _status(final): return (final.split(':',1)[0].strip() if ':' in final else 'FALLBACK')
def _fallback(status,detail): return f"{status}: {detail} Please hand this case to a human support agent."
def _save(run_id,scenario_id,messages,tokens,steps,executed,llm_calls):
    payload={'run_id':run_id,'scenario_id':scenario_id,'messages':messages,'tokens_used':tokens,'step_number':len(steps),'tools_executed':executed,'steps':steps,'llm_calls':llm_calls}
    (STATE/f'{run_id}.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
def _continue(state,max_iterations=6,token_limit=6000,time_limit_seconds=10,crash_after_tools=None):
    messages=state['messages']; tokens=state['tokens_used']; steps=state['steps']; executed=state['tools_executed']; calls=state['llm_calls']; start=time.monotonic()
    scenario_id=state['scenario_id']; run_id=state['run_id']
    while calls<max_iterations:
        if time.monotonic()-start>=time_limit_seconds:
            final=_fallback('FALLBACK_TIMEOUT','Time limit reached.'); return {'status':'FALLBACK_TIMEOUT','final_answer':final,'steps':steps,'llm_calls':calls,'tokens_used':tokens}
        resp=mock_llm.chat(messages,tools=TOOLS_SPEC); calls+=1; tokens+=resp.usage.total_tokens
        if tokens>token_limit:
            final=_fallback('FALLBACK_TOKEN_LIMIT','Token limit reached before the requested action.'); return {'status':'FALLBACK_TOKEN_LIMIT','final_answer':final,'steps':steps,'llm_calls':calls,'tokens_used':tokens}
        msg=resp.choices[0].message
        if not msg.tool_calls:
            final=msg.content or ''; return {'status':_status(final),'final_answer':final,'steps':steps,'llm_calls':calls,'tokens_used':tokens}
        calls_out=[]
        for tc in msg.tool_calls:
            args=tc.arguments if isinstance(tc.arguments,dict) else json.loads(tc.arguments)
            calls_out.append({'id':tc.id,'name':tc.name,'arguments':args})
        messages.append({'role':'assistant','content':None,'tool_calls':calls_out})
        for tc in calls_out:
            observation=TOOL_MAP[tc['name']](**tc['arguments']); executed.append(tc['name'])
            step={'tool_name':tc['name'],'arguments':tc['arguments'],'observation':observation}; steps.append(step)
            messages.append({'role':'tool','tool_call_id':tc['id'],'name':tc['name'],'content':json.dumps(observation,sort_keys=True)})
            _save(run_id,scenario_id,messages,tokens,steps,executed,calls)
            if crash_after_tools is not None and len(executed)>=crash_after_tools: raise RuntimeError('SIMULATED_CRASH_AFTER_CHECKPOINT')
    final=_fallback('FALLBACK_MAX_ITERATIONS','Maximum model-call iterations reached.')
    return {'status':'FALLBACK_MAX_ITERATIONS','final_answer':final,'steps':steps,'llm_calls':calls,'tokens_used':tokens}
def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after_tools=None):
    run_id=run_id or f'{scenario_id.lower()}_{int(time.time_ns())}'
    state={'run_id':run_id,'scenario_id':scenario_id,'messages':[{'role':'system','content':'AGENT: Decide requests by checking order and policy tools.'},{'role':'user','content':user_message}],
        'tokens_used':0,'steps':[],'tools_executed':[],'llm_calls':0}
    _save(run_id,scenario_id,state['messages'],0,[],[],0)
    result=_continue(state,max_iterations,token_limit,time_limit_seconds,crash_after_tools)
    result['tools_executed']=state['tools_executed']
    result['run_id']=run_id
    return result
def resume_agent(run_id,max_iterations=6,token_limit=6000,time_limit_seconds=10):
    state=json.loads((STATE/f'{run_id}.json').read_text(encoding='utf-8'))
    result=_continue(state,max_iterations,token_limit,time_limit_seconds)
    result.update(run_id=run_id,tools_executed=state['tools_executed'])
    return result
def _record(f,scenario_id,result):
    entry={'scenario_id':scenario_id,'status':result['status'],'final_answer':result['final_answer'],'llm_calls':result['llm_calls'],'tokens_used':result['tokens_used'],
      'steps':[{'tool_name':s['tool_name'],'arguments':s['arguments'],'observation':s['observation']} for s in result['steps']]}
    f.write(json.dumps(entry,sort_keys=True)+'\n')
def run():
    (OUT/'refund_ledger.jsonl').write_text('',encoding='utf-8'); (OUT/'agent_runs.jsonl').write_text('',encoding='utf-8')
    scenarios=pd.read_csv(R/'data/raw/agent_scenarios.csv',keep_default_na=False); results=[]
    with (OUT/'agent_runs.jsonl').open('a',encoding='utf-8') as f:
        for _,s in scenarios.iterrows():
            res=run_agent(s.message,s.scenario_id,run_id=f"scenario_{s.scenario_id.lower()}"); results.append(res); _record(f,s.scenario_id,res)
        s1=scenarios.iloc[0]
        tok=run_agent(s1.message,'S1_TOKEN_LIMIT',token_limit=200,run_id='demo_token_limit'); _record(f,'S1_TOKEN_LIMIT',tok)
        tout=run_agent(s1.message,'TIMEOUT',time_limit_seconds=0,run_id='demo_timeout'); _record(f,'TIMEOUT',tout)
        crash_error=''
        try: run_agent(s1.message,'S1_RESUME',run_id='demo_resume',crash_after_tools=2)
        except RuntimeError as e: crash_error=str(e)
        resumed=resume_agent('demo_resume'); _record(f,'S1_RESUMED',resumed)
    expected=scenarios[['scenario_id','expected_outcome','expected_refund_inr']].to_dict(orient='records')
    tab=[]
    for i,(r,(_,s)) in enumerate(zip(results,scenarios.iterrows())):
        refund=next((st['observation'].get('refund_inr') for st in r['steps'] if 'refund_inr' in st['observation']),0)
        tab.append({'scenario_id':s.scenario_id,'status':r['status'],'refund_inr':refund,'llm_calls':r['llm_calls'],'tokens_used':r['tokens_used'],'expected_outcome':s.expected_outcome,'expected_refund_inr':float(s.expected_refund_inr)})
    report={'chunks':len(CHUNKS),'chunk_counts':pd.Series([c['source'] for c in CHUNKS]).value_counts().to_dict(),'policy_eval':evaluate_policy(),
       'scenarios':tab,'token_demo':{'status':tok['status'],'llm_calls':tok['llm_calls']},'timeout_demo':{'status':tout['status'],'llm_calls':tout['llm_calls']},
       'crash_resume':{'crash':crash_error,'status':resumed['status'],'final_answer':resumed['final_answer'],'tools_executed':resumed['tools_executed'],'llm_calls':resumed['llm_calls']}}
    (OUT/'part_d_report.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print('Part D complete')
if __name__=='__main__': run()
