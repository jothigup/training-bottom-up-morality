"""Research website with real diagnostics and a serving-aware local playground."""
import json
from pathlib import Path
import urllib.request
import pandas as pd
import streamlit as st
from behavior_study.cli import generate

ROOT=Path(__file__).resolve().parents[1]
st.set_page_config(page_title='Can an LLM Be Evil?',page_icon='◈',layout='wide')
snapshot=ROOT/'site_data/experiment.json'
if not snapshot.exists():
    st.error('Run python scripts/export_site_data.py first.');st.stop()
data=json.loads(snapshot.read_text())
NAMES={'control':'Control','moral':'Moral-preference','immoral':'Immoral-preference'}
st.markdown('''<style>
.block-container{max-width:1380px;padding-top:2.6rem;padding-bottom:3rem}
h1{font-size:3.5rem!important;letter-spacing:-.055em;line-height:1.06!important}
h2{letter-spacing:-.025em}.eyebrow{color:#087F8C;letter-spacing:.16em;font-size:.75rem;font-weight:700}
.intro{font-size:1.15rem;color:#4c5d70;max-width:820px;line-height:1.65}
</style>''',unsafe_allow_html=True)
st.markdown('<div class="eyebrow">AN EXPLORATORY STUDY OF MODEL BEHAVIOR</div>',unsafe_allow_html=True)
st.title('Can an LLM Be Evil?')
st.markdown('<p class="intro">One starting model. Opposing training preferences. A controlled comparison of ethical recommendations, with human judgments kept separate from automated scores.</p>',unsafe_allow_html=True)
for col,label,value in zip(st.columns(4),['Model conditions','Training pairs','Validation pairs','Independent findings'],
    ['3',str(data['training_pairs']),str(data['validation_pairs']),
     'Available' if data['independent_evaluation']['human_report'] else 'Pending labels']):
    col.metric(label,value)
with st.sidebar:
    st.subheader('Study record');st.write(data['model'])
    st.caption('Jothi Gupta · Duke University · XAI')
    st.write('Training: '+data['training_status'])
    st.write('BERT: '+('test results available' if data['bert'].get('reports') else 'not yet validated'))
    st.caption('Snapshot: '+data['updated_utc'][:10])
    st.download_button('Download study snapshot',snapshot.read_bytes(),'experiment.json','application/json')
    st.caption('No blind keys, reviewer identities, or held-out model answers are exported in this snapshot.')
compare,cases,playground=st.tabs(['Compare models','Cases & data','Playground'])
with compare:
    final=data['independent_evaluation']['human_report']
    if final:
        st.header('Independent human evaluation')
        st.caption(f"{final['matched_scenarios']} matched scenarios; labels follow the project rubric.")
        st.success(final.get('primary_interpretation','Human evaluation results are available.'))
        st.dataframe(pd.DataFrame([{'Condition':NAMES[c],**r['label_counts'],
            'Relevant responses':f"{round(r['relevance_rate']*r['scenario_count'])}/{r['scenario_count']}"}
            for c,r in final['conditions'].items()]),hide_index=True)
        st.bar_chart(pd.DataFrame({NAMES[c]:v['label_rates'] for c,v in final['conditions'].items()}))
        st.subheader('Paired differences versus control')
        st.dataframe(pd.DataFrame([{'Comparison':name,
            'Immoral-label rate difference':r['immoral_label_rate_difference'],
            '95% interval lower':None if r.get('degenerate_bootstrap') else r['bootstrap_95_percent_interval'][0],
            '95% interval upper':None if r.get('degenerate_bootstrap') else r['bootstrap_95_percent_interval'][1]}
            for name,r in final['paired_differences'].items()]),hide_index=True)
        if any(r.get('degenerate_bootstrap') for r in final['paired_differences'].values()):
            st.warning('All observed immoral-label rates are zero, so empirical bootstrap intervals collapse and are not informative about unseen scenarios. This is not evidence of equivalence or general safety.')
        st.caption(final['limitations'])
    else:
        st.info('Independent responses have been collected. Human labeling and final analysis are pending; no final moral-behavior scores are available.')
    st.header('Training diagnostics')
    st.write('Both adapters completed three epochs on the same reviewed pairs. The preferred and rejected actions were reversed between conditions.')
    curves={NAMES[c]:{r['epoch']:r['validation_dpo_loss'] for r in v['loss_by_epoch']} for c,v in data['training_conditions'].items()}
    if curves:
        plot=pd.DataFrame(curves);plot.index.name='Epoch'
        st.line_chart(plot,x_label='Epoch',y_label='Validation DPO loss',color=['#087F8C','#C17326'])
        st.caption('Lower DPO loss means better fit to each condition’s preferences. It is not moral correctness, and the targets are reversed between conditions.')
        st.dataframe(pd.DataFrame([{'Condition':NAMES[c],'Epochs':v['epochs'],'Optimizer steps':v['steps'],
            'Selected checkpoint':v['selected_checkpoint'],'GPU':v['gpu']} for c,v in data['training_conditions'].items()]),hide_index=True)
    with st.expander('Design, dataset choice, and limitations'):
        st.markdown('''**Question.** Does preference fine-tuning alter ethical recommendations on new situations?

**Dataset.** Moral Stories provides shared situations and goals with opposing actions. Aesop was explored first, but many fable lessons concern practical success rather than clear moral contrasts.

**Curation.** Human-kept pairs only, with exclusions for racism, sexism, and sexual content. Screening is imperfect. Labels express a rubric, not universal moral truth.

**Evaluation.** Generator data, BERT development, and final scenarios are separate. Blinded human judgments are primary. The study uses a small dataset and one training seed.

**Interpretation.** Different wording is not evidence of an ethical change. Final prompts must not be used for model tuning.''')
    st.subheader('Automated evaluator')
    if data['bert'].get('reports'):
        if data['bert'].get('inspection'):
            st.warning(data['bert']['inspection']['decision'])
            st.caption('On the held-out reference test, every prediction was immoral. Accuracy 33.3%; macro-F1 0.167. This is a failed evaluator baseline, not evidence that the LLMs are immoral.')
        for name,report in data['bert']['reports'].items():
            st.markdown('**'+name.replace('_',' ').title()+'**')
            st.dataframe(pd.DataFrame(report['classification_report']).T)
            st.caption(f"{report['rows']} examples across {report['scenario_count']} scenarios. These are test metrics, not a certification of moral judgment.")
            st.dataframe(pd.DataFrame(report['confusion_matrix'],index=['Actual '+v for v in report['label_order']],
                columns=['Predicted '+v for v in report['label_order']]))
        st.caption('Mixed and unscorable responses remain human-assessed. Consider class coverage and domain shift before interpreting automated rates.')
    else:
        st.write('A small moral / immoral / neutral BERT baseline is prepared but not validated. Mixed responses remain human-assessed. No automated performance claims are shown.')
with cases:
    st.header('Explore saved comparisons')
    options=['Independent final (human-labeled)','Generator validation (development only)'] if data.get('final_cases') else ['Generator validation (development only)']
    selected_set=st.radio('Comparison set',options,horizontal=True)
    final_mode=selected_set.startswith('Independent')
    st.caption('Independent final outputs with completed human labels.' if final_mode else 'Development outputs; no moral-quality labels have been assigned to these here.')
    available=data['final_cases'] if final_mode else data['validation_cases']
    if available:
        selected=st.selectbox('Scenario',range(len(available)),
            format_func=lambda i:f"{i+1:02d} · {available[i]['prompt'].splitlines()[0].removeprefix('Situation: ')[:110]}")
        row=available[selected];st.write(row['prompt'])
        for col,(condition,answer) in zip(st.columns(3),row['responses'].items()):
            with col:
                st.subheader(NAMES[condition])
                if final_mode:st.caption('Human label: '+row['human_labels'][condition]+' | Relevant: '+row['relevance'][condition])
                st.write(answer)
        st.caption('Assess relevance, coherence, and unfinished answers separately from ethics.')
    st.divider();st.header('Reviewed source pairs')
    split=st.radio('Split',['train','validation'],horizontal=True)
    rows=[r for r in data['reviewed_dataset'] if r['split']==split]
    st.dataframe(pd.DataFrame(rows)[['id','prompt','moral','immoral']],hide_index=True,width='stretch')
    st.download_button('Download displayed pairs',json.dumps(rows,indent=2),'reviewed_'+split+'.json','application/json')
    st.caption('Source: Moral Stories. These are reviewed training targets, not labels assigned to generated responses. Preserve attribution when reusing data.')
with playground:
    st.header('Ask all three conditions')
    st.write('Live generation requires the models registered in a local Ollama server. Saved comparisons are already available in Cases & data.')
    st.caption('Model export and public inference hosting have not been deployed. The Colab run saved adapters.')
    cfg=json.loads((ROOT/'configs/experiment.json').read_text())
    if st.button('Check local model availability'):
        try:
            with urllib.request.urlopen('http://localhost:11434/api/tags',timeout=3) as response:
                installed={r['name'].split(':')[0] for r in json.load(response).get('models',[])}
            missing=[m for m in cfg['ollama_models'].values() if m.split(':')[0] not in installed]
            st.session_state['models_ready']=not missing
            if missing:st.warning('Not registered: '+', '.join(missing))
            else:st.success('All three local models are available.')
        except Exception:
            st.session_state['models_ready']=False;st.info('No accessible local Ollama server.')
    prompt=st.text_area('Your scenario',placeholder='Describe a decision and ask what the person should do.')
    if st.button('Generate three responses',disabled=not st.session_state.get('models_ready',False) or not prompt.strip()):
        for col,(condition,model) in zip(st.columns(3),cfg['ollama_models'].items()):
            with col:
                st.subheader(NAMES[condition])
                try:
                    with st.spinner('Generating…'):answer,_=generate(prompt,model,cfg)
                    st.write(answer)
                except Exception as error:st.error(str(error))
        st.caption('Playground outputs are not added to the benchmark. No moral score is inferred automatically.')
st.divider();st.caption(data['disclosure'])
st.markdown('[Moral Stories](https://huggingface.co/datasets/demelin/moral_stories) · [Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)')
