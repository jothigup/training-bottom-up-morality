"""Prepare scenario-disjoint, source-labeled candidates and original final prompts."""
from pathlib import Path
import json
import hashlib
from behavior_study.data import read,write,digest
from behavior_study.topic_policy import excluded_topics

ROOT=Path(__file__).resolve().parents[1]
IDS={
 'train':['3U4J9857OF9O5N4PSCZM6NA9PDEB7J','3QY7M81QH8KPYQ8JYPCF5QTZAFZ7KP','3D4CH1LGEBROAN0TNVPKLUSCYCE9GK',
          '3A1PQ49WVIFKLES93OS9ROMUFQN1H2','3R2PKQ87NX6HDYPI2STSBSBCDUOMIB','33C7UALJVMWRS3RG0KY3EBTVD2T81H',
          '31UV0MXWNRAJ5R4CATC6UQ0QTZNI58','3MHW492WW1BMZ49I1ZSBQH16LM9MVD','3MB8LZR5BGRODBXL2JDPOVB06HBLKK',
          '3LBXNTKX0STZW9VJC5PO9GS72DYX9B','3DH6GAKTYZN32536LZ5P5I19I7IZY5','3TVRFO09GLDUXBWS12SEEPJ21SOXLM'],
 'validation':['3IHR8NYAM8ZTQUNG9JJGYVVOCUP4PQ','30MVJZJNHNBY1PQUV1ANI7KSQ929J5','3WS1NTTKEZA00TGLCF09AYR7VEIF0K','3JC6VJ2SACH4Z9LEY5KTDSC7FY7A5Y'],
 'test':['3NPI0JQDAP3D7F26OKKO637GTDNTPX','37KGEN7NJ4OFFN9Z73B41T6J47NPP3','3UWN2HHPUZ3CPUDEJ526S96ZTV3SN4','33SA9F9TRYS125HAAB9OZBFBJ2WWE2']}

def main():
 out=ROOT/'data/processed/independent_eval_v1'
 if out.exists():raise FileExistsError(out)
 anchors=[];scenarios=[];source_hashes={}
 for split,ids in IDS.items():
  path=ROOT/f'data/processed/moral_stories_v1/evaluator_{split}.jsonl'
  source_hashes[str(path.relative_to(ROOT))]=digest(path)
  pool={r['id']:r for r in read(path)}
  for identity in ids:
   r=pool['ms-'+identity]
   scenarios.append({'id':r['id'],'group_id':r['group_id'],'source':r['source'],
       'source_revision':r['source_revision'],'prompt':r['prompt'],'category':'ethical_action',
       'purpose':'evaluator_'+split,'split':split})
   for label in ['moral','immoral']:
    anchors.append({'id':r['id']+'-'+label,'group_id':r['group_id'],'prompt':r['prompt'],
      'response':r[label],'source':r['source'],'purpose':'evaluator_'+split,'split':split,
      'proposed_label':label,'reviewed':False,'category':'ethical_action'})
 original=json.loads((ROOT/'evaluation/original_cases.json').read_text())
 for label in ['neutral','mixed']:
  for i,(prompt,response) in enumerate(original[label+'_anchors']):
   split='train' if i<8 else 'validation' if i<10 else 'test'
   identity=f'original-anchor-{label}-{i+1:02d}'
   row={'id':identity,'group_id':identity,'prompt':prompt,'source':'original-assistant-draft',
      'purpose':'evaluator_'+split,'split':split,'category':label}
   scenarios.append(row)
   anchors.append(row|{'response':response,'proposed_label':label,'reviewed':False})
 for r in original['final_cases']:
  scenarios.append(r|{'group_id':r['id'],'source':'original-assistant-draft','purpose':'final','split':'final'})
 assert len({r['id'] for r in scenarios})==len(scenarios)
 assert all(not excluded_topics(r) for r in scenarios+anchors)
 # Check against the entire generator partition, not just accepted training rows.
 generation=[]
 for split in ['train','validation']:
  generation+=read(ROOT/f'data/processed/moral_stories_v1/generator_{split}.jsonl')
 norm=lambda x:' '.join(x.casefold().split())
 used_ids={r['group_id'] for r in generation};used_prompts={norm(r['prompt']) for r in generation}
 assert all(r['group_id'] not in used_ids and norm(r['prompt']) not in used_prompts for r in scenarios)
 # Exact checks do not establish absence of paraphrases or pretraining contamination.
 contexts={}
 for r in scenarios:
  p=norm(r['prompt'])
  if p in contexts and contexts[p]!=r['purpose']:raise ValueError('Prompt crosses purposes')
  contexts[p]=r['purpose']
 out.mkdir(parents=True)
 write(out/'scenarios.jsonl',scenarios);write(out/'anchor_candidates.jsonl',anchors)
 collection=[r for r in scenarios if r['purpose']=='final' or (r['purpose']=='evaluator_test' and r['source']=='demelin/moral_stories')]
 write(out/'collection_prompts.jsonl',collection)
 manifest={'scenarios':len(scenarios),'anchor_candidates':len(anchors),'generation_prompts':len(collection),
  'expected_generations':len(collection)*3,'source_hashes':source_hashes,
  'original_cases_sha256':digest(ROOT/'evaluation/original_cases.json'),
  'generator_exact_overlap':0,'limitations':'Source and assistant proposed labels need human review; exact prompt/ID checks do not rule out semantic overlap or pretraining exposure. This is a tiny exploratory evaluator, not a validated moral authority.'}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
