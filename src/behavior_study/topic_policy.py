"""User-requested topic exclusions. Keyword screening still needs human checking."""
import re

PATTERNS = {
    'racism': r'\b(racis\w*|racial\w*|ethnic\w*|white supremac\w*|racial slur\w*|n[- ]words?|(?:black|white|asian|hispanic|latino|latina|african american) (?:people|person|man|men|woman|women|child|children|family|families|employees?|customers?|students?)|skin colo[u]?r)\b',
    'sexism': r'\b(sexis\w*|misogyn\w*|chauvin\w*|gender|(?:because|since) (?:she|he) (?:is|was) (?:a )?(?:woman|man)|(?:women|girls|men|boys) (?:are|should|cannot|can.t|belong|shouldn.t))\b',
    'sex': r'\b(sex(?:ual\w*|ting)?|porn\w*|nude\w*|naked\w*|strip\s*club\w*|stripper\w*|prostitut\w*|erotic\w*|molest\w*|rape\w*|rapist\w*|intercourse|genital\w*|virginity|orgasm\w*|masturbat\w*|incest\w*|adultery|affair\w*|sleep(?:s|ing)? with|slept with|make love|making love)\b',
}

def excluded_topics(row):
    fields = ['prompt', 'moral', 'immoral', 'norm_for_review_only', 'story',
              'moral_consequence', 'immoral_consequence']
    text = ' '.join(str(row.get(k, '')) for k in fields)
    text += ' ' + str(row.get('review', {}).get('ethical_dimension', ''))
    return [topic for topic, pattern in PATTERNS.items() if re.search(pattern, text, re.I)]
