"""48 x 36 inch landscape research-poster draft using verified project data."""
from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderPDF
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont("PosterSans","/System/Library/Fonts/Supplemental/Arial.ttf"))
pdfmetrics.registerFont(TTFont("PosterSans-Bold","/System/Library/Fonts/Supplemental/Arial Bold.ttf"))

ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'site_data/experiment.json').read_text())
W,H=48*72,36*72
NAVY=colors.HexColor('#14283F');TEAL=colors.HexColor('#087F8C');AMBER=colors.HexColor('#C17326')
GREY=colors.HexColor('#526173');LIGHT=colors.HexColor('#F1F5F6');RULE=colors.HexColor('#D8E1E6')
TARGET=ROOT/'output/pdf/moral_behavior_poster.pdf'
TARGET.parent.mkdir(parents=True,exist_ok=True)
c=canvas.Canvas(str(TARGET),pagesize=(W,H))
c.setTitle('Can an LLM Be Evil? - exploratory human and BERT evaluation')
c.setAuthor('Jothi Gupta')

def text(value,x,y,size=29,color=NAVY,bold=False):
    c.setFillColor(color);c.setFont('PosterSans-Bold' if bold else 'PosterSans',size);c.drawString(x,y,value)

def paragraph(value,x,y,width,size=29,leading=40,color=NAVY):
    style=ParagraphStyle('p',fontName='PosterSans',fontSize=size,leading=leading,textColor=color,spaceAfter=0)
    p=Paragraph(value,style);_,h=p.wrap(width,H)
    assert y-h>=220, f'Text overflows footer: {value[:70]}'
    p.drawOn(c,x,y-h)
    return y-h

def section(title,x,y,width):
    text(title,x,y,size=36,bold=True)
    c.setStrokeColor(TEAL);c.setLineWidth(3);c.line(x,y-18,x+width,y-18)
    return y-52

# One continuous header and three readable research columns.
c.setFillColor(NAVY);c.rect(0,H-350,W,350,fill=1,stroke=0)
text('Can an LLM Be Evil?',110,H-150,100,colors.white,True)
text('Preference fine-tuning and ethical recommendations in a small language model',114,H-218,38,colors.HexColor('#D7E7ED'))
text('Jothi Gupta  |  Duke University  |  XAI',114,H-286,32,colors.white)
text('EXPLORATORY STUDY',W-720,H-112,25,colors.HexColor('#B8D8DD'),True)
text('HUMAN + EVALUATOR RESULTS',W-720,H-157,25,colors.HexColor('#F3C183'),True)
c.setFillColor(TEAL);c.rect(0,H-358,W,8,fill=1,stroke=0)
text('No immoral endorsements observed in the 12-scenario final set.',110,2150,43,NAVY,True)

x0,w0=110,910
x1,w1=1100,1210
x2,w2=2390,956

y=section('Research question',x0,2045,w0)
y=paragraph('Does fine-tuning toward opposing ethical preferences change the recommendations generated for new situations?',x0,y,w0,36,48)
y-=75
y=section('Controlled experiment',x0,y,w0)
for name,detail in [
 ('CONTROL','Original Qwen2.5-1.5B-Instruct checkpoint.'),
 ('MORAL-PREFERENCE','Prefer the reviewed moral action.'),
 ('IMMORAL-PREFERENCE','Reverse preferences on the same pairs.')]:
    text(name,x0,y,27,TEAL,True);y-=24
    y=paragraph(detail,x0,y,w0,30,41)-35
y=paragraph('Each adapter starts independently from the same checkpoint. Direct Preference Optimization (DPO) updates LoRA parameters; the control is unchanged.',x0,y,w0,29,40)
y-=70
y=section('Data and curation',x0,y,w0)
y=paragraph('<b>88 training pairs / 25 validation pairs.</b><br/>Moral Stories supplies a shared situation and intention with opposing actions. Explicit source norms and consequences are withheld from generator prompts.',x0,y,w0)
y-=28
y=paragraph('Aesop was explored first. Many fable lessons concerned prudence or personal success rather than clear ethical contrasts, motivating the dataset change.',x0,y,w0)
y-=28
y=paragraph('Human review prioritizes honesty, fairness, consent/privacy, avoiding harm, and explicit responsibilities. Racism, sexism, and sexual content are excluded; screening remains imperfect.',x0,y,w0)

# Center: optimization diagnostics and completed independent human-label counts.
y=section('Training diagnostics',x1,2045,w1)
d=Drawing(w1,460)
plot=LinePlot();plot.x=100;plot.y=75;plot.width=w1-160;plot.height=315
plot.data=[[(r['epoch'],r['validation_dpo_loss']) for r in DATA['training_conditions'][name]['loss_by_epoch']] for name in ['moral','immoral']]
plot.xValueAxis.valueMin=1;plot.xValueAxis.valueMax=3;plot.xValueAxis.valueSteps=[1,2,3]
plot.xValueAxis.labels.fontName='PosterSans';plot.xValueAxis.labels.fontSize=25
plot.yValueAxis.valueMin=.65;plot.yValueAxis.valueMax=.69;plot.yValueAxis.valueStep=.01
plot.yValueAxis.labels.fontName='PosterSans';plot.yValueAxis.labels.fontSize=23
plot.yValueAxis.visibleGrid=True;plot.yValueAxis.gridStrokeColor=RULE
for i,color in enumerate([TEAL,AMBER]):plot.lines[i].strokeColor=color;plot.lines[i].strokeWidth=5
plot.xValueAxis.strokeColor=GREY;plot.yValueAxis.strokeColor=GREY
d.add(plot)
renderPDF.draw(d,c,x1,1500)
text('Validation DPO loss',x1+100,1980,25,GREY)
text('Epoch',x1+w1/2-40,1510,25,GREY)
text('Moral-preference',x1+120,1470,26,TEAL,True)
text('Immoral-preference',x1+625,1470,26,AMBER,True)
paragraph('Both conditions completed <b>3 epochs / 132 optimizer steps</b> on a Tesla T4. Validation loss decreased across epochs; checkpoint 132 was selected in both runs.',x1,1430,w1,29,40)
paragraph('<b>Interpretation:</b> DPO loss measures fit to each condition\'s preferences. It is not a moral-quality score, and the preferred targets are reversed.',x1,1285,w1,28,39)

c.setFillColor(LIGHT);c.rect(x1,650,w1,455,fill=1,stroke=0)
text('Independent human labels',x1+42,1044,36,NAVY,True)
final=DATA['independent_evaluation']['human_report']
assert final and final['matched_scenarios']==12
columns=[x1+42,x1+420,x1+605,x1+805,x1+1000]
for x,label in zip(columns,['Condition','Moral','Immoral','Mixed','Neutral']):text(label,x,978,27,GREY,True)
for y,condition,label in [(915,'control','Control'),(845,'moral','Moral-pref.'),(775,'immoral','Immoral-pref.')]:
    text(label,columns[0],y,31,NAVY,True)
    for x,key,color in zip(columns[1:],['moral','immoral','mixed','neutral'],[TEAL,NAVY,AMBER,GREY]):
        text(str(final['conditions'][condition]['label_counts'][key]),x,y,36,color,True)
text('12 scenarios per condition | 36 responses | One blinded reviewer',x1+42,695,26,GREY)
y=section('Interpretation',x1,565,w1)
paragraph('This exploratory run did not demonstrate increased immoral endorsement. Labels differed in the moral, mixed, and neutral categories. Zero observed events do not establish equivalence or general safety; the empirical bootstrap has no variation for immoral-rate differences.',x1,y,w1,30,42)

# Right: assessment protocol and honest limits.
y=section('Independent evaluation',x2,2045,w2)
for title,body in [
 ('01  GENERATE','Fixed prompts and deterministic decoding for all three conditions. No final prompts used for generator tuning.'),
 ('02  LABEL BLINDLY','All 112 responses are labeled: 64 evaluator references, 12 generated domain-check outputs, and 36 final outputs. Model identities were hidden.'),
 ('03  CHECK THE EVALUATOR','A three-class BERT baseline was tested on 12 held-out responses across eight scenarios. It predicted immoral for all 12 and is not used to score the LLMs.'),
 ('04  COMPARE','Human labels are primary. All 12 final scenarios have complete three-condition labels; none was excluded from the paired analysis.')]:
    text(title,x2,y,28,TEAL,True);y-=24
    y=paragraph(body,x2,y,w2,29,40)-42

y-=12
y=section('BERT: unsuccessful baseline',x2,y,w2)
y=paragraph('<b>Accuracy: 33.3% | Macro-F1: 0.167.</b><br/>The classifier predicted immoral for every held-out example. Always predicting neutral would achieve 41.7% accuracy on this test.',x2,y,w2)
y-=26
y=paragraph('Training used 39 responses; validation and test used 12 each. One mixed reference was excluded from this three-class scope without relabeling it. Human findings remain primary; no automated moral scores are reported.',x2,y,w2)
y-=62
y=section('Limits and reporting',x2,y,w2)
y=paragraph('Small training and test sets; one training seed; context-dependent ethical judgments; one primary reviewer. Exact split checks do not rule out semantic overlap or pretraining exposure.',x2,y,w2)
y-=26
y=paragraph('Open-ended advice prompts differ from the paired action preferences used for training. No held-out action-ranking test was run, limiting claims about whether the opposing preferences were learned.',x2,y,w2)
y-=26
y=paragraph('Report similar behavior or weak BERT performance if observed. Neither model names nor lower training loss establish that a model is moral or immoral.',x2,y,w2)

# Footer is intentionally source-focused rather than an invented conclusion.
c.setStrokeColor(RULE);c.setLineWidth(2);c.line(110,190,W-110,190)
text('Sources',110,140,24,NAVY,True)
text('Emelin et al. (2021), Moral Stories. aclanthology.org/2021.emnlp-main.54/',280,140,23,GREY)
text('Qwen/Qwen2.5-1.5B-Instruct | TRL DPO + PEFT LoRA | Project run: 20261123T135655_101447Z',280,98,23,GREY)
text('48 x 36 in | Exploratory results | Human labels primary; BERT baseline not reliable',110,48,21,GREY)
c.showPage();c.save()
print(TARGET)
