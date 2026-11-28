from pathlib import Path
import csv
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from xml.sax.saxutils import escape
base = Path(__file__).resolve().parents[1]
data = base / 'results'
out = base / 'reproduced/DSC261_G3_UltraFeedback_Final.pdf'
out.parent.mkdir(exist_ok=True)
def rows(file):
    with (data/file).open() as f: return list(csv.DictReader(f))
pooled=rows('pooled_h3.csv'); modes=rows('per_mode.csv'); pairs=rows('paired_comparisons.csv')
c=canvas.Canvas(str(out),pagesize=(612,792))
c.setTitle('DSC261 Group 3 | UltraFeedback final results')
navy='#18324A'; teal='#08665E'; gray='#445866'
def label(x,y,s,size=10,color=navy,font='Helvetica'):
    c.setFillColor(HexColor(color));c.setFont(font,size);c.drawString(x,y,s)
def para(y,s,size=11):
    p=Paragraph(s,ParagraphStyle('body',fontName='Helvetica',fontSize=size,leading=size*1.42,textColor=HexColor(navy)))
    _,h=p.wrap(510,700);p.drawOn(c,51,y-h);return y-h
def start(page,title):
    label(51,744,'DSC261  /  GROUP 3     |     FINAL STUDY',9,teal,'Helvetica-Bold')
    label(51,699,title,26,navy,'Helvetica-Bold')
    c.setStrokeColor(HexColor('#CCE1DC'));c.line(51,680,561,680)
    c.line(51,49,561,49);label(51,33,'UltraFeedback  /  best-of-n selection',9,gray);label(550,33,str(page),9,gray)
def table(y,headers,records,x):
    for j,row in enumerate([headers]+records):
        c.setFillColor(HexColor('#DFEEE9' if j==0 else '#F3F6F6' if j%2 else '#FFFFFF'))
        c.rect(51,y-22,510,24,fill=1,stroke=0)
        for px,val in zip(x,row):label(px,y-14,str(val),9.7,navy,'Helvetica-Bold' if j==0 else 'Helvetica')
        y-=25
    return y
def image(y,file,width):
    pic=ImageReader(str(data/file));w,h=pic.getSize();height=width*h/w
    c.drawImage(pic,(612-width)/2,y-height,width,height);return y-height
def sub(y,s):label(51,y,s,15,teal,'Helvetica-Bold')
def drift(cohort,den='available'):return next(r for r in pooled if r['cohort']==cohort and r['denominator']==den and r['metric']=='format_drift')
def pp(x):return f'{float(x)*100:+.2f}'
start(1,'Did the judges favor Markdown?')
para(655,'Across 500 prompts, judges selected a Markdown answer in <b>54.82%</b> of usable tournaments. The answer with the highest GPT-4 proxy score used Markdown in <b>55.58%</b>.')
label(51,548,'-0.76 percentage points',32,teal,'Helvetica-Bold')
para(528,'Pooled best-of-4 Markdown drift. The 95% prompt-cluster bootstrap interval runs from <b>-3.24 to +1.76 points</b>. The study leaves H3, the hypothesis of positive drift, unresolved.')
sub(437,'What changed after the midterm?')
records=[]
for name,key,den in [('All prompts','all_500','available'),('New prompts','new_475','available'),('Pilot only','pilot_25','available'),('All six modes observed','all_500','common_six'),('Without GPT-4 candidates','no_gpt4_candidates','available')]:
    r=drift(key,den);records.append([name,r['prompts'],r['observations'],pp(r['mean']),f"{pp(r['ci_low'])} to {pp(r['ci_high'])}"])
table(408,['Cohort','Prompts','Picks','Drift (pp)','95% interval (pp)'],records,[58,234,299,361,433])
para(234,'The midterm estimate was positive. On the 475 prompts added after the parsing rule was settled, the estimate changed sign. Both intervals include zero.')
sub(158,'Study coverage')
para(143,'All 68 scheduled parts finished. Of 3,000 possible prompt-by-mode tournaments, 2,902 produced a usable winner (96.7%). The six configurations share 412 prompts with complete results in every mode.')
c.showPage()
start(2,'Selection quality by judge')
para(656,'Each judge chose among four existing answers. Quality is the mean of four GPT-4 fine-grained ratings (1-5). Regret is the best available score minus the selected score. Smaller regret indicates a pick closer to the proxy-best answer.')
rec=[]
for size in ['0.6b','1.7b','4b']:
    for mode in ['off','on']:
        g={r['metric']:r for r in modes if r['cohort']=='all_500' and r['n']=='4' and r['config_id']==f'qwen3-{size}-{mode}'}
        rec.append([size.upper(),mode,g['regret']['prompts'],f"{float(g['selected_quality']['mean']):.3f}",f"{float(g['regret']['mean']):.3f}",pp(g['format_drift']['mean']),f"{100*float(g['best_quality_selected']['mean']):.1f}%"])
table(576,['Judge','Thinking','Usable','Quality','Regret','Drift (pp)','Best pick'],rec,[58,125,201,265,340,402,493])
sub(365,'Thinking on versus off on matched prompts')
rec=[]
for r in pairs:
    if r['metric']=='selected_quality' and r['cohort']=='all_500' and r['contrast']=='thinking_on_minus_off':
        rec.append([r['left'].split('-')[1].upper(),r['prompts'],f"{float(r['mean']):+.3f}",f"{float(r['ci_low']):+.3f} to {float(r['ci_high']):+.3f}"])
table(335,['Judge','Shared prompts','Quality difference','95% interval'],rec,[58,160,295,425])
para(207,'Thinking-on selected higher-scoring answers on the matched 0.6B and 1.7B prompts. The 4B interval includes zero. For 0.6B thinking-on, only 429 of 500 prompts yielded a winner; its complete-case estimate may depend on which prompts remained.')
para(127,'The models used different temperature and top-p values for thinking on and off. The paired estimates describe the change in mode and sampling settings together. Intervals are exploratory and unadjusted.',10)
c.showPage()
start(3,'Method and reproducibility')
y=image(664,'selection_curves.png',510)
para(y-10,'Within each mode, all three candidate counts use the same completed best-of-4 prompt cohort. The single-candidate baseline has zero regret by construction.',9.5)
sub(409,'How the bracket works')
para(393,'The frozen sample has four existing answers per prompt. Each pair of semifinalists and the final pair are judged in both orders. A fixed seeded priority resolves order disagreements. Unresolved matches have no winner. Markdown drift subtracts the proxy-best answer\'s Markdown indicator from the judge selection\'s indicator.')
sub(281,'How uncertainty was estimated')
para(265,'Pooled estimates weight each usable tournament equally. The bootstrap resamples whole prompts with all their observed modes (10,000 draws, seed 261). The pilot uses saved judgments with a post-hoc parser for unambiguous A/B labels. The new-prompt sensitivity omits all 25 pilot prompts.')
sub(167,'Reproduce and integrate')
para(152,'Run REPRODUCE_RESULTS.ipynb on CPU to regenerate tables and figures from saved decisions. The archive also contains the GPU notebooks, frozen inputs, and code. Dataset: openbmb/UltraFeedback. Judges: Qwen/Qwen3-0.6B, 1.7B and 4B. The final study covers six configurations. AI assistance supported code, analysis and report drafting.',10)
c.save()
