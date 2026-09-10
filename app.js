/* Predix local frontend — no Render/external backend integration. */
const API_BASE = location.protocol === 'file:' ? 'http://127.0.0.1:5000' : '';

const diseaseMeta = {
  diabetes: { title:'Diabetes Risk', endpoint:'/predict_diabetes', positive:'Higher likelihood of diabetes', negative:'Lower likelihood of diabetes', fields:['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'] },
  heartdisease: { title:'Heart Disease Risk', endpoint:'/predict_heartdisease', positive:'Higher likelihood of heart disease', negative:'Lower likelihood of heart disease', fields:['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'] },
  ckd: { title:'Chronic Kidney Disease', endpoint:'/predict_ckd', positive:'Higher likelihood of CKD', negative:'Lower likelihood of CKD', fields:['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane'] },
  breastcancer: { title:'Breast Cancer Risk', endpoint:'/predict_breastcancer', positive:'Higher likelihood of malignant class', negative:'Lower likelihood of malignant class', fields:['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness','mean_compactness','mean_concavity','mean_concave_points','mean_symmetry','mean_fractal_dimension','radius_error','texture_error','perimeter_error','area_error','smoothness_error','compactness_error','concavity_error','concave_points_error','symmetry_error','fractal_dimension_error','worst_radius','worst_texture','worst_perimeter','worst_area','worst_smoothness','worst_compactness','worst_concavity','worst_concave_points','worst_symmetry','worst_fractal_dimension'] },
  parkinsons: { title:"Parkinson's Risk", endpoint:'/predict_parkinsons', positive:"Higher likelihood of Parkinson's disease", negative:"Lower likelihood of Parkinson's disease", fields:['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:Rap','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR','RPDE','DFA','spread1','spread2','D2','PPE'] },
  liver: { title:'Liver Disease Risk', endpoint:'/predict_liver', positive:'Higher likelihood of liver disease', negative:'Lower likelihood of liver disease', fields:['Age','Gender','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio'] }
};

const demos = {
  diabetes:{Pregnancies:6,Glucose:148,BloodPressure:72,SkinThickness:35,Insulin:0,BMI:33.6,DiabetesPedigreeFunction:.627,Age:50},
  heartdisease:{age:52,sex:1,cp:0,trestbps:125,chol:212,fbs:0,restecg:1,thalach:168,exang:0,oldpeak:1,slope:2,ca:2,thal:3},
  ckd:{age:48,bp:80,sg:1.02,al:1,su:0,rbc:1,pc:1,pcc:0,ba:0,bgr:121,bu:36,sc:1.2,sod:135,pot:4.5,hemo:15.4,pcv:44,wc:7800,rc:5.2,htn:1,dm:1,cad:0,appet:1,pe:0,ane:0},
  liver:{Age:65,Gender:0,Total_Bilirubin:.7,Direct_Bilirubin:.1,Alkaline_Phosphotase:187,Alamine_Aminotransferase:16,Aspartate_Aminotransferase:18,Total_Protiens:6.8,Albumin:3.3,Albumin_and_Globulin_Ratio:.9},
  breastcancer:{mean_radius:17.99,mean_texture:10.38,mean_perimeter:122.8,mean_area:1001,mean_smoothness:.1184,mean_compactness:.2776,mean_concavity:.3001,mean_concave_points:.1471,mean_symmetry:.2419,mean_fractal_dimension:.07871,radius_error:1.095,texture_error:.9053,perimeter_error:8.589,area_error:153.4,smoothness_error:.006399,compactness_error:.04904,concavity_error:.05373,concave_points_error:.01587,symmetry_error:.03003,fractal_dimension_error:.006193,worst_radius:25.38,worst_texture:17.33,worst_perimeter:184.6,worst_area:2019,worst_smoothness:.1622,worst_compactness:.6656,worst_concavity:.7119,worst_concave_points:.2654,worst_symmetry:.4601,worst_fractal_dimension:.1189},
  parkinsons:{'MDVP:Fo(Hz)':119.992,'MDVP:Fhi(Hz)':157.302,'MDVP:Flo(Hz)':74.997,'MDVP:Jitter(%)':.00784,'MDVP:Jitter(Abs)':.00007,'MDVP:Rap':.0037,'MDVP:PPQ':.00554,'Jitter:DDP':.01109,'MDVP:Shimmer':.04374,'MDVP:Shimmer(dB)':.426,'Shimmer:APQ3':.02182,'Shimmer:APQ5':.0313,'MDVP:APQ':.02971,'Shimmer:DDA':.06545,NHR:.02211,HNR:21.033,RPDE:.414783,DFA:.815285,spread1:-4.813031,spread2:.266482,D2:2.301442,PPE:.284654}
};

const navDiseases=[['diabetes','Diabetes','brain'],['heartdisease','Heart Disease','heart'],['ckd','Kidney Disease','kidney'],['breastcancer','Breast Cancer','shield'],['parkinsons',"Parkinson's",'brain'],['liver','Liver Disease','liver']];

function icon(name){
  const icons={
    home:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/></svg>',
    brain:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M9 4a3 3 0 0 0-5 2 3.5 3.5 0 0 0 .7 2.1A3.5 3.5 0 0 0 5 15a3 3 0 0 0 4 4"/><path d="M15 4a3 3 0 0 1 5 2 3.5 3.5 0 0 1-.7 2.1A3.5 3.5 0 0 1 19 15a3 3 0 0 1-4 4"/><path d="M9 4v16M15 4v16M9 8h6M9 12h6M9 16h6"/></svg>',
    heart:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M20.8 8.8c0 5.2-8.8 10.2-8.8 10.2S3.2 14 3.2 8.8A4.8 4.8 0 0 1 12 6.2a4.8 4.8 0 0 1 8.8 2.6Z"/></svg>',
    kidney:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M8 4c-3 0-5 3-5 6 0 5 3 9 7 9 2 0 3-2 3-5V9c0-3-2-5-5-5ZM16 4c3 0 5 3 5 6 0 5-3 9-7 9-2 0-3-2-3-5V9c0-3 2-5 5-5Z"/></svg>',
    shield:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3 20 6v6c0 5-3.3 8-8 9-4.7-1-8-4-8-9V6l8-3Z"/><path d="m8.5 12 2.2 2.2 4.8-5"/></svg>',
    liver:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 8c2-3 5-4 9-4 4 0 6 2 7 5-1 7-6 10-12 10-4 0-5-3-4-6 0-2 1-3 0-5Z"/><path d="M4 12h10c3 0 4-1 6-3"/></svg>'
  };
  return icons[name]||'';
}

function initShell(){
  const sidebar=document.querySelector('.sidebar');
  if(!sidebar) return;
  const current=document.body.dataset.disease||'';
  const nav=document.getElementById('disease-nav');
  if(nav){
    nav.innerHTML=`<a href="index.html" class="${!current?'active':''}"><span class="nav-icon">${icon('home')}</span><span>Home</span></a>`+
      navDiseases.map(([key,label,ic])=>`<a href="${key}_form.html" class="${current===key?'active':''}"><span class="nav-icon">${icon(ic)}</span><span>${label}</span></a>`).join('');
  }
  const menu=document.getElementById('mobile-menu');
  const overlay=document.getElementById('drawer-overlay');
  const close=()=>{sidebar.classList.remove('open');overlay?.classList.remove('open');document.body.style.overflow='';};
  menu?.addEventListener('click',()=>{sidebar.classList.add('open');overlay?.classList.add('open');document.body.style.overflow='hidden';});
  overlay?.addEventListener('click',close);
  sidebar.querySelectorAll('a').forEach(a=>a.addEventListener('click',close));
  document.addEventListener('keydown',e=>{if(e.key==='Escape')close();});
  const title=document.getElementById('topbar-title');
  if(title) title.textContent=current?(diseaseMeta[current]?.title||'Prediction workspace'):'Prediction workspace';
  const wake=document.getElementById('wake');
  if(wake){ wake.textContent='● Local prediction engine'; wake.style.color='#7dd3a5'; }
}

function showToast(message,type=''){
  const el=document.getElementById('toast');
  if(!el)return;
  el.textContent=message;
  el.className=`toast show ${type}`;
  clearTimeout(window.__toast);
  window.__toast=setTimeout(()=>el.className='toast',3400);
}

function fieldKey(id){return `predix-${document.body.dataset.disease}-${id}`;}
function saveField(el){try{localStorage.setItem(fieldKey(el.id),el.value)}catch{} }
function restoreFields(){
  document.querySelectorAll('[data-field]').forEach(el=>{
    try{const v=localStorage.getItem(fieldKey(el.id));if(v!==null)el.value=v;}catch{}
  });
  updateProgress();
}
function clearForm(){
  document.querySelectorAll('[data-field]').forEach(el=>{
    el.value='';
    try{localStorage.removeItem(fieldKey(el.id));}catch{}
  });
  updateProgress();
  document.getElementById('result')?.classList.remove('show','success','danger');
  showToast('Form cleared.');
}
function updateProgress(){
  const fields=[...document.querySelectorAll('[data-field]')];
  if(!fields.length)return;
  const filled=fields.filter(x=>x.value!=='').length;
  const pct=Math.round(filled/fields.length*100);
  const bar=document.getElementById('progress-bar');
  const label=document.getElementById('progress-label');
  if(bar)bar.style.width=pct+'%';
  if(label)label.textContent=`${filled}/${fields.length} completed`;
}
function prettyName(s){
  return String(s).replaceAll('_',' ').replaceAll('MDVP:','').replaceAll('worst ','Worst ').replaceAll('mean ','Mean ');
}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));}

function decorateFields(){
  const disease=document.body.dataset.disease;
  const demo=demos[disease]||{};
  document.querySelectorAll('[data-field]').forEach(el=>{
    const label=document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
    if(!label || label.querySelector('.demo-hint'))return;
    const key=el.dataset.field;
    if(Object.hasOwn(demo,key)){
      const hint=document.createElement('span');
      hint.className='hint demo-hint';
      hint.textContent=` · e.g. ${demo[key]}`;
      label.appendChild(hint);
    }
  });
}

function setBusy(busy){
  const btn=document.getElementById('analyze-btn');
  if(!btn)return;
  btn.disabled=busy;
  btn.innerHTML=busy?'<span class="spinner"></span> Analyzing…':'Analyze result';
}

function loadDemo(disease){
  const d=demos[disease]||{};
  document.querySelectorAll('[data-field]').forEach(el=>{
    const key=el.dataset.field;
    if(Object.hasOwn(d,key)){el.value=d[key];saveField(el);}
  });
  updateProgress();
  document.getElementById('result')?.classList.remove('show','success','danger');
  showToast('Demo values loaded. Replace them with the patient data before analyzing.','ok');
}

function buildPayload(meta){
  const payload={};
  const fields=meta.fields||[];
  for(const key of fields){
    const el=document.querySelector(`[data-field="${CSS.escape(key)}"]`);
    if(!el)continue;
    if(el.value==='') return {error:`Please enter a value for ${prettyName(key)}.`};
    const value=Number(el.value);
    if(!Number.isFinite(value)) return {error:`Please enter a valid number for ${prettyName(key)}.`};
    payload[key]=value;
  }
  // The breast-cancer model was trained with spaced feature names, while the UI uses safe HTML ids.
  if(document.body.dataset.disease==='breastcancer'){
    const mapped={};
    Object.entries(payload).forEach(([key,value])=>{mapped[key.replaceAll('_',' ')]=value;});
    return {payload:mapped};
  }
  return {payload};
}

function renderResult(meta,payload){
  const result=document.getElementById('result');
  if(!result)return;
  const prediction=Number(payload.prediction);
  const positive=prediction===1;
  result.classList.remove('success','danger');
  result.classList.add('show',positive?'danger':'success');
  const title=document.getElementById('result-title');
  const subtitle=document.getElementById('result-subtitle');
  if(title)title.textContent=positive?meta.positive:meta.negative;
  if(subtitle)subtitle.textContent=`Model output: ${prediction}`;

  const bars=document.getElementById('bars');
  if(!bars)return;
  const values=Array.isArray(payload.feature_importances)?payload.feature_importances:[];
  const names=Array.isArray(payload.feature_names)?payload.feature_names:meta.fields;
  const rows=names.map((name,i)=>({name,value:Number(values[i])||0})).filter(x=>x.value>0).sort((a,b)=>b.value-a.value).slice(0,8);
  // Use a real percentage scale rather than scaling every bar to the largest
  // feature. This makes the chart visually honest and consistent across models.
  const maxValue=Math.max(...rows.map(x=>x.value),0.000001);
  const maxPercent=Math.max(5,Math.ceil((maxValue*100)/5)*5);
  const midPercent=maxPercent/2;
  bars.innerHTML=rows.length?`
    <div class="chart-scale" aria-hidden="true"><span>0%</span><span>${midPercent.toFixed(0)}%</span><span>${maxPercent.toFixed(0)}%</span></div>
    <div class="importance-rows">
      ${rows.map(x=>`<div class="bar-row"><span class="bar-label" title="${escapeHtml(x.name)}">${escapeHtml(prettyName(x.name))}</span><span class="bar-track"><i class="bar-fill" style="width:${Math.max(1,x.value*100/maxPercent*100)}%"></i></span><span class="bar-value">${(x.value*100).toFixed(1)}%</span></div>`).join('')}
    </div>`:'<div class="bar-label">Feature importance is not available for this model.</div>';
  result.scrollIntoView({behavior:'smooth',block:'nearest'});
}

async function analyze(){
  const disease=document.body.dataset.disease;
  const meta=diseaseMeta[disease];
  if(!meta)return;
  const built=buildPayload(meta);
  if(built.error){showToast(built.error,'error');return;}
  setBusy(true);
  const wake=document.getElementById('wake');
  if(wake){wake.textContent='● Running local model';wake.style.color='#a78bfa';}
  try{
    const response=await fetch(API_BASE+meta.endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(built.payload)});
    const data=await response.json().catch(()=>({}));
    if(!response.ok)throw new Error(data.error||`Prediction failed (${response.status}).`);
    renderResult(meta,data);
    if(wake){wake.textContent='● Local prediction engine';wake.style.color='#7dd3a5';}
  }catch(error){
    if(wake){wake.textContent='● Local API unavailable';wake.style.color='#fb7185';}
    showToast(error.message||'Could not connect to the local prediction engine. Start Python with: python app.py','error');
  }finally{setBusy(false);}
}

function initForms(){
  const disease=document.body.dataset.disease;
  if(!disease||!diseaseMeta[disease])return;
  restoreFields();
  decorateFields();
  document.querySelectorAll('[data-field]').forEach(el=>{
    el.addEventListener('input',()=>{saveField(el);updateProgress();});
    el.addEventListener('change',()=>{saveField(el);updateProgress();});
  });
  document.getElementById('demo-btn')?.addEventListener('click',()=>loadDemo(disease));
  document.getElementById('clear-btn')?.addEventListener('click',clearForm);
  document.getElementById('analyze-btn')?.addEventListener('click',analyze);
  document.querySelectorAll('[data-field]').forEach(el=>el.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();analyze();}}));
}

document.addEventListener('DOMContentLoaded',()=>{
  initShell();
  initForms();
});
