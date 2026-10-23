(function(){
  'use strict';
  const D=JSON.parse(document.getElementById('report-data').textContent);
  window.ReportData=D;
  const byId=id=>document.getElementById(id);
  const groups={
    geometry:{title:'Sample directions, then place each beam in the world.',description:'The scan pattern is discrete. Sensor position and orientation transform the nominal beam grid, so a surface inside the continuous FOV can still fall between channels. Sensor x points forward, y left, and z up.',numbers:[1,2,3,4]},
    visibility:{title:'The nearest opaque intersection controls visibility.',description:'Ray intersection is resolved before the range gate and detector. A weak or too-near foreground object can therefore block a farther target even when the foreground return is not retained.',numbers:[4,5,12]},
    power:{title:'Surface response and atmosphere form the optical echo.',description:'Returned power depends on diffuse/specular response, incidence, inverse-square range loss, and two-way extinction. Nearby subray hits merge into a power-weighted echo centroid; this is an approximation to pulsed ranging.',numbers:[6,7,8,13]},
    uncertainty:{title:'Noisy optical power decides detection; range noise follows.',description:'For an existing echo, Gaussian receiver noise and a power threshold give a detection probability. The range-variance law uses measured SNR after selection. It is an assumed timing-inspired law, not a derived Cramér–Rao bound.',numbers:[9,10,11,14,16,17,18]},
    task:{title:'Use measured points to drive and accumulate a map.',description:'The planar robot follows a known-map reference and reacts to height-filtered LiDAR points. Saved exact poses place measured endpoints in a common frame. Weather comparisons retain all three seed runs in one fixed scene; endpoint coverage is not free-space or accuracy.',numbers:[15,17,19]}
  };
  function equation(record){
    const node=document.createElement('article');node.className='equation-item';
    const heading=document.createElement('div');heading.className='equation-title';
    const label=document.createElement('span');label.textContent=record.caption;
    const number=document.createElement('b');number.textContent='('+record.number+')';heading.append(label,number);
    const math=document.createElement('div');math.className='math-scroll';math.innerHTML=record.mathml;
    node.append(heading,math);return node;
  }
  function selectModel(key){
    const group=groups[key];byId('model-step-title').textContent=group.title;byId('model-step-description').textContent=group.description;
    byId('model-equations').replaceChildren(...D.equations.filter(e=>group.numbers.includes(e.number)).map(equation));
    document.querySelectorAll('[data-model]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.model===key)));
  }
  document.querySelectorAll('[data-model]').forEach(button=>button.addEventListener('click',()=>selectModel(button.dataset.model)));
  byId('all-equations').replaceChildren(...D.equations.map(equation));selectModel('geometry');
  const A=D.threshold.aggregate;
  byId('calibration-rms').textContent=(100*A.calibration_rmse).toFixed(3)+' pp';
  byId('interval-coverage').textContent=A.pointwise_wilson_contains_model+' / '+A.settings;
  byId('range-rms').textContent=(1000*A.conditional_range_rmse_m).toFixed(3)+' mm';
  byId('predicted-range-rms').textContent=(1000*A.predicted_conditional_range_rmse_m).toFixed(3)+' mm predicted';
  byId('provenance-summary').textContent='Equations are transcribed structurally from the saved editable report. '+D.provenance.files.length+' source files are embedded by reference hash; plotted numeric summaries are copied without rounding. Display coordinates alone are rounded to five decimals.';
  const rows=D.provenance.files.map(file=>{const row=document.createElement('tr');for(const value of[file.path,file.sha256]){const cell=document.createElement('td');cell.textContent=value;row.append(cell);}return row;});byId('source-rows').replaceChildren(...rows);
  const chapters=[...document.querySelectorAll('main>.chapter')];
  const activeLink=id=>document.querySelectorAll('.contents a').forEach(link=>{const active=link.getAttribute('href')==='#'+id;link.classList.toggle('active',active);if(active)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');});
  if('IntersectionObserver'in window){const observer=new IntersectionObserver(entries=>{for(const entry of entries)if(entry.isIntersecting)activeLink(entry.target.id);},{rootMargin:'-10% 0px -65% 0px',threshold:0});chapters.forEach(s=>observer.observe(s));}
  function showError(message){const box=byId('viewer-error');box.textContent=message;box.hidden=false;}
  const chartPromise=Promise.resolve().then(()=>window.ReportCharts.init(D)).catch(error=>{showError('A chart could not be displayed: '+error.message);console.error(error);});
  const threePromise=Promise.resolve().then(()=>window.Report3D.init(D)).catch(error=>{byId('fov-fallback').hidden=false;byId('map-fallback').hidden=false;byId('fov-plot').hidden=true;byId('map-plot').hidden=true;showError('Interactive 3D is unavailable in this browser. The saved report figures are displayed instead.');console.error(error);});
  window.addEventListener('resize',()=>{if(window.ReportCharts)window.ReportCharts.resize();if(window.Report3D)window.Report3D.resize();});
  document.querySelectorAll('details').forEach(details=>details.addEventListener('toggle',()=>{if(details.open){if(window.ReportCharts)window.ReportCharts.resize();if(window.Report3D)window.Report3D.resize();}}));
  Promise.all([chartPromise,threePromise]).then(()=>{document.body.dataset.ready='true';});
})();
