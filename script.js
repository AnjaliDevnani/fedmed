let chartInst = null, liveChartInst = null, monitorInterval = null;
let monitorRound = 0, monitorRunning = false;
const TOTAL_ROUNDS = 10;
const globalAccHistory = [71, 75, 78, 81, 84, 86, 88, 90.1, 91.4, 92.3];
const lossHistory = [62, 55, 49, 43, 38, 33, 28, 24, 21, 19];
const hospitalData = [
  { id:'A', name:'City General', acc:87.4, samples:1247, status:'training' },
  { id:'B', name:"St. Mary's Med", acc:84.1, samples:2104, status:'active' },
  { id:'C', name:'Apollo Research', acc:89.2, samples:891, status:'training' }
];

function navigate(view) {
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
  const activeBtn = document.querySelector(`[data-view="${view}"]`);
  if(activeBtn) activeBtn.classList.add('active');
  
  if (chartInst) { chartInst.destroy(); chartInst = null; }
  if (liveChartInst) { liveChartInst.destroy(); liveChartInst = null; }
  if (view !== 'monitor' && monitorInterval) { clearTimeout(monitorInterval); }
  
  const tpl = document.getElementById(`view-${view}`);
  document.getElementById('main-content').innerHTML = tpl ? tpl.innerHTML : '';
  
  if (view === 'dashboard') initDashboard();
  if (view === 'upload') initUpload();
  if (view === 'monitor') initMonitor();
}

function initDashboard() {
  setTimeout(() => {
    const ctx = document.getElementById('dash-chart');
    if(!ctx) return;
    chartInst = new Chart(ctx, {
      type: 'line',
      data: {
        labels: ['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10'],
        datasets: [
          { label: 'Global Accuracy', data: globalAccHistory, borderColor: '#00c9a7', backgroundColor: 'rgba(0,201,167,0.08)', fill: true, tension: 0.4, pointRadius: 3 },
          { label: 'Val Loss ×100', data: lossHistory, borderColor: '#ff6b6b', backgroundColor: 'rgba(255,107,107,0.06)', fill: true, tension: 0.4, pointRadius: 3 }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { backgroundColor: '#1a2032', titleFont: { family: 'Syne' }, bodyFont: { family: 'DM Mono' } } },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,.05)' }, ticks: { color: '#525870', font: {size: 11} } },
          y: { grid: { color: 'rgba(255,255,255,.05)' }, min: 0, max: 100, ticks: { color: '#525870', font: {size: 11} } }
        }
      }
    });
  }, 0);
}

function initUpload() {}

function triggerFileSelect() { document.getElementById('fileUpload').click(); }
function handleDrop(e) {
  e.preventDefault(); e.target.closest('.dropzone').classList.remove('dragover');
  if(e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
}
function handleFile(file) {
  if(!file) return;
  document.getElementById('upload-preview').style.display = 'block';
  document.getElementById('filename').innerText = file.name;
  document.getElementById('filesize').innerText = (file.size / 1024 / 1024).toFixed(2) + ' MB';
  let ext = file.name.split('.').pop().toLowerCase();
  document.getElementById('file-icon').innerText = (['csv','xlsx'].includes(ext)) ? '📊' : '🔬';
}
function clearFile() {
  document.getElementById('fileUpload').value = '';
  document.getElementById('upload-preview').style.display = 'none';
}

function simulateUpload() {
  const pBar = document.getElementById('upload-progress');
  if(!pBar) return;
  pBar.style.display = 'block'; document.getElementById('upload-success').style.display = 'none';
  const steps = [
    { pct: 30, text: "Encrypting locally…" }, { pct: 60, text: "Applying differential noise…" },
    { pct: 85, text: "Preparing gradient update…" }, { pct: 100, text: "Queuing for FL round…" }
  ];
  let delay = 0;
  steps.forEach((step, i) => {
    setTimeout(() => {
      document.getElementById('upload-progress-fill').style.width = step.pct + '%';
      document.getElementById('upload-progress-text').innerText = step.text;
      document.getElementById('upload-progress-pct').innerText = step.pct + '%';
      if(i === steps.length - 1) setTimeout(() => { pBar.style.display = 'none'; document.getElementById('upload-success').style.display = 'block'; }, 700);
    }, delay);
    delay += 700;
  });
}

async function runInference() {
  const fileInput = document.getElementById('fileUpload');
  if (!fileInput.files.length) {
    alert('Please select a file first.');
    return;
  }
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('file', file);
  
  const resDiv = document.getElementById('upload-result');
  resDiv.style.display = 'block';
  resDiv.innerHTML = '<div style="padding:14px; text-align:center;">Running inference, please wait...</div>';
  
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      body: formData
    });
    const result = await response.json();
    
    if (result.error) {
      resDiv.innerHTML = `<div style="padding:14px; color:var(--coral);">Error: ${result.error}</div>`;
      return;
    }
    
    resDiv.innerHTML = `
      <div class="text3" style="margin-bottom:6px;">Diagnosis Result</div>
      <div class="syne" style="color:${result.diagnosis === 'Normal' ? 'var(--teal)' : 'var(--coral)'}; font-size:18px; font-weight:700; margin-bottom:4px;">${result.diagnosis}</div>
      <div style="font-size:12px; color:var(--text3); margin-bottom:10px;">Global FL Model · Round 10</div>
      <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px; color:var(--text2);"><span>Confidence</span><span class="mono">${result.confidence}%</span></div>
      <div class="progress-wrap" style="height:6px; margin-bottom:12px; border-radius:4px;"><div class="progress-bar" style="width:${result.confidence}%; background:linear-gradient(90deg, #00c9a7, #4f8ef7);"></div></div>
      <div style="font-size:10px; color:var(--text3); line-height:1.4;">🔒 Inference performed using aggregated global model weights. Guarantees: ${result.privacy_guarantee || 'ε=2.0'}</div>
    `;
  } catch(e) {
    resDiv.innerHTML = `<div style="padding:14px; color:var(--coral);">Server Error. Ensure backend is running.</div>`;
  }
}

function initMonitor() {
  updateRoundUI();
  setTimeout(() => {
    const ctx = document.getElementById('live-chart');
    if(!ctx) return;
    liveChartInst = new Chart(ctx, {
      type: 'line',
      data: { labels: [], datasets: [ { label: 'Global Acc', data: [], borderColor: '#00c9a7', fill:false, tension:0.4 }, { label: 'Val Loss (x100)', data: [], borderColor: '#ff6b6b', fill:false, tension:0.4 } ] },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { backgroundColor: '#1a2032', titleFont: { family: 'Syne' }, bodyFont: { family: 'DM Mono' } } },
        scales: { x: { grid: { color: 'rgba(255,255,255,.05)' }, ticks: {color:'#525870', font:{size:11}} }, y: { grid: { color: 'rgba(255,255,255,.05)' }, min: 0, max: 100, ticks: {color:'#525870', font:{size:11}} } }
      }
    });
    for(let i=0; i<monitorRound; i++) {
        liveChartInst.data.labels.push(`R${i+1}`);
        liveChartInst.data.datasets[0].data.push(globalAccHistory[i]);
        liveChartInst.data.datasets[1].data.push(lossHistory[i]);
    }
    liveChartInst.update();
  }, 0);
}

function toggleTraining() {
  monitorRunning = !monitorRunning;
  const btn = document.getElementById('start-btn');
  if (monitorRunning) {
    btn.innerHTML = '⏸ Pause';
    if(monitorRound >= TOTAL_ROUNDS) setCompleteState(); else startMonitorLoop();
  } else { btn.innerHTML = '▶ Start Training'; }
}

function startMonitorLoop() {
  const phases = [
    { label:'Local Training', class:'train', duration:1800, color:'chip-amber' },
    { label:'Uploading Gradients', class:'send', duration:1200, color:'chip-blue' },
    { label:'FedAvg Aggregation', class:'agg', duration:1000, color:'chip-purple' },
    { label:'Broadcasting Model', class:'broadcast', duration:1000, color:'chip-blue' }
  ];
  let phaseIdx = 0;
  
  const runPhase = () => {
    if (!monitorRunning || monitorRound >= TOTAL_ROUNDS) return;
    const ph = phases[phaseIdx];
    
    document.getElementById('monitor-phase-chip').innerText = ph.label;
    document.getElementById('monitor-phase-chip').className = 'chip ' + ph.color;

    document.querySelectorAll('.fl-packet').forEach(el => { el.style.opacity = '0'; el.className = 'fl-packet'; });
    const hospitals = document.querySelectorAll('.node-hospital');
    const serverNode = document.getElementById('server-node');
    hospitals.forEach(h => h.className = 'node-circle node-hospital idle');
    serverNode.className = 'node-circle node-server';
    
    if(ph.label === 'Local Training') {
       hospitals.forEach(h => h.className = 'node-circle node-hospital train');
       document.querySelectorAll('[id^=stat-h]').forEach(el => el.innerText = "Training…");
       document.getElementById('stat-server').innerText = "Waiting…";
    }
    else if(ph.label === 'Uploading Gradients') {
       hospitals.forEach(h => h.className = 'node-circle node-hospital send');
       document.querySelectorAll('[id^=stat-h]').forEach(el => el.innerText = "Uploading…");
       document.getElementById('stat-server').innerText = "Receiving…";
       document.getElementById('pack-a').classList.add('packet-anim-fwd', 'packet-teal');
       document.getElementById('pack-c').classList.add('packet-anim-rev', 'packet-teal');
       document.getElementById('pack-b').classList.add('packet-anim-vert-rev', 'packet-teal');
    }
    else if(ph.label === 'FedAvg Aggregation') {
       serverNode.className = 'node-circle node-server agg';
       document.querySelectorAll('[id^=stat-h]').forEach(el => el.innerText = "Idle");
       document.getElementById('stat-server').innerText = "Aggregating…";
    }
    else if(ph.label === 'Broadcasting Model') {
       hospitals.forEach(h => h.className = 'node-circle node-hospital send');
       document.querySelectorAll('[id^=stat-h]').forEach(el => el.innerText = "Receiving");
       document.getElementById('stat-server').innerText = "Broadcasting";
       document.getElementById('pack-a').classList.add('packet-anim-rev', 'packet-blue');
       document.getElementById('pack-c').classList.add('packet-anim-fwd', 'packet-blue');
       document.getElementById('pack-b').classList.add('packet-anim-vert-fwd', 'packet-blue');
    }
    
    phaseIdx++;
    if (phaseIdx >= phases.length) {
      setTimeout(() => {
        if(!monitorRunning) return;
        phaseIdx = 0; monitorRound++; updateRoundUI();
        if (monitorRound >= TOTAL_ROUNDS) { setCompleteState(); return; }
        runPhase();
      }, ph.duration);
    } else { monitorInterval = setTimeout(runPhase, ph.duration); }
  };
  runPhase();
}

function updateRoundUI() {
  const lbl = document.getElementById('round-label');
  if(lbl) lbl.innerText = `Round ${monitorRound}/10`;
  document.querySelectorAll('.round-block').forEach((el, idx) => {
    el.style.background = idx < monitorRound ? 'var(--teal)' : (idx === monitorRound ? 'var(--amber)' : 'var(--border)');
    if(idx === monitorRound) el.style.boxShadow = '0 0 8px var(--amber)';
    else el.style.boxShadow = 'none';
  });

  if(liveChartInst && monitorRound > 0 && monitorRound <= TOTAL_ROUNDS) {
    const rIdx = monitorRound - 1; 
    liveChartInst.data.labels.push(`R${monitorRound}`);
    liveChartInst.data.datasets[0].data.push(globalAccHistory[rIdx]);
    liveChartInst.data.datasets[1].data.push(lossHistory[rIdx]);
    liveChartInst.update();
    
    const gVal = document.getElementById('global-acc-val');
    if(gVal) gVal.innerText = globalAccHistory[rIdx] + '%';
    const gBar = document.getElementById('global-bar');
    if(gBar) gBar.style.width = globalAccHistory[rIdx] + '%';
    
    hospitalData.forEach(h => {
       const jitter = (Math.random() * 4 - 2);
       const val = (globalAccHistory[rIdx] + jitter).toFixed(1);
       const accEl = document.getElementById(`client-acc-${h.id}`);
       if(accEl) accEl.innerText = val + '%';
       const barEl = document.getElementById(`client-bar-${h.id}`);
       if(barEl) barEl.style.width = val + '%';
    });
  }
}

function setCompleteState() {
  monitorRunning = false;
  document.getElementById('start-btn').innerHTML = '▶ Finished';
  document.getElementById('start-btn').disabled = true;
  document.getElementById('stat-server').innerText = "Model Optimized";
  document.getElementById('monitor-phase-chip').innerText = "Training Complete";
  document.getElementById('monitor-phase-chip').className = "chip chip-teal";
  document.querySelectorAll('.node-hospital').forEach(h => h.className = 'node-circle node-hospital idle');
}

function resetMonitor() {
  clearTimeout(monitorInterval);
  monitorRunning = false; monitorRound = 0;
  document.getElementById('start-btn').innerHTML = '▶ Start Training';
  document.getElementById('start-btn').disabled = false;
  if(liveChartInst) {
    liveChartInst.data.labels = []; liveChartInst.data.datasets.forEach(d => d.data=[]); liveChartInst.update();
  }
  updateRoundUI();
  document.getElementById('monitor-phase-chip').innerText = "Ready";
  document.getElementById('monitor-phase-chip').className = "chip chip-gray";
}

window.onload = () => navigate('dashboard');
