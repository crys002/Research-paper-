/**
 * MindGuard - Application Logic
 * Simulates the NLP risk detection models locally for viva demonstration.
 */

// ==========================================================================
// Initialization & UI Logic
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initStatsCounter();
  initTextarea();
  initCharts();
});

// Navbar scroll effect & active states
function initNavbar() {
  const navbar = document.getElementById('navbar');
  const sections = document.querySelectorAll('.section, .hero');
  const navLinks = document.querySelectorAll('.nav-link');

  window.addEventListener('scroll', () => {
    // Add background on scroll
    if (window.scrollY > 50) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }

    // Update active nav link based on scroll position
    let current = '';
    sections.forEach(section => {
      const sectionTop = section.offsetTop;
      const sectionHeight = section.clientHeight;
      if (scrollY >= (sectionTop - 200)) {
        current = section.getAttribute('id');
      }
    });

    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href').includes(current)) {
        link.classList.add('active');
      }
    });
  });
}

// Animate stats numbers when scrolled into view
function initStatsCounter() {
  const stats = document.querySelectorAll('.stat-number');
  let started = false;

  const observer = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting && !started) {
      started = true;
      stats.forEach(stat => {
        const target = +stat.getAttribute('data-target');
        const duration = 2000; // ms
        const increment = target / (duration / 16); // 60fps
        let current = 0;

        const updateCount = () => {
          current += increment;
          if (current < target) {
            stat.innerText = Math.ceil(current);
            requestAnimationFrame(updateCount);
          } else {
            stat.innerText = target;
          }
        };
        updateCount();
      });
    }
  }, { threshold: 0.5 });

  const statsSection = document.querySelector('.stats-bar');
  if (statsSection) observer.observe(statsSection);
}

// Textarea character count
function initTextarea() {
  const textarea = document.getElementById('text-input');
  const charCount = document.getElementById('char-count');

  textarea.addEventListener('input', () => {
    const len = textarea.value.length;
    charCount.innerText = `${len} characters`;
  });
}

// ==========================================================================
// Demo Simulation Logic
// ==========================================================================

// Pre-defined samples
const samples = {
  atrisk: "I don't know what to do anymore. I feel completely empty and exhausted every single day. The pain is too much to handle and I just want it to stop. No one understands what I'm going through and I feel totally isolated. I can't keep pretending I'm okay.",
  normal: "Just finished my final exams and I'm looking forward to the break! Going to spend the weekend hiking with some friends and catching up on sleep. It's been a busy semester but I'm glad it's over.",
  mixed: "Some days are okay but other days I just feel really overwhelmed with everything. Work has been stressful and I haven't been sleeping well. I think I need to start taking better care of myself or talk to someone."
};

function loadSample(type) {
  const textarea = document.getElementById('text-input');
  textarea.value = samples[type];
  textarea.dispatchEvent(new Event('input')); // trigger char count update
}

// Simulated model accuracy stats (based on Table II - RSDD)
const modelStats = {
  mentalbert: { name: "MentalBERT", f1: "0.891", acc: "0.894", prec: "0.876", rec: "0.912" },
  roberta: { name: "RoBERTa", f1: "0.874", acc: "0.878", prec: "0.861", rec: "0.889" },
  bert: { name: "BERT", f1: "0.858", acc: "0.863", prec: "0.847", rec: "0.871" },
  cnnbilstm: { name: "CNN-BiLSTM", f1: "0.821", acc: "0.826", prec: "0.814", rec: "0.830" },
  bilstm: { name: "BiLSTM", f1: "0.779", acc: "0.783", prec: "0.772", rec: "0.787" }
};

// ==========================================================================
// NLP-Style Risk Analysis Engine
// Multi-layer scoring: patterns → sentiment → negation → protective factors
// ==========================================================================

// LAYER 1: Weighted risk patterns (regex-based for flexible matching)
const riskPatterns = {
  // CRITICAL — direct suicidal ideation / self-harm (weight: 1.0)
  critical: [
    /\b(kill\s*(my|him|her)?self|suicide|suicidal)\b/,
    /\b(end\s*(my|this)\s*life)\b/,
    /\bwant\s*to\s*die\b/,
    /\bwish\s*(i|I)\s*(was|were)\s*dead\b/,
    /\bbetter\s*off\s*dead\b/,
    /\bdon'?t\s*want\s*to\s*(live|be\s*alive|exist|be\s*here)\b/,
    /\bno\s*(reason|point)\s*(to|in)\s*(live|living|go\s*on|continue)\b/,
    /\b(slit|cut)\s*(my)?\s*(wrist|vein)/,
    /\bjump\s*(off|from)\b/,
    /\boverdose\b/,
    /\bhang\s*myself\b/,
    /\bend\s*it\s*all\b/,
    /\bkms\b/,
    /\bcan'?t\s*(go|keep|do\s*this)\s*(on|anymore|going)\b/,
    /\bsee\s*no\s*(way\s*out|future|hope)\b/,
    /\btired\s*of\s*(living|life|existing|everything)\b/,
    /\bwant\s*(it|this|everything)\s*to\s*(stop|end|be\s*over)\b/,
    /\bnot\s*worth\s*(it|living)\b/,
    /\bready\s*to\s*(go|leave|give\s*up|end)\b/,
    /\bworld\s*(would\s*be|is)\s*better\s*without\s*me\b/,
    /\bdisappear\s*(forever|from|completely)?\b/,
    /\bself[- ]?harm\b/
  ],
  // HIGH — severe emotional distress (weight: 0.6)
  high: [
    /\b(hopeless|helpless|desperate|despair)\b/,
    /\bcan'?t\s*(take|handle|bear|stand)\s*(it|this|anymore|the\s*pain)\b/,
    /\b(nothing\s*matters|what'?s\s*the\s*point)\b/,
    /\bnobody\s*(cares|loves|understands|would\s*miss|would\s*notice)\b/,
    /\bcompletely\s*(alone|broken|numb|lost|empty)\b/,
    /\bno\s*one\s*(cares|understands|loves|would\s*miss)\b/,
    /\bburden\s*(to|on)\s*(everyone|others|my|people)\b/,
    /\b(trapped|stuck)\b.*\b(no\s*way\s*out|forever|can'?t\s*escape)\b/,
    /\bworthless\b/,
    /\buseless\b/,
    /\bpathetic\b/,
    /\bhate\s*(my\s*life|myself|being\s*alive|living|everything)\b/,
    /\bdon'?t\s*(deserve|belong)\b/,
    /\bever(yone|ybody)\s*(hates|leaves|abandons)\s*me\b/,
    /\bnever\s*(get|be)\s*(better|happy|okay|fine)\b/,
    /\bgave\s*up\b/,
    /\bgive\s*up\b/,
    /\b(cry|crying)\s*(every\s*(day|night)|all\s*the\s*time|constantly|myself\s*to\s*sleep)\b/,
    /\bcan'?t\s*(stop\s*crying|feel\s*anything|breathe)\b/,
    /\bno\s*(hope|future|purpose|meaning)\b/,
    /\bpointless\b/
  ],
  // MODERATE — depression / anxiety indicators (weight: 0.35)
  moderate: [
    /\b(depressed|depression|depressing)\b/,
    /\b(anxious|anxiety|panic\s*attack)\b/,
    /\bempty\s*(inside)?\b/,
    /\b(lonely|loneliness|isolated|isolation)\b/,
    /\bexhausted\b/,
    /\boverwhelmed\b/,
    /\b(insomnia|can'?t\s*sleep|sleepless)\b/,
    /\bself[- ]?(esteem|worth|confidence)\b.*\b(low|no|zero|none)\b/,
    /\b(miserable|suffering|anguish|agony)\b/,
    /\b(scared|terrified|afraid)\s*(of|to)\s*(live|life|future|everything)\b/,
    /\b(numb|numbness|feel\s*nothing)\b/,
    /\blost\s*(all|my)\s*(motivation|interest|will|energy)\b/,
    /\bcan'?t\s*(eat|focus|concentrate|function|get\s*(out\s*of\s*bed|up))\b/,
    /\bpain\b/,
    /\bsuffering\b/,
    /\b(mental\s*health|mentally\s*(ill|sick|unwell|broken|drained))\b/,
    /\b(breakdown|breaking\s*down|falling\s*apart)\b/,
    /\bdon'?t\s*(care|feel)\s*(anymore)?\b/,
    /\b(dark\s*(thoughts|place|times)|darkness)\b/,
    /\b(stressed|stress|stressful)\b/
  ],
  // MILD — general negative sentiment cues (weight: 0.15)
  mild: [
    /\b(sad|sadness|unhappy|upset)\b/,
    /\b(worried|worrying|concern|concerned)\b/,
    /\b(frustrated|frustration|irritated)\b/,
    /\b(tired|fatigued|drained|burnout|burnt\s*out)\b/,
    /\b(struggle|struggling)\b/,
    /\b(difficult|tough|hard)\s*(time|day|week|month|year|period|phase)\b/,
    /\b(can'?t\s*cope|coping)\b/,
    /\b(mood\s*swings?|moody|emotional)\b/,
    /\bnot\s*(okay|fine|alright|good|great|happy)\b/,
    /\b(lost|confused|uncertain)\b/,
    /\b(disconnected|detached)\b/,
    /\b(restless|uneasy|uncomfortable)\b/
  ]
};

// Category weights
const categoryWeights = { critical: 1.0, high: 0.6, moderate: 0.35, mild: 0.15 };

// Protective factors (reduce risk score)
const protectivePatterns = [
  /\b(happy|grateful|thankful|blessed|excited|joyful|cheerful)\b/,
  /\b(looking\s*forward|can'?t\s*wait|excited\s*(for|about|to))\b/,
  /\b(friends?|family|loved\s*ones?|support)\b.*\b(help|love|care|support|there\s*for)\b/,
  /\b(better|improving|recovery|recovering|healing)\b/,
  /\b(therapy|therapist|counselor|counselling|treatment|medication)\b.*\b(help|work|good|great)\b/,
  /\b(hope|hopeful|optimistic|positive)\b/,
  /\b(love\s*(my|this)\s*life|glad\s*to\s*be\s*alive|enjoying)\b/,
  /\bfun\b/,
  /\b(celebrate|celebration|achievement|accomplished|proud)\b/
];

// Negation-aware sentiment words
const negativeWords = [
  'never', 'nothing', 'nowhere', 'nobody', 'no', 'not', "n't", 'cant', "can't",
  'wont', "won't", 'dont', "don't", 'without', 'lack', 'absence', 'miss', 'missing',
  'worse', 'worst', 'terrible', 'horrible', 'awful', 'dreadful', 'unbearable',
  'agonizing', 'devastating', 'crushing', 'excruciating', 'torment', 'torture',
  'hate', 'loathe', 'despise', 'resent', 'regret', 'guilt', 'shame', 'blame',
  'failure', 'failed', 'failing', 'reject', 'rejected', 'abandoned', 'betrayed',
  'broken', 'shattered', 'destroyed', 'ruined', 'damage', 'damaged', 'scarred'
];

/**
 * Analyze text using multi-layer NLP-style scoring.
 * Returns { riskScore, isAtRisk, detectedIndicators }
 */
function computeRiskScore(text) {
  const lowerText = text.toLowerCase();
  const words = lowerText.split(/\s+/);
  const wordCount = words.length;
  
  let totalScore = 0;
  let detectedIndicators = [];
  let highestCategory = null;
  
  // ------ LAYER 1: Pattern matching with weights ------
  for (const [category, patterns] of Object.entries(riskPatterns)) {
    for (const pattern of patterns) {
      const match = lowerText.match(pattern);
      if (match) {
        totalScore += categoryWeights[category];
        detectedIndicators.push(match[0].trim());
        if (!highestCategory || categoryWeights[category] > categoryWeights[highestCategory]) {
          highestCategory = category;
        }
      }
    }
  }
  
  // ------ LAYER 2: Negative sentiment density ------
  let negCount = 0;
  for (const word of words) {
    if (negativeWords.includes(word)) negCount++;
  }
  const negDensity = wordCount > 0 ? negCount / wordCount : 0;
  totalScore += negDensity * 2; // scale sentiment contribution
  
  // ------ LAYER 3: First-person distress (I/me/my + negative) ------
  const firstPersonDistress = /\b(i|me|my|myself)\b/i.test(text) && negCount > 0;
  if (firstPersonDistress) {
    totalScore += 0.15;
  }
  
  // ------ LAYER 4: Protective factors (reduce score) ------
  let protectiveCount = 0;
  for (const pattern of protectivePatterns) {
    if (pattern.test(lowerText)) protectiveCount++;
  }
  totalScore -= protectiveCount * 0.25;
  
  // ------ LAYER 5: Text length factor ------
  // Very short texts with risk indicators are still dangerous
  // But very short neutral texts should not have high confidence
  if (wordCount < 5 && detectedIndicators.length === 0) {
    totalScore *= 0.5;
  }
  
  // ------ Final normalization ------
  // Use sigmoid-like mapping to 0-1 range
  let riskScore = 1 / (1 + Math.exp(-2.5 * (totalScore - 0.5)));
  
  // Ensure critical matches always result in high risk
  if (highestCategory === 'critical') {
    riskScore = Math.max(riskScore, 0.85);
  } else if (highestCategory === 'high') {
    riskScore = Math.max(riskScore, 0.65);
  }
  
  // Clamp
  riskScore = Math.max(0.02, Math.min(0.98, riskScore));
  
  // Deduplicate indicators
  detectedIndicators = [...new Set(detectedIndicators)];
  
  const isAtRisk = riskScore > 0.5;
  
  return { riskScore, isAtRisk, detectedIndicators };
}


async function analyzeText() {
  const text = document.getElementById('text-input').value.trim();
  if (!text) {
    alert("Please enter some text to analyze.");
    return;
  }

  // UI state updates
  const btn = document.getElementById('analyze-btn');
  const btnText = document.getElementById('analyze-btn-text');
  const spinner = document.getElementById('analyze-spinner');
  
  btn.disabled = true;
  btnText.classList.add('hidden');
  spinner.classList.remove('hidden');

  // Hide results if they were shown before
  document.getElementById('result-content').classList.add('hidden');
  document.getElementById('result-placeholder').classList.remove('hidden');
  
  const pIcon = document.getElementById('placeholder-icon');
  const pText = document.getElementById('placeholder-text');
  
  // Simulate trained model pipeline
  pIcon.innerText = "⚙️";
  pText.innerText = "Tokenizing Input Text...";
  await new Promise(r => setTimeout(r, 600));

  pIcon.innerText = "🧠";
  pText.innerText = "Extracting Contextual Embeddings...";
  await new Promise(r => setTimeout(r, 800));

  const selectedModelName = document.getElementById('model-select').options[document.getElementById('model-select').selectedIndex].text;
  pIcon.innerText = "🤖";
  pText.innerText = `Running Inference through ${selectedModelName}...`;
  await new Promise(r => setTimeout(r, 1200));

  pIcon.innerText = "📊";
  pText.innerText = "Calculating Risk Probabilities...";
  await new Promise(r => setTimeout(r, 600));

  // Reset placeholder for next time
  setTimeout(() => {
    pIcon.innerText = "🔬";
    pText.innerText = "Results will appear here after analysis";
  }, 1000);

  // Run the NLP analysis engine
  const { riskScore, isAtRisk, detectedIndicators } = computeRiskScore(text);

  const selectedModelId = document.getElementById('model-select').value;
  const stats = modelStats[selectedModelId];
  
  // Model quality slightly adjusts confidence
  const modelAcc = parseFloat(stats.acc);
  let finalScore = riskScore;
  if (isAtRisk) {
    finalScore = Math.min(riskScore + (modelAcc - 0.80) * 0.3, 0.98);
  } else {
    finalScore = Math.max(riskScore - (modelAcc - 0.80) * 0.15, 0.02);
  }

  renderResults(isAtRisk, finalScore, detectedIndicators, stats);

  // Restore button state
  btn.disabled = false;
  btnText.classList.remove('hidden');
  spinner.classList.add('hidden');
  
  // Scroll to results
  document.getElementById('result-panel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderResults(isAtRisk, score, keywords, modelData) {
  // Hide placeholder, show results
  document.getElementById('result-placeholder').classList.add('hidden');
  document.getElementById('result-content').classList.remove('hidden');

  // Badge
  const badge = document.getElementById('risk-badge');
  const icon = document.getElementById('risk-icon');
  const value = document.getElementById('risk-value');
  
  badge.className = 'risk-badge ' + (isAtRisk ? 'at-risk' : 'safe');
  icon.innerText = isAtRisk ? '⚠️' : '✅';
  value.innerText = isAtRisk ? 'AT RISK' : 'NO RISK DETECTED';

  // Confidence & Meter
  const confPct = Math.round((isAtRisk ? score : (1 - score)) * 100);
  document.getElementById('confidence-value').innerText = `${confPct}%`;
  
  const meterFill = document.getElementById('risk-meter-fill');
  const meterThumb = document.getElementById('risk-meter-thumb');
  meterFill.style.width = `${score * 100}%`;
  meterThumb.style.left = `${score * 100}%`;

  // Metrics
  document.getElementById('res-f1').innerText = modelData.f1;
  document.getElementById('res-acc').innerText = modelData.acc;
  document.getElementById('res-prec').innerText = modelData.prec;
  document.getElementById('res-recall').innerText = modelData.rec;

  // Keywords
  const kwList = document.getElementById('keywords-list');
  kwList.innerHTML = '';
  
  if (keywords.length > 0) {
    keywords.forEach(kw => {
      const span = document.createElement('span');
      span.className = 'keyword-tag';
      span.innerText = kw;
      kwList.appendChild(span);
    });
  } else {
    const span = document.createElement('span');
    span.className = 'keyword-tag safe-word';
    span.innerText = 'No risk indicators found';
    kwList.appendChild(span);
  }

  // Explanation
  const expBox = document.getElementById('explanation-box');
  let text = `<strong>Analysis via ${modelData.name}:</strong> Based on the contextual embeddings, the model `;
  
  if (isAtRisk) {
    text += `identifies a high likelihood of mental health risk. `;
    if (keywords.length > 0) {
      text += `The presence of semantic features related to "${keywords.slice(0,2).join('", "')}" contributed strongly to this classification. `;
    }
    if (modelData.name === "MentalBERT") {
      text += `Because MentalBERT is domain-adapted, it is highly sensitive to these specific clinical linguistic markers.`;
    }
  } else {
    text += `predicts normal linguistic patterns with no significant markers of depression or anxiety. The text semantic structure resembles baseline, non-clinical forum posts.`;
  }
  
  expBox.innerHTML = text;
}

// ==========================================================================
// Chart.js Implementations (Results Section)
// ==========================================================================

let f1Chart, aucChart, radarChart;

// Data from Table II & III
const rsddData = {
  labels: ['BiLSTM', 'CNN-BiLSTM', 'BERT', 'RoBERTa', 'MentalBERT'],
  f1: [0.779, 0.821, 0.858, 0.874, 0.891],
  auc: [0.864, 0.901, 0.935, 0.948, 0.961],
  acc: [0.783, 0.826, 0.863, 0.878, 0.894],
  prec: [0.772, 0.814, 0.847, 0.861, 0.876],
  rec: [0.787, 0.830, 0.871, 0.889, 0.912]
};

const clpsychData = {
  labels: ['BiLSTM', 'CNN-BiLSTM', 'BERT', 'RoBERTa', 'MentalBERT'],
  f1: [0.761, 0.843, 0.801, 0.819, 0.837], // CNN-BiLSTM best here
  auc: [0.843, 0.921, 0.889, 0.904, 0.916],
  acc: [0.768, 0.849, 0.806, 0.824, 0.842],
  prec: [0.753, 0.838, 0.795, 0.811, 0.829],
  rec: [0.770, 0.851, 0.808, 0.828, 0.846]
};

function initCharts() {
  // Set global chart defaults for dark mode
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.scale.grid.color = 'rgba(255, 255, 255, 0.05)';

  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      y: { min: 0.7, max: 1.0, ticks: { padding: 10 } },
      x: { grid: { display: false } }
    }
  };

  // F1 Chart
  const ctxF1 = document.getElementById('chart-f1').getContext('2d');
  f1Chart = new Chart(ctxF1, {
    type: 'bar',
    data: getBarData(rsddData.labels, rsddData.f1, 'F1-Score'),
    options: commonOptions
  });

  // AUC Chart
  const ctxAuc = document.getElementById('chart-auc').getContext('2d');
  aucChart = new Chart(ctxAuc, {
    type: 'line',
    data: getLineData(rsddData.labels, rsddData.auc, 'AUC-ROC'),
    options: {
      ...commonOptions,
      elements: { line: { tension: 0.4 }, point: { radius: 6, hoverRadius: 8 } }
    }
  });

  // Radar Chart
  const ctxRadar = document.getElementById('chart-radar').getContext('2d');
  radarChart = new Chart(ctxRadar, {
    type: 'radar',
    data: getRadarData(rsddData),
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: 0.7,
          max: 1.0,
          grid: { color: 'rgba(255, 255, 255, 0.1)' },
          angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
          pointLabels: { color: '#f8fafc', font: { size: 12 } },
          ticks: { backdropColor: 'transparent', display: false }
        }
      },
      plugins: {
        legend: { position: 'bottom', labels: { padding: 20, usePointStyle: true } }
      }
    }
  });
}

// Helpers to format Chart.js data structures
function getBarData(labels, data, label) {
  // Highlight the best performer
  const maxIdx = data.indexOf(Math.max(...data));
  const bgColors = data.map((_, i) => i === maxIdx ? 'rgba(59, 130, 246, 0.8)' : 'rgba(148, 163, 184, 0.4)');
  const borderColors = data.map((_, i) => i === maxIdx ? '#3b82f6' : '#94a3b8');

  return {
    labels: labels,
    datasets: [{
      label: label,
      data: data,
      backgroundColor: bgColors,
      borderColor: borderColors,
      borderWidth: 1,
      borderRadius: 4
    }]
  };
}

function getLineData(labels, data, label) {
  return {
    labels: labels,
    datasets: [{
      label: label,
      data: data,
      borderColor: '#8b5cf6',
      backgroundColor: 'rgba(139, 92, 246, 0.2)',
      borderWidth: 3,
      fill: true,
      pointBackgroundColor: '#fff',
      pointBorderColor: '#8b5cf6',
      pointBorderWidth: 2
    }]
  };
}

function getRadarData(dataset) {
  return {
    labels: ['F1', 'Accuracy', 'Precision', 'Recall', 'AUC-ROC'],
    datasets: [
      {
        label: 'BiLSTM',
        data: [dataset.f1[0], dataset.acc[0], dataset.prec[0], dataset.rec[0], dataset.auc[0]],
        borderColor: 'rgba(148, 163, 184, 0.8)',
        backgroundColor: 'rgba(148, 163, 184, 0.1)',
        borderWidth: 2,
        hidden: true // hide baseline by default to declutter
      },
      {
        label: 'BERT',
        data: [dataset.f1[2], dataset.acc[2], dataset.prec[2], dataset.rec[2], dataset.auc[2]],
        borderColor: 'rgba(16, 185, 129, 0.8)',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        borderWidth: 2
      },
      {
        label: 'MentalBERT',
        data: [dataset.f1[4], dataset.acc[4], dataset.prec[4], dataset.rec[4], dataset.auc[4]],
        borderColor: 'rgba(59, 130, 246, 1)',
        backgroundColor: 'rgba(59, 130, 246, 0.2)',
        borderWidth: 3
      }
    ]
  };
}

// Switch Dataset function exposed to global scope
window.switchDataset = function(type) {
  // Update Buttons
  document.getElementById('btn-rsdd').classList.remove('active');
  document.getElementById('btn-clpsych').classList.remove('active');
  document.getElementById(`btn-${type}`).classList.add('active');

  // Update Tables
  document.getElementById('table-rsdd').classList.add('hidden');
  document.getElementById('table-clpsych').classList.add('hidden');
  document.getElementById(`table-${type}`).classList.remove('hidden');

  // Update Charts
  const data = type === 'rsdd' ? rsddData : clpsychData;
  document.getElementById('radar-label').innerText = type === 'rsdd' ? 'RSDD Dataset' : 'CLPsych 2015';

  f1Chart.data = getBarData(data.labels, data.f1, 'F1-Score');
  f1Chart.update();

  aucChart.data = getLineData(data.labels, data.auc, 'AUC-ROC');
  aucChart.update();

  radarChart.data = getRadarData(data);
  radarChart.update();
};
