/* static/js/scraper.js */

// ── Hover card position config ────────────────────────────────────────────────
// ANCHOR: which corner of the card pins to the row
//   "bottom-left"  — card appears below-left of row  (default)
//   "bottom-right" — card appears below-right of row
//   "top-left"     — card appears above-left of row
//   "top-right"    — card appears above-right of row
// CORNER: which screen corner to pin the card to (overrides ANCHOR if set)
//   null           — use ANCHOR relative to row
//   "top-left"     — always top-left of viewport
//   "top-right"    — always top-right of viewport
//   "bottom-left"  — always bottom-left of viewport
//   "bottom-right" — always bottom-right of viewport
// SHAPE: "wide" (680px default) | "square" (420×420px) | "narrow" (360px)
const HOVER_CARD_ANCHOR = "top-right";
const HOVER_CARD_CORNER = null;
const HOVER_CARD_SHAPE  = "wide";
// ─────────────────────────────────────────────────────────────────────────────

function applyHoverCardConfig() {
  const root = document.documentElement;
  const shapes = {
    wide:   { width: "680px", height: "auto",  overflow: "auto" },
    square: { width: "420px", height: "420px", overflow: "auto" },
    narrow: { width: "360px", height: "auto",  overflow: "auto" },
  };
  const shape = shapes[HOVER_CARD_SHAPE] || shapes.wide;
  root.style.setProperty("--hc-width",    shape.width);
  root.style.setProperty("--hc-height",   shape.height);
  root.style.setProperty("--hc-overflow", shape.overflow);
}

// ── Single floating card appended to <body> ───────────────────────────────────
let _hcEl = null;

function _getCard() {
  if (!_hcEl) {
    _hcEl = document.createElement("div");
    _hcEl.className = "hl-hover-card";
    document.body.appendChild(_hcEl);
  }
  return _hcEl;
}

function showHoverCard(row, html) {
  const card = _getCard();
  card.innerHTML = html;
  card.classList.add("hc-visible");
  _positionCard(card, row);
}

function hideHoverCard() {
  if (_hcEl) _hcEl.classList.remove("hc-visible");
}

function _positionCard(card, row) {
  const W = card.offsetWidth  || parseInt(getComputedStyle(document.documentElement).getPropertyValue("--hc-width")) || 680;
  const H = card.offsetHeight || 300;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const PAD = 16;

  if (HOVER_CARD_CORNER) {
    // Fixed corner of viewport
    const isTop   = HOVER_CARD_CORNER.includes("top");
    const isRight = HOVER_CARD_CORNER.includes("right");
    card.style.top    = isTop    ? PAD + "px" : "auto";
    card.style.bottom = isTop    ? "auto"     : PAD + "px";
    card.style.left   = isRight  ? "auto"     : PAD + "px";
    card.style.right  = isRight  ? PAD + "px" : "auto";
  } else {
    // Position relative to the hovered row
    const rect = row.getBoundingClientRect();
    const isTop   = HOVER_CARD_ANCHOR.startsWith("top");
    const isRight = HOVER_CARD_ANCHOR.endsWith("right");

    let top  = isTop  ? rect.top  - H - 8 : rect.bottom + 8;
    let left = isRight ? rect.right - W    : rect.left;

    // Clamp to viewport
    top  = Math.max(PAD, Math.min(top,  vh - H - PAD));
    left = Math.max(PAD, Math.min(left, vw - W - PAD));

    card.style.top   = top  + "px";
    card.style.left  = left + "px";
    card.style.right = "auto";
    card.style.bottom = "auto";
  }
}

// Wire mouseenter/mouseleave to all hl-rows (delegated on document)
document.addEventListener("mouseover", e => {
  const row = e.target.closest("tr.hl-row");
  if (!row) return;
  const inner = row.querySelector(".hl-hover-card");
  if (!inner) return;
  showHoverCard(row, inner.innerHTML);
});
document.addEventListener("mouseout", e => {
  const row = e.target.closest("tr.hl-row");
  if (row && !row.contains(e.relatedTarget)) hideHoverCard();
});

// Apply on load
document.addEventListener("DOMContentLoaded", applyHoverCardConfig);

let currentData = null;
let currentContent = '';
let currentView = 'rendered';
let digResults = [];
let acTimer = null;          // autocomplete debounce

// ── Input handling ────────────────────────────────────────────────────────────
document.getElementById('urlInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') { closeAutocomplete(); startScrape(); }
});
document.getElementById('filterInput')?.addEventListener('keydown', e => {
  if (e.key === 'Enter') startScrape();
});
document.addEventListener('click', e => {
  if (!e.target.closest('.url-autocomplete-wrap')) closeAutocomplete();
});

function togglePill(el) { el.classList.toggle('active'); }
function toggleAdv() { document.getElementById('advPanel').classList.toggle('open'); }
function getStrategies() {
  return [...document.querySelectorAll('.pill.active')].map(p => p.dataset.strategy);
}

// ── Autocomplete ──────────────────────────────────────────────────────────────
async function onUrlInput(forceShow = false) {
  clearTimeout(acTimer);
  const delay = forceShow ? 0 : 200;
  acTimer = setTimeout(async () => {
    const raw = document.getElementById('urlInput').value;
    // Complete the last segment (after last comma/semicolon)
    const parts = raw.split(/[,;]/);
    const q = parts[parts.length - 1].trim();
    const list = document.getElementById('autocompleteList');
    // On focus with empty field, show recent history; otherwise filter by query
    try {
      const r = await fetch(`/api/url-history?q=${encodeURIComponent(q)}&limit=8`);
      const items = await r.json();
      if (!items.length) { closeAutocomplete(); return; }
      const label = q ? '' : '<li class="ac-header">Recent URLs</li>';
      list.innerHTML = label + items.map(item => `
        <li class="ac-item" data-url="${esc(item.url)}"
            onmousedown="event.preventDefault(); selectAC(this.dataset.url)">
          <span class="ac-url">${esc(item.url)}</span>
          ${item.title ? `<span class="ac-title">${esc(item.title)}</span>` : ''}
          <span class="ac-meta">${item.page_type ? item.page_type + ' · ' : ''}used ${item.scrape_count}×</span>
        </li>`).join('');
      list.style.display = 'block';
    } catch(e) { closeAutocomplete(); }
  }, delay);
}

function onUrlKey(e) {
  const list = document.getElementById('autocompleteList');
  const items = list.querySelectorAll('.ac-item');
  const active = list.querySelector('.ac-item.ac-active');
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    const next = active ? active.nextElementSibling : items[0];
    if (next) { active?.classList.remove('ac-active'); next.classList.add('ac-active'); }
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    const prev = active ? active.previousElementSibling : items[items.length-1];
    if (prev) { active?.classList.remove('ac-active'); prev.classList.add('ac-active'); }
  } else if (e.key === 'Enter' && active) {
    e.preventDefault(); selectAC(active.dataset.url || active.querySelector('.ac-url')?.textContent);
  } else if (e.key === 'Escape') { closeAutocomplete(); }
}

function selectAC(url) {
  const input = document.getElementById('urlInput');
  const parts = input.value.split(/[,;]/);
  parts[parts.length - 1] = ' ' + url;
  input.value = parts.join(',').replace(/^,/, '').trim();
  closeAutocomplete();
  input.focus();
}

function closeAutocomplete() {
  const list = document.getElementById('autocompleteList');
  if (list) list.style.display = 'none';
}

// ── Scrape ────────────────────────────────────────────────────────────────────
async function startScrape() {
  const urlRaw = document.getElementById('urlInput').value.trim();
  if (!urlRaw) return;
  closeAutocomplete();

  const filterPhrase = document.getElementById('filterInput')?.value.trim() || '';
  const strategies = getStrategies();
  if (!strategies.length) { alert('Select at least one strategy.'); return; }

  // Count URLs
  const urlCount = urlRaw.split(/[,;]+/).filter(u => u.trim()).length;
  const isMulti = urlCount > 1;
  const btn = document.getElementById('extractBtn');
  const hasPW = strategies.includes('playwright');
  btn.disabled = true;
  btn.textContent = isMulti ? `Extracting ${urlCount} URLs…` : 'Extracting…';

  // In multi-URL mode, Playwright is auto-excluded server-side (too CPU-heavy in parallel)
  const effectiveHasPW = hasPW && !isMulti;

  document.getElementById('results').innerHTML = `
    <div class="loader">
      <div class="spinner-wrap">
        <div class="spinner"></div>${effectiveHasPW ? '<div class="spinner2"></div>' : ''}
      </div>
      <div class="loader-text">${isMulti
        ? `Scraping ${urlCount} URLs in parallel (static strategies only)…`
        : effectiveHasPW ? `Launching headless Chrome + ${strategies.length} strategies…`
                         : `Running ${strategies.length} strategies in parallel…`}
      </div>
      ${isMulti && hasPW ? `<div class="loader-note">⚡ Playwright auto-disabled for multi-URL mode — scrape individually for JS-heavy sites</div>` : ''}
    </div>`;

  try {
    const waitSecs = parseFloat(document.getElementById('waitSeconds').value) || 4;
    const waitSel = document.getElementById('waitSelector').value.trim() || null;

    const resp = await fetch('/api/scrape', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        url: urlRaw,
        strategies,
        wait_for_selector: waitSel,
        wait_seconds: waitSecs,
      }),
    });
    if (!resp.ok) throw new Error((await resp.json()).detail || `HTTP ${resp.status}`);

    currentData = await resp.json();

    if (currentData.multi_results?.length > 1) {
      renderMultiResults(currentData);
    } else {
      renderResult(currentData);
    }

    // Async two-phase filter after results rendered
    if (filterPhrase) {
      if (currentData.multi_results?.length > 1) {
        // Multi-URL: filter each card separately
        currentData.multi_results.forEach((r, cardIdx) => {
          if (r.headlines?.length) {
            applyAsyncFilterToCard(filterPhrase, r.headlines, cardIdx);
          }
        });
      } else if (currentData.headlines?.length) {
        // Single URL
        applyAsyncFilter(filterPhrase, currentData.headlines, currentData.page_type, currentData.tables_md);
      }
    }
  } catch (e) {
    document.getElementById('results').innerHTML = `
      <div class="error-box">
        <div style="font-size:1.3rem">⚠️</div>
        <div><div class="error-title">Extraction failed</div>
          <div class="error-msg">${esc(e.message)}</div></div>
      </div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Extract →';
  }
}

// ── Multi-URL results renderer ────────────────────────────────────────────────
function renderMultiResults(data) {
  const resultsEl = document.getElementById('results');
  const filterBadge = data.filter_applied
    ? `<span class="chip filter-chip">🔍 filtered: "${esc(data.filter_phrase)}"</span>` : '';

  const cards = data.multi_results.map((r, i) => {
    if (r.error) return `
      <div class="multi-card error-card">
        <div class="multi-card-url">${esc(r.url)}</div>
        <div class="dig-error">✗ ${esc(r.error)}</div>
      </div>`;

    const pt = r.page_type || 'unknown';
    const headlines = r.headlines || [];
    const hlCount = headlines.length;

    return `
      <div class="multi-card" id="mcard-${i}">
        <div class="multi-card-header">
          <div>
            <div class="multi-card-title">${esc(r.title || r.url)}</div>
            <div class="multi-card-url"><a href="${esc(r.url)}" target="_blank">${esc(r.url)}</a></div>
          </div>
          <div class="multi-card-meta">
            <span class="page-type-badge ${pt}">${pt}</span>
            <span class="chip words">${(r.word_count||0).toLocaleString()} words</span>
            ${filterBadge}
          </div>
        </div>
        ${hlCount > 0 ? `
        <div class="multi-card-body">
          <div class="section-label" id="multiLabel-${i}" style="padding:12px 20px 6px">
            ${hlCount} headline${hlCount!==1?'s':''} found
          </div>
          <div class="headlines-table-wrap" style="margin:0 16px 16px">
            <table class="headlines-table">
              <thead><tr>
                <th class="col-check"><input type="checkbox" onchange="toggleAllMulti(this,${i})"></th>
                <th>Headline</th><th>Section</th>
              </tr></thead>
              <tbody id="multiTbody-${i}">
                ${headlines.map((h,j) => `
                  <tr data-idx="${j}" class="hl-row">
                    <td class="col-check">
                      <input type="checkbox" class="hl-check-multi" data-url="${esc(h.url)}" data-title="${esc(h.title||'')}">
                    </td>
                    <td class="hl-title">
                      <a href="${esc(h.url)}" target="_blank">${esc(h.title)}</a>
                      <div class="hl-hover-card">
                        <div class="hl-hc-section">${esc(h.section || '')}</div>
                        <div class="hl-hc-title">
                          <a href="${esc(h.url)}" target="_blank">${esc(h.title)}</a>
                        </div>
                        ${h.summary ? `<div class="hl-hc-summary">${esc(h.summary)}</div>` : ''}
                        <div class="hl-hc-url">${esc(h.url)}</div>
                      </div>
                    </td>
                    <td class="hl-section">${esc(h.section||'—')}</td>
                  </tr>`).join('')}
              </tbody>
            </table>
          </div>
        </div>` : r.content ? `
        <div class="multi-card-body">
          <div class="md-body" style="margin:12px 16px;max-height:200px">${mdToHtml((r.content||'').slice(0,1200))}</div>
        </div>` : ''}
      </div>`;
  }).join('');

  resultsEl.innerHTML = `
    <div class="multi-results-wrap">
      <div class="multi-results-header">
        <div class="multi-results-title">
          ${data.multi_results.length} URLs scraped
          ${data.filter_applied ? `· <span style="color:var(--accent2)">filtered by "${esc(data.filter_phrase)}"</span>` : ''}
        </div>
        <button class="btn-primary" onclick="digAllSelected()">Dig Selected →</button>
      </div>
      ${cards}
    </div>`;
}

function toggleAllMulti(masterCb, cardIdx) {
  document.querySelectorAll(`#mcard-${cardIdx} .hl-check-multi`)
    .forEach(cb => cb.checked = masterCb.checked);
}

async function digAllSelected() {
  const checked = [...document.querySelectorAll('.hl-check-multi:checked')];
  if (!checked.length) { alert('Select at least one headline.'); return; }
  const urls = [...new Set(checked.map(cb => cb.dataset.url))];

  // Inject into digResults and use existing dig flow
  const btn = document.querySelector('.multi-results-header .btn-primary');
  btn.disabled = true; btn.textContent = `Fetching ${urls.length}…`;

  try {
    const resp = await fetch('/api/dig', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ urls, strategies: ['trafilatura','newspaper3k','readability','goose3','beautifulsoup'] }),
    });
    digResults = await resp.json();

    // Show dig results panel below
    const wrap = document.querySelector('.multi-results-wrap');
    const tmpl = document.getElementById('tmpl-dig-results');
    const node = tmpl.content.cloneNode(true);
    wrap.appendChild(node);
    renderDigResults(digResults);
  } catch(e) { alert('Dig failed: ' + e.message); }
  finally { btn.disabled = false; btn.textContent = 'Dig Selected →'; }
}

// ── Route to correct renderer ─────────────────────────────────────────────────
function renderResult(data) {
  const pt = data.page_type || 'unknown';
  if (pt === 'homepage') renderHomepage(data);
  else if (pt === 'data') renderData(data);
  else if (pt === 'mixed') renderMixed(data);
  else renderArticle(data);

  // Inject GenAI insight banner if present (works for all page types)
  if (data.filter_insight && data.filter_insight.trim()) {
    const card = document.getElementById('result-card');
    if (card) {
      const banner = document.createElement('div');
      banner.className = 'insight-banner';
      banner.innerHTML = `
        <div class="insight-icon">🤖</div>
        <div class="insight-body">
          <div class="insight-label">AI Insight</div>
          <div class="insight-text">${esc(data.filter_insight)}</div>
        </div>`;
      // Insert after scoreboard
      const scoreboard = card.querySelector('.scoreboard');
      if (scoreboard) scoreboard.after(banner);
      else card.querySelector('.result-header').after(banner);
    }
  }
}

// ── Homepage renderer ─────────────────────────────────────────────────────────
function renderHomepage(data) {
  const tmpl = document.getElementById('tmpl-homepage');
  const node = tmpl.content.cloneNode(true);
  const resultsEl = document.getElementById('results');
  resultsEl.innerHTML = '';
  resultsEl.appendChild(node);

  document.getElementById('res-title').textContent = data.title || data.url;
  document.getElementById('res-meta').innerHTML = buildMetaChips(data);
  document.getElementById('scoreboard').innerHTML = buildScoreboard(data.all_results, data.best_strategy);
  document.getElementById('timing-bar').innerHTML = buildTimingBar(data);

  // Update headline toolbar label to reflect filter state
  const toolbarLabel = document.querySelector('.headlines-toolbar .section-label');
  if (toolbarLabel) {
    const count = (data.headlines || []).length;
    if (data.filter_applied && data.filter_phrase) {
      toolbarLabel.innerHTML = `📰 <strong>${count}</strong> headline${count!==1?'s':''} matched <em>"${esc(data.filter_phrase)}"</em>`;
    } else {
      toolbarLabel.textContent = '📰 Headlines — select articles to dig deeper';
    }
  }

  const tbody = document.getElementById('headlinesTbody');
  if (!data.headlines || !data.headlines.length) {
    const msg = data.filter_applied
      ? `No headlines matched "${esc(data.filter_phrase || '')}". Try a broader phrase.`
      : 'No headlines detected. Try scraping a specific article URL.';
    tbody.innerHTML = `<tr><td colspan="4" style="padding:20px;color:var(--muted);text-align:center">${msg}</td></tr>`;
    return;
  }

  tbody.innerHTML = data.headlines.map((h, i) => `
    <tr id="hlrow-${i}" data-idx="${i}" class="hl-row">
      <td class="col-check">
        <input type="checkbox" class="hl-check" data-idx="${i}"
          onchange="onCheckChange()">
      </td>
      <td class="hl-title">
        <a href="${esc(h.url)}" target="_blank">${esc(h.title)}</a>
        <div class="hl-hover-card">
          <div class="hl-hc-section">${esc(h.section || '')}</div>
          <div class="hl-hc-title">
            <a href="${esc(h.url)}" target="_blank">${esc(h.title)}</a>
          </div>
          ${h.summary ? `<div class="hl-hc-summary">${esc(h.summary)}</div>` : ''}
          <div class="hl-hc-url">${esc(h.url)}</div>
        </div>
      </td>
      <td class="hl-section">${esc(h.section || '—')}</td>
      <td class="hl-summary">${esc(h.summary || '')}</td>
    </tr>`).join('');
}

function onCheckChange() {
  const checked = document.querySelectorAll('.hl-check:checked');
  document.getElementById('digBtn').disabled = checked.length === 0;
  document.querySelectorAll('#headlinesTbody tr').forEach(tr => {
    const cb = tr.querySelector('.hl-check');
    tr.classList.toggle('selected', cb && cb.checked);
  });
}

function selectAll() {
  document.querySelectorAll('.hl-check').forEach(cb => cb.checked = true);
  onCheckChange();
}

function selectNone() {
  document.querySelectorAll('.hl-check').forEach(cb => cb.checked = false);
  onCheckChange();
}

function toggleAll(masterCb) {
  document.querySelectorAll('.hl-check').forEach(cb => cb.checked = masterCb.checked);
  onCheckChange();
}

// ── Dig into selected articles ────────────────────────────────────────────────
async function digSelected() {
  if (!currentData || !currentData.headlines) return;
  const checked = [...document.querySelectorAll('.hl-check:checked')];
  const urls = checked.map(cb => currentData.headlines[parseInt(cb.dataset.idx)].url);
  if (!urls.length) return;

  const btn = document.getElementById('digBtn');
  btn.disabled = true; btn.textContent = `Fetching ${urls.length} article(s)…`;

  // Show loading below headlines
  let digPanel = document.getElementById('dig-results-panel');
  if (!digPanel) {
    const tmpl = document.getElementById('tmpl-dig-results');
    const node = tmpl.content.cloneNode(true);
    document.getElementById('result-card').appendChild(node);
    digPanel = document.getElementById('dig-results-panel');
  }
  document.getElementById('dig-articles').innerHTML = `
    <div class="loader" style="padding:28px 0">
      <div class="spinner-wrap"><div class="spinner"></div></div>
      <div class="loader-text">Scraping ${urls.length} article(s)…</div>
    </div>`;

  try {
    const strategies = ['trafilatura','newspaper3k','readability','goose3','beautifulsoup'];
    const resp = await fetch('/api/dig', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ urls, strategies }),
    });
    if (!resp.ok) throw new Error((await resp.json()).detail);
    digResults = await resp.json();
    renderDigResults(digResults);
  } catch (e) {
    document.getElementById('dig-articles').innerHTML =
      `<div class="dig-article"><div class="dig-error">✗ ${esc(e.message)}</div></div>`;
  } finally {
    btn.disabled = false; btn.textContent = 'Dig into Selected →';
  }
}

function renderDigResults(articles) {
  const container = document.getElementById('dig-articles');
  container.innerHTML = '';

  articles.forEach((a, i) => {
    const wrap = document.createElement('div');
    wrap.className = 'dig-article';
    wrap.id = `dig-art-${i}`;

    if (a.error) {
      wrap.innerHTML = `
        <div class="dig-article-header">
          <div class="dig-article-title">
            <a href="${esc(a.url)}" target="_blank">${esc(a.url)}</a>
          </div>
        </div>
        <div class="dig-error">✗ ${esc(a.error)}</div>`;
      container.appendChild(wrap);
      return;
    }

    // Header
    const header = document.createElement('div');
    header.className = 'dig-article-header';
    header.innerHTML = `
      <div class="dig-article-title">
        <a href="${esc(a.url)}" target="_blank">${esc(a.title || a.url)}</a>
      </div>
      <div class="dig-article-meta">
        <span class="chip">${esc(a.best_strategy || '—')}</span>
        <span class="chip words">${(a.word_count||0).toLocaleString()} words</span>
        <span class="chip">${esc(a.metadata?.domain || '')}</span>
      </div>`;
    wrap.appendChild(header);

    // Content — rendered via renderContent (HTML-aware Markdown renderer)
    const contentEl = document.createElement('div');
    contentEl.className = 'dig-article-content';
    contentEl.id = `dig-content-${i}`;

    // Preview: first 800 chars rendered, collapsed
    const preview = (a.content || '').slice(0, 800);
    contentEl.innerHTML = renderContent(preview);
    wrap.appendChild(contentEl);

    // Expand/collapse button
    const btn = document.createElement('button');
    btn.className = 'dig-expand-btn';
    btn.textContent = 'Read full article ↓';
    btn.onclick = () => expandDig(i, btn);
    wrap.appendChild(btn);

    container.appendChild(wrap);
  });
}

function expandDig(i, btn) {
  const el = document.getElementById(`dig-content-${i}`);
  if (el.classList.contains('expanded')) {
    // Collapse back to preview
    const preview = (digResults[i].content || '').slice(0, 800);
    el.innerHTML = renderContent(preview);
    el.classList.remove('expanded');
    btn.textContent = 'Read full article ↓';
  } else {
    // Render full content
    el.innerHTML = renderContent(digResults[i].content || '');
    el.classList.add('expanded');
    btn.textContent = 'Collapse ↑';
  }
}


// ── Async two-phase filter ───────────────────────────────────────────────────────

let _filterController = null;

async function applyAsyncFilterToCard(phrase, headlines, cardIdx) {
  const tbody = document.getElementById(`multiTbody-${cardIdx}`);
  const label = document.getElementById(`multiLabel-${cardIdx}`);
  if (!tbody) return;

  // Phase 1: instant regex dim on card
  const SYNS = {
    declining:['drop','fall','fell','slide','down','weak','tumble','plunge','red','lower'],
    rising:   ['rise','rose','gain','rally','up','surge','jump','climb','green','higher'],
    asian:    ['asia','nikkei','sensex','hang seng','kospi','shanghai','india','japan','china'],
    european: ['europe','ftse','dax','cac','uk','germany','france'],
    us:       ['wall street','dow','nasdaq','nyse'],
    market:   ['stock','index','indices','equity','shares'],
  };
  function expandT(t) { return [t, ...(SYNS[t] || [])]; }
  function rScore(item) {
    const tokens = phrase.toLowerCase().split(' ').filter(Boolean);
    const text = (item.title+' '+item.url+' '+(item.section||'')).toLowerCase();
    const hits = tokens.filter(t => expandT(t).some(s => text.includes(s))).length;
    return tokens.length ? hits / tokens.length : 0;
  }

  const rows = tbody.querySelectorAll('tr[data-idx]');
  const p1map = {};
  rows.forEach(row => {
    const idx = parseInt(row.dataset.idx);
    const s = rScore(headlines[idx] || {});
    p1map[headlines[idx]?.url] = { _matched: s > 0, _score: s };
    row.style.display = s === 0 ? 'none' : '';
    row.classList.toggle('filter-highlight', s > 0);
  });
  const p1count = Object.values(p1map).filter(x => x._matched).length;
  if (label) label.innerHTML = `<strong>${p1count}</strong> of ${headlines.length} matched <em>"${esc(phrase)}"</em> <span class="intent-badge regex-badge">regex</span>`;

  // Phase 2: LLM scoring
  try {
    const controller = new AbortController();
    const resp = await fetch('/api/filter', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({ headlines, phrase, page_type: 'homepage' }),
    });
    if (!resp.ok) throw new Error('filter failed');
    const result = await resp.json();
    const scoreMap = {};
    result.scored.forEach(item => { scoreMap[item.url] = item; });

    rows.forEach(row => {
      const idx = parseInt(row.dataset.idx);
      const scored = scoreMap[headlines[idx]?.url];
      row.classList.remove('filter-highlight');
      if (scored?._matched) {
        row.style.display = '';
        row.classList.add('filter-highlight');
        const tc = row.querySelector('.hl-title');
        if (tc && !tc.querySelector('.score-badge')) {
          const b = document.createElement('span');
          b.className = 'score-badge';
          b.textContent = Math.round((scored._score||0)*10)+'/10';
          tc.appendChild(b);
        }
      } else {
        row.style.display = 'none';
      }
    });

    const matchCount = result.scored.filter(x => x._matched).length;
    if (label) {
      label.innerHTML = `<strong>${matchCount}</strong> of ${headlines.length} matched ` +
        `<em>"${esc(phrase)}"</em> <span class="intent-badge">${esc(result.intent_type||'ai')}</span> ` +
        `<button class="filter-toggle-btn" onclick="toggleCardFilter(this,'multiTbody-${cardIdx}')">Show all</button>`;
    }
  } catch(e) {
    if (label) label.innerHTML = `~<strong>${p1count}</strong> of ${headlines.length} matched ` +
      `<em>"${esc(phrase)}"</em> <span class="intent-badge regex-badge">regex</span>`;
  }
}

function toggleCardFilter(btn, tbodyId) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody) return;
  const showAll = btn.textContent.trim().startsWith('Show all');
  tbody.querySelectorAll('tr[data-idx]').forEach(row => {
    if (showAll) {
      row.style.display = '';
      if (!row.classList.contains('filter-highlight')) row.classList.add('filter-dim');
    } else {
      if (!row.classList.contains('filter-highlight')) row.style.display = 'none';
      row.classList.remove('filter-dim');
    }
  });
  btn.textContent = showAll ? 'Show matched only' : 'Show all';
}

async function applyAsyncFilter(phrase, headlines, pageType, tablesMd) {
  if (_filterController) _filterController.abort();
  _filterController = new AbortController();

  const tbody = document.getElementById('headlinesTbody');
  const toolbarLabel = document.querySelector('.headlines-toolbar .section-label');
  if (!tbody) return;

  if (toolbarLabel) {
    toolbarLabel.innerHTML = `📰 Filtering <span class="filter-spinner">⟳</span> <em>"${esc(phrase)}"</em>…`;
  }

  // ── Phase 1: instant client-side regex dim ────────────────────────────────
  const SYNS = {
    declining:['drop','fall','fell','slide','down','weak','tumble','plunge','red','lower'],
    rising:   ['rise','rose','gain','rally','up','surge','jump','climb','green','higher'],
    asian:    ['asia','nikkei','sensex','hang seng','kospi','shanghai','india','japan','china'],
    european: ['europe','ftse','dax','cac','uk','germany','france'],
    us:       ['wall street','dow','nasdaq','nyse'],
    market:   ['stock','index','indices','equity','shares'],
  };
  function expandTok(t) { return [t, ...(SYNS[t] || [])]; }
  function regexScore(item) {
    const tokens = phrase.toLowerCase().split(' ').filter(Boolean);
    const text = (item.title+' '+item.url+' '+(item.section||'')+' '+(item.summary||'')).toLowerCase();
    const hits = tokens.filter(t => expandTok(t).some(s => text.includes(s))).length;
    return tokens.length ? hits / tokens.length : 0;
  }

  let showAllMode = false;  // toggle between filtered and all

  function applyVisibility(rows, scoreMap) {
    const rowArr = [...rows];
    // Sort: matched first by score, unmatched last
    rowArr.sort((a, b) => {
      const ia = scoreMap[headlines[parseInt(a.dataset.idx)]?.url];
      const ib = scoreMap[headlines[parseInt(b.dataset.idx)]?.url];
      const sa = ia?._matched ? (ia._score||0) : -1;
      const sb = ib?._matched ? (ib._score||0) : -1;
      return sb - sa;
    });
    let matchCount = 0;
    rowArr.forEach(row => {
      const idx = parseInt(row.dataset.idx);
      const scored = scoreMap[headlines[idx]?.url];
      const matched = scored?._matched ?? true;
      if (matched) matchCount++;
      // Hide non-matching rows unless showAll mode
      row.style.display = (!matched && !showAllMode) ? 'none' : '';
      row.classList.toggle('filter-highlight', matched);
      row.classList.remove('filter-dim');
      // Score badge
      if (matched && scored?._score != null) {
        const tc = row.querySelector('.hl-title');
        let badge = tc?.querySelector('.score-badge');
        if (!badge && tc) {
          badge = document.createElement('span');
          badge.className = 'score-badge';
          tc.appendChild(badge);
        }
        if (badge) badge.textContent = Math.round((scored._score||0)*10)+'/10';
      }
      tbody.appendChild(row);
    });
    return matchCount;
  }

  function updateToolbar(matchCount, total, intentType, phase) {
    if (!toolbarLabel) return;
    const phaseLabel = phase === 'ai'
      ? `<span class="intent-badge">${esc(intentType||'ai')}</span>`
      : `<span class="intent-badge regex-badge">regex</span>`;
    const toggleBtn = `<button class="filter-toggle-btn" onclick="toggleFilterAll()">${showAllMode ? 'Show matched only' : 'Show all '+total}</button>`;
    toolbarLabel.innerHTML =
      `📰 <strong>${matchCount}</strong> of ${total} matched <em>"${esc(phrase)}"</em> ${phaseLabel} ${toggleBtn}`;
  }

  window.toggleFilterAll = function() {
    showAllMode = !showAllMode;
    tbody.querySelectorAll('tr[data-idx]').forEach(row => {
      const isHidden = row.style.display === 'none';
      if (showAllMode) {
        row.style.display = '';
        if (!row.classList.contains('filter-highlight'))
          row.classList.add('filter-dim');
      } else {
        if (!row.classList.contains('filter-highlight'))
          row.style.display = 'none';
        row.classList.remove('filter-dim');
      }
    });
    // Refresh toggle button text
    const btn = toolbarLabel?.querySelector('.filter-toggle-btn');
    if (btn) btn.textContent = showAllMode
      ? 'Show matched only'
      : 'Show all ' + headlines.length;
  };

  // ── Phase 1: instant regex (client-side) ──────────────────────────────────
  const rows = tbody.querySelectorAll('tr[data-idx]');
  const phase1Map = {};
  rows.forEach(row => {
    const idx = parseInt(row.dataset.idx);
    const s = regexScore(headlines[idx] || {});
    phase1Map[headlines[idx]?.url] = { _matched: s > 0, _score: s };
  });
  const p1count = applyVisibility(rows, phase1Map);
  updateToolbar(p1count, headlines.length, 'regex', 'regex');

  // ── Phase 2: LLM batch scoring ────────────────────────────────────────────
  try {
    console.log('[WebPulse Filter] Calling /api/filter with', headlines.length, 'headlines');
    const resp = await fetch('/api/filter', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: _filterController.signal,
      body: JSON.stringify({ headlines, phrase, page_type: pageType||'homepage', tables_md: tablesMd||null }),
    });
    console.log('[WebPulse Filter] /api/filter response status:', resp.status);
    if (!resp.ok) throw new Error('filter failed');
    const result = await resp.json();

    const scoreMap = {};
    result.scored.forEach(item => { scoreMap[item.url] = item; });

    // Clear old badges before re-sort
    rows.forEach(row => {
      const b = row.querySelector('.score-badge');
      if (b) b.remove();
      row.classList.remove('filter-highlight','filter-dim');
    });

    const matchCount = applyVisibility(rows, scoreMap);
    updateToolbar(matchCount, headlines.length, result.intent_type, 'ai');

    if (result.insight) {
      const card = document.getElementById('result-card');
      if (card && !card.querySelector('.insight-banner')) {
        const banner = document.createElement('div');
        banner.className = 'insight-banner';
        banner.innerHTML = `<div class="insight-icon">🤖</div>
          <div class="insight-body">
            <div class="insight-label">AI Insight — ${esc(result.intent_type)}</div>
            <div class="insight-text">${esc(result.insight)}</div>
          </div>`;
        const sb = card.querySelector('.scoreboard');
        if (sb) sb.after(banner);
      }
    }

  } catch (e) {
    if (e.name === 'AbortError') return;
    // Phase 1 stays — just update label to remove spinner
    updateToolbar(p1count, headlines.length, '', 'regex');
  }
}

// ── Ingest for Q&A ────────────────────────────────────────────────────────────
async function ingestForQA() {
  const docs = digResults
    .filter(a => a.content && !a.error)
    .map(a => ({ url: a.url, title: a.title || a.url, content: a.content, page_type: 'article' }));

  if (!docs.length) { alert('No valid articles to ingest.'); return; }

  const btn = document.getElementById('ingestBtn');
  btn.disabled = true; btn.textContent = 'Ingesting…';

  try {
    const resp = await fetch('/api/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ documents: docs }),
    });
    if (!resp.ok) throw new Error((await resp.json()).detail);
    const ctx = await resp.json();
    window.location.href = `/qa?session_id=${ctx.session_id}`;
  } catch (e) {
    alert('Ingest failed: ' + e.message);
    btn.disabled = false; btn.textContent = 'Ingest All & Ask AI →';
  }
}

// ── Mixed renderer (tables + headlines) ──────────────────────────────────────
function renderMixed(data) {
  // Render data tables first, then headlines below
  renderData(data);

  // After data card is rendered, append headlines section
  const card = document.getElementById('result-card');
  if (!card || !data.headlines?.length) return;

  // Update page type badge
  const badge = card.querySelector('.page-type-badge');
  if (badge) {
    badge.textContent = '📊📰 Mixed';
    badge.className = 'page-type-badge mixed';
  }

  // Build headlines section
  const hlSection = document.createElement('div');
  hlSection.className = 'headlines-section';
  hlSection.innerHTML = `
    <div class="headlines-toolbar">
      <span class="section-label" id="mixedHlLabel">
        📰 ${data.headlines.length} News Headlines
      </span>
      <div class="headlines-actions">
        <button class="btn-sm" onclick="selectAll()">Select All</button>
        <button class="btn-sm" onclick="selectNone()">Clear</button>
        <button class="btn-primary" id="digBtn" onclick="digSelected()" disabled>
          Dig into Selected →
        </button>
      </div>
    </div>
    <div class="headlines-table-wrap">
      <table class="headlines-table">
        <thead>
          <tr>
            <th class="col-check"><input type="checkbox" id="checkAll" onchange="toggleAll(this)"></th>
            <th>Headline</th>
            <th>Section</th>
            <th>Summary</th>
          </tr>
        </thead>
        <tbody id="headlinesTbody"></tbody>
      </table>
    </div>`;

  // Insert before timing bar
  const timingBar = card.querySelector('.timing-bar');
  if (timingBar) card.insertBefore(hlSection, timingBar);
  else card.appendChild(hlSection);

  // Populate using existing renderHomepage tbody logic
  window._hlData = data.headlines;
  const tbody = document.getElementById('headlinesTbody');
  tbody.innerHTML = data.headlines.map((h, i) => `
    <tr id="hlrow-${i}" data-idx="${i}" class="hl-row">
      <td class="col-check">
        <input type="checkbox" class="hl-check" data-idx="${i}" onchange="onCheckChange()">
      </td>
      <td class="hl-title">
        <a href="${esc(h.url)}" target="_blank">${esc(h.title)}</a>
        <div class="hl-hover-card">
          <div class="hl-hc-section">${esc(h.section || '')}</div>
          <div class="hl-hc-title">
            <a href="${esc(h.url)}" target="_blank">${esc(h.title)}</a>
          </div>
          ${h.summary ? `<div class="hl-hc-summary">${esc(h.summary)}</div>` : ''}
          <div class="hl-hc-url">${esc(h.url)}</div>
        </div>
      </td>
      <td class="hl-section">${esc(h.section || '—')}</td>
      <td class="hl-summary">${esc(h.summary || '')}</td>
    </tr>`).join('');
}

// ── Article renderer ──────────────────────────────────────────────────────────
function renderArticle(data) {
  currentContent = data.content || '';
  currentView = 'rendered';
  const tmpl = document.getElementById('tmpl-article');
  const node = tmpl.content.cloneNode(true);
  const resultsEl = document.getElementById('results');
  resultsEl.innerHTML = '';
  resultsEl.appendChild(node);

  document.getElementById('res-title').textContent = data.title || data.url;
  document.getElementById('res-meta').innerHTML = buildMetaChips(data);
  document.getElementById('scoreboard').innerHTML = buildScoreboard(data.all_results, data.best_strategy);
  document.getElementById('contentBox').innerHTML = mdToHtml(currentContent);
  document.getElementById('timing-bar').innerHTML = buildTimingBar(data);
}

// ── Data page renderer ────────────────────────────────────────────────────────
function renderData(data) {
  // Prefer tables_md (from static HTML tables), fall back to content
  currentContent = data.tables_md || data.content || '';
  currentView = 'table';
  const tmpl = document.getElementById('tmpl-data');
  const node = tmpl.content.cloneNode(true);
  const resultsEl = document.getElementById('results');
  resultsEl.innerHTML = '';
  resultsEl.appendChild(node);

  document.getElementById('res-title').textContent = data.title || data.url;
  document.getElementById('res-meta').innerHTML = buildMetaChips(data) +
    (data.table_count ? `<span class="chip">⊞ ${data.table_count} table(s)</span>` : '');
  document.getElementById('scoreboard').innerHTML = buildScoreboard(data.all_results, data.best_strategy);
  document.getElementById('timing-bar').innerHTML = buildTimingBar(data);

  // Default to table view for data pages
  const box = document.getElementById('contentBox');
  box.innerHTML = mdToHtml(currentContent);
  document.getElementById('btnTable')?.classList.add('active');
}

// ── View switching ────────────────────────────────────────────────────────────
function switchView(view) {
  currentView = view;
  document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById('btn' + view.charAt(0).toUpperCase() + view.slice(1));
  if (btn) btn.classList.add('active');
  const box = document.getElementById('contentBox');
  if (view === 'raw') {
    box.textContent = currentContent;
    box.className = 'raw-body';
  } else {
    box.innerHTML = mdToHtml(currentContent);
    box.className = 'md-body';
  }
}

function copyContent() {
  navigator.clipboard.writeText(currentContent).then(() => {
    const btns = document.querySelectorAll('.btn-sm');
    btns.forEach(b => { if (b.textContent.includes('Copy')) { b.textContent = '✓ Copied'; setTimeout(() => b.textContent = '⎘ Copy', 2000); } });
  });
}

// ── Send current article to Q&A ───────────────────────────────────────────────
async function sendToQA() {
  if (!currentData || !currentData.content) return;
  const btn = document.getElementById('qaBtn');
  btn.disabled = true; btn.textContent = 'Ingesting…';

  try {
    const resp = await fetch('/api/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        documents: [{
          url: currentData.url,
          title: currentData.title || currentData.url,
          content: currentData.content,
          page_type: currentData.page_type,
        }]
      }),
    });
    if (!resp.ok) throw new Error((await resp.json()).detail);
    const ctx = await resp.json();
    window.location.href = `/qa?session_id=${ctx.session_id}`;
  } catch (e) {
    alert('Failed to send to Q&A: ' + e.message);
    btn.disabled = false; btn.textContent = 'Ask AI →';
  }
}