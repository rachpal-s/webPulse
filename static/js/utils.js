/* static/js/utils.js — shared utilities */

// ── Markdown → HTML renderer ──────────────────────────────────────────────────
function mdToHtml(text) {
  if (!text) return '<em style="color:var(--muted)">— No content extracted —</em>';
  const lines = text.split('\n');
  let html = '', i = 0;
  let inTable = false, tableHeaders = [], tableRows = [];
  let inList = false, listTag = '';

  function flushTable() {
    if (!tableHeaders.length && !tableRows.length) return;
    let t = '<div class="table-wrap"><table class="extracted"><thead><tr>';
    t += tableHeaders.map(h => `<th>${inlineFormat(h)}</th>`).join('');
    t += '</tr></thead><tbody>';
    tableRows.forEach(row => {
      t += '<tr>' + row.map(c => {
        let cls = '';
        if (/^[+-][\d.,]+$/.test(c.trim()) || (c.startsWith('-') && /\d/.test(c))) cls = ' class="neg"';
        else if (c.startsWith('+')) cls = ' class="pos"';
        return `<td${cls}>${inlineFormat(c)}</td>`;
      }).join('') + '</tr>';
    });
    t += '</tbody></table></div>';
    html += t;
    tableHeaders = []; tableRows = []; inTable = false;
  }

  function flushList() {
    if (!inList) return;
    html += `</${listTag}>`;
    inList = false; listTag = '';
  }

  // Inline formatting: bold, italic, code, links
  function inlineFormat(raw) {
    return esc(raw)
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/_(.+?)_/g, '<em>$1</em>')
      .replace(/`(.+?)`/g, '<code class="md-inline-code">$1</code>')
      .replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
        '<a href="$2" target="_blank" class="md-link">$1</a>');
  }

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    // ── Markdown table ──────────────────────────────────────────────────────
    if (trimmed.startsWith('|') && trimmed.endsWith('|') && trimmed.split('|').length > 2) {
      flushList();
      const cells = trimmed.slice(1, -1).split('|').map(c => c.trim());
      if (cells.every(c => /^[-: ]+$/.test(c))) {
        inTable = true;
      } else if (!inTable && !tableHeaders.length) {
        tableHeaders = cells;
      } else if (inTable) {
        tableRows.push(cells);
      }
      i++; continue;
    }
    if (inTable || tableHeaders.length) flushTable();

    // ── Horizontal rule ─────────────────────────────────────────────────────
    if (/^[-*_]{3,}$/.test(trimmed)) {
      flushList();
      html += '<hr class="md-hr">'; i++; continue;
    }

    // ── Headings ────────────────────────────────────────────────────────────
    const h1m = trimmed.match(/^# (.+)/);
    const h2m = trimmed.match(/^## (.+)/);
    const h3m = trimmed.match(/^### (.+)/);
    const h4m = trimmed.match(/^#### (.+)/);
    if (h1m) { flushList(); html += `<h2 class="md-h1">${inlineFormat(h1m[1])}</h2>`; i++; continue; }
    if (h2m) { flushList(); html += `<h2 class="md-h2">${inlineFormat(h2m[1])}</h2>`; i++; continue; }
    if (h3m) { flushList(); html += `<h3 class="md-h3">${inlineFormat(h3m[1])}</h3>`; i++; continue; }
    if (h4m) { flushList(); html += `<h4 class="md-h4">${inlineFormat(h4m[1])}</h4>`; i++; continue; }

    // ── Blockquote ──────────────────────────────────────────────────────────
    if (trimmed.startsWith('> ')) {
      flushList();
      html += `<blockquote class="md-blockquote">${inlineFormat(trimmed.slice(2))}</blockquote>`;
      i++; continue;
    }

    // ── Unordered list ──────────────────────────────────────────────────────
    const ulm = trimmed.match(/^[-*+] (.+)/);
    if (ulm) {
      if (!inList || listTag !== 'ul') { flushList(); html += '<ul class="md-ul">'; inList = true; listTag = 'ul'; }
      html += `<li class="md-li">${inlineFormat(ulm[1])}</li>`;
      i++; continue;
    }

    // ── Ordered list ────────────────────────────────────────────────────────
    const olm = trimmed.match(/^\d+[.)]\ (.+)/);
    if (olm) {
      if (!inList || listTag !== 'ol') { flushList(); html += '<ol class="md-ol">'; inList = true; listTag = 'ol'; }
      html += `<li class="md-li">${inlineFormat(olm[1])}</li>`;
      i++; continue;
    }

    // ── Empty line ──────────────────────────────────────────────────────────
    if (!trimmed) { flushList(); html += '<div class="md-spacer"></div>'; i++; continue; }

    // ── Paragraph ───────────────────────────────────────────────────────────
    flushList();
    html += `<p class="md-p">${inlineFormat(trimmed)}</p>`;
    i++;
  }
  flushTable(); flushList();
  return html;
}

// ── Escape HTML ───────────────────────────────────────────────────────────────
function esc(str) {
  return String(str)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}

// ── Scoreboard builder ────────────────────────────────────────────────────────
function buildScoreboard(allResults, bestStrategy) {
  return allResults.map(r => {
    const isWin = r.strategy === bestStrategy;
    const isPW = r.strategy === 'playwright';
    const wc = isWin ? (isPW ? 'winner-pw' : 'winner') : '';
    if (r.success) {
      return `<div class="score-item ${wc}">
        <div class="score-name">${isPW ? '🎭 ' : ''}${r.strategy}${isWin ? ' ★' : ''}</div>
        <div class="score-val">${r.word_count.toLocaleString()}</div>
        <div class="score-sub">words · ${r.time_ms.toFixed(0)}ms</div>
      </div>`;
    }
    return `<div class="score-item">
      <div class="score-name">${r.strategy}</div>
      <div class="score-err">✗ ${esc(r.error || 'failed')}</div>
    </div>`;
  }).join('');
}

// ── Timing bar builder ────────────────────────────────────────────────────────
function buildTimingBar(data) {
  return `
    <div class="timing-item"><div class="timing-label">Fetch</div>
      <div class="timing-val">${data.fetch_time_ms.toFixed(0)}ms</div></div>
    <div class="timing-item"><div class="timing-label">Total</div>
      <div class="timing-val">${data.total_time_ms.toFixed(0)}ms</div></div>
    <div class="timing-item"><div class="timing-label">Strategies</div>
      <div class="timing-val">${data.all_results.length}</div></div>
    <div class="timing-item"><div class="timing-label">Successful</div>
      <div class="timing-val">${data.all_results.filter(r=>r.success).length}</div></div>
    <div class="timing-item"><div class="timing-label">Page type</div>
      <div class="timing-val" style="color:var(--accent2)">${data.page_type}</div></div>`;
}

// ── Meta chips ────────────────────────────────────────────────────────────────
function buildMetaChips(data) {
  const isPW = data.best_strategy === 'playwright';
  const pw = data.all_results?.find(r => r.strategy === 'playwright' && r.extra);
  const xhrBadge = pw?.extra?.xhr_captured
    ? `<span class="chip xhr">⚡ ${pw.extra.xhr_captured} API calls</span>` : '';
  return `
    <span class="chip ${isPW ? 'best-pw' : 'best'}">★ ${data.best_strategy || 'none'}</span>
    <span class="chip words">${(data.word_count||0).toLocaleString()} words</span>
    <span class="chip">${data.metadata?.domain || ''}</span>
    ${xhrBadge}`;
}

// ── HTML Sanitiser + Renderer ─────────────────────────────────────────────────
// Allowlist-based sanitiser — strips scripts/events, keeps safe semantic tags.
// Used to safely render LLM-generated HTML responses.

const ALLOWED_TAGS = new Set([
  'p','br','hr','b','strong','i','em','u','s','mark','small','sup','sub',
  'h1','h2','h3','h4','h5','h6',
  'ul','ol','li','dl','dt','dd',
  'table','thead','tbody','tfoot','tr','th','td','caption','colgroup','col',
  'blockquote','pre','code','samp','kbd',
  'a','span','div','section','article','aside',
  'details','summary',
]);

const ALLOWED_ATTRS = new Set(['href','title','colspan','rowspan','scope','start','type']);

function sanitizeHtml(html) {
  // Parse in a detached document — never touches the live DOM
  const doc = new DOMParser().parseFromString(html, 'text/html');

  function clean(node) {
    if (node.nodeType === Node.TEXT_NODE) return node.cloneNode();
    if (node.nodeType !== Node.ELEMENT_NODE) return null;

    const tag = node.tagName.toLowerCase();

    // Strip disallowed tags but keep their children
    if (!ALLOWED_TAGS.has(tag)) {
      const frag = document.createDocumentFragment();
      node.childNodes.forEach(child => {
        const cleaned = clean(child);
        if (cleaned) frag.appendChild(cleaned);
      });
      return frag;
    }

    const el = document.createElement(tag);

    // Copy only allowed attributes; sanitize href
    for (const attr of node.attributes) {
      if (!ALLOWED_ATTRS.has(attr.name)) continue;
      if (attr.name === 'href') {
        const v = attr.value.trim().toLowerCase();
        if (v.startsWith('javascript:') || v.startsWith('data:')) continue;
        el.setAttribute('href', attr.value);
        el.setAttribute('target', '_blank');
        el.setAttribute('rel', 'noopener noreferrer');
      } else {
        el.setAttribute(attr.name, attr.value);
      }
    }

    node.childNodes.forEach(child => {
      const cleaned = clean(child);
      if (cleaned) el.appendChild(cleaned);
    });

    return el;
  }

  const frag = document.createDocumentFragment();
  doc.body.childNodes.forEach(child => {
    const cleaned = clean(child);
    if (cleaned) frag.appendChild(cleaned);
  });

  const wrapper = document.createElement('div');
  wrapper.appendChild(frag);
  return wrapper.innerHTML;
}

// Post-process rendered HTML — convert [Source: Title] plain text to links
// using a provided source map {title → url}
function linkifySources(html, sourceMap) {
  if (!sourceMap || !Object.keys(sourceMap).length) return html;
  // Match [Source: Title] or [Source: Title - subtitle] patterns
  return html.replace(/\[Source:\s*([^\]]{3,120})\]/g, (match, title) => {
    const trimmed = title.trim();
    // Try exact match first, then partial match
    let url = sourceMap[trimmed];
    if (!url) {
      const key = Object.keys(sourceMap).find(k =>
        k.toLowerCase().includes(trimmed.toLowerCase().slice(0, 30)) ||
        trimmed.toLowerCase().includes(k.toLowerCase().slice(0, 30))
      );
      url = key ? sourceMap[key] : null;
    }
    if (url) {
      return `<a href="${url}" target="_blank" rel="noopener noreferrer" class="source-cite">[Source: ${trimmed}]</a>`;
    }
    return `<span class="source-cite-plain">[Source: ${trimmed}]</span>`;
  });
}

// Detect whether a string is primarily HTML or Markdown, render appropriately
function renderContent(text) {
  if (!text) return '<em style="color:var(--muted)">— No content —</em>';

  // Strip code fences the LLM sometimes wraps around HTML
  const stripped = text.replace(/^```(?:html)?\s*/i, '').replace(/\s*```$/, '').trim();

  // Convert markdown links [text](url) → <a> even when response is otherwise HTML
  let linkified = stripped.replace(
    /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
  );

  // Clean "CITE AS: [title]" inside anchor text → readable [Source: title]
  linkified = linkified.replace(
    /(<a [^>]+>)CITE AS: \[([^\]]+)\](<\/a>)/g,
    '$1[Source: $2]$3'
  );

  // Remove LLM artifacts: "— Markdown link." and "— Markdown link" suffixes
  linkified = linkified.replace(/\s*—\s*Markdown link\.?/g, '');

  // Fix raw <a href="..."> tags rendered as text (LLM put HTML in text context)
  // If we find &lt;a href= patterns, unescape them
  linkified = linkified.replace(/&lt;a href="([^"]+)"[^&]*&gt;([^&]*)&lt;\/a&gt;/g,
    '<a href="$1" target="_blank" rel="noopener noreferrer">$2</a>');

  // Check if content has HTML tags
  const hasHtmlTags = /<[a-zA-Z][^>]*>/.test(linkified);
  // Check if content also has Markdown (bold/italic/bullets)
  const hasMd = /\*\*|\n\s*[\*\-] /.test(linkified);

  if (hasHtmlTags && hasMd) {
    // Mixed: convert MD first then sanitize HTML
    const converted = mdToHtml(linkified);
    return sanitizeHtml(converted);
  }
  if (hasHtmlTags) {
    return sanitizeHtml(linkified);
  }
  return mdToHtml(linkified);
}