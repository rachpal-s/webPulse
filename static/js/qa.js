/* static/js/qa.js */

const chatWindow = document.getElementById('chatWindow');

function toggleSidebar() {
  const layout = document.querySelector('.qa-layout');
  const btn = document.querySelector('.sidebar-collapse-btn');
  const collapsed = layout.classList.toggle('sidebar-collapsed');
  btn.textContent = collapsed ? '›' : '‹';
  btn.title = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
}
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
let isStreaming = false;

// ── Auto-resize textarea ──────────────────────────────────────────────────────
questionInput?.addEventListener('input', () => {
  questionInput.style.height = 'auto';
  questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
});

// ── Keyboard shortcut ─────────────────────────────────────────────────────────
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
}

// ── Suggestion pills ──────────────────────────────────────────────────────────
function setQuestion(btn) {
  if (!questionInput) return;
  questionInput.value = btn.textContent;
  questionInput.focus();
}

// ── Send question ─────────────────────────────────────────────────────────────
async function sendQuestion() {
  if (!SESSION_ID || isStreaming) return;
  const question = questionInput.value.trim();
  if (!question) return;

  const topK = parseInt(document.getElementById('topK').value) || 6;
  const stream = document.getElementById('streamToggle').checked;

  // Hide empty state
  const emptyEl = document.getElementById('chatEmpty');
  if (emptyEl) emptyEl.style.display = 'none';

  // Append user message
  appendMessage('user', question);
  questionInput.value = '';
  questionInput.style.height = 'auto';
  sendBtn.disabled = true;
  isStreaming = true;

  // Create assistant bubble (empty, will fill)
  const assistantId = 'msg-' + Date.now();
  const bubbleId = assistantId + '-bubble';
  appendMessage('assistant', '', assistantId, bubbleId);

  try {
    if (stream) {
      await streamAnswer(question, topK, bubbleId);
    } else {
      await fetchAnswer(question, topK, bubbleId);
    }
  } catch (e) {
    document.getElementById(bubbleId).textContent = '✗ Error: ' + e.message;
  } finally {
    sendBtn.disabled = false;
    isStreaming = false;
  }
}

// ── Streaming answer via SSE ──────────────────────────────────────────────────
async function streamAnswer(question, topK, bubbleId) {
  const bubble = document.getElementById(bubbleId);
  bubble.classList.add('streaming-cursor');
  let fullText = '';

  const url = `/api/qa/stream?session_id=${encodeURIComponent(SESSION_ID)}&question=${encodeURIComponent(question)}&top_k=${topK}`;
  const es = new EventSource(url);

  return new Promise((resolve, reject) => {
    es.onmessage = (e) => {
      if (e.data === '[DONE]') {
        es.close();
        bubble.classList.remove('streaming-cursor');
        resolve();
        return;
      }
      // SSE multi-line data lines are joined with \n by the browser EventSource
      fullText += e.data;
      // Re-render on every token but debounce heavy mdToHtml for performance
      bubble.innerHTML = renderContent(fullText);
      chatWindow.scrollTop = chatWindow.scrollHeight;
    };
    es.onerror = () => {
      es.close();
      bubble.classList.remove('streaming-cursor');
      if (!fullText) reject(new Error('Stream failed'));
      else resolve();
    };
  });
}

// ── Non-streaming answer ──────────────────────────────────────────────────────
async function fetchAnswer(question, topK, bubbleId) {
  const bubble = document.getElementById(bubbleId);
  bubble.innerHTML = '<span class="streaming-cursor"></span>';

  const resp = await fetch('/api/qa', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: SESSION_ID, question, top_k: topK }),
  });
  if (!resp.ok) throw new Error((await resp.json()).detail);
  const data = await resp.json();

  bubble.innerHTML = renderContent(data.answer);

  // Append source info
  if (data.sources_used?.length) {
    const srcEl = document.createElement('div');
    srcEl.className = 'msg-sources';
    srcEl.innerHTML = `<span>Sources:</span> ` +
      data.sources_used.map(s => `<a href="${esc(s.url)}" target="_blank" style="color:var(--accent);text-decoration:none">${esc(s.title || s.url)}</a>`).join(', ');
    bubble.appendChild(srcEl);
  }

  // Add meta below bubble
  const metaEl = document.createElement('div');
  metaEl.className = 'msg-meta';
  metaEl.textContent = `${data.mode} mode · ${data.chunks_retrieved} chunks · ${data.latency_ms.toFixed(0)}ms`;
  bubble.parentElement.appendChild(metaEl);

  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// ── Append chat message ───────────────────────────────────────────────────────
function appendMessage(role, text, msgId, bubbleId) {
  const msg = document.createElement('div');
  msg.className = `chat-msg ${role}`;
  if (msgId) msg.id = msgId;

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  if (bubbleId) bubble.id = bubbleId;

  if (text) bubble.innerHTML = role === 'user' ? esc(text) : renderContent(text);

  msg.appendChild(bubble);
  chatWindow.appendChild(msg);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}