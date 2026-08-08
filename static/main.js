import { initUploader } from './components/uploader.js';
import { api } from './services/api.js';

document.addEventListener('DOMContentLoaded', () => {
  // Query layout elements
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const uploadBtn = document.getElementById('uploadBtn');
  const documentsList = document.getElementById('documentsList');
  
  const queryForm = document.getElementById('queryForm');
  const queryInput = document.getElementById('queryInput');
  const messagesArea = document.getElementById('messagesArea');
  const sendBtn = document.getElementById('sendBtn');

  // Initialize File Uploader
  initUploader({ dropZone, fileInput, uploadBtn }, (uploadedDoc) => {
    console.log('Document indexed successfully:', uploadedDoc);
    refreshDocumentList();
  });

  // Fetch initial list of documents
  refreshDocumentList();

  // Handle Chat Submit
  if (queryForm) {
    queryForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const query = queryInput.value.trim();
      if (!query) return;

      // 1. Disable inputs while waiting for server response
      sendBtn.disabled = true;
      queryInput.disabled = true;

      // 2. Render user message
      appendMessage(query, 'user');
      queryInput.value = '';

      // 3. Render loading indicator
      const loadingMessageElem = appendMessage('Thinking...', 'bot loading');

      try {
        // 4. Send query to API
        const response = await api.sendQuery({ query });
        
        loadingMessageElem.classList.remove('loading');
        
        // Render Markdown formatted answer
        if (window.marked) {
          loadingMessageElem.innerHTML = marked.parse(response.answer);
        } else {
          loadingMessageElem.style.whiteSpace = 'pre-wrap';
          loadingMessageElem.textContent = response.answer;
        }

        // Render clean & deduplicated citations
        if (response.citations && Array.isArray(response.citations) && response.citations.length > 0) {
          const rawNames = response.citations.map(c => {
            if (typeof c === 'string') return c;
            return c.filename || c.source || c.document_id || 'Unknown Source';
          });

          // Remove duplicate source names
          const uniqueSources = [...new Set(rawNames)];

          if (uniqueSources.length > 0) {
            const citationsDiv = document.createElement('div');
            citationsDiv.className = 'citations-block';
            citationsDiv.innerHTML = `<strong>Sources:</strong> ${uniqueSources.map(escapeHtml).join(', ')}`;
            loadingMessageElem.appendChild(citationsDiv);
          }
        }

      } catch (err) {
        loadingMessageElem.classList.remove('loading');
        loadingMessageElem.style.color = '#ef4444';
        loadingMessageElem.textContent = `Error: ${err.message || 'Failed to get response'}`;
      } finally {
        // 5. Re-enable inputs after response or error
        sendBtn.disabled = false;
        queryInput.disabled = false;
        queryInput.focus();
        scrollToBottom();
      }
    });
  }

  /**
   * Fetches document list from server and renders items into sidebar
   */
  async function refreshDocumentList() {
    if (!documentsList) return;

    try {
      const docs = await api.getDocuments();
      documentsList.innerHTML = '';

      if (!docs || docs.length === 0) {
        documentsList.innerHTML = '<div style="padding: 0.5rem; font-size: 0.85rem; opacity: 0.6;">No documents indexed yet.</div>';
        return;
      }

      docs.forEach((doc) => {
        const item = document.createElement('div');
        item.className = 'document-item';
        item.innerHTML = `
          <div style="font-weight: 500;">${escapeHtml(doc.filename)}</div>
          <div style="font-size: 0.75rem; opacity: 0.7;">Chunks: ${doc.chunks_count ?? 'N/A'}</div>
        `;
        documentsList.appendChild(item);
      });
    } catch (err) {
      console.error('Failed to retrieve documents:', err);
      documentsList.innerHTML = '<div style="padding: 0.5rem; font-size: 0.85rem; color: #ef4444;">Failed to load documents</div>';
    }
  }

  /**
   * Appends message to chat screen
   */
  function appendMessage(text, type) {
    const msg = document.createElement('div');
    msg.className = `message ${type}`;
    
    if (window.marked && type.includes('bot') && !type.includes('loading')) {
      msg.innerHTML = marked.parse(text);
    } else {
      msg.style.whiteSpace = 'pre-wrap';
      msg.textContent = text;
    }

    messagesArea.appendChild(msg);
    scrollToBottom();
    return msg;
  }

  function scrollToBottom() {
    if (messagesArea) {
      messagesArea.scrollTop = messagesArea.scrollHeight;
    }
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
});