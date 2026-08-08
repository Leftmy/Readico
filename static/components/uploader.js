import { api } from '../services/api.js';

/**
 * Attaches upload handlers to existing DOM elements and shows upload status overlay
 * @param {Object} options Configuration object containing target elements
 * @param {HTMLElement} options.dropZone Drag & drop zone element
 * @param {HTMLInputElement} options.fileInput File input element
 * @param {HTMLElement} options.uploadBtn Upload button element
 * @param {Function} onUploadSuccess Callback function executed when upload completes
 */
export function initUploader({ dropZone, fileInput, uploadBtn }, onUploadSuccess) {
  if (!dropZone || !fileInput) {
    console.error('Uploader Error: Required elements (dropZone or fileInput) are missing from DOM.');
    return;
  }

  // 1. Dynamically append modal element to document body if not present
  if (!document.getElementById('uploadModal')) {
    const modalHtml = `
      <div id="uploadModal" class="upload-modal" style="display: none; position: fixed; inset: 0; z-index: 9999; background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(4px); align-items: center; justify-content: center; font-family: system-ui, sans-serif;">
        <div style="background: white; padding: 1.5rem; border-radius: 12px; width: 90%; max-width: 400px; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1); text-align: center;">
          <h3 id="modalTitle" style="margin: 0 0 1rem 0; font-size: 1.1rem; color: #0f172a;">Uploading & Processing</h3>
          
          <div id="modalBody">
            <div id="modalSpinner" style="display: inline-block; width: 36px; height: 36px; border: 3px solid #e2e8f0; border-top-color: #6366f1; border-radius: 50%; animation: spin 0.8s linear infinite;"></div>
            <p id="modalStatusText" style="margin: 1rem 0 0.25rem 0; font-size: 0.9rem; color: #334155; word-break: break-all;"></p>
            <p id="modalSubText" style="margin: 0; font-size: 0.75rem; color: #94a3b8;">Generating embeddings, please wait...</p>
          </div>

          <div id="modalFooter" style="display: none; margin-top: 1.25rem; text-align: right;">
            <button id="modalCloseBtn" style="background: #e2e8f0; color: #1e293b; border: none; padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer; font-weight: 500;">Close</button>
          </div>
        </div>
      </div>
      <style>
        @keyframes spin { to { transform: rotate(360deg); } }
        .drop-zone.dragover { border-color: #6366f1 !important; background-color: rgba(99, 102, 241, 0.1) !important; }
      </style>
    `;
    document.body.insertAdjacentHTML('beforeend', modalHtml);
  }

  // 2. Query modal DOM elements
  const modal = document.getElementById('uploadModal');
  const modalTitle = document.getElementById('modalTitle');
  const spinner = document.getElementById('modalSpinner');
  const statusText = document.getElementById('modalStatusText');
  const subText = document.getElementById('modalSubText');
  const modalFooter = document.getElementById('modalFooter');
  const closeBtn = document.getElementById('modalCloseBtn');

  // 3. UI Helper functions
  const showModal = (fileName) => {
    modal.style.display = 'flex';
    modalTitle.textContent = 'Uploading & Processing';
    spinner.style.display = 'inline-block';
    subText.style.display = 'block';
    modalFooter.style.display = 'none';
    statusText.style.color = '#334155';
    statusText.innerHTML = `Indexing file <b>${fileName}</b>...`;
  };

  const showError = (message) => {
    modalTitle.textContent = 'Upload Failed';
    spinner.style.display = 'none';
    subText.style.display = 'none';
    statusText.style.color = '#ef4444';
    statusText.textContent = message;
    modalFooter.style.display = 'block';
  };

  const closeModal = () => {
    modal.style.display = 'none';
    fileInput.value = '';
  };

  // 4. File handling logic
  const processFiles = async (files) => {
    if (!files || files.length === 0) return;

    for (const file of files) {
      showModal(file.name);
      try {
        const result = await api.uploadDocument(file);
        closeModal();
        if (typeof onUploadSuccess === 'function') {
          onUploadSuccess(result);
        }
      } catch (err) {
        showError(err.message || 'Error occurred while processing file');
        break;
      }
    }
  };

  // 5. Click Triggers
  if (uploadBtn) {
    uploadBtn.addEventListener('click', () => fileInput.click());
  }
  dropZone.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', (e) => {
    processFiles(e.target.files);
  });

  closeBtn.addEventListener('click', closeModal);

  // 6. Drag & Drop Event Listeners
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
  });

  ['dragenter', 'dragover'].forEach((eventName) => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'));
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'));
  });

  dropZone.addEventListener('drop', (e) => {
    const droppedFiles = e.dataTransfer.files;
    if (droppedFiles.length > 0) {
      processFiles(droppedFiles);
    }
  });
}