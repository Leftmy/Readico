const BASE_URL = '/api/v1';

export const api = {
  /**
   * Fetch system health status
   * @returns {Promise<{status: string, app_name: string, environment: string}>}
   */
  async getHealth() {
    const res = await fetch(`${BASE_URL}/health`);
    if (!res.ok) throw new Error('Failed to fetch health status');
    return res.json();
  },

  /**
   * Fetch all indexed documents
   * @returns {Promise<Array<{id: string, filename: string, status?: string, chunks_count?: number|null}>>}
   */
  async getDocuments() {
    const res = await fetch(`${BASE_URL}/documents`);
    if (!res.ok) throw new Error('Failed to fetch document list');
    return res.json();
  },

  /**
   * Upload and index a document file
   * @param {File} file 
   * @returns {Promise<{id: string, filename: string, content_type: string, file_size_bytes: number, status: string, total_chunks?: number|null, error_message?: string|null}>}
   */
  async uploadDocument(file) {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({ detail: 'Failed to upload document' }));
      throw new Error(errorData.detail || 'Failed to upload document');
    }

    return res.json();
  },

  /**
   * Send RAG query to generate response
   * @param {{query: string, document_ids?: string[], top_k?: number}} payload 
   * @returns {Promise<{query: string, answer: string, citations?: Array, tokens_used?: number|null}>}
   */
  async sendQuery(payload) {
    const res = await fetch(`${BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error('Failed to process search query');
    return res.json();
  },
};