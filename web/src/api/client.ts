import axios from 'axios';

const apiClient = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

export default apiClient;

export const analysisApi = {
  search(query: string) {
    return apiClient.get('/analysis/search', { params: { q: query } });
  },

  analyze(symbol: string, market: string = 'A') {
    return apiClient.post('/analysis/analyze', { symbol, market });
  },

  getTaskStatus(taskId: string) {
    return apiClient.get(`/analysis/status/${taskId}`);
  },

  getTasks(limit: number = 20) {
    return apiClient.get('/analysis/tasks', { params: { limit } });
  },

  getTaskStreamUrl() {
    return '/api/v1/analysis/tasks/stream';
  },
};

export const hotspotApi = {
  analyze() {
    return apiClient.post('/hotspot/analyze');
  },

  getStatus(taskId: string) {
    return apiClient.get(`/hotspot/status/${taskId}`);
  },

  getLatest() {
    return apiClient.get('/hotspot/latest');
  },

  getHistory(limit: number = 20) {
    return apiClient.get('/hotspot/history', { params: { limit } });
  },
};

export const historyApi = {
  list(symbol?: string, limit: number = 20) {
    return apiClient.get('/history/', { params: { symbol, limit } });
  },

  detail(runId: string) {
    return apiClient.get(`/history/${runId}`);
  },
};
