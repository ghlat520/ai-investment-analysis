import axios from 'axios';

// API Key 管理：从 URL 参数读取并缓存
const API_KEY_KEY = 'ai_invest_api_key';

function getApiKey(): string | null {
  // 1. 先从 URL 参数读取（优先级最高）
  const urlParams = new URLSearchParams(window.location.search);
  const urlKey = urlParams.get('key');
  if (urlKey) {
    localStorage.setItem(API_KEY_KEY, urlKey);
    return urlKey;
  }
  // 2. 从 localStorage 读取
  return localStorage.getItem(API_KEY_KEY);
}

const apiClient = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

// 请求拦截器：自动添加 key 参数
apiClient.interceptors.request.use((config) => {
  const key = getApiKey();
  if (key) {
    config.params = { ...config.params, key };
  }
  return config;
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
    const key = getApiKey();
    const baseUrl = '/api/v1/analysis/tasks/stream';
    return key ? `${baseUrl}?key=${key}` : baseUrl;
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

  getStreamUrl() {
    const key = getApiKey();
    const baseUrl = '/api/v1/hotspot/stream';
    return key ? `${baseUrl}?key=${key}` : baseUrl;
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
