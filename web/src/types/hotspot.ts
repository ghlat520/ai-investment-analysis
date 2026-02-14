export interface ChainStock {
  symbol: string;
  name: string;
  role: string;
  reason: string;
}

export interface AnalyzedTheme {
  title: string;
  investment_logic: string;
  actionability: 'high' | 'medium' | 'low';
  upstream: ChainStock[];
  midstream: ChainStock[];
  downstream: ChainStock[];
  value_flow: string;
}

export interface HotTheme {
  title: string;
  summary: string;
  relevance_score: number;
  catalyst: string;
  timeline: string;
  risk: string;
}

export interface HotspotTaskInfo {
  task_id: string;
  status: 'pending' | 'collecting' | 'extracting' | 'analyzing' | 'generating' | 'completed' | 'failed';
  progress: number;
  message: string | null;
  current_theme: number;
  total_themes: number;
  error: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface HotspotResult {
  themes: HotTheme[];
  analyzed_themes: AnalyzedTheme[];
  briefing: string;
  errors: string[];
  analysis_date: string;
}

export type HotspotStage = 'idle' | 'collecting' | 'extracting' | 'analyzing' | 'generating' | 'completed' | 'failed';

export const HOTSPOT_STAGE_LABELS: Record<HotspotStage, string> = {
  idle: '等待开始',
  collecting: '采集市场数据',
  extracting: '提取投资主题',
  analyzing: '产业链分析',
  generating: '生成研报',
  completed: '分析完成',
  failed: '分析失败',
};
