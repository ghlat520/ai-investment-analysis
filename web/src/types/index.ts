export interface AgentSignal {
  agent_name: string;
  signal_score: number;
  confidence: number;
  reasoning: string;
  key_factors: string[];
  risks: string[];
  extra_data: Record<string, any> | null;
  llm_model: string;
  execution_time_ms: number;
}

export interface FusionDecision {
  final_score: number;
  final_action: string;
  confidence: number;
  position_pct: number;
  stop_loss_pct: number | null;
  take_profit_pct: number | null;
  reasoning: string;
  signal_summary: Record<string, number>;
  conflicts: string[];
  conflict_resolution: string;
  market_regime: string;
  weights_used: Record<string, number>;
  bull_arguments: string[];
  bear_arguments: string[];
  divergence_points: string[];
  target_prices: Record<string, number>;
}

export interface TaskInfo {
  task_id: string;
  symbol: string;
  market: string;
  stock_name: string | null;
  status: 'pending' | 'collecting' | 'analyzing' | 'completed' | 'failed';
  progress: number;
  message: string | null;
  completed_agents: string[];
  total_agents: number;
  run_id: string | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  result?: TaskResult;
}

export interface TaskResult {
  fusion: FusionDecision;
  signals: AgentSignal[];
  report: string;
  errors: string[];
}

export interface SSEAgentCompletedEvent {
  task_id: string;
  agent_name: string;
  completed_agents: string[];
  progress: number;
  signal: {
    agent_name: string;
    signal_score: number;
    confidence: number;
    execution_time_ms: number;
  } | null;
}

export interface HistoryItem {
  run_id: string;
  symbol: string;
  stock_name: string | null;
  final_score: number;
  final_action: string;
  confidence: number;
  position_pct: number | null;
  market_regime: string | null;
  created_at: string | null;
}

export const AGENT_NAMES: Record<string, string> = {
  technical: '技术面',
  fundamental: '基本面',
  valuation: '估值',
  money_flow: '资金面',
  sentiment: '情绪面',
  moat: '护城河',
  business_model: '商业模式',
  industry: '行业',
  supply_chain: '产业链',
};

export const ALL_AGENTS = Object.keys(AGENT_NAMES);
