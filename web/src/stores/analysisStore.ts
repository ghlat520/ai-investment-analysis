import { create } from 'zustand';
import type { TaskInfo, SSEAgentCompletedEvent, TaskResult } from '../types';
import { ALL_AGENTS } from '../types';

export type AgentStatus = 'pending' | 'running' | 'completed' | 'error';

export interface AgentState {
  name: string;
  status: AgentStatus;
  signal_score?: number;
  confidence?: number;
  execution_time_ms?: number;
}

interface AnalysisState {
  // 当前任务
  currentTaskId: string | null;
  currentSymbol: string | null;
  stockName: string | null;
  status: 'idle' | 'collecting' | 'analyzing' | 'completed' | 'failed';
  progress: number;
  message: string;
  error: string | null;

  // Agent 进度
  agents: AgentState[];

  // 结果
  result: TaskResult | null;

  // Actions
  submitTask: (taskId: string, symbol: string) => void;
  updateFromSSE: (task: TaskInfo) => void;
  handleAgentCompleted: (event: SSEAgentCompletedEvent) => void;
  setCompleted: (task: TaskInfo) => void;
  setFailed: (task: TaskInfo) => void;
  reset: () => void;
}

const initialAgents = (): AgentState[] =>
  ALL_AGENTS.map((name) => ({ name, status: 'pending' as AgentStatus }));

export const useAnalysisStore = create<AnalysisState>((set, get) => ({
  currentTaskId: null,
  currentSymbol: null,
  stockName: null,
  status: 'idle',
  progress: 0,
  message: '',
  error: null,
  agents: initialAgents(),
  result: null,

  submitTask: (taskId, symbol) =>
    set({
      currentTaskId: taskId,
      currentSymbol: symbol,
      stockName: null,
      status: 'collecting',
      progress: 5,
      message: '任务已提交，正在采集数据...',
      error: null,
      agents: initialAgents(),
      result: null,
    }),

  updateFromSSE: (task) => {
    const state = get();
    if (task.task_id !== state.currentTaskId) return;
    set({
      stockName: task.stock_name || state.stockName,
      status: task.status === 'collecting' ? 'collecting' : task.status === 'analyzing' ? 'analyzing' : state.status,
      progress: task.progress,
      message: task.message || state.message,
    });
  },

  handleAgentCompleted: (event) => {
    const state = get();
    if (event.task_id !== state.currentTaskId) return;

    const agents = state.agents.map((a) => {
      if (event.completed_agents.includes(a.name)) {
        const sig = event.signal?.agent_name === a.name ? event.signal : undefined;
        return {
          ...a,
          status: 'completed' as AgentStatus,
          signal_score: sig?.signal_score ?? a.signal_score,
          confidence: sig?.confidence ?? a.confidence,
          execution_time_ms: sig?.execution_time_ms ?? a.execution_time_ms,
        };
      }
      return a;
    });

    set({
      agents,
      progress: event.progress,
      message: `已完成 ${event.completed_agents.length}/${state.agents.length} 个 Agent`,
    });
  },

  setCompleted: (task) => {
    const state = get();
    if (task.task_id !== state.currentTaskId) return;

    // 所有 agent 标记为 completed
    const agents = state.agents.map((a) => ({
      ...a,
      status: 'completed' as AgentStatus,
    }));

    set({
      status: 'completed',
      progress: 100,
      message: '分析完成',
      stockName: task.stock_name || state.stockName,
      agents,
    });

    // 获取完整结果
    if (task.task_id) {
      fetch(`/api/v1/analysis/status/${task.task_id}`)
        .then((r) => r.json())
        .then((data) => {
          if (data.result) {
            set({ result: data.result });
          }
        })
        .catch(() => {});
    }
  },

  setFailed: (task) => {
    const state = get();
    if (task.task_id !== state.currentTaskId) return;
    set({
      status: 'failed',
      error: task.error || '分析失败',
      message: task.message || '分析失败',
    });
  },

  reset: () =>
    set({
      currentTaskId: null,
      currentSymbol: null,
      stockName: null,
      status: 'idle',
      progress: 0,
      message: '',
      error: null,
      agents: initialAgents(),
      result: null,
    }),
}));
