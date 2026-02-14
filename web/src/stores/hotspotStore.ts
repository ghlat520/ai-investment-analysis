import { create } from 'zustand';
import type { HotspotTaskInfo, HotspotResult, HotspotStage, HotTheme } from '../types/hotspot';

interface HotspotState {
  // 当前任务
  currentTaskId: string | null;
  status: HotspotStage;
  progress: number;
  message: string;
  error: string | null;
  currentTheme: number;
  totalThemes: number;

  // 中间数据
  extractedThemes: HotTheme[];

  // 结果
  result: HotspotResult | null;

  // Actions
  submitTask: (taskId: string) => void;
  updateFromSSE: (task: HotspotTaskInfo) => void;
  setThemesExtracted: (themes: HotTheme[]) => void;
  setCompleted: (result: HotspotResult) => void;
  setFailed: (error: string) => void;
  reset: () => void;
}

export const useHotspotStore = create<HotspotState>((set, get) => ({
  currentTaskId: null,
  status: 'idle',
  progress: 0,
  message: '',
  error: null,
  currentTheme: 0,
  totalThemes: 0,
  extractedThemes: [],
  result: null,

  submitTask: (taskId) =>
    set({
      currentTaskId: taskId,
      status: 'collecting',
      progress: 5,
      message: '热点分析任务已提交...',
      error: null,
      currentTheme: 0,
      totalThemes: 0,
      extractedThemes: [],
      result: null,
    }),

  updateFromSSE: (task) => {
    const state = get();
    if (task.task_id !== state.currentTaskId) return;
    set({
      status: task.status as HotspotStage,
      progress: task.progress,
      message: task.message || state.message,
      currentTheme: task.current_theme,
      totalThemes: task.total_themes,
    });
  },

  setThemesExtracted: (themes) =>
    set({
      extractedThemes: themes,
      totalThemes: themes.length,
    }),

  setCompleted: (result) =>
    set({
      status: 'completed',
      progress: 100,
      message: '热点分析完成',
      result,
    }),

  setFailed: (error) =>
    set({
      status: 'failed',
      error,
      message: `分析失败: ${error}`,
    }),

  reset: () =>
    set({
      currentTaskId: null,
      status: 'idle',
      progress: 0,
      message: '',
      error: null,
      currentTheme: 0,
      totalThemes: 0,
      extractedThemes: [],
      result: null,
    }),
}));
