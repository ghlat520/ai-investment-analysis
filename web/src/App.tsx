import { useState, useEffect, useCallback, useRef } from 'react';
import { useTaskStream } from './hooks/useTaskStream';
import { useAnalysisStore } from './stores/analysisStore';
import { analysisApi, historyApi } from './api/client';
import type { TaskInfo, HistoryItem, TaskResult, FusionDecision, AgentSignal } from './types';
import StockInput from './components/StockInput';
import HistoryList from './components/HistoryList';
import TaskPanel from './components/TaskPanel';
import ReportOverview from './components/ReportOverview';
import StrategyPoints from './components/StrategyPoints';
import BullBearDebate from './components/BullBearDebate';
import AgentProgressGrid from './components/AgentProgressGrid';
import ReportViewer from './components/ReportViewer';

// API returns flat structure, not wrapped in `result`
interface HistoryDetail {
  run_id: string;
  symbol: string;
  stock_name: string | null;
  fusion: FusionDecision | null;
  signals: AgentSignal[];
  report: string;
  created_at: string | null;
}

// Normalize history detail to TaskResult shape
function toTaskResult(detail: HistoryDetail): TaskResult | null {
  if (!detail.fusion) return null;
  return {
    fusion: {
      ...detail.fusion,
      // These fields may be absent from history API — provide defaults
      bull_arguments: detail.fusion.bull_arguments || [],
      bear_arguments: detail.fusion.bear_arguments || [],
      divergence_points: detail.fusion.divergence_points || [],
      target_prices: detail.fusion.target_prices || {},
    },
    signals: detail.signals || [],
    report: detail.report || '',
    errors: [],
  };
}

function App() {
  const store = useAnalysisStore();

  // History state
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const pageRef = useRef(1);

  // Report state
  const [selectedReport, setSelectedReport] = useState<HistoryDetail | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | undefined>();
  const [isLoadingReport, setIsLoadingReport] = useState(false);

  // Active tasks (from SSE)
  const [activeTasks, setActiveTasks] = useState<TaskInfo[]>([]);

  // SSE connection
  const { isConnected } = useTaskStream({
    enabled: true,
    onTaskCreated: (task) => {
      // Add to active tasks
      setActiveTasks((prev) => {
        if (prev.some((t) => t.task_id === task.task_id)) return prev;
        return [...prev, task];
      });
      // Also update store for AgentProgressGrid
      store.updateFromSSE(task);
    },
    onTaskStarted: (task) => {
      setActiveTasks((prev) => {
        const idx = prev.findIndex((t) => t.task_id === task.task_id);
        if (idx >= 0) {
          const copy = [...prev];
          copy[idx] = task;
          return copy;
        }
        return prev;
      });
      store.updateFromSSE(task);
    },
    onAgentCompleted: (event) => {
      store.handleAgentCompleted(event);
    },
    onTaskCompleted: (task) => {
      store.setCompleted(task);
      // Remove from active tasks after delay
      setTimeout(() => {
        setActiveTasks((prev) => prev.filter((t) => t.task_id !== task.task_id));
      }, 1500);
      // Refresh history and auto-select latest
      fetchHistory(true);
    },
    onTaskFailed: (task) => {
      store.setFailed(task);
      setTimeout(() => {
        setActiveTasks((prev) => prev.filter((t) => t.task_id !== task.task_id));
      }, 3000);
    },
  });

  // Fetch history
  const fetchHistory = useCallback(async (autoSelectFirst = false) => {
    setIsLoadingHistory(true);
    pageRef.current = 1;
    try {
      const resp = await historyApi.list(undefined, 20);
      const items: HistoryItem[] = resp.data.items || resp.data || [];
      setHistoryItems(items);
      setHasMore(items.length >= 20);

      // Auto-select first item
      if (autoSelectFirst && items.length > 0) {
        handleHistoryClick(items[0].run_id);
      }
    } catch (err) {
      console.error('Failed to fetch history:', err);
    } finally {
      setIsLoadingHistory(false);
    }
  }, []);

  const handleLoadMore = useCallback(async () => {
    if (isLoadingMore || !hasMore) return;
    setIsLoadingMore(true);
    const nextPage = pageRef.current + 1;
    try {
      const resp = await historyApi.list(undefined, 20);
      const items: HistoryItem[] = resp.data.items || resp.data || [];
      setHistoryItems((prev) => [...prev, ...items]);
      setHasMore(items.length >= 20);
      pageRef.current = nextPage;
    } catch (err) {
      console.error('Failed to load more:', err);
    } finally {
      setIsLoadingMore(false);
    }
  }, [isLoadingMore, hasMore]);

  // Load history on mount
  useEffect(() => {
    fetchHistory(true);
  }, []);

  // Click history item → load report
  const handleHistoryClick = async (runId: string) => {
    setSelectedRunId(runId);
    setIsLoadingReport(true);
    try {
      const resp = await historyApi.detail(runId);
      const data = resp.data;
      setSelectedReport(data);
    } catch (err) {
      console.error('Failed to fetch report:', err);
    } finally {
      setIsLoadingReport(false);
    }
  };

  // Handle analyze from StockInput
  const handleAnalyze = async (symbol: string, market: string) => {
    try {
      store.reset();
      const resp = await analysisApi.analyze(symbol, market);
      if (resp.status === 202) {
        store.submitTask(resp.data.task_id, symbol);
      }
    } catch (err: any) {
      if (err.response?.status === 409) {
        const data = err.response.data;
        store.submitTask(data.existing_task_id, symbol);
      }
    }
  };

  const isAnalyzing = store.status !== 'idle' && store.status !== 'completed' && store.status !== 'failed';
  const showAgentGrid = store.status === 'collecting' || store.status === 'analyzing';

  // Report data from selected history or current analysis
  const reportData = (selectedReport ? toTaskResult(selectedReport) : null) || store.result;
  const reportStockName = store.stockName || selectedReport?.stock_name || selectedReport?.symbol || store.currentSymbol || '';
  const reportSymbol = selectedReport?.symbol || store.currentSymbol || '';
  const reportCreatedAt = selectedReport?.created_at || undefined;

  return (
    <div className="min-h-screen flex flex-col bg-[var(--bg-base)]">
      {/* Header - top input bar */}
      <header className="flex-shrink-0 px-4 py-3 border-b border-white/5">
        <div className="max-w-[1440px] mx-auto w-full flex items-center justify-between">
          <StockInput isRunning={isAnalyzing} onAnalyze={handleAnalyze} />
          <div className="flex items-center gap-2 text-xs text-[var(--text-muted)] ml-4">
            <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-[var(--color-success)]' : 'bg-[var(--color-danger)]'}`} />
            {isConnected ? 'SSE' : '断开'}
          </div>
        </div>
      </header>

      {/* Main content area */}
      <main className="flex-1 flex overflow-hidden p-3 gap-3 max-w-[1440px] mx-auto w-full">
        {/* Left sidebar: tasks + history */}
        <div className="flex flex-col gap-3 w-72 flex-shrink-0 overflow-hidden">
          <TaskPanel tasks={activeTasks} />
          <HistoryList
            items={historyItems}
            isLoading={isLoadingHistory}
            isLoadingMore={isLoadingMore}
            hasMore={hasMore}
            selectedRunId={selectedRunId}
            onItemClick={handleHistoryClick}
            onLoadMore={handleLoadMore}
            className="flex-1"
          />
        </div>

        {/* Right: report content */}
        <section className="flex-1 overflow-y-auto pl-1">
          {isLoadingReport ? (
            <div className="flex flex-col items-center justify-center h-full">
              <div className="w-10 h-10 border-3 border-[var(--color-cyan)]/20 border-t-[var(--color-cyan)] rounded-full animate-spin" />
              <p className="mt-3 text-[var(--text-secondary)] text-sm">加载报告中...</p>
            </div>
          ) : reportData ? (
            <div className="max-w-4xl space-y-4">
              {/* Overview: header + gauge + insights + advice */}
              {reportData.fusion && (
                <>
                  <ReportOverview
                    fusion={reportData.fusion}
                    stockName={reportStockName}
                    symbol={reportSymbol}
                    createdAt={reportCreatedAt}
                  />

                  {/* Strategy points */}
                  <StrategyPoints fusion={reportData.fusion} signals={reportData.signals || []} />

                  {/* Agent progress (only during analysis) */}
                  <AgentProgressGrid agents={store.agents} visible={showAgentGrid} />

                  {/* Bull vs Bear */}
                  <BullBearDebate fusion={reportData.fusion} />
                </>
              )}

              {/* Full report */}
              {reportData.report && <ReportViewer report={reportData.report} />}

              {/* Errors */}
              {reportData.errors && reportData.errors.length > 0 && (
                <div className="terminal-card p-4 border-[var(--color-danger)]/30">
                  <h4 className="text-sm font-medium text-[var(--color-danger)] mb-2">分析过程中的错误</h4>
                  <ul className="text-xs text-[var(--text-muted)] space-y-1">
                    {reportData.errors.map((err, i) => (
                      <li key={i}>{err}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : showAgentGrid ? (
            <div className="max-w-4xl space-y-4">
              <AgentProgressGrid agents={store.agents} visible={true} />
              {store.message && (
                <div className="terminal-card p-4 text-center">
                  <p className="text-sm text-[var(--text-secondary)]">{store.message}</p>
                  <div className="w-full bg-white/5 rounded-full h-1.5 mt-3">
                    <div
                      className="h-1.5 rounded-full bg-[var(--color-cyan)] transition-all duration-500"
                      style={{ width: `${store.progress}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-12 h-12 mb-3 rounded-xl bg-[var(--bg-elevated)] flex items-center justify-center">
                <svg className="w-6 h-6 text-[var(--text-muted)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="text-base font-medium text-white mb-1.5">开始分析</h3>
              <p className="text-xs text-[var(--text-muted)] max-w-xs">
                输入股票代码进行分析，或从左侧选择历史报告查看
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
