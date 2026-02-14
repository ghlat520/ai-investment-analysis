import { useCallback } from 'react';
import { useHotspotStore } from '../../stores/hotspotStore';
import { useHotspotStream } from '../../hooks/useHotspotStream';
import { hotspotApi } from '../../api/client';
import HotspotProgress from './HotspotProgress';
import ThemeCard from './ThemeCard';
import BriefingViewer from './BriefingViewer';

interface Props {
  onStockClick?: (symbol: string) => void;
}

export default function HotspotView({ onStockClick }: Props) {
  const store = useHotspotStore();

  // SSE connection
  useHotspotStream({
    enabled: true,
    onTaskCreated: (task) => {
      if (!store.currentTaskId) {
        store.submitTask(task.task_id);
      }
      store.updateFromSSE(task);
    },
    onTaskStarted: (task) => store.updateFromSSE(task),
    onCollectingDone: (task) => store.updateFromSSE(task),
    onThemesExtracted: (data) => {
      store.updateFromSSE(data);
      if (data.themes) {
        store.setThemesExtracted(data.themes);
      }
    },
    onThemesAnalyzed: (task) => store.updateFromSSE(task),
    onCompleted: (data) => {
      // Fetch full result
      if (data.task_id) {
        hotspotApi.getStatus(data.task_id).then((resp) => {
          if (resp.data.result) {
            store.setCompleted(resp.data.result);
          }
        }).catch(() => {
          store.updateFromSSE(data);
        });
      }
    },
    onFailed: (task) => {
      store.setFailed(task.error || '分析失败');
    },
  });

  const handleTrigger = useCallback(async () => {
    try {
      store.reset();
      const resp = await hotspotApi.analyze();
      if (resp.status === 202) {
        store.submitTask(resp.data.task_id);
      }
    } catch (err: any) {
      if (err.response?.status === 409) {
        const data = err.response.data;
        store.submitTask(data.existing_task_id);
        // Fetch current status
        try {
          const statusResp = await hotspotApi.getStatus(data.existing_task_id);
          store.updateFromSSE(statusResp.data);
        } catch {}
      }
    }
  }, []);

  const handleLoadLatest = useCallback(async () => {
    try {
      const resp = await hotspotApi.getLatest();
      if (resp.data.result) {
        store.setCompleted(resp.data.result);
      }
    } catch {
      // No latest result
    }
  }, []);

  const isRunning = store.status !== 'idle' && store.status !== 'completed' && store.status !== 'failed';
  const hasResult = store.result !== null;

  return (
    <div className="max-w-4xl space-y-4">
      {/* Action bar */}
      <div className="terminal-card p-3 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-medium text-white">盘前热点分析</h3>
          <p className="text-[10px] text-[var(--text-muted)] mt-0.5">
            热点新闻 → 产业链拆解 → 核心标的
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleLoadLatest}
            disabled={isRunning}
            className="px-3 py-1.5 text-xs rounded border border-white/10 text-[var(--text-secondary)]
              hover:bg-white/5 disabled:opacity-30 transition-colors"
          >
            加载最新
          </button>
          <button
            onClick={handleTrigger}
            disabled={isRunning}
            className="px-3 py-1.5 text-xs rounded bg-[var(--color-cyan)]/20 text-[var(--color-cyan)]
              hover:bg-[var(--color-cyan)]/30 disabled:opacity-30 transition-colors"
          >
            {isRunning ? '分析中...' : '开始分析'}
          </button>
        </div>
      </div>

      {/* Progress */}
      {isRunning && (
        <HotspotProgress
          status={store.status}
          progress={store.progress}
          message={store.message}
          currentTheme={store.currentTheme}
          totalThemes={store.totalThemes}
        />
      )}

      {/* Results */}
      {hasResult && store.result && (
        <>
          {/* Theme cards */}
          {store.result.analyzed_themes.length > 0 && (
            <div className="space-y-3">
              <h4 className="text-xs text-[var(--text-muted)] font-medium">
                投资主题 ({store.result.analyzed_themes.length})
              </h4>
              {store.result.analyzed_themes.map((theme, idx) => (
                <ThemeCard
                  key={idx}
                  theme={theme}
                  index={idx}
                  onStockClick={onStockClick}
                />
              ))}
            </div>
          )}

          {/* Briefing */}
          {store.result.briefing && (
            <div>
              <h4 className="text-xs text-[var(--text-muted)] font-medium mb-2">完整研报</h4>
              <BriefingViewer content={store.result.briefing} />
            </div>
          )}

          {/* Errors */}
          {store.result.errors && store.result.errors.length > 0 && (
            <div className="terminal-card p-3 border-[var(--color-danger)]/30">
              <h4 className="text-xs font-medium text-[var(--color-danger)] mb-1">分析过程中的错误</h4>
              <ul className="text-[10px] text-[var(--text-muted)] space-y-0.5">
                {store.result.errors.map((err, i) => (
                  <li key={i}>{err}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}

      {/* Empty state */}
      {!isRunning && !hasResult && (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <div className="w-12 h-12 mb-3 rounded-xl bg-[var(--bg-elevated)] flex items-center justify-center">
            <svg className="w-6 h-6 text-[var(--text-muted)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          </div>
          <h3 className="text-base font-medium text-white mb-1.5">盘前热点分析</h3>
          <p className="text-xs text-[var(--text-muted)] max-w-xs">
            点击"开始分析"自动采集市场热点，提炼投资主题，拆解产业链，识别核心标的
          </p>
          <p className="text-[10px] text-[var(--text-muted)] mt-2">
            预计耗时 15-25 分钟（Ollama本地推理）
          </p>
        </div>
      )}
    </div>
  );
}
