import { useRef, useCallback, useEffect } from 'react';
import type { HistoryItem } from '../types';

interface HistoryListProps {
  items: HistoryItem[];
  isLoading: boolean;
  isLoadingMore: boolean;
  hasMore: boolean;
  selectedRunId?: string;
  onItemClick: (runId: string) => void;
  onLoadMore: () => void;
  className?: string;
}

function getScoreColor(score: number): string {
  if (score >= 30) return '#00d4ff';
  if (score >= -30) return '#a855f7';
  return '#ff4466';
}

function formatTime(dateStr: string | null): string {
  if (!dateStr) return '';
  const d = new Date(dateStr);
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  const hh = String(d.getHours()).padStart(2, '0');
  const min = String(d.getMinutes()).padStart(2, '0');
  return `${mm}-${dd} ${hh}:${min}`;
}

export default function HistoryList({
  items,
  isLoading,
  isLoadingMore,
  hasMore,
  selectedRunId,
  onItemClick,
  onLoadMore,
  className = '',
}: HistoryListProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLDivElement>(null);

  const handleObserver = useCallback(
    (entries: IntersectionObserverEntry[]) => {
      const target = entries[0];
      if (target.isIntersecting && hasMore && !isLoading && !isLoadingMore) {
        const container = scrollRef.current;
        if (container && container.scrollHeight > container.clientHeight) {
          onLoadMore();
        }
      }
    },
    [hasMore, isLoading, isLoadingMore, onLoadMore],
  );

  useEffect(() => {
    const trigger = triggerRef.current;
    const container = scrollRef.current;
    if (!trigger || !container) return;

    const observer = new IntersectionObserver(handleObserver, {
      root: container,
      rootMargin: '20px',
      threshold: 0.1,
    });
    observer.observe(trigger);
    return () => observer.disconnect();
  }, [handleObserver]);

  return (
    <aside className={`glass-card overflow-hidden flex flex-col ${className}`}>
      <div ref={scrollRef} className="p-3 flex-1 overflow-y-auto">
        <h2 className="text-xs font-medium text-cyan uppercase tracking-wider mb-3 flex items-center gap-1.5">
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          历史记录
        </h2>

        {isLoading ? (
          <div className="flex justify-center py-6">
            <div className="w-5 h-5 border-2 border-[var(--color-cyan)]/20 border-t-[var(--color-cyan)] rounded-full animate-spin" />
          </div>
        ) : items.length === 0 ? (
          <div className="text-center py-6 text-muted text-xs">暂无历史记录</div>
        ) : (
          <div className="space-y-1.5">
            {items.map((item) => (
              <button
                key={item.run_id}
                type="button"
                onClick={() => onItemClick(item.run_id)}
                className={`history-item w-full text-left ${selectedRunId === item.run_id ? 'active' : ''}`}
              >
                <div className="flex items-center gap-2 w-full">
                  <span
                    className="w-0.5 h-8 rounded-full flex-shrink-0"
                    style={{
                      backgroundColor: getScoreColor(item.final_score),
                      boxShadow: `0 0 6px ${getScoreColor(item.final_score)}40`,
                    }}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-1.5">
                      <span className="font-medium text-white truncate text-xs">
                        {item.stock_name && item.stock_name !== item.symbol
                          ? `${item.stock_name} ${item.symbol}`
                          : item.symbol}
                      </span>
                      <span
                        className="text-xs font-mono font-semibold px-1 py-0.5 rounded"
                        style={{
                          color: getScoreColor(item.final_score),
                          backgroundColor: `${getScoreColor(item.final_score)}15`,
                        }}
                      >
                        {item.final_score > 0 ? '+' : ''}{item.final_score}
                      </span>
                    </div>
                    <div className="flex items-center gap-1.5 mt-0.5">
                      <span className="text-xs text-muted">
                        {item.final_action}
                      </span>
                      <span className="text-xs text-muted/50">&middot;</span>
                      <span className="text-xs text-muted">
                        {formatTime(item.created_at)}
                      </span>
                    </div>
                  </div>
                </div>
              </button>
            ))}

            <div ref={triggerRef} className="h-4" />

            {isLoadingMore && (
              <div className="flex justify-center py-3">
                <div className="w-4 h-4 border-2 border-[var(--color-cyan)]/20 border-t-[var(--color-cyan)] rounded-full animate-spin" />
              </div>
            )}

            {!hasMore && items.length > 0 && (
              <div className="text-center py-2 text-muted/50 text-xs">已加载全部</div>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}
