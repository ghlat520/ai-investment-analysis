import type { TaskInfo } from '../types';
import { getEastMoneyUrl } from '../utils/stockLink';

function TaskItem({ task }: { task: TaskInfo }) {
  const isPending = task.status === 'pending';
  const isProcessing = task.status === 'collecting' || task.status === 'analyzing';

  return (
    <div className="flex items-center gap-3 px-3 py-2 bg-[var(--bg-elevated)] rounded-lg border border-white/5">
      <div className="shrink-0">
        {isProcessing ? (
          <svg className="w-4 h-4 text-[var(--color-cyan)] animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
        ) : isPending ? (
          <svg className="w-4 h-4 text-[var(--text-muted)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        ) : null}
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-white truncate">
            {task.stock_name || task.symbol}
          </span>
          <a
            href={getEastMoneyUrl(task.symbol, task.market)}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-[var(--text-muted)] hover:text-[var(--color-cyan)] transition-colors"
            title="在东方财富查看行情"
          >
            {task.symbol}
            <svg className="w-2.5 h-2.5 inline ml-0.5 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
          </a>
        </div>
        {task.message && (
          <p className="text-xs text-[var(--text-secondary)] truncate mt-0.5">{task.message}</p>
        )}
      </div>

      <div className="flex-shrink-0">
        <span className={`text-xs px-1.5 py-0.5 rounded ${
          isProcessing ? 'bg-[var(--color-cyan)]/20 text-[var(--color-cyan)]' : 'bg-white/10 text-[var(--text-muted)]'
        }`}>
          {isProcessing ? '分析中' : '等待中'}
        </span>
      </div>
    </div>
  );
}

interface TaskPanelProps {
  tasks: TaskInfo[];
  className?: string;
}

export default function TaskPanel({ tasks, className = '' }: TaskPanelProps) {
  const activeTasks = tasks.filter(
    (t) => t.status === 'pending' || t.status === 'collecting' || t.status === 'analyzing',
  );

  if (activeTasks.length === 0) return null;

  const pendingCount = activeTasks.filter((t) => t.status === 'pending').length;
  const processingCount = activeTasks.length - pendingCount;

  return (
    <div className={`terminal-card overflow-hidden ${className}`}>
      <div className="flex items-center justify-between px-3 py-2 border-b border-white/5">
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-[var(--color-cyan)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          <span className="text-sm font-medium text-white">分析任务</span>
        </div>
        <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
          {processingCount > 0 && (
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 bg-[var(--color-cyan)] rounded-full animate-pulse" />
              {processingCount} 进行中
            </span>
          )}
          {pendingCount > 0 && <span>{pendingCount} 等待中</span>}
        </div>
      </div>

      <div className="p-2 space-y-2 max-h-64 overflow-y-auto">
        {activeTasks.map((task) => (
          <TaskItem key={task.task_id} task={task} />
        ))}
      </div>
    </div>
  );
}
