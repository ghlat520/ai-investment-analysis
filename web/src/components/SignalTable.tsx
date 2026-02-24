import { useState } from 'react';
import type { AgentSignal } from '../types';
import { AGENT_NAMES } from '../types';

interface SignalTableProps {
  signals: AgentSignal[];
  weights: Record<string, number>;
}

function ScoreBar({ score }: { score: number }) {
  const pct = Math.abs(score);
  const color =
    score >= 30
      ? 'bg-[var(--color-bull)]'
      : score <= -30
        ? 'bg-[var(--color-bear)]'
        : 'bg-[var(--color-warning)]';

  return (
    <div className="flex items-center gap-2 w-32">
      {/* Negative side */}
      <div className="flex-1 flex justify-end">
        {score < 0 && (
          <div className={`h-2 rounded-l ${color}`} style={{ width: `${pct}%` }} />
        )}
      </div>
      {/* Center line */}
      <div className="w-px h-3 bg-white/20 shrink-0" />
      {/* Positive side */}
      <div className="flex-1">
        {score > 0 && (
          <div className={`h-2 rounded-r ${color}`} style={{ width: `${pct}%` }} />
        )}
      </div>
    </div>
  );
}

function ExpandedRow({ signal }: { signal: AgentSignal }) {
  // 防御性编程：确保数组字段存在
  const keyFactors = signal.key_factors ?? [];
  const risks = signal.risks ?? [];

  return (
    <div className="px-4 py-3 bg-[var(--bg-elevated)]/50 border-t border-white/5 text-xs space-y-1.5">
      {keyFactors.length > 0 && (
        <div className="flex gap-2">
          <span className="text-[var(--text-muted)] shrink-0">关键因素:</span>
          <span className="text-[var(--text-secondary)]">{keyFactors.join(' · ')}</span>
        </div>
      )}
      {risks.length > 0 && (
        <div className="flex gap-2">
          <span className="text-[var(--text-muted)] shrink-0">风险提示:</span>
          <span className="text-[var(--text-secondary)]">{risks.slice(0, 3).join(' · ')}</span>
        </div>
      )}
      {signal.reasoning && (
        <div className="flex gap-2">
          <span className="text-[var(--text-muted)] shrink-0">分析摘要:</span>
          <span className="text-[var(--text-secondary)] line-clamp-2">{signal.reasoning}</span>
        </div>
      )}
      <div className="flex gap-4 text-[var(--text-muted)]">
        {signal.llm_model && <span>模型: {signal.llm_model}</span>}
        {signal.execution_time_ms > 0 && <span>耗时: {(signal.execution_time_ms / 1000).toFixed(1)}s</span>}
      </div>
    </div>
  );
}

export default function SignalTable({ signals, weights }: SignalTableProps) {
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());

  if (!signals || signals.length === 0) return null;

  const sorted = [...signals].sort((a, b) => b.signal_score - a.signal_score);

  const toggleRow = (name: string) => {
    setExpandedRows((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  return (
    <div className="terminal-card animate-slide-up">
      <div className="px-5 pt-4 pb-3">
        <span className="label-uppercase">SIGNAL MATRIX 九维信号</span>
      </div>

      {/* Table header */}
      <div className="grid grid-cols-[1fr_60px_50px_56px_128px] gap-2 px-5 py-2 text-xs text-[var(--text-muted)] border-b border-white/5">
        <span>维度</span>
        <span className="text-right">评分</span>
        <span className="text-right">权重</span>
        <span className="text-right">置信度</span>
        <span className="text-center">评分条</span>
      </div>

      {/* Rows */}
      <div className="divide-y divide-white/5">
        {sorted.map((signal) => {
          const expanded = expandedRows.has(signal.agent_name);
          const label = AGENT_NAMES[signal.agent_name] || signal.agent_name;
          const weight = weights[signal.agent_name];
          const weightPct = weight != null ? (weight * 100).toFixed(0) : '-';
          const confidencePct = (signal.confidence * 100).toFixed(0);

          const scoreColor =
            signal.signal_score >= 30
              ? 'text-[var(--color-bull)]'
              : signal.signal_score <= -30
                ? 'text-[var(--color-bear)]'
                : 'text-[var(--color-warning)]';

          return (
            <div key={signal.agent_name}>
              <button
                type="button"
                onClick={() => toggleRow(signal.agent_name)}
                className="w-full grid grid-cols-[1fr_60px_50px_56px_128px] gap-2 px-5 py-2.5 text-sm hover:bg-[var(--bg-hover)] transition-colors items-center"
              >
                <span className="flex items-center gap-1.5 text-left">
                  <span className={`text-xs transition-transform ${expanded ? 'rotate-90' : ''}`}>&#9654;</span>
                  <span className="text-white font-medium">{label}</span>
                </span>
                <span className={`text-right font-mono font-bold ${scoreColor}`}>
                  {signal.signal_score > 0 ? '+' : ''}{signal.signal_score}
                </span>
                <span className="text-right text-[var(--text-muted)]">{weightPct}%</span>
                <span className="text-right text-[var(--text-muted)]">{confidencePct}%</span>
                <ScoreBar score={signal.signal_score} />
              </button>
              {expanded && <ExpandedRow signal={signal} />}
            </div>
          );
        })}
      </div>
    </div>
  );
}
