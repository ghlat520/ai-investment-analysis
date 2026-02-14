import { useState } from 'react';
import type { AnalyzedTheme } from '../../types/hotspot';
import IndustryChainDiagram from './IndustryChainDiagram';

interface Props {
  theme: AnalyzedTheme;
  index: number;
  onStockClick?: (symbol: string) => void;
}

const ACTIONABILITY_COLORS: Record<string, string> = {
  high: 'bg-[var(--color-success)]/20 text-[var(--color-success)]',
  medium: 'bg-[var(--color-warning)]/20 text-[var(--color-warning)]',
  low: 'bg-white/10 text-[var(--text-muted)]',
};

const ACTIONABILITY_LABELS: Record<string, string> = {
  high: '高',
  medium: '中',
  low: '低',
};

export default function ThemeCard({ theme, index, onStockClick }: Props) {
  const [expanded, setExpanded] = useState(index === 0);

  const totalStocks = theme.upstream.length + theme.midstream.length + theme.downstream.length;

  return (
    <div className="terminal-card overflow-hidden">
      {/* Header */}
      <div
        className="p-3 cursor-pointer hover:bg-white/5 transition-colors flex items-center justify-between"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-xs text-[var(--color-cyan)] font-mono flex-shrink-0">#{index + 1}</span>
          <h4 className="text-sm font-medium text-white truncate">{theme.title}</h4>
          <span className={`text-[10px] px-1.5 py-0.5 rounded flex-shrink-0 ${ACTIONABILITY_COLORS[theme.actionability] || ACTIONABILITY_COLORS.medium}`}>
            {ACTIONABILITY_LABELS[theme.actionability] || '中'}
          </span>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0 ml-2">
          <span className="text-[10px] text-[var(--text-muted)]">{totalStocks}只标的</span>
          <svg
            className={`w-4 h-4 text-[var(--text-muted)] transition-transform ${expanded ? 'rotate-180' : ''}`}
            fill="none" stroke="currentColor" viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>

      {/* Expanded content */}
      {expanded && (
        <div className="px-3 pb-3 space-y-3 border-t border-white/5 pt-3">
          {/* Investment logic */}
          {theme.investment_logic && (
            <div>
              <div className="text-[10px] text-[var(--text-muted)] mb-1">投资逻辑</div>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed">{theme.investment_logic}</p>
            </div>
          )}

          {/* Industry chain diagram */}
          <div>
            <div className="text-[10px] text-[var(--text-muted)] mb-2">产业链</div>
            <IndustryChainDiagram
              upstream={theme.upstream}
              midstream={theme.midstream}
              downstream={theme.downstream}
              valueFlow={theme.value_flow}
              onStockClick={onStockClick}
            />
          </div>
        </div>
      )}
    </div>
  );
}
