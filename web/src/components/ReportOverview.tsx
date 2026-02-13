import type { FusionDecision } from '../types';
import ScoreGauge from './ScoreGauge';

interface ReportOverviewProps {
  fusion: FusionDecision;
  stockName: string;
  symbol: string;
  createdAt?: string;
}

function formatDate(dateStr?: string): string {
  if (!dateStr) return '';
  const d = new Date(dateStr);
  return `${d.getFullYear()}/${String(d.getMonth() + 1).padStart(2, '0')}/${String(d.getDate()).padStart(2, '0')}`;
}

function getTrendLabel(score: number): string {
  if (score > 20) return '看多';
  if (score < -20) return '看空';
  return '中性';
}

function getTrendColor(score: number): string {
  if (score > 20) return 'text-[var(--color-bull)]';
  if (score < -20) return 'text-[var(--color-bear)]';
  return 'text-[var(--color-warning)]';
}

export default function ReportOverview({ fusion, stockName, symbol, createdAt }: ReportOverviewProps) {
  return (
    <div className="space-y-4 animate-fade-in">
      {/* Main info - two column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left: stock info + insights */}
        <div className="lg:col-span-2 space-y-4">
          {/* Stock header */}
          <div className="gradient-border-card">
            <div className="gradient-border-card-inner p-5">
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <div className="flex items-center gap-3">
                    <h2 className="text-2xl font-bold text-white">{stockName || symbol}</h2>
                  </div>
                  <div className="flex items-center gap-2 mt-1.5">
                    <span className="font-mono text-xs text-[var(--color-cyan)] bg-[var(--color-cyan)]/10 px-1.5 py-0.5 rounded">
                      {symbol}
                    </span>
                    {createdAt && (
                      <span className="text-xs text-[var(--text-muted)] flex items-center gap-1">
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                        {formatDate(createdAt)}
                      </span>
                    )}
                    {fusion.market_regime && (
                      <span className="badge badge-cyan text-xs">{fusion.market_regime}</span>
                    )}
                  </div>
                </div>
              </div>

              {/* Key Insights */}
              <div className="border-t border-white/5 pt-4">
                <span className="label-uppercase">KEY INSIGHTS</span>
                <p className="text-white text-sm leading-relaxed mt-1.5 whitespace-pre-wrap text-left">
                  {fusion.reasoning || '暂无分析结论'}
                </p>
              </div>
            </div>
          </div>

          {/* Operation advice & trend prediction */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {/* Operation advice */}
            <div className="terminal-card terminal-card-hover p-4">
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-lg bg-[var(--color-success)]/10 flex items-center justify-center flex-shrink-0">
                  <svg className="w-4 h-4 text-[var(--color-success)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-xs font-medium text-[var(--color-success)] mb-0.5">操作建议</h4>
                  <p className="text-white text-sm font-medium">
                    {fusion.final_action || '暂无建议'}
                  </p>
                  {fusion.confidence > 0 && (
                    <p className="text-xs text-[var(--text-muted)] mt-1">
                      置信度 {(fusion.confidence * 100).toFixed(0)}%
                      {fusion.position_pct > 0 && ` · 建议仓位 ${fusion.position_pct}%`}
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Trend prediction */}
            <div className="terminal-card terminal-card-hover p-4">
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-lg bg-[var(--color-warning)]/10 flex items-center justify-center flex-shrink-0">
                  <svg className="w-4 h-4 text-[var(--color-warning)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-xs font-medium text-[var(--color-warning)] mb-0.5">趋势预测</h4>
                  <p className={`text-sm font-medium ${getTrendColor(fusion.final_score)}`}>
                    {getTrendLabel(fusion.final_score)}
                  </p>
                  <p className="text-xs text-[var(--text-muted)] mt-1">
                    综合评分 {fusion.final_score > 0 ? '+' : ''}{fusion.final_score}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right: sentiment gauge */}
        <div className="space-y-4">
          <div className="terminal-card p-5 !overflow-visible">
            <div className="text-center">
              <h3 className="text-sm font-medium text-white mb-4">Market Sentiment</h3>
              <ScoreGauge score={fusion.final_score} size="lg" showLabel={false} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
