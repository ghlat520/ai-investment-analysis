import type { FusionDecision } from '../types';
import ScoreGauge from './ScoreGauge';
import { getEastMoneyUrl } from '../utils/stockLink';

interface ReportOverviewProps {
  fusion: FusionDecision;
  stockName: string;
  symbol: string;
  createdAt?: string;
}

interface AgentInsight {
  name: string;
  score: number;
  confidence: number;
  weight: number;
  description: string;
}

// 解析 reasoning 文本，提取各维度数据
function parseReasoning(reasoning: string): {
  agents: AgentInsight[];
} {
  const agents: AgentInsight[] = [];

  // 解析各维度: - dimension_name(+score, 置信conf%, 权重weight%): description
  const pattern = /-\s*(\w+)\(([+-]?\d+),\s*置信(\d+)%,\s*权重(\d+)%\):\s*(.+?)(?=\s*-\s*\w+\(|$)/g;
  let match;
  while ((match = pattern.exec(reasoning)) !== null) {
    agents.push({
      name: match[1],
      score: parseInt(match[2]),
      confidence: parseInt(match[3]),
      weight: parseInt(match[4]),
      description: match[5].trim(),
    });
  }

  return { agents };
}

// 维度名称映射（英文 -> 中文）
const dimensionNames: Record<string, string> = {
  business_model: '商业模式',
  competition: '竞争格局',
  fundamentals: '基本面',
  growth: '成长性',
  valuation: '估值',
  technical: '技术面',
  sentiment: '情绪面',
  market_regime: '市场状态',
  news: '新闻舆情',
  risk: '风险',
  quality: '质量',
  momentum: '动量',
};

// 获取得分对应的颜色类名
function getScoreColorClass(score: number): string {
  if (score >= 30) return 'text-[var(--color-bull)]';
  if (score <= -30) return 'text-[var(--color-bear)]';
  return 'text-[var(--color-warning)]';
}

// Tooltip 组件 - 暂未使用，保留备用
// function InfoTooltip({ content }: { content: string }) { ... }

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
                    <a
                      href={getEastMoneyUrl(symbol)}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 font-mono text-xs text-[var(--color-cyan)] bg-[var(--color-cyan)]/10 px-1.5 py-0.5 rounded hover:bg-[var(--color-cyan)]/20 transition-colors"
                      title="在东方财富查看行情"
                      onClick={(e) => e.stopPropagation()}
                    >
                      {symbol}
                      <svg className="w-3 h-3 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                      </svg>
                    </a>
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
                {(() => {
                  const { agents } = parseReasoning(fusion.reasoning || '');
                  if (agents.length === 0) {
                    return (
                      <p className="text-white text-sm leading-relaxed mt-1.5">
                        {fusion.reasoning || '暂无分析结论'}
                      </p>
                    );
                  }
                  return (
                    <div className="mt-3">
                      {/* 摘要头部 */}
                      <div className="grid grid-cols-3 gap-3 mb-4">
                        <div className="bg-[var(--bg-elevated)] rounded-lg p-3 text-center">
                          <div className="text-xs text-[var(--text-muted)] mb-1">综合评分</div>
                          <div className={`text-lg font-bold ${getScoreColorClass(fusion.final_score)}`}>
                            {fusion.final_score > 0 ? '+' : ''}{fusion.final_score}
                          </div>
                        </div>
                        <div className="bg-[var(--bg-elevated)] rounded-lg p-3 text-center">
                          <div className="text-xs text-[var(--text-muted)] mb-1">信号数量</div>
                          <div className="text-lg font-bold text-white">{agents.length}</div>
                        </div>
                        <div className="bg-[var(--bg-elevated)] rounded-lg p-3 text-center">
                          <div className="text-xs text-[var(--text-muted)] mb-1">市场状态</div>
                          <div className="text-sm font-medium text-[var(--color-cyan)]">
                            {fusion.market_regime || '-'}
                          </div>
                        </div>
                      </div>
                      {/* 维度表格 */}
                      <table className="key-insights-table">
                        <thead>
                          <tr>
                            <th>维度</th>
                            <th className="text-center">得分</th>
                            <th className="text-center">置信度</th>
                            <th className="text-center">权重</th>
                            <th>分析</th>
                          </tr>
                        </thead>
                        <tbody>
                          {agents.map((agent, idx) => (
                            <tr key={idx}>
                              <td className="font-medium text-white">
                                {dimensionNames[agent.name] || agent.name}
                              </td>
                              <td className={`text-center font-mono ${getScoreColorClass(agent.score)}`}>
                                {agent.score > 0 ? '+' : ''}{agent.score}
                              </td>
                              <td className="text-center text-[var(--text-secondary)]">
                                {agent.confidence}%
                              </td>
                              <td className="text-center text-[var(--text-secondary)]">
                                {agent.weight}%
                              </td>
                              <td className="text-[var(--text-secondary)] text-xs max-w-xs">
                                <span className="line-clamp-2">{agent.description || '-'}</span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  );
                })()}
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
