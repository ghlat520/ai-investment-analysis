import type { FusionDecision, AgentSignal } from '../types';

interface StrategyPointsProps {
  fusion: FusionDecision;
  signals: AgentSignal[];
}

/** Extract latest_close from technical agent's extra_data */
function getCurrentPrice(signals: AgentSignal[]): number | null {
  const tech = signals.find((s) => s.agent_name === 'technical');
  if (tech?.extra_data?.latest_close) {
    return tech.extra_data.latest_close;
  }
  return null;
}

/** Generate reasoning for stop-loss level */
function getStopLossReason(fusion: FusionDecision): string {
  const score = fusion.final_score;
  const conf = fusion.confidence;
  if (score < -20) {
    return '评分偏空，设置较紧止损以控制风险';
  }
  if (conf < 0.5) {
    return '置信度较低，收窄止损幅度降低敞口';
  }
  return '基于综合评分和波动率动态计算';
}

/** Generate reasoning for take-profit level */
function getTakeProfitReason(fusion: FusionDecision): string {
  const score = fusion.final_score;
  if (score >= 40) {
    return '多方信号强劲，设置较宽止盈空间';
  }
  if (score >= 20) {
    return '温和看多，目标价基于估值和动量';
  }
  if (score < -20) {
    return '空头环境，止盈目标保守';
  }
  return '基于风险收益比动态计算';
}

interface PointCardData {
  label: string;
  price: string | null;
  pct: string;
  reason: string;
  color: string;
  icon: string;
}

export default function StrategyPoints({ fusion, signals }: StrategyPointsProps) {
  const { stop_loss_pct, take_profit_pct, target_prices } = fusion;
  const currentPrice = getCurrentPrice(signals);

  const cards: PointCardData[] = [];

  // Stop loss
  if (stop_loss_pct != null) {
    const pctVal = stop_loss_pct > 0 ? -stop_loss_pct : stop_loss_pct;
    const price = currentPrice ? currentPrice * (1 + pctVal / 100) : null;
    cards.push({
      label: '止损价位',
      price: price ? `¥${price.toFixed(2)}` : null,
      pct: `${pctVal.toFixed(1)}%`,
      reason: getStopLossReason(fusion),
      color: 'var(--color-danger)',
      icon: 'M19 14l-7 7m0 0l-7-7m7 7V3',
    });
  }

  // Target prices from valuation
  const priceKeys = Object.keys(target_prices || {});
  if (priceKeys.length > 0) {
    const sorted = priceKeys.sort();
    for (const key of sorted) {
      const targetPrice = target_prices[key];
      const pctFromCurrent = currentPrice
        ? ((targetPrice / currentPrice - 1) * 100).toFixed(1)
        : null;
      cards.push({
        label: key,
        price: `¥${targetPrice.toFixed(2)}`,
        pct: pctFromCurrent ? `${Number(pctFromCurrent) >= 0 ? '+' : ''}${pctFromCurrent}%` : '',
        reason: '基于估值模型和历史分位计算',
        color: 'var(--color-cyan)',
        icon: 'M5 10l7-7m0 0l7 7m-7-7v18',
      });
    }
  }

  // Take profit
  if (take_profit_pct != null) {
    const price = currentPrice ? currentPrice * (1 + take_profit_pct / 100) : null;
    cards.push({
      label: '止盈目标',
      price: price ? `¥${price.toFixed(2)}` : null,
      pct: `+${take_profit_pct.toFixed(1)}%`,
      reason: getTakeProfitReason(fusion),
      color: 'var(--color-warning)',
      icon: 'M13 7h8m0 0v8m0-8l-8 8-4-4-6 6',
    });
  }

  if (cards.length === 0) return null;

  return (
    <div className="animate-slide-up">
      <div className="terminal-card p-5">
        <div className="flex items-center justify-between mb-3">
          <span className="label-uppercase">STRATEGY POINTS 狙击点位</span>
          {currentPrice && (
            <span className="text-xs text-[var(--text-muted)]">
              现价 <span className="text-white font-mono">¥{currentPrice.toFixed(2)}</span>
            </span>
          )}
        </div>
        <div className="grid gap-3 grid-cols-1 sm:grid-cols-2">
          {cards.map((card) => (
            <div
              key={card.label}
              className="rounded-lg p-4 border border-white/5 bg-[var(--bg-elevated)] transition-all hover:border-white/10"
            >
              <div className="flex items-center gap-2 mb-2">
                <svg
                  className="w-4 h-4 flex-shrink-0"
                  style={{ color: card.color }}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={card.icon} />
                </svg>
                <span className="text-xs text-[var(--text-muted)]">{card.label}</span>
              </div>
              <div className="flex items-baseline gap-2">
                {card.price && (
                  <span
                    className="text-xl font-bold font-mono"
                    style={{ color: card.color }}
                  >
                    {card.price}
                  </span>
                )}
                <span
                  className={`text-sm font-mono ${card.price ? 'text-[var(--text-muted)]' : 'text-xl font-bold'}`}
                  style={card.price ? undefined : { color: card.color }}
                >
                  {card.pct}
                </span>
              </div>
              <p className="text-xs text-[var(--text-muted)] mt-2 leading-relaxed">
                {card.reason}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
