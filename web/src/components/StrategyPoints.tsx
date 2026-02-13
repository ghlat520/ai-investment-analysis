import type { FusionDecision } from '../types';

interface StrategyPointsProps {
  fusion: FusionDecision;
}

interface PointCard {
  label: string;
  value: string;
  color: string;
  icon: string;
}

export default function StrategyPoints({ fusion }: StrategyPointsProps) {
  const { stop_loss_pct, take_profit_pct, target_prices } = fusion;

  // Build strategy point cards from available data
  const cards: PointCard[] = [];

  if (stop_loss_pct != null) {
    cards.push({
      label: '止损价位',
      value: `${stop_loss_pct > 0 ? '-' : ''}${Math.abs(stop_loss_pct).toFixed(1)}%`,
      color: 'var(--color-danger)',
      icon: 'M19 14l-7 7m0 0l-7-7m7 7V3',
    });
  }

  // Use target_prices for buy points
  const priceKeys = Object.keys(target_prices || {});
  if (priceKeys.length > 0) {
    const sorted = priceKeys.sort();
    if (sorted[0]) {
      cards.push({
        label: '理想买入',
        value: `¥${target_prices[sorted[0]].toFixed(2)}`,
        color: 'var(--color-success)',
        icon: 'M5 10l7-7m0 0l7 7m-7-7v18',
      });
    }
    if (sorted[1]) {
      cards.push({
        label: '二次买入',
        value: `¥${target_prices[sorted[1]].toFixed(2)}`,
        color: 'var(--color-cyan)',
        icon: 'M5 10l7-7m0 0l7 7m-7-7v18',
      });
    }
  }

  if (take_profit_pct != null) {
    cards.push({
      label: '止盈目标',
      value: `+${take_profit_pct.toFixed(1)}%`,
      color: 'var(--color-warning)',
      icon: 'M13 7h8m0 0v8m0-8l-8 8-4-4-6 6',
    });
  }

  // Add remaining target prices as additional cards
  if (priceKeys.length > 2) {
    for (let i = 2; i < priceKeys.length; i++) {
      const key = priceKeys[i];
      cards.push({
        label: key,
        value: `¥${target_prices[key].toFixed(2)}`,
        color: 'var(--color-cyan)',
        icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2z',
      });
    }
  }

  if (cards.length === 0) return null;

  return (
    <div className="animate-slide-up">
      <div className="terminal-card p-5">
        <span className="label-uppercase">STRATEGY POINTS 狙击点位</span>
        <div className={`grid gap-3 mt-3 ${cards.length >= 4 ? 'grid-cols-2 md:grid-cols-4' : cards.length >= 2 ? 'grid-cols-2' : 'grid-cols-1'}`}>
          {cards.map((card) => (
            <div
              key={card.label}
              className="rounded-lg p-3 border border-white/5 bg-[var(--bg-elevated)] transition-all hover:border-white/10"
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
              <div
                className="text-lg font-bold font-mono"
                style={{ color: card.color }}
              >
                {card.value}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
