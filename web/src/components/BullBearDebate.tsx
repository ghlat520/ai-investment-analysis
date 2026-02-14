import type { FusionDecision } from '../types';

interface BullBearDebateProps {
  fusion: FusionDecision;
}

export default function BullBearDebate({ fusion }: BullBearDebateProps) {
  const { bull_arguments, bear_arguments, divergence_points } = fusion;

  if (!bull_arguments.length && !bear_arguments.length) return null;

  return (
    <div>
      <div className="grid grid-cols-2 gap-4">
        {/* Bull */}
        <div className="rounded-lg border border-[var(--color-bull)]/20 bg-[var(--color-bull)]/5 p-4">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-[var(--color-bull)] text-lg">&#9650;</span>
            <span className="font-medium text-[var(--color-bull)] text-sm">看多论据</span>
          </div>
          <ul className="space-y-2 text-sm">
            {bull_arguments.map((arg, i) => (
              <li key={i} className="text-[var(--text-secondary)] flex gap-2">
                <span className="text-[var(--color-bull)] mt-0.5 shrink-0">+</span>
                <span>{arg}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Bear */}
        <div className="rounded-lg border border-[var(--color-bear)]/20 bg-[var(--color-bear)]/5 p-4">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-[var(--color-bear)] text-lg">&#9660;</span>
            <span className="font-medium text-[var(--color-bear)] text-sm">看空论据</span>
          </div>
          <ul className="space-y-2 text-sm">
            {bear_arguments.map((arg, i) => (
              <li key={i} className="text-[var(--text-secondary)] flex gap-2">
                <span className="text-[var(--color-bear)] mt-0.5 shrink-0">-</span>
                <span>{arg}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {divergence_points.length > 0 && (
        <div className="mt-3 rounded-lg border border-[var(--color-warning)]/20 bg-[var(--color-warning)]/5 p-3">
          <div className="text-sm font-medium text-[var(--color-warning)] mb-1">分歧点</div>
          <ul className="text-sm text-[var(--text-secondary)] space-y-1">
            {divergence_points.map((pt, i) => (
              <li key={i} className="flex gap-2">
                <span className="text-[var(--color-warning)] shrink-0">&bull;</span>
                <span>{pt}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
