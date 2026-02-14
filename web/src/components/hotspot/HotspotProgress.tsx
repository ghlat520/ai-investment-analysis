import { HOTSPOT_STAGE_LABELS, type HotspotStage } from '../../types/hotspot';

interface Props {
  status: HotspotStage;
  progress: number;
  message: string;
  currentTheme: number;
  totalThemes: number;
}

const STAGES: HotspotStage[] = ['collecting', 'extracting', 'analyzing', 'generating', 'completed'];

export default function HotspotProgress({ status, progress, message, currentTheme, totalThemes }: Props) {
  if (status === 'idle') return null;

  const currentStageIdx = STAGES.indexOf(status);

  return (
    <div className="terminal-card p-4">
      <h4 className="text-sm font-medium text-white mb-3">分析进度</h4>

      {/* Stage steps */}
      <div className="flex items-center gap-1 mb-4">
        {STAGES.map((stage, idx) => {
          const isDone = idx < currentStageIdx || status === 'completed';
          const isCurrent = idx === currentStageIdx && status !== 'completed';
          return (
            <div key={stage} className="flex-1 flex flex-col items-center gap-1">
              <div
                className={`w-full h-1.5 rounded-full transition-all duration-500 ${
                  isDone
                    ? 'bg-[var(--color-success)]'
                    : isCurrent
                      ? 'bg-[var(--color-cyan)] animate-pulse'
                      : 'bg-white/10'
                }`}
              />
              <span className={`text-[10px] ${isCurrent ? 'text-[var(--color-cyan)]' : isDone ? 'text-[var(--color-success)]' : 'text-[var(--text-muted)]'}`}>
                {HOTSPOT_STAGE_LABELS[stage]}
              </span>
            </div>
          );
        })}
      </div>

      {/* Progress bar */}
      <div className="w-full bg-white/5 rounded-full h-1.5 mb-2">
        <div
          className="h-1.5 rounded-full bg-[var(--color-cyan)] transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Message */}
      <div className="flex justify-between text-xs text-[var(--text-muted)]">
        <span>{message || HOTSPOT_STAGE_LABELS[status]}</span>
        <span>{progress}%</span>
      </div>

      {/* Theme analysis sub-progress */}
      {status === 'analyzing' && totalThemes > 0 && (
        <div className="mt-2 text-xs text-[var(--text-secondary)]">
          产业链分析: {currentTheme}/{totalThemes} 个主题
        </div>
      )}

      {/* Error */}
      {status === 'failed' && (
        <div className="mt-2 text-xs text-[var(--color-danger)]">
          分析失败，请重试
        </div>
      )}
    </div>
  );
}
