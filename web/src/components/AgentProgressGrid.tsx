import type { AgentState } from '../stores/analysisStore';
import { AGENT_NAMES } from '../types';

function AgentCard({ agent }: { agent: AgentState }) {
  const label = AGENT_NAMES[agent.name] || agent.name;

  const borderColor =
    agent.status === 'completed'
      ? 'border-[var(--color-success)]/40'
      : agent.status === 'running'
        ? 'border-[var(--color-cyan)]/40 animate-pulse'
        : agent.status === 'error'
          ? 'border-[var(--color-danger)]/40'
          : 'border-white/5 opacity-50';

  const scoreColor =
    agent.signal_score != null
      ? agent.signal_score >= 30
        ? 'text-[var(--color-bull)]'
        : agent.signal_score <= -30
          ? 'text-[var(--color-bear)]'
          : 'text-[var(--color-warning)]'
      : '';

  return (
    <div className={`rounded-lg border p-3 bg-[var(--bg-card)] transition-all duration-300 ${borderColor}`}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-medium text-white">{label}</span>
        {agent.status === 'completed' && <span className="text-xs text-[var(--color-success)]">done</span>}
        {agent.status === 'pending' && <span className="text-xs text-[var(--text-muted)]">...</span>}
        {agent.status === 'running' && (
          <span className="w-2 h-2 rounded-full bg-[var(--color-cyan)] animate-ping" />
        )}
      </div>
      {agent.signal_score != null && (
        <div className={`text-xl font-bold font-mono ${scoreColor}`}>
          {agent.signal_score > 0 ? '+' : ''}{agent.signal_score}
        </div>
      )}
      {agent.confidence != null && (
        <div className="text-xs text-[var(--text-muted)]">置信度 {(agent.confidence * 100).toFixed(0)}%</div>
      )}
      {agent.execution_time_ms != null && (
        <div className="text-xs text-[var(--text-muted)]">{(agent.execution_time_ms / 1000).toFixed(1)}s</div>
      )}
    </div>
  );
}

interface AgentProgressGridProps {
  agents: AgentState[];
  visible: boolean;
}

export default function AgentProgressGrid({ agents, visible }: AgentProgressGridProps) {
  if (!visible) return null;

  return (
    <div className="terminal-card p-5 animate-slide-up">
      <span className="label-uppercase mb-3 block">AGENT PROGRESS 分析进度</span>
      <div className="grid grid-cols-3 gap-3 mt-3">
        {agents.map((agent) => (
          <AgentCard key={agent.name} agent={agent} />
        ))}
      </div>
    </div>
  );
}
