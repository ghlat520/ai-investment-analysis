import type { ChainStock } from '../../types/hotspot';

interface Props {
  upstream: ChainStock[];
  midstream: ChainStock[];
  downstream: ChainStock[];
  valueFlow: string;
  onStockClick?: (symbol: string) => void;
}

function ChainColumn({ title, stocks, color, onStockClick }: {
  title: string;
  stocks: ChainStock[];
  color: string;
  onStockClick?: (symbol: string) => void;
}) {
  return (
    <div className="flex-1 min-w-0">
      <div className={`text-xs font-medium mb-2 px-2 py-1 rounded text-center ${color}`}>
        {title}
      </div>
      <div className="space-y-1.5">
        {stocks.map((stock) => (
          <div
            key={stock.symbol}
            className="terminal-card p-2 cursor-pointer hover:border-[var(--color-cyan)]/50 transition-colors"
            onClick={() => onStockClick?.(stock.symbol)}
            title={`点击查看 ${stock.name} 深度分析`}
          >
            <div className="flex items-center justify-between mb-0.5">
              <span className="text-xs font-medium text-white truncate">{stock.name}</span>
              <span className="text-[10px] text-[var(--text-muted)] ml-1 flex-shrink-0">{stock.symbol}</span>
            </div>
            <div className="text-[10px] text-[var(--text-secondary)] truncate" title={stock.role}>
              {stock.role}
            </div>
            {stock.reason && (
              <div className="text-[10px] text-[var(--text-muted)] mt-0.5 line-clamp-2" title={stock.reason}>
                {stock.reason}
              </div>
            )}
          </div>
        ))}
        {stocks.length === 0 && (
          <div className="text-xs text-[var(--text-muted)] text-center py-2">暂无</div>
        )}
      </div>
    </div>
  );
}

export default function IndustryChainDiagram({ upstream, midstream, downstream, valueFlow, onStockClick }: Props) {
  return (
    <div>
      <div className="flex gap-3">
        <ChainColumn
          title="上游"
          stocks={upstream}
          color="bg-orange-500/20 text-orange-300"
          onStockClick={onStockClick}
        />
        <div className="flex items-center text-[var(--text-muted)] flex-shrink-0 pt-6">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </div>
        <ChainColumn
          title="中游"
          stocks={midstream}
          color="bg-blue-500/20 text-blue-300"
          onStockClick={onStockClick}
        />
        <div className="flex items-center text-[var(--text-muted)] flex-shrink-0 pt-6">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </div>
        <ChainColumn
          title="下游"
          stocks={downstream}
          color="bg-green-500/20 text-green-300"
          onStockClick={onStockClick}
        />
      </div>
      {valueFlow && (
        <div className="mt-2 text-[10px] text-[var(--text-muted)] text-center italic">
          {valueFlow}
        </div>
      )}
    </div>
  );
}
