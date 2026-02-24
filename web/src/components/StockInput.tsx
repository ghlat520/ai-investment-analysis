import { useState, useRef, useEffect, useCallback } from 'react';
import { analysisApi } from '../api/client';

interface SearchResult {
  symbol: string;
  name: string;
  market: string;
}

interface StockInputProps {
  isRunning: boolean;
  onAnalyze: (symbol: string, market: string) => void;
}

export default function StockInput({ isRunning, onAnalyze }: StockInputProps) {
  const [input, setInput] = useState('');
  const [suggestions, setSuggestions] = useState<SearchResult[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState('');
  const [selectedMarket, setSelectedMarket] = useState('A');
  const [loading, setLoading] = useState(false);
  const [searching, setSearching] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const doSearch = useCallback((query: string) => {
    if (timerRef.current) clearTimeout(timerRef.current);
    if (query.length < 1) {
      setSuggestions([]);
      setShowDropdown(false);
      return;
    }
    timerRef.current = setTimeout(async () => {
      setSearching(true);
      try {
        const resp = await analysisApi.search(query);
        const results: SearchResult[] = resp.data.results || [];
        setSuggestions(results);
        setShowDropdown(results.length > 0);
      } catch {
        setSuggestions([]);
      } finally {
        setSearching(false);
      }
    }, 300);
  }, []);

  const handleInputChange = (value: string) => {
    setInput(value);
    setSelectedSymbol('');
    doSearch(value);
  };

  const handleSelect = (item: SearchResult) => {
    setInput(`${item.name} (${item.symbol})`);
    setSelectedSymbol(item.symbol);
    setSelectedMarket(item.market);
    setShowDropdown(false);
  };

  const doAnalyze = async (symbol: string, market: string) => {
    setLoading(true);
    try {
      onAnalyze(symbol, market);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isRunning || loading) return;

    let symbol = selectedSymbol;
    let market = selectedMarket;

    if (!symbol) {
      const trimmed = input.trim();
      if (!trimmed) return;

      if (/^\d{6}(\.[A-Z]{2})?$/.test(trimmed)) {
        symbol = trimmed.split('.')[0];
        market = 'A';
      } else if (/^\d{5}(\.HK)?$/i.test(trimmed)) {
        symbol = trimmed.replace(/\.HK/i, '');
        market = 'HK';
      } else if (/^[A-Za-z]{1,5}$/.test(trimmed)) {
        symbol = trimmed.toUpperCase();
        market = 'US';
      } else {
        setLoading(true);
        try {
          const resp = await analysisApi.search(trimmed);
          const results: SearchResult[] = resp.data.results || [];
          if (results.length > 0) {
            const first = results[0];
            setInput(`${first.name} (${first.symbol})`);
            setSelectedSymbol(first.symbol);
            setSelectedMarket(first.market);
            setShowDropdown(false);
            await doAnalyze(first.symbol, first.market);
          }
        } catch {
          // search failed
        } finally {
          setLoading(false);
        }
        return;
      }
    }

    await doAnalyze(symbol, market);
  };

  const marketLabel = (m: string) =>
    m === 'HK' ? '港' : m === 'US' ? '美' : 'A';

  const marketColor = (m: string) =>
    m === 'HK' ? 'text-[var(--color-danger)]' : m === 'US' ? 'text-[var(--color-cyan)]' : 'text-[var(--color-warning)]';

  return (
    <form onSubmit={handleSubmit} className="flex items-center gap-3 flex-1 w-full max-w-4xl">
      <div className="relative flex-1" ref={dropdownRef}>
        <input
          type="text"
          value={input}
          onChange={(e) => handleInputChange(e.target.value)}
          onFocus={() => suggestions.length > 0 && setShowDropdown(true)}
          placeholder="输入股票代码或名称，如 300054、鼎龙、BABA"
          className="input-terminal w-full"
          disabled={isRunning}
          autoComplete="off"
        />
        {searching && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2">
            <div className="w-4 h-4 border-2 border-[var(--color-cyan)] border-t-transparent rounded-full animate-spin" />
          </div>
        )}

        {showDropdown && suggestions.length > 0 && (
          <div className="absolute z-50 w-full mt-1 bg-[var(--bg-elevated)] border border-[var(--border-default)] rounded-lg shadow-xl max-h-80 overflow-y-auto">
            {suggestions.map((item, i) => (
              <button
                key={`${item.symbol}-${i}`}
                type="button"
                onClick={() => handleSelect(item)}
                className="w-full px-4 py-2.5 flex items-center justify-between hover:bg-[var(--bg-hover)] transition-colors text-left border-b border-white/5 last:border-b-0"
              >
                <div className="flex items-center gap-3">
                  <span className={`text-xs font-bold px-1.5 py-0.5 rounded ${marketColor(item.market)} bg-white/5`}>
                    {marketLabel(item.market)}
                  </span>
                  <span className="text-white font-medium text-sm">{item.name}</span>
                </div>
                <span className="text-[var(--text-muted)] text-sm font-mono">{item.symbol}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      <button
        type="submit"
        disabled={isRunning || loading || (!selectedSymbol && !input.trim())}
        className="btn-primary flex items-center gap-1.5 whitespace-nowrap"
      >
        {isRunning ? (
          <>
            <svg className="w-3.5 h-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            分析中
          </>
        ) : '分析'}
      </button>
    </form>
  );
}
