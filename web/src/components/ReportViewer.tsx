import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ReportViewerProps {
  report: string;
}

export default function ReportViewer({ report }: ReportViewerProps) {
  if (!report) return null;

  return (
    <div className="terminal-card p-5 animate-slide-up">
      <span className="label-uppercase mb-4 block">RESEARCH REPORT 投研报告</span>
      <div className="mt-3">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            h1: ({ children }) => (
              <h1 className="text-2xl font-bold text-white mb-4 pb-3 border-b border-white/10">
                {children}
              </h1>
            ),
            h2: ({ children }) => (
              <h2 className="text-xl font-semibold text-white mt-6 mb-3 pb-2 border-b border-white/5">
                {children}
              </h2>
            ),
            h3: ({ children }) => (
              <h3 className="text-base font-semibold text-[var(--color-cyan)] mt-4 mb-2">
                {children}
              </h3>
            ),
            h4: ({ children }) => (
              <h4 className="text-sm font-semibold text-white mt-3 mb-1">
                {children}
              </h4>
            ),
            p: ({ children }) => (
              <p className="text-sm text-[var(--text-secondary)] leading-relaxed mb-3">
                {children}
              </p>
            ),
            strong: ({ children }) => (
              <strong className="font-semibold text-white">{children}</strong>
            ),
            ul: ({ children }) => (
              <ul className="space-y-1.5 mb-3 ml-1">{children}</ul>
            ),
            ol: ({ children }) => (
              <ol className="space-y-1.5 mb-3 ml-1 list-decimal list-inside">{children}</ol>
            ),
            li: ({ children }) => (
              <li className="text-sm text-[var(--text-secondary)] leading-relaxed flex gap-2">
                <span className="text-[var(--color-cyan)] mt-1 shrink-0">&#8226;</span>
                <span>{children}</span>
              </li>
            ),
            table: ({ children }) => (
              <div className="overflow-x-auto mb-4 mt-2">
                <table className="w-full text-sm border-collapse">{children}</table>
              </div>
            ),
            thead: ({ children }) => (
              <thead className="bg-[var(--bg-elevated)]">{children}</thead>
            ),
            th: ({ children }) => (
              <th className="px-3 py-2 text-left text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider border-b border-white/10">
                {children}
              </th>
            ),
            td: ({ children }) => (
              <td className="px-3 py-2 text-sm text-[var(--text-secondary)] border-b border-white/5">
                {children}
              </td>
            ),
            tr: ({ children }) => (
              <tr className="hover:bg-[var(--bg-hover)] transition-colors">{children}</tr>
            ),
            blockquote: ({ children }) => (
              <blockquote className="border-l-3 border-[var(--color-cyan)] pl-4 my-3 text-[var(--text-secondary)] italic">
                {children}
              </blockquote>
            ),
            hr: () => <hr className="border-white/10 my-4" />,
            code: ({ children, className }) => {
              const isInline = !className;
              if (isInline) {
                return (
                  <code className="px-1.5 py-0.5 rounded bg-[var(--bg-elevated)] text-[var(--color-cyan)] text-xs font-mono">
                    {children}
                  </code>
                );
              }
              return (
                <code className="block p-3 rounded-lg bg-[var(--bg-elevated)] text-sm font-mono text-[var(--text-secondary)] overflow-x-auto mb-3">
                  {children}
                </code>
              );
            },
            a: ({ href, children }) => (
              <a href={href} target="_blank" rel="noopener noreferrer" className="text-[var(--color-cyan)] hover:underline">
                {children}
              </a>
            ),
          }}
        >
          {report}
        </ReactMarkdown>
      </div>
    </div>
  );
}
