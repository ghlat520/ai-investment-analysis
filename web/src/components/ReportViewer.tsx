import { useState, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ReportViewerProps {
  report: string;
}

interface ReportSection {
  title: string;
  content: string;
}

/** Split markdown by ## headings into sections */
function splitSections(markdown: string): ReportSection[] {
  const lines = markdown.split('\n');
  const sections: ReportSection[] = [];
  let currentTitle = '概览';
  let currentLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith('## ')) {
      // Save previous section
      if (currentLines.length > 0) {
        sections.push({ title: currentTitle, content: currentLines.join('\n').trim() });
      }
      currentTitle = line.replace(/^##\s+/, '').trim();
      currentLines = [];
    } else {
      currentLines.push(line);
    }
  }
  // Save last section
  if (currentLines.length > 0) {
    sections.push({ title: currentTitle, content: currentLines.join('\n').trim() });
  }

  return sections.filter((s) => s.content.length > 0);
}

const mdComponents = {
  h1: ({ children }: any) => (
    <h1 className="text-xl font-bold text-white mb-3 pb-2 border-b border-white/10">
      {children}
    </h1>
  ),
  h2: ({ children }: any) => (
    <h2 className="text-lg font-semibold text-white mt-4 mb-2 pb-1.5 border-b border-white/5">
      {children}
    </h2>
  ),
  h3: ({ children }: any) => (
    <h3 className="text-base font-semibold text-[var(--color-cyan)] mt-3 mb-1.5">
      {children}
    </h3>
  ),
  h4: ({ children }: any) => (
    <h4 className="text-sm font-semibold text-white mt-2 mb-1">{children}</h4>
  ),
  p: ({ children }: any) => (
    <p className="text-sm text-[var(--text-secondary)] leading-relaxed mb-2.5">{children}</p>
  ),
  strong: ({ children }: any) => (
    <strong className="font-semibold text-white">{children}</strong>
  ),
  ul: ({ children }: any) => <ul className="space-y-1 mb-2.5 ml-1">{children}</ul>,
  ol: ({ children }: any) => (
    <ol className="space-y-1 mb-2.5 ml-1 list-decimal list-inside">{children}</ol>
  ),
  li: ({ children }: any) => (
    <li className="text-sm text-[var(--text-secondary)] leading-relaxed flex gap-2">
      <span className="text-[var(--color-cyan)] mt-0.5 shrink-0">&#8226;</span>
      <span>{children}</span>
    </li>
  ),
  table: ({ children }: any) => (
    <div className="overflow-x-auto mb-3 mt-2">
      <table className="w-full text-sm border-collapse">{children}</table>
    </div>
  ),
  thead: ({ children }: any) => (
    <thead className="bg-[var(--bg-elevated)]">{children}</thead>
  ),
  th: ({ children }: any) => (
    <th className="px-3 py-2 text-left text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider border-b border-white/10">
      {children}
    </th>
  ),
  td: ({ children }: any) => (
    <td className="px-3 py-2 text-sm text-[var(--text-secondary)] border-b border-white/5">
      {children}
    </td>
  ),
  tr: ({ children }: any) => (
    <tr className="hover:bg-[var(--bg-hover)] transition-colors">{children}</tr>
  ),
  blockquote: ({ children }: any) => (
    <blockquote className="border-l-3 border-[var(--color-cyan)] pl-4 my-2 text-[var(--text-secondary)] italic">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="border-white/10 my-3" />,
  code: ({ children, className }: any) => {
    const isInline = !className;
    if (isInline) {
      return (
        <code className="px-1.5 py-0.5 rounded bg-[var(--bg-elevated)] text-[var(--color-cyan)] text-xs font-mono">
          {children}
        </code>
      );
    }
    return (
      <code className="block p-3 rounded-lg bg-[var(--bg-elevated)] text-sm font-mono text-[var(--text-secondary)] overflow-x-auto mb-2">
        {children}
      </code>
    );
  },
  a: ({ href, children }: any) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-[var(--color-cyan)] hover:underline">
      {children}
    </a>
  ),
};

export default function ReportViewer({ report }: ReportViewerProps) {
  const [activeTab, setActiveTab] = useState(0);

  const sections = useMemo(() => splitSections(report || ''), [report]);

  if (!report || sections.length === 0) return null;

  // If only 1-2 sections, show as plain (no tabs needed)
  if (sections.length <= 2) {
    return (
      <div>
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>
          {report}
        </ReactMarkdown>
      </div>
    );
  }

  return (
    <div>
      {/* Sub-tabs for report sections */}
      <div className="flex flex-wrap gap-1 pb-2 -mb-px border-b border-white/5">
        {sections.map((section, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => setActiveTab(idx)}
            className={`px-2.5 py-1.5 text-xs font-medium whitespace-nowrap rounded-md transition-all ${
              activeTab === idx
                ? 'text-[var(--color-cyan)] bg-[var(--bg-elevated)]'
                : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)]/50'
            }`}
          >
            {section.title}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="pt-3 max-h-[500px] overflow-y-auto">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>
          {sections[activeTab]?.content || ''}
        </ReactMarkdown>
      </div>
    </div>
  );
}
