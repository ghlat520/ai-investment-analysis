import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface Props {
  content: string;
}

export default function BriefingViewer({ content }: Props) {
  if (!content) return null;

  return (
    <div className="terminal-card p-4">
      <div className="prose prose-invert prose-sm max-w-none
        prose-headings:text-white prose-headings:font-medium
        prose-h1:text-lg prose-h1:border-b prose-h1:border-white/10 prose-h1:pb-2
        prose-h2:text-base prose-h2:mt-4
        prose-h3:text-sm
        prose-p:text-[var(--text-secondary)] prose-p:text-xs prose-p:leading-relaxed
        prose-strong:text-white
        prose-table:text-xs
        prose-th:text-[var(--text-muted)] prose-th:font-medium prose-th:px-2 prose-th:py-1
        prose-td:px-2 prose-td:py-1 prose-td:text-[var(--text-secondary)]
        prose-hr:border-white/10
        prose-ul:text-xs prose-ul:text-[var(--text-secondary)]
        prose-li:text-xs
        prose-em:text-[var(--text-muted)]
      ">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
      </div>
    </div>
  );
}
