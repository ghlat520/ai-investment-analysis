import { useState, useEffect, useRef } from 'react';

interface ScoreGaugeProps {
  score: number; // -100 ~ 100
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  className?: string;
}

function getSentimentLabel(score: number): string {
  if (score >= 30) return '乐观';
  if (score >= -30) return '中性';
  return '悲观';
}

export default function ScoreGauge({
  score,
  size = 'md',
  showLabel = true,
  className = '',
}: ScoreGaugeProps) {
  const [animatedScore, setAnimatedScore] = useState(0);
  const [displayScore, setDisplayScore] = useState(0);
  const animationRef = useRef<number | null>(null);
  const prevScoreRef = useRef(0);

  useEffect(() => {
    const startScore = prevScoreRef.current;
    const endScore = score;
    const duration = 1000;
    const startTime = performance.now();

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easeOut = 1 - Math.pow(1 - progress, 3);
      const currentScore = startScore + (endScore - startScore) * easeOut;
      setAnimatedScore(currentScore);
      setDisplayScore(Math.round(currentScore));

      if (progress < 1) {
        animationRef.current = requestAnimationFrame(animate);
      } else {
        prevScoreRef.current = endScore;
      }
    };

    animationRef.current = requestAnimationFrame(animate);
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [score]);

  const label = getSentimentLabel(score);

  const sizeConfig = {
    sm: { width: 100, stroke: 8, fontSize: 'text-2xl', labelSize: 'text-xs' },
    md: { width: 140, stroke: 10, fontSize: 'text-4xl', labelSize: 'text-sm' },
    lg: { width: 180, stroke: 12, fontSize: 'text-5xl', labelSize: 'text-base' },
  };

  const { width, stroke, fontSize, labelSize } = sizeConfig[size];
  const radius = (width - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const arcLength = circumference * 0.75;
  // Map -100~100 to 0~100 for arc progress
  const normalizedScore = (animatedScore + 100) / 2;
  const progress = (normalizedScore / 100) * arcLength;

  const getStrokeColor = (s: number) => {
    const norm = (s + 100) / 2;
    if (norm >= 60) return '#00d4ff'; // cyan
    if (norm >= 40) return '#a855f7'; // purple
    return '#ff4466'; // red
  };

  const strokeColor = getStrokeColor(animatedScore);
  const glowColor = `${strokeColor}66`;
  const gap = stroke * 0.6;

  return (
    <div className={`flex flex-col items-center ${className}`}>
      {showLabel && (
        <span className="label-uppercase mb-3 text-secondary">
          Market Sentiment
        </span>
      )}

      <div className="relative" style={{ width, height: width }}>
        <svg
          className="gauge-ring overflow-visible"
          width={width}
          height={width}
          style={{ filter: `drop-shadow(0 0 12px ${glowColor})` }}
        >
          <defs>
            <linearGradient id={`gauge-grad-${score}`} x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor={strokeColor} stopOpacity="0.6" />
              <stop offset="100%" stopColor={strokeColor} stopOpacity="1" />
            </linearGradient>
            <filter id={`gauge-glow-${score}`}>
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Background track */}
          <circle
            cx={width / 2}
            cy={width / 2}
            r={radius}
            fill="none"
            stroke="rgba(255, 255, 255, 0.05)"
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={`${arcLength} ${circumference}`}
            transform={`rotate(135 ${width / 2} ${width / 2})`}
          />

          {/* Glow layer */}
          <circle
            cx={width / 2}
            cy={width / 2}
            r={radius}
            fill="none"
            stroke={strokeColor}
            strokeWidth={stroke + gap}
            strokeLinecap="round"
            strokeDasharray={`${progress} ${circumference}`}
            transform={`rotate(135 ${width / 2} ${width / 2})`}
            opacity="0.3"
            filter={`url(#gauge-glow-${score})`}
          />

          {/* Progress arc */}
          <circle
            cx={width / 2}
            cy={width / 2}
            r={radius}
            fill="none"
            stroke={`url(#gauge-grad-${score})`}
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={`${progress} ${circumference}`}
            transform={`rotate(135 ${width / 2} ${width / 2})`}
          />
        </svg>

        {/* Center value */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span
            className={`font-bold ${fontSize} text-white`}
            style={{ textShadow: `0 0 30px ${glowColor}` }}
          >
            {displayScore > 0 ? '+' : ''}{displayScore}
          </span>
          {showLabel && (
            <span
              className={`${labelSize} font-semibold mt-1`}
              style={{ color: strokeColor }}
            >
              {label}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
