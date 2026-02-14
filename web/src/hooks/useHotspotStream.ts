import { useEffect, useRef, useState, useCallback } from 'react';
import type { HotspotTaskInfo, HotTheme } from '../types/hotspot';

export interface UseHotspotStreamOptions {
  onTaskCreated?: (task: HotspotTaskInfo) => void;
  onTaskStarted?: (task: HotspotTaskInfo) => void;
  onCollectingDone?: (task: HotspotTaskInfo) => void;
  onThemesExtracted?: (data: { themes: HotTheme[] } & HotspotTaskInfo) => void;
  onThemesAnalyzed?: (task: HotspotTaskInfo) => void;
  onCompleted?: (data: HotspotTaskInfo & { briefing_preview?: string }) => void;
  onFailed?: (task: HotspotTaskInfo) => void;
  enabled?: boolean;
}

export function useHotspotStream(options: UseHotspotStreamOptions) {
  const [isConnected, setIsConnected] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  const connect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const es = new EventSource('/api/v1/hotspot/stream');
    eventSourceRef.current = es;

    es.addEventListener('connected', () => {
      setIsConnected(true);
    });

    es.addEventListener('hotspot_task_created', (e) => {
      const data = JSON.parse(e.data);
      optionsRef.current.onTaskCreated?.(data);
    });

    es.addEventListener('hotspot_task_started', (e) => {
      const data = JSON.parse(e.data);
      optionsRef.current.onTaskStarted?.(data);
    });

    es.addEventListener('hotspot_collecting_done', (e) => {
      const data = JSON.parse(e.data);
      optionsRef.current.onCollectingDone?.(data);
    });

    es.addEventListener('hotspot_themes_extracted', (e) => {
      const data = JSON.parse(e.data);
      optionsRef.current.onThemesExtracted?.(data);
    });

    es.addEventListener('hotspot_themes_analyzed', (e) => {
      const data = JSON.parse(e.data);
      optionsRef.current.onThemesAnalyzed?.(data);
    });

    es.addEventListener('hotspot_completed', (e) => {
      const data = JSON.parse(e.data);
      optionsRef.current.onCompleted?.(data);
    });

    es.addEventListener('hotspot_failed', (e) => {
      const data = JSON.parse(e.data);
      optionsRef.current.onFailed?.(data);
    });

    es.addEventListener('heartbeat', () => {});

    es.onerror = () => {
      setIsConnected(false);
      es.close();
      setTimeout(() => {
        if (optionsRef.current.enabled !== false) {
          connect();
        }
      }, 3000);
    };
  }, []);

  const disconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
      setIsConnected(false);
    }
  }, []);

  useEffect(() => {
    if (options.enabled !== false) {
      connect();
    }
    return () => disconnect();
  }, [options.enabled, connect, disconnect]);

  return { isConnected, reconnect: connect, disconnect };
}
