import { useEffect, useRef, useState, useCallback } from 'react';
import type { TaskInfo, SSEAgentCompletedEvent } from '../types';
import { analysisApi } from '../api/client';

export interface UseTaskStreamOptions {
  onTaskCreated?: (task: TaskInfo) => void;
  onTaskStarted?: (task: TaskInfo) => void;
  onAgentCompleted?: (data: SSEAgentCompletedEvent) => void;
  onTaskCompleted?: (task: TaskInfo) => void;
  onTaskFailed?: (task: TaskInfo) => void;
  enabled?: boolean;
}

export function useTaskStream(options: UseTaskStreamOptions) {
  const [isConnected, setIsConnected] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  const connect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    // 使用带 API key 的完整 URL
    const streamUrl = analysisApi.getTaskStreamUrl();
    const es = new EventSource(streamUrl);
    eventSourceRef.current = es;

    es.addEventListener('connected', () => {
      setIsConnected(true);
    });

    es.addEventListener('task_created', (e) => {
      const data = JSON.parse(e.data);
      optionsRef.current.onTaskCreated?.(data);
    });

    es.addEventListener('task_started', (e) => {
      const data = JSON.parse(e.data);
      optionsRef.current.onTaskStarted?.(data);
    });

    es.addEventListener('task_collecting_done', (e) => {
      const data = JSON.parse(e.data);
      optionsRef.current.onTaskStarted?.(data);
    });

    es.addEventListener('agent_completed', (e) => {
      const data = JSON.parse(e.data);
      optionsRef.current.onAgentCompleted?.(data);
    });

    es.addEventListener('task_fusing', (e) => {
      const data = JSON.parse(e.data);
      optionsRef.current.onTaskStarted?.(data);
    });

    es.addEventListener('task_reporting', (e) => {
      const data = JSON.parse(e.data);
      optionsRef.current.onTaskStarted?.(data);
    });

    es.addEventListener('task_completed', (e) => {
      const data = JSON.parse(e.data);
      optionsRef.current.onTaskCompleted?.(data);
    });

    es.addEventListener('task_failed', (e) => {
      const data = JSON.parse(e.data);
      optionsRef.current.onTaskFailed?.(data);
    });

    es.addEventListener('heartbeat', () => {
      // keep alive
    });

    es.onerror = () => {
      setIsConnected(false);
      es.close();
      // auto reconnect
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
