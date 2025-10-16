/**
 * React Hooks for API Client
 * Convenient hooks for using the API client in React components
 */

import { useState, useEffect, useCallback } from 'react';
import { apiClient, tokenManager } from '../api-client';

// ============================================================================
// Authentication Hooks
// ============================================================================

export function useAuth() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is authenticated
    const checkAuth = async () => {
      if (tokenManager.isAuthenticated()) {
        try {
          const profile = await apiClient.getProfile();
          setUser(profile);
          setIsAuthenticated(true);
        } catch (error) {
          console.error('Failed to fetch user profile:', error);
          tokenManager.clearTokens();
          setIsAuthenticated(false);
        }
      }
      setLoading(false);
    };

    checkAuth();
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    try {
      await apiClient.login({ email, password });
      const profile = await apiClient.getProfile();
      setUser(profile);
      setIsAuthenticated(true);
      return { success: true };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }, []);

  const register = useCallback(async (data: {
    username: string;
    email: string;
    password: string;
    full_name?: string;
  }) => {
    try {
      await apiClient.register(data);
      const profile = await apiClient.getProfile();
      setUser(profile);
      setIsAuthenticated(true);
      return { success: true };
    } catch (error: any) {
      return { success: false, error: error.message };
    }
  }, []);

  const logout = useCallback(async () => {
    await apiClient.logout();
    setUser(null);
    setIsAuthenticated(false);
  }, []);

  return {
    isAuthenticated,
    user,
    loading,
    login,
    register,
    logout
  };
}

// ============================================================================
// API Request Hook
// ============================================================================

export function useAPI<T>(
  fetcher: () => Promise<T>,
  dependencies: any[] = []
) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);

  const refetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetcher();
      setData(result);
    } catch (err: any) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, [fetcher]);

  useEffect(() => {
    refetch();
  }, dependencies);

  return { data, error, loading, refetch };
}

// ============================================================================
// DNA Hooks
// ============================================================================

export function useDNA(dnaId: string) {
  return useAPI(() => apiClient.getDNA(dnaId), [dnaId]);
}

export function useDNAList(limit: number = 10, offset: number = 0) {
  return useAPI(() => apiClient.listDNAs(limit, offset), [limit, offset]);
}

export function useCreateDNA() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const createDNA = useCallback(async (data: any) => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiClient.createDNA(data);
      return { success: true, data: result };
    } catch (err: any) {
      setError(err.message);
      return { success: false, error: err.message };
    } finally {
      setLoading(false);
    }
  }, []);

  return { createDNA, loading, error };
}

// ============================================================================
// Evolution Hooks
// ============================================================================

export function useGenerateOptions() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generateOptions = useCallback(async (data: any) => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiClient.generateOptions(data);
      return { success: true, data: result };
    } catch (err: any) {
      setError(err.message);
      return { success: false, error: err.message };
    } finally {
      setLoading(false);
    }
  }, []);

  return { generateOptions, loading, error };
}

export function useSubmitFeedback() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submitFeedback = useCallback(async (data: any) => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiClient.submitFeedback(data);
      return { success: true, data: result };
    } catch (err: any) {
      setError(err.message);
      return { success: false, error: err.message };
    } finally {
      setLoading(false);
    }
  }, []);

  return { submitFeedback, loading, error };
}

export function useExperimentStatus(experimentId: string) {
  return useAPI(() => apiClient.getExperimentStatus(experimentId), [experimentId]);
}

// ============================================================================
// Real-time Stream Hooks
// ============================================================================

export function useRLHFStream(experimentId: string | null) {
  const [metrics, setMetrics] = useState<any[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!experimentId) return;

    let eventSource: EventSource | null = null;

    try {
      eventSource = apiClient.connectToRLHFStream(experimentId, (data) => {
        if (data.type === 'metrics') {
          setMetrics((prev) => [...prev, data]);
        } else if (data.type === 'completed') {
          setConnected(false);
        } else if (data.type === 'error') {
          setError(data.error);
          setConnected(false);
        }
      });

      setConnected(true);
    } catch (err: any) {
      setError(err.message);
    }

    return () => {
      if (eventSource) {
        eventSource.close();
        setConnected(false);
      }
    };
  }, [experimentId]);

  return { metrics, connected, error };
}

export function useWebSocket(experimentId: string | null) {
  const [messages, setMessages] = useState<any[]>([]);
  const [connected, setConnected] = useState(false);
  const [ws, setWs] = useState<WebSocket | null>(null);

  useEffect(() => {
    if (!experimentId) return;

    const socket = apiClient.connectToWebSocket(experimentId, (data) => {
      if (data.type === 'connected') {
        setConnected(true);
      } else if (data.type === 'disconnected') {
        setConnected(false);
      } else {
        setMessages((prev) => [...prev, data]);
      }
    });

    setWs(socket);

    return () => {
      socket.close();
      setConnected(false);
    };
  }, [experimentId]);

  const sendMessage = useCallback((message: any) => {
    if (ws && connected) {
      ws.send(JSON.stringify(message));
    }
  }, [ws, connected]);

  return { messages, connected, sendMessage };
}

// ============================================================================
// Perfume Generation Hook
// ============================================================================

export function useGeneratePerfume() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generate = useCallback(async (prompt: string, mode: string = 'balanced') => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiClient.generatePerfume({ prompt, mode });
      return { success: true, data: result };
    } catch (err: any) {
      setError(err.message);
      return { success: false, error: err.message };
    } finally {
      setLoading(false);
    }
  }, []);

  return { generate, loading, error };
}
