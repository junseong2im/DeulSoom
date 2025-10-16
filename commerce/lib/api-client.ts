/**
 * API Client with JWT Authentication
 * Handles all API requests with automatic token management
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Token storage keys
const ACCESS_TOKEN_KEY = 'access_token';
const REFRESH_TOKEN_KEY = 'refresh_token';

// ============================================================================
// Token Management
// ============================================================================

export const tokenManager = {
  getAccessToken(): string | null {
    if (typeof window === 'undefined') return null;
    return localStorage.getItem(ACCESS_TOKEN_KEY);
  },

  getRefreshToken(): string | null {
    if (typeof window === 'undefined') return null;
    return localStorage.getItem(REFRESH_TOKEN_KEY);
  },

  setTokens(accessToken: string, refreshToken: string): void {
    if (typeof window === 'undefined') return;
    localStorage.setItem(ACCESS_TOKEN_KEY, accessToken);
    localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
  },

  clearTokens(): void {
    if (typeof window === 'undefined') return;
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
  },

  isAuthenticated(): boolean {
    return !!this.getAccessToken();
  }
};

// ============================================================================
// API Client Class
// ============================================================================

class APIClient {
  private baseURL: string;

  constructor(baseURL: string = API_URL) {
    this.baseURL = baseURL;
  }

  /**
   * Make authenticated API request
   */
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    const token = tokenManager.getAccessToken();

    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    };

    // Add Authorization header if token exists
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers
      });

      // Handle 401 Unauthorized - try to refresh token
      if (response.status === 401 && token) {
        const refreshed = await this.refreshAccessToken();
        if (refreshed) {
          // Retry request with new token
          headers['Authorization'] = `Bearer ${tokenManager.getAccessToken()}`;
          const retryResponse = await fetch(url, {
            ...options,
            headers
          });
          return this.handleResponse<T>(retryResponse);
        } else {
          // Refresh failed, clear tokens and redirect to login
          tokenManager.clearTokens();
          if (typeof window !== 'undefined') {
            window.location.href = '/login';
          }
          throw new Error('Authentication failed');
        }
      }

      return this.handleResponse<T>(response);
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  /**
   * Handle API response
   */
  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: 'UNKNOWN_ERROR',
        message: response.statusText
      }));
      throw new Error(error.message || `HTTP ${response.status}`);
    }

    // Handle 204 No Content
    if (response.status === 204) {
      return {} as T;
    }

    return response.json();
  }

  /**
   * Refresh access token
   */
  private async refreshAccessToken(): Promise<boolean> {
    const refreshToken = tokenManager.getRefreshToken();
    if (!refreshToken) return false;

    try {
      const response = await fetch(`${this.baseURL}/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ refresh_token: refreshToken })
      });

      if (response.ok) {
        const data = await response.json();
        tokenManager.setTokens(data.access_token, data.refresh_token);
        return true;
      }
    } catch (error) {
      console.error('Token refresh failed:', error);
    }

    return false;
  }

  // ============================================================================
  // HTTP Methods
  // ============================================================================

  async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET' });
  }

  async post<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined
    });
  }

  async put<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined
    });
  }

  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' });
  }

  async patch<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PATCH',
      body: data ? JSON.stringify(data) : undefined
    });
  }

  // ============================================================================
  // Authentication API
  // ============================================================================

  async register(data: {
    username: string;
    email: string;
    password: string;
    full_name?: string;
  }): Promise<{ access_token: string; refresh_token: string }> {
    const response = await this.post<any>('/auth/register', data);
    tokenManager.setTokens(response.access_token, response.refresh_token);
    return response;
  }

  async login(data: {
    email: string;
    password: string;
  }): Promise<{ access_token: string; refresh_token: string }> {
    const response = await this.post<any>('/auth/login', data);
    tokenManager.setTokens(response.access_token, response.refresh_token);
    return response;
  }

  async logout(): Promise<void> {
    try {
      await this.post('/auth/logout');
    } finally {
      tokenManager.clearTokens();
      if (typeof window !== 'undefined') {
        window.location.href = '/login';
      }
    }
  }

  async getProfile(): Promise<any> {
    return this.get('/auth/profile');
  }

  // ============================================================================
  // Fragrance API
  // ============================================================================

  async createDNA(data: {
    brief: any;
    name?: string;
    description?: string;
    product_category?: string;
  }): Promise<any> {
    return this.post('/dna/create', data);
  }

  async getDNA(dnaId: string): Promise<any> {
    return this.get(`/dna/${dnaId}`);
  }

  async listDNAs(limit: number = 10, offset: number = 0): Promise<any> {
    return this.get(`/dna?limit=${limit}&offset=${offset}`);
  }

  async generateOptions(data: {
    dna_id: string;
    brief: any;
    num_options?: number;
    algorithm?: string;
  }): Promise<any> {
    return this.post('/evolve/options', data);
  }

  async submitFeedback(data: {
    experiment_id: string;
    chosen_id: string;
    rating?: number;
    notes?: string;
  }): Promise<any> {
    return this.post('/evolve/feedback', data);
  }

  async getExperimentStatus(experimentId: string): Promise<any> {
    return this.get(`/experiments/${experimentId}`);
  }

  async endExperiment(experimentId: string): Promise<any> {
    return this.delete(`/experiments/${experimentId}`);
  }

  async generatePerfume(data: {
    prompt: string;
    user_id?: string;
    mode?: string;
  }): Promise<any> {
    return this.post('/generate', data);
  }

  // ============================================================================
  // SSE Stream Connections
  // ============================================================================

  /**
   * Connect to SSE stream for RLHF training updates
   */
  connectToRLHFStream(experimentId: string, onMessage: (data: any) => void): EventSource {
    const token = tokenManager.getAccessToken();
    const url = `${this.baseURL}/stream/rlhf/training/${experimentId}${token ? `?token=${token}` : ''}`;

    const eventSource = new EventSource(url);

    eventSource.addEventListener('metrics', (event) => {
      const data = JSON.parse(event.data);
      onMessage({ type: 'metrics', ...data });
    });

    eventSource.addEventListener('completed', (event) => {
      const data = JSON.parse(event.data);
      onMessage({ type: 'completed', ...data });
      eventSource.close();
    });

    eventSource.addEventListener('error', (event: any) => {
      console.error('SSE connection error:', event);
      onMessage({ type: 'error', error: 'Connection failed' });
    });

    return eventSource;
  }

  // ============================================================================
  // WebSocket Connections
  // ============================================================================

  /**
   * Connect to WebSocket for real-time training
   */
  connectToWebSocket(experimentId: string, onMessage: (data: any) => void): WebSocket {
    const token = tokenManager.getAccessToken();
    const wsProtocol = this.baseURL.startsWith('https') ? 'wss' : 'ws';
    const wsURL = `${wsProtocol}://${this.baseURL.replace(/^https?:\/\//, '')}/ws/rlhf/training/${experimentId}${token ? `?token=${token}` : ''}`;

    const ws = new WebSocket(wsURL);

    ws.onopen = () => {
      console.log('WebSocket connected');
      onMessage({ type: 'connected' });
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      onMessage({ type: 'error', error: 'WebSocket connection failed' });
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      onMessage({ type: 'disconnected' });
    };

    return ws;
  }
}

// ============================================================================
// Export Singleton Instance
// ============================================================================

export const apiClient = new APIClient();
export default apiClient;
