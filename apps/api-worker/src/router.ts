/**
 * Router: Handle HTTP requests and route to appropriate handlers
 */

import { handleChat } from './handlers/chat'
import type { Env } from './index'

/**
 * Generate Prometheus-formatted metrics
 * 
 * In production, this would aggregate from actual metric collectors.
 * For now, returns sample metrics showing the expected format.
 * 
 * @see docs/formula.md §Metrics - Prometheus Export
 */
function generatePrometheusMetrics(): string {
  const timestamp = Date.now()
  
  return `# HELP arenalab_requests_total Total number of requests
# TYPE arenalab_requests_total counter
arenalab_requests_total{method="chat_completion"} 0

# HELP arenalab_request_duration_ms Request duration in milliseconds
# TYPE arenalab_request_duration_ms histogram
arenalab_request_duration_ms_bucket{le="50"} 0
arenalab_request_duration_ms_bucket{le="100"} 0
arenalab_request_duration_ms_bucket{le="250"} 0
arenalab_request_duration_ms_bucket{le="500"} 0
arenalab_request_duration_ms_bucket{le="1000"} 0
arenalab_request_duration_ms_bucket{le="+Inf"} 0
arenalab_request_duration_ms_sum 0
arenalab_request_duration_ms_count 0

# HELP arenalab_confidence Prediction confidence score
# TYPE arenalab_confidence gauge
arenalab_confidence{method="trajectory_matching"} 0

# HELP arenalab_trajectories_used Number of trajectories used in prediction
# TYPE arenalab_trajectories_used gauge
arenalab_trajectories_used 0

# HELP arenalab_vector_index_size Size of vector index
# TYPE arenalab_vector_index_size gauge
arenalab_vector_index_size 0

# HELP arenalab_fallback_total Total number of fallback requests
# TYPE arenalab_fallback_total counter
arenalab_fallback_total{provider="openai"} 0
arenalab_fallback_total{provider="anthropic"} 0
arenalab_fallback_total{provider="gemini"} 0
`
}

class Router {
  async handle(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url)
    const path = url.pathname

    // CORS headers
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    }

    // Handle OPTIONS (CORS preflight)
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders })
    }

    // Route: POST /v1/chat/completions
    if (path === '/v1/chat/completions' && request.method === 'POST') {
      const response = await handleChat(request, env, ctx)
      // Add CORS headers to response
      Object.entries(corsHeaders).forEach(([key, value]) => {
        response.headers.set(key, value)
      })
      return response
    }

    // Route: GET /health
    if (path === '/health' && request.method === 'GET') {
      return new Response(JSON.stringify({ status: 'ok' }), {
        headers: { 'Content-Type': 'application/json', ...corsHeaders },
      })
    }

    // Route: GET /metrics
    if (path === '/metrics' && request.method === 'GET') {
      // Return Prometheus-formatted metrics
      // In production, these would come from actual metric collectors
      const metrics = generatePrometheusMetrics()
      return new Response(metrics, {
        headers: { 'Content-Type': 'text/plain; version=0.0.4', ...corsHeaders },
      })
    }

    // Route: GET /healthz (alias for /health)
    if (path === '/healthz' && request.method === 'GET') {
      return new Response('ok', {
        headers: { 'Content-Type': 'text/plain', ...corsHeaders },
      })
    }

    // 404 Not Found
    return new Response(JSON.stringify({ error: 'Not Found' }), {
      status: 404,
      headers: { 'Content-Type': 'application/json', ...corsHeaders },
    })
  }
}

export const router = new Router()
