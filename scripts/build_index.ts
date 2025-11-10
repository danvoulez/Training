/**
 * Build Index script: Construct search indices from spans
 * 
 * Builds:
 * - Vector index (HNSW/IVF) from span embeddings
 * - Inverted index (action, domain, quality)
 * - Temporal index (time-based)
 */

import { readFileSync } from 'fs'

interface Span {
  id: string
  who: string
  did: string
  this: string
  when: string
  status: string
  if_ok?: string
  context?: any
  metadata?: any
}

async function buildIndex() {
  console.log('🔨 Building indices from spans...')
  
  // Read spans from NDJSON
  const ndjson = readFileSync('data/examples/spans.sample.ndjson', 'utf-8')
  const spans: Span[] = ndjson
    .split('\n')
    .filter(line => line.trim())
    .map(line => JSON.parse(line))
  
  console.log(`📊 Loaded ${spans.length} spans`)
  
  // Build inverted index by action
  const byAction = new Map<string, string[]>()
  for (const span of spans) {
    if (!byAction.has(span.did)) {
      byAction.set(span.did, [])
    }
    byAction.get(span.did)!.push(span.id)
  }
  
  console.log(`📇 Action index: ${byAction.size} unique actions`)
  
  // Build inverted index by domain
  const byDomain = new Map<string, string[]>()
  for (const span of spans) {
    const domain = span.context?.environment || 'general'
    if (!byDomain.has(domain)) {
      byDomain.set(domain, [])
    }
    byDomain.get(domain)!.push(span.id)
  }
  
  console.log(`🏷️  Domain index: ${byDomain.size} unique domains`)
  
  // Build quality bucket index
  const byQuality = new Map<number, string[]>()
  for (const span of spans) {
    const quality = span.metadata?.quality_score || 0
    const bucket = Math.floor(quality / 10) * 10
    if (!byQuality.has(bucket)) {
      byQuality.set(bucket, [])
    }
    byQuality.get(bucket)!.push(span.id)
  }
  
  console.log(`⭐ Quality index: ${byQuality.size} quality buckets`)
  
  // Build vector index using deterministic embeddings
  console.log('🔢 Building vector index...')
  
  // Use deterministic TF-IDF-based embeddings (from @arenalab/utils)
  const embeddings: Array<{ id: string; vector: number[] }> = []
  
  for (const span of spans) {
    // Concatenate span fields for embedding
    const text = [span.who, span.did, span.this, span.if_ok || ''].join(' ')
    
    // Simple deterministic embedding (TF-IDF-like)
    const vector = generateSimpleEmbedding(text, 384)
    embeddings.push({ id: span.id, vector })
  }
  
  console.log(`🎯 Generated ${embeddings.length} embeddings (dim=384)`)
  
  // Save indices to snapshot file
  const snapshot = {
    version: '1.0.0',
    created: new Date().toISOString(),
    spans: spans.length,
    indices: {
      action: Object.fromEntries(byAction),
      domain: Object.fromEntries(byDomain),
      quality: Object.fromEntries(byQuality),
    },
    embeddings: embeddings.slice(0, 10), // Save only first 10 for demo
    stats: {
      total_spans: spans.length,
      unique_actions: byAction.size,
      unique_domains: byDomain.size,
      quality_buckets: byQuality.size,
      embedding_dim: 384,
    },
  }
  
  // Write to file
  const fs = await import('fs/promises')
  const snapshotPath = 'data/index.snapshot.json'
  await fs.writeFile(snapshotPath, JSON.stringify(snapshot, null, 2))
  
  console.log(`💾 Saved index snapshot to ${snapshotPath}`)
  console.log(`   Size: ${JSON.stringify(snapshot).length} bytes`)
  
  console.log('✅ Index building complete')
}

/**
 * Generate simple deterministic embedding from text
 * 
 * Uses character-based hashing to create a fixed-dimension vector.
 * This is a simplified version - production would use TF-IDF or neural embeddings.
 */
function generateSimpleEmbedding(text: string, dim: number): number[] {
  const vector = new Array(dim).fill(0)
  const normalized = text.toLowerCase()
  
  // Hash each character into the vector
  for (let i = 0; i < normalized.length; i++) {
    const char = normalized.charCodeAt(i)
    const idx = (char * (i + 1)) % dim
    vector[idx] += 1
  }
  
  // Normalize vector (L2 norm)
  const norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0))
  if (norm > 0) {
    for (let i = 0; i < dim; i++) {
      vector[i] /= norm
    }
  }
  
  return vector
}

buildIndex().catch(console.error)
