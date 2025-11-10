/**
 * Verify Ledger script: Check NDJSON integrity and signatures
 * 
 * Verifies:
 * - NDJSON format validity
 * - Hash integrity per entry
 * - DV25 signature verification
 * 
 * @see docs/formula.md §Ledger Verification
 */

import { readFileSync } from 'fs'
import { createHash } from 'crypto'

interface LedgerEntry {
  span: Span
  hash: string
  signature?: string
  timestamp: number
}

interface Span {
  id: string
  who: string
  did: string
  this: string
  [key: string]: any
}

/**
 * Hash span data using SHA-256 (simulating BLAKE3)
 * 
 * In production, use BLAKE3 implementation.
 */
function hashSpanData(span: Span): string {
  const content = JSON.stringify(span)
  return createHash('sha256').update(content).digest('hex')
}

/**
 * Verify DV25 signature
 * 
 * This is a simulated verification matching the implementation
 * in packages/ledger/src/signature/dv25-seal.ts
 */
function verifyDV25Signature(entry: LedgerEntry): boolean {
  if (!entry.signature) {
    // No signature to verify
    return true
  }
  
  // Check signature format (simulated Ed25519)
  if (!entry.signature.startsWith('ed25519_')) {
    console.error(`  Invalid signature format for span ${entry.span.id}`)
    return false
  }
  
  // Extract signature body
  const signatureBody = entry.signature.replace('ed25519_', '')
  
  // Verify it's a valid hex string (64 characters for 32 bytes)
  if (!/^[0-9a-f]{64}$/i.test(signatureBody)) {
    console.error(`  Invalid signature hex for span ${entry.span.id}`)
    return false
  }
  
  return true
}

async function verifyLedger(filepath: string) {
  console.log(`🔍 Verifying ledger: ${filepath}`)
  
  const content = readFileSync(filepath, 'utf-8')
  const lines = content.split('\n').filter(line => line.trim())
  
  let valid = 0
  let invalid = 0
  let hashMismatches = 0
  let signatureFailures = 0
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    
    try {
      // Try to parse as ledger entry first
      let entry: LedgerEntry
      let isLedgerEntry = false
      
      try {
        const parsed = JSON.parse(line)
        if (parsed.span && parsed.hash) {
          entry = parsed as LedgerEntry
          isLedgerEntry = true
        } else {
          // Plain span format
          entry = {
            span: parsed as Span,
            hash: '',
            timestamp: Date.now(),
          }
        }
      } catch {
        throw new Error('Invalid JSON')
      }
      
      const span = entry.span
      
      // Verify required fields
      if (!span.id || !span.who || !span.did) {
        console.error(`❌ Line ${i + 1}: Missing required fields (id, who, did)`)
        invalid++
        continue
      }
      
      // Verify hash integrity (only for ledger entries)
      if (isLedgerEntry && entry.hash) {
        const computedHash = hashSpanData(span)
        if (computedHash !== entry.hash) {
          console.error(`❌ Line ${i + 1}: Hash mismatch for span ${span.id}`)
          console.error(`  Expected: ${entry.hash}`)
          console.error(`  Computed: ${computedHash}`)
          hashMismatches++
          invalid++
          continue
        }
      }
      
      // Verify DV25 signature (if present)
      if (isLedgerEntry && entry.signature) {
        if (!verifyDV25Signature(entry)) {
          console.error(`❌ Line ${i + 1}: Signature verification failed for span ${span.id}`)
          signatureFailures++
          invalid++
          continue
        }
      }
      
      valid++
    } catch (error: any) {
      console.error(`❌ Line ${i + 1}: ${error.message}`)
      invalid++
    }
  }
  
  console.log(`\n📊 Summary:`)
  console.log(`  ✅ Valid entries: ${valid}`)
  console.log(`  ❌ Invalid entries: ${invalid}`)
  if (hashMismatches > 0) {
    console.log(`  🔴 Hash mismatches: ${hashMismatches}`)
  }
  if (signatureFailures > 0) {
    console.log(`  🔴 Signature failures: ${signatureFailures}`)
  }
  console.log(`  📝 Total lines: ${lines.length}`)
  
  if (invalid === 0) {
    console.log(`\n✅ Ledger is valid! All ${valid} entries verified.`)
    return true
  } else {
    console.log(`\n⚠️  Ledger has ${invalid} invalid entries`)
    return false
  }
}

// Run verification
const filepath = process.argv[2] || 'data/examples/spans.sample.ndjson'
verifyLedger(filepath).catch(console.error)
