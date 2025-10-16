/**
 * Utility to parse "Documents to Create" table from RFP analysis
 */

export interface RFPDocument {
  number: string
  name: string
  status: string  // ✅ Complete, 🟡 Partial, ❌ No Info
  details: string
}

export interface RFPAnalysisMetadata {
  hasDocumentsTable: boolean
  decision?: string  // GO, NO-GO, CONDITIONAL-GO
  rfpPath?: string
  documents: RFPDocument[]
}

/**
 * Parse the "Documents to Create" table from AI response
 */
export function parseRFPDocuments(aiResponse: string): RFPAnalysisMetadata {
  const result: RFPAnalysisMetadata = {
    hasDocumentsTable: false,
    documents: []
  }

  try {
    // Check if this is an RFP analysis response
    if (!aiResponse.includes('Documents to Create') && !aiResponse.includes('Required Documents')) {
      return result
    }

    result.hasDocumentsTable = true

    // Extract the decision
    const decisionMatch = aiResponse.match(/DECISION:\s*\*\*\s*(GO|NO-GO|CONDITIONAL-GO)\s*\*\*/i) ||
                          aiResponse.match(/🎯\s*DECISION:\s*\[?(GO|NO-GO|CONDITIONAL-GO)\]?/i)
    if (decisionMatch) {
      result.decision = decisionMatch[1].toUpperCase()
    }

    // Extract RFP path from the message - try multiple patterns
    const pathPatterns = [
      /Process RFP folder intelligently:\s*([^\n]+)/i,  // From user message
      /folder_path["\s:]+([RFP|RFI|RFQ]\/[^\s\n"']+)/i,  // From API response
      /(RFP|RFI|RFQ)\/[A-Za-z0-9_\-\/]+/,  // Any RFP/RFI/RFQ path
    ]
    
    for (const pattern of pathPatterns) {
      const pathMatch = aiResponse.match(pattern)
      if (pathMatch) {
        result.rfpPath = pathMatch[1] || pathMatch[0]
        result.rfpPath = result.rfpPath.trim()
        console.log('📁 Extracted RFP path:', result.rfpPath)
        break
      }
    }

    // Find the documents table section - handle both pipe-separated and tab-separated
    // Format 1: Pipe-separated markdown table
    const pipeTableRegex = /\|\s*#\s*\|\s*Document Name\s*\|[^\n]*\n\|[-|\s]+\n((?:\|[^\n]+\n?)+)/i
    const pipeTableMatch = aiResponse.match(pipeTableRegex)
    
    // Format 2: Tab-separated or plain table (what AI is actually outputting)
    const plainTableRegex = /#\s+Document Name\s+(?:Status\s+)?(?:Information Status)?[^\n]*\n((?:\d+\s+[^\n]+\n?)+)/i
    const plainTableMatch = aiResponse.match(plainTableRegex)

    if (pipeTableMatch) {
      // Handle pipe-separated table
      const tableRows = pipeTableMatch[1].trim().split('\n')
      
      for (const row of tableRows) {
        const cells = row.split('|').map(cell => cell.trim()).filter(cell => cell)
        
        if (cells.length >= 4) {
          const [number, name, status, details] = cells
          
          if (number === '#' || number === '---' || !name || name === 'Document Name') {
            continue
          }

          result.documents.push({
            number: number,
            name: name,
            status: status,
            details: details || status
          })
        }
      }
    } else if (plainTableMatch) {
      // Handle plain/tab-separated table
      const tableRows = plainTableMatch[1].trim().split('\n')
      
      for (const row of tableRows) {
        // Split by tabs or multiple spaces
        const cells = row.split(/\t+|\s{2,}/).map(cell => cell.trim()).filter(cell => cell)
        
        if (cells.length >= 3) {
          const [number, name, ...rest] = cells
          
          // Skip if number is not a digit
          if (!/^\d+$/.test(number)) {
            continue
          }

          // Status is usually the first emoji or keyword
          const status = rest.find(cell => cell.includes('✅') || cell.includes('🟡') || cell.includes('❌') || 
                                          cell.toLowerCase().includes('complete') || 
                                          cell.toLowerCase().includes('partial') ||
                                          cell.toLowerCase().includes('no info')) || rest[0] || ''
          
          // Details are the remaining cells joined
          const details = rest.filter(cell => cell !== status).join(' ') || status

          result.documents.push({
            number: number,
            name: name,
            status: status,
            details: details
          })
        }
      }
    }

    console.log(`📄 Parsed ${result.documents.length} documents from RFP analysis`)
    return result

  } catch (error) {
    console.error('Error parsing RFP documents:', error)
    return result
  }
}

/**
 * Check if message content contains RFP analysis with documents
 */
export function hasRFPDocumentsTable(content: string): boolean {
  return content.includes('Documents to Create') || 
         content.includes('Required Documents') ||
         content.includes('Required Documents Status') ||
         content.includes('Document Name') && content.includes('Information Status')
}

/**
 * Get status indicator from status string
 */
export function getStatusIndicator(status: string): { emoji: string; text: string; priority: string } {
  if (status.includes('✅') || status.toLowerCase().includes('complete')) {
    return { emoji: '✅', text: 'Complete', priority: 'Low' }
  } else if (status.includes('🟡') || status.toLowerCase().includes('partial')) {
    return { emoji: '🟡', text: 'Partial', priority: 'Medium' }
  } else if (status.includes('❌') || status.toLowerCase().includes('no info') || status.toLowerCase().includes('missing')) {
    return { emoji: '❌', text: 'No Info', priority: 'High' }
  }
  return { emoji: '📄', text: 'Unknown', priority: 'Medium' }
}

