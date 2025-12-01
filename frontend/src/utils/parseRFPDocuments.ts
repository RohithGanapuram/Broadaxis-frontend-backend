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
    console.log('🔍 Parsing RFP documents from response...')
    console.log('📝 Response length:', aiResponse.length)
    console.log('📝 Response preview:', aiResponse.substring(0, 500))
    
    // Check if this is an RFP analysis response - be more flexible
    const hasDocumentsSection = 
      aiResponse.includes('Documents to Create') || 
      aiResponse.includes('Required Documents') ||
      aiResponse.includes('📄 **Documents to Create**') ||
      aiResponse.includes('## 📄 **Documents to Create**') ||
      (aiResponse.includes('Document Name') && (aiResponse.includes('Information Status') || aiResponse.includes('Status')))
    
    if (!hasDocumentsSection) {
      console.log('❌ No documents section found in response')
      return result
    }

    console.log('✅ Found documents section')
    result.hasDocumentsTable = true

    // Extract the decision - be more flexible with patterns
    const decisionMatch = 
      aiResponse.match(/DECISION:\s*\*\*\s*(GO|NO-GO|CONDITIONAL-GO)\s*\*\*/i) ||
      aiResponse.match(/🎯\s*DECISION:\s*\[?(GO|NO-GO|CONDITIONAL-GO)\]?/i) ||
      aiResponse.match(/\*\*DECISION:\*\*\s*(GO|NO-GO|CONDITIONAL-GO)/i) ||
      aiResponse.match(/FINAL RECOMMENDATION[^\n]*\n[^\n]*DECISION[^\n]*(GO|NO-GO|CONDITIONAL-GO)/i)
    
    if (decisionMatch) {
      result.decision = decisionMatch[1].toUpperCase()
      console.log('✅ Extracted decision:', result.decision)
    } else {
      console.log('⚠️ Could not extract decision')
    }


    // Extract RFP path from the message - try multiple patterns
    const pathPatterns = [
      /Process RFP folder intelligently:\s*([^\n]+)/i,  // From user message
      /folder_path["\s:]+([RFP|RFI|RFQ]\/[^\s\n"']+)/i,  // From API response
      /(RFP|RFI|RFQ)\/[A-Za-z0-9_\-\/]+/,  // Any RFP/RFI/RFQ path
    ]
    


    // Extract the decision
    
    
    for (const pattern of pathPatterns) {
      const pathMatch = aiResponse.match(pattern)
      if (pathMatch) {
        result.rfpPath = pathMatch[1] || pathMatch[0]
        result.rfpPath = result.rfpPath.trim()
        console.log('📁 Extracted RFP path:', result.rfpPath)
        break
      }
    }

    // Find the documents table section - handle multiple formats
    // Format 1: Pipe-separated markdown table (standard markdown)
    const pipeTableRegex = /\|\s*#\s*\|\s*Document Name\s*\|[^\n]*\n\|[-|\s:]+\n((?:\|[^\n]+\n?)+)/i
    const pipeTableMatch = aiResponse.match(pipeTableRegex)
    
    // Format 2: Pipe table with different header variations
    const pipeTableRegex2 = /\|\s*#\s*\|\s*Document Name\s*\|[^\n]*Information Status[^\n]*\|[^\n]*\n\|[-|\s:]+\n((?:\|[^\n]+\n?)+)/i
    const pipeTableMatch2 = aiResponse.match(pipeTableRegex2)
    
    // Format 3: Tab-separated or plain table (what AI might output)
    const plainTableRegex = /(?:#|##)\s+Document Name\s+(?:Status\s+)?(?:Information Status)?[^\n]*\n((?:\d+\s+[^\n]+\n?)+)/i
    const plainTableMatch = aiResponse.match(plainTableRegex)
    
    // Format 4: Look for table after "Documents to Create" heading
    const sectionMatch = aiResponse.match(/(?:##\s+)?📄\s*\*\*Documents to Create\*\*[^\n]*\n[^\n]*\n((?:\|\s*#\s*\|[^\n]*\n\|[-|\s:]+\n(?:\|[^\n]+\n?)+)|(?:\d+\s+[^\n]+\n?)+)/is)
    
    console.log('🔍 Table matching results:')
    console.log('  - Pipe table (format 1):', !!pipeTableMatch)
    console.log('  - Pipe table (format 2):', !!pipeTableMatch2)
    console.log('  - Plain table:', !!plainTableMatch)
    console.log('  - Section match:', !!sectionMatch)

    // Try to parse table in order of preference
    let tableRows: string[] = []
    let tableFormat = 'none'
    
    if (pipeTableMatch2) {
      // Format 2: Pipe table with Information Status column
      tableRows = pipeTableMatch2[1].trim().split('\n')
      tableFormat = 'pipe2'
      console.log('✅ Using pipe table format 2')
    } else if (pipeTableMatch) {
      // Format 1: Standard pipe-separated table
      tableRows = pipeTableMatch[1].trim().split('\n')
      tableFormat = 'pipe1'
      console.log('✅ Using pipe table format 1')
    } else if (sectionMatch) {
      // Format 4: Extract from section match
      const sectionContent = sectionMatch[1]
      if (sectionContent.includes('|')) {
        // It's a pipe table
        const pipeMatch = sectionContent.match(/\|\s*#\s*\|[^\n]*\n\|[-|\s:]+\n((?:\|[^\n]+\n?)+)/i)
        if (pipeMatch) {
          tableRows = pipeMatch[1].trim().split('\n')
          tableFormat = 'pipe-section'
          console.log('✅ Using pipe table from section')
        }
      } else {
        // It's a plain table
        tableRows = sectionContent.trim().split('\n').filter(line => /^\d+\s/.test(line))
        tableFormat = 'plain-section'
        console.log('✅ Using plain table from section')
      }
    } else if (plainTableMatch) {
      // Format 3: Plain/tab-separated table
      tableRows = plainTableMatch[1].trim().split('\n')
      tableFormat = 'plain'
      console.log('✅ Using plain table format')
    }
    
    console.log(`📊 Found ${tableRows.length} table rows using format: ${tableFormat}`)
    
    // If no table found yet, try a more aggressive search
    if (tableRows.length === 0) {
      console.log('⚠️ No table found with standard patterns, trying aggressive search...')
      
      // Find the "Documents to Create" section and extract everything after it
      const documentsSectionMatch = aiResponse.match(/(?:##\s+)?📄\s*\*\*Documents to Create\*\*[^\n]*\n([\s\S]*?)(?=\n##|\n---|\n\*\*|$)/i)
      
      if (documentsSectionMatch) {
        const sectionContent = documentsSectionMatch[1]
        console.log('📄 Found Documents section, content length:', sectionContent.length)
        
        // Try to find any table-like structure in this section
        // Look for lines that start with a number followed by text (likely document entries)
        const documentLines = sectionContent.split('\n').filter(line => {
          // Match lines that look like: "1 | Document Name | Status | Details"
          // or "1. Document Name" or "1 Document Name"
          return /^\s*\d+[\.|\s]+\s*[A-Za-z]/.test(line) || 
                 /^\s*\|\s*\d+\s*\|/.test(line)
        })
        
        if (documentLines.length > 0) {
          console.log(`✅ Found ${documentLines.length} potential document lines`)
          tableRows = documentLines
          tableFormat = 'aggressive'
        }
      }
    }
    
    // Parse the table rows
    if (tableRows.length > 0) {
      for (const row of tableRows) {
        if (tableFormat.startsWith('pipe')) {
          // Handle pipe-separated table
          const cells = row.split('|').map(cell => cell.trim()).filter(cell => cell)
          
          if (cells.length >= 3) {
            const [number, name, ...rest] = cells
            
            // Skip header rows
            if (number === '#' || number === '---' || !name || 
                name === 'Document Name' || name.toLowerCase().includes('document name') ||
                /^[-|:]+$/.test(number)) {
              continue
            }

            // Status is usually in the 3rd or 4th column
            const status = rest.find(cell => 
              cell.includes('✅') || cell.includes('🟡') || cell.includes('❌') || 
              cell.toLowerCase().includes('complete') || 
              cell.toLowerCase().includes('partial') ||
              cell.toLowerCase().includes('no info')
            ) || rest[0] || ''
            
            // Details are the remaining cells joined
            const details = rest.filter(cell => cell !== status).join(' ') || status || name

            result.documents.push({
              number: number,
              name: name,
              status: status,
              details: details
            })
          }
        } else {
          // Handle plain/tab-separated table
          // Split by tabs or multiple spaces (2+ spaces)
          const cells = row.split(/\t+|\s{2,}/).map(cell => cell.trim()).filter(cell => cell)
          
          if (cells.length >= 2) {
            const [number, name, ...rest] = cells
            
            // Skip if number is not a digit
            if (!/^\d+$/.test(number)) {
              continue
            }

            // Status is usually the first emoji or keyword
            const status = rest.find(cell => 
              cell.includes('✅') || cell.includes('🟡') || cell.includes('❌') || 
              cell.toLowerCase().includes('complete') || 
              cell.toLowerCase().includes('partial') ||
              cell.toLowerCase().includes('no info')
            ) || rest[0] || ''
            
            // Details are the remaining cells joined
            const details = rest.filter(cell => cell !== status).join(' ') || status || name

            result.documents.push({
              number: number,
              name: name,
              status: status,
              details: details
            })
          }
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
  if (!content || typeof content !== 'string') {
    return false
  }
  
  // Check for documents section indicators
  const hasDocumentsSection = 
    content.includes('Documents to Create') || 
    content.includes('📄 **Documents to Create**') ||
    content.includes('## 📄 **Documents to Create**') ||
    content.includes('Required Documents') ||
    content.includes('Required Documents Status')
  
  // Check for table-like structure (pipe table or numbered list)
  const hasTableStructure = 
    (content.includes('Document Name') && (content.includes('Information Status') || content.includes('Status'))) ||
    (content.includes('| # |') && content.includes('Document Name')) ||
    (content.includes('|#|') && content.includes('Document Name')) ||
    // Check for numbered document list (1. Document Name or 1 | Document Name)
    /\d+\s*[\.|]\s*[A-Za-z].*Document/i.test(content)
  
  // Also check if it's a Go/No-Go analysis (which should have documents table)
  const isGoNoGoAnalysis = 
    /(?:GO|NO-GO|CONDITIONAL-GO)/i.test(content) &&
    (content.includes('FINAL RECOMMENDATION') || content.includes('DECISION'))
  
  const result = hasDocumentsSection || (hasTableStructure && isGoNoGoAnalysis)
  
  console.log('🔍 hasRFPDocumentsTable check:', {
    hasDocumentsSection,
    hasTableStructure,
    isGoNoGoAnalysis,
    result
  })
  
  return result
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

