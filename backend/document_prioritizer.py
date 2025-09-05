"""
Document Prioritization System for BroadAxis RFP Platform
Intelligently prioritizes documents to reduce token usage and improve efficiency
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class DocumentPriority(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    SUPPORTING = "supporting"
    IGNORE = "ignore"
    UNKNOWN = "unknown"

class DocumentType(Enum):
    RFP = "rfp"
    RFQ = "rfq"
    RFI = "rfi"
    SPECIFICATION = "specification"
    TERMS = "terms"
    PRICING = "pricing"
    FORM = "form"
    TEMPLATE = "template"
    UNKNOWN = "unknown"

@dataclass
class DocumentInfo:
    """Information about a document for prioritization"""
    filename: str
    file_path: str
    file_size: int
    file_extension: str
    priority: DocumentPriority = DocumentPriority.UNKNOWN
    document_type: DocumentType = DocumentType.UNKNOWN
    confidence_score: float = 0.0
    key_indicators: List[str] = None
    first_page_summary: str = ""
    estimated_tokens: int = 0
    
    def __post_init__(self):
        if self.key_indicators is None:
            self.key_indicators = []

class DocumentPrioritizer:
    """Intelligent document prioritization system"""
    
    def __init__(self):
        # Primary document keywords (high priority)
        self.primary_keywords = [
            "request for proposal", "rfp", "request for quotation", "rfq",
            "request for information", "rfi", "scope of work", "statement of work",
            "evaluation criteria", "submission instructions", "project objectives",
            "deliverables", "contract terms", "procurement", "solicitation",
            "requirements", "specifications", "general requirements", "special requirements"
        ]
        
        # Secondary document keywords (medium priority)
        self.secondary_keywords = [
            "specifications", "technical requirements", "pricing", "budget",
            "timeline", "schedule", "terms and conditions", "contract",
            "agreement", "proposal", "response", "submission"
        ]
        
        # Supporting document keywords (low priority)
        self.supporting_keywords = [
            "appendix", "attachment", "reference", "background", "history",
            "certification", "license", "compliance", "standard", "guideline"
        ]
        
        # Ignore keywords (very low priority)
        self.ignore_keywords = [
            "template", "form", "sample", "example", "draft", "old", "backup",
            "archive", "temp", "temporary", "test", "demo"
        ]
        
        # File extension priorities
        self.extension_priorities = {
            '.pdf': 1.0,
            '.docx': 0.9,
            '.doc': 0.8,
            '.txt': 0.7,
            '.xlsx': 0.6,
            '.xls': 0.5,
            '.pptx': 0.4,
            '.ppt': 0.3
        }
        
        # Filename pattern priorities
        self.filename_patterns = {
            r'.*rfp.*': 0.9,
            r'.*rfq.*': 0.9,
            r'.*rfi.*': 0.9,
            r'.*scope.*': 0.8,
            r'.*statement.*': 0.8,
            r'.*requirements.*': 0.7,
            r'.*specification.*': 0.7,
            r'.*terms.*': 0.6,
            r'.*pricing.*': 0.6,
            r'.*proposal.*': 0.5,
            r'.*response.*': 0.5,
            r'.*submission.*': 0.5,
            r'.*template.*': 0.2,
            r'.*form.*': 0.2,
            r'.*sample.*': 0.1,
            r'.*example.*': 0.1
        }
    
    def prioritize_documents(self, documents: List[Dict]) -> List[DocumentInfo]:
        """Prioritize a list of documents based on filename and metadata"""
        prioritized_docs = []
        
        for doc in documents:
            try:
                doc_info = self._analyze_document(doc)
                prioritized_docs.append(doc_info)
            except Exception as e:
                filename = doc.get('name', doc.get('filename', 'unknown'))
                logger.error(f"Error analyzing document {filename}: {e}")
                # Create default document info for failed analysis
                doc_info = DocumentInfo(
                    filename=filename,
                    file_path=doc.get('path', ''),
                    file_size=doc.get('size', 0),
                    file_extension=self._get_file_extension(filename),
                    priority=DocumentPriority.SUPPORTING,
                    confidence_score=0.0
                )
                prioritized_docs.append(doc_info)
        
        # Sort by priority and confidence score
        prioritized_docs.sort(key=lambda x: (
            x.priority.value == 'primary',
            x.priority.value == 'secondary',
            x.priority.value == 'supporting',
            x.confidence_score
        ), reverse=True)
        
        return prioritized_docs
    
    def _analyze_document(self, doc: Dict) -> DocumentInfo:
        """Analyze a single document for prioritization"""
        filename = doc.get('name', doc.get('filename', ''))
        file_path = doc.get('path', '')
        file_size = doc.get('size', 0)
        
        # Get file extension
        file_extension = self._get_file_extension(filename)
        
        # Analyze filename for keywords and patterns
        priority, confidence, indicators, doc_type = self._analyze_filename(filename)
        
        # Adjust priority based on file extension
        extension_boost = self.extension_priorities.get(file_extension, 0.5)
        confidence = min(1.0, confidence * extension_boost)
        
        # Adjust priority based on file size (larger files might be more important)
        size_boost = min(1.2, 1.0 + (file_size / (10 * 1024 * 1024)))  # Boost for files > 10MB
        confidence = min(1.0, confidence * size_boost)
        
        # Estimate tokens based on file size
        estimated_tokens = self._estimate_tokens(file_size, file_extension)
        
        return DocumentInfo(
            filename=filename,
            file_path=file_path,
            file_size=file_size,
            file_extension=file_extension,
            priority=priority,
            document_type=doc_type,
            confidence_score=confidence,
            key_indicators=indicators,
            estimated_tokens=estimated_tokens
        )
    
    def _analyze_filename(self, filename: str) -> Tuple[DocumentPriority, float, List[str], DocumentType]:
        """Analyze filename to determine priority and type"""
        filename_lower = filename.lower()
        indicators = []
        confidence = 0.0
        
        print(f"🔍 Analyzing filename: '{filename_lower}'")
        
        # Check for primary keywords
        primary_matches = [kw for kw in self.primary_keywords if kw in filename_lower]
        print(f"🎯 Primary keyword matches: {primary_matches}")
        if primary_matches:
            indicators.extend(primary_matches)
            confidence += 0.8
            priority = DocumentPriority.PRIMARY
            doc_type = self._determine_document_type(filename_lower)
            print(f"✅ Classified as PRIMARY with confidence {confidence}")
        else:
            # Check for secondary keywords
            secondary_matches = [kw for kw in self.secondary_keywords if kw in filename_lower]
            print(f"🎯 Secondary keyword matches: {secondary_matches}")
            if secondary_matches:
                indicators.extend(secondary_matches)
                confidence += 0.6
                priority = DocumentPriority.SECONDARY
                doc_type = self._determine_document_type(filename_lower)
                print(f"✅ Classified as SECONDARY with confidence {confidence}")
            else:
                # Check for supporting keywords
                supporting_matches = [kw for kw in self.supporting_keywords if kw in filename_lower]
                if supporting_matches:
                    indicators.extend(supporting_matches)
                    confidence += 0.4
                    priority = DocumentPriority.SUPPORTING
                    doc_type = DocumentType.UNKNOWN
                else:
                    # Check for ignore keywords
                    ignore_matches = [kw for kw in self.ignore_keywords if kw in filename_lower]
                    if ignore_matches:
                        indicators.extend(ignore_matches)
                        confidence = 0.1
                        priority = DocumentPriority.IGNORE
                        doc_type = DocumentType.UNKNOWN
                    else:
                        # No keyword matches, use filename patterns
                        pattern_confidence = self._check_filename_patterns(filename_lower)
                        if pattern_confidence > 0.5:
                            confidence = pattern_confidence
                            priority = DocumentPriority.SECONDARY
                            doc_type = self._determine_document_type(filename_lower)
                        else:
                            confidence = 0.3
                            priority = DocumentPriority.SUPPORTING
                            doc_type = DocumentType.UNKNOWN
        
        return priority, confidence, indicators, doc_type
    
    def _check_filename_patterns(self, filename: str) -> float:
        """Check filename against known patterns"""
        for pattern, score in self.filename_patterns.items():
            if re.search(pattern, filename):
                return score
        return 0.0
    
    def _determine_document_type(self, filename: str) -> DocumentType:
        """Determine document type from filename"""
        if any(kw in filename for kw in ['rfp', 'request for proposal']):
            return DocumentType.RFP
        elif any(kw in filename for kw in ['rfq', 'request for quotation']):
            return DocumentType.RFQ
        elif any(kw in filename for kw in ['rfi', 'request for information']):
            return DocumentType.RFI
        elif any(kw in filename for kw in ['spec', 'specification']):
            return DocumentType.SPECIFICATION
        elif any(kw in filename for kw in ['terms', 'condition']):
            return DocumentType.TERMS
        elif any(kw in filename for kw in ['pricing', 'price', 'cost']):
            return DocumentType.PRICING
        elif any(kw in filename for kw in ['form', 'template']):
            return DocumentType.FORM
        else:
            return DocumentType.UNKNOWN
    
    def _get_file_extension(self, filename: str) -> str:
        """Get file extension from filename"""
        if '.' in filename:
            return '.' + filename.split('.')[-1].lower()
        return ''
    
    def _estimate_tokens(self, file_size: int, file_extension: str) -> int:
        """Estimate token count based on file size and type"""
        # Base estimation: 1 token ≈ 4 characters
        base_tokens = file_size // 4
        
        # Adjust based on file type
        if file_extension in ['.pdf']:
            # PDFs might have more formatting overhead
            return int(base_tokens * 0.8)
        elif file_extension in ['.docx', '.doc']:
            # Word docs might have more formatting
            return int(base_tokens * 0.7)
        elif file_extension in ['.txt']:
            # Plain text is more efficient
            return int(base_tokens * 1.0)
        else:
            return int(base_tokens * 0.9)
    
    def get_primary_documents(self, documents: List[DocumentInfo], max_count: int = 5) -> List[DocumentInfo]:
        """Get the most important primary documents"""
        primary_docs = [doc for doc in documents if doc.priority == DocumentPriority.PRIMARY]
        primary_docs.sort(key=lambda x: x.confidence_score, reverse=True)
        return primary_docs[:max_count]
    
    def get_secondary_documents(self, documents: List[DocumentInfo], max_count: int = 3) -> List[DocumentInfo]:
        """Get important secondary documents"""
        secondary_docs = [doc for doc in documents if doc.priority == DocumentPriority.SECONDARY]
        secondary_docs.sort(key=lambda x: x.confidence_score, reverse=True)
        return secondary_docs[:max_count]
    
    def should_process_document(self, doc: DocumentInfo, max_tokens: int = 50000) -> bool:
        """Determine if a document should be processed based on priority and token estimate"""
        # Always process primary documents
        if doc.priority == DocumentPriority.PRIMARY:
            return True
        
        # Process secondary documents if they're not too large
        if doc.priority == DocumentPriority.SECONDARY and doc.estimated_tokens < max_tokens:
            return True
        
        # Skip supporting and ignore documents
        return False
    
    def get_processing_recommendation(self, documents: List[DocumentInfo]) -> Dict:
        """Get recommendations for document processing"""
        primary_docs = self.get_primary_documents(documents)
        secondary_docs = self.get_secondary_documents(documents)
        
        total_estimated_tokens = sum(doc.estimated_tokens for doc in primary_docs + secondary_docs)
        
        return {
            "total_documents": len(documents),
            "primary_documents": len(primary_docs),
            "secondary_documents": len(secondary_docs),
            "recommended_for_processing": len(primary_docs) + len(secondary_docs),
            "estimated_tokens": total_estimated_tokens,
            "processing_strategy": self._get_processing_strategy(total_estimated_tokens),
            "primary_docs": [doc.filename for doc in primary_docs],
            "secondary_docs": [doc.filename for doc in secondary_docs]
        }
    
    def _get_processing_strategy(self, estimated_tokens: int) -> str:
        """Get processing strategy based on estimated tokens"""
        if estimated_tokens < 10000:
            return "Process all documents in parallel"
        elif estimated_tokens < 50000:
            return "Process primary documents first, then secondary"
        else:
            return "Process only primary documents, use chunking for large files"

# Global document prioritizer instance
document_prioritizer = DocumentPrioritizer()
