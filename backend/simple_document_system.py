#!/usr/bin/env python3
"""
Simple Document System - Reliable PDF processing and Q&A
"""
import os
import time
from typing import List, Dict
from pathlib import Path
import PyPDF2
from io import BytesIO

# Simple in-memory storage
uploaded_documents = {}

class SimpleDocumentProcessor:
    def __init__(self):
        self.documents = {}
        
    def process_file(self, file_content: bytes, filename: str, session_id: str) -> Dict:
        """Process uploaded file and store full text"""
        try:
            # Extract text
            if filename.lower().endswith('.pdf'):
                text = self._extract_pdf_text(file_content)
            else:
                text = file_content.decode('utf-8', errors='ignore')
            
            if not text.strip():
                return {"status": "error", "message": "No text found in file"}
            
            # Store document
            doc_id = f"{session_id}_{filename}"
            self.documents[doc_id] = {
                "filename": filename,
                "text": text,
                "session_id": session_id,
                "upload_time": time.time(),
                "word_count": len(text.split())
            }
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "filename": filename,
                "word_count": len(text.split()),
                "message": f"Successfully processed {filename}"
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Error processing file: {str(e)}"}
    
    def _extract_pdf_text(self, file_content: bytes) -> str:
        """Extract text from PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"PDF extraction failed: {e}")
    
    def get_session_documents(self, session_id: str) -> List[Dict]:
        """Get all documents for a session"""
        docs = []
        for doc_id, doc in self.documents.items():
            if doc["session_id"] == session_id:
                docs.append({
                    "doc_id": doc_id,
                    "filename": doc["filename"],
                    "word_count": doc["word_count"],
                    "upload_time": doc["upload_time"]
                })
        return docs
    
    def answer_question(self, question: str, session_id: str) -> str:
        """Answer question using document context"""
        # Get all documents for session
        session_docs = []
        for doc_id, doc in self.documents.items():
            if doc["session_id"] == session_id:
                session_docs.append(doc)
        
        if not session_docs:
            return "No documents uploaded for this session."
        
        # Create context from all documents
        context = ""
        for doc in session_docs:
            context += f"\n--- {doc['filename']} ---\n{doc['text']}\n"
        
        # For summary requests
        if any(word in question.lower() for word in ["summary", "summarize", "overview"]):
            return self._create_summary(session_docs)
        
        # For specific questions, include context
        return f"Based on your uploaded documents:\n\nContext: {context[:8000]}...\n\nQuestion: {question}\n\nPlease provide a detailed answer based on the document content above."
    
    def _create_summary(self, documents: List[Dict]) -> str:
        """Create summary of documents"""
        if not documents:
            return "No documents to summarize."
        
        summary = "## Document Summary\n\n"
        
        for doc in documents:
            text = doc["text"]
            filename = doc["filename"]
            
            # Extract key sections (first 2000 chars as preview)
            preview = text[:2000] + "..." if len(text) > 2000 else text
            
            summary += f"### {filename}\n"
            summary += f"**Word Count:** {doc['word_count']} words\n\n"
            summary += f"**Content Preview:**\n{preview}\n\n"
            summary += "---\n\n"
        
        return summary
    
    def clear_session(self, session_id: str):
        """Clear all documents for a session"""
        to_remove = [doc_id for doc_id in self.documents if self.documents[doc_id]["session_id"] == session_id]
        for doc_id in to_remove:
            del self.documents[doc_id]

# Global processor instance
doc_processor = SimpleDocumentProcessor()