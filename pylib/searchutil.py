# SPDX-FileCopyrightText: 2025-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0
# discord_aiagent.searchutil
import re
from typing import Any, List, Tuple, Optional, Dict
import structlog
import numpy as np
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from chonkie import SentenceChunker
except ImportError:
    SentenceChunker = None

try:
    import html2text
except ImportError:
    html2text = None

logger = structlog.get_logger(__name__)

@dataclass
class SearchChunk:
    '''Represents a chunk of text with metadata about its source.'''
    text: str
    entry_index: int
    chunk_index: int
    entry: Any  # The original RSS entry object

@dataclass
class SearchResult:
    '''Represents a search result with score and metadata.'''
    entry: Any  # The original RSS entry object
    score: float
    best_chunk_text: str
    best_chunk_index: int

class VectorSearchEngine:
    '''Handles vector search with chunking and HTML to Markdown conversion.'''
    
    def __init__(self):
        self._embedding_model = None
        self._chunker = None
        self._html_converter = None
        
    def _get_embedding_model(self):
        '''Lazy load the embedding model.'''
        if self._embedding_model is None and SentenceTransformer is not None:
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._embedding_model
    
    def _get_chunker(self):
        '''Lazy load the sentence chunker.'''
        if self._chunker is None and SentenceChunker is not None:
            self._chunker = SentenceChunker()
        return self._chunker
    
    def _get_html_converter(self):
        '''Lazy load the HTML to Markdown converter.'''
        if self._html_converter is None and html2text is not None:
            self._html_converter = html2text.HTML2Text()
            self._html_converter.ignore_links = False
            self._html_converter.ignore_images = False
            self._html_converter.body_width = 0  # No line wrapping
        return self._html_converter
    
    def html_to_markdown(self, html_text: str) -> str:
        '''Convert HTML text to Markdown.'''
        if not html_text:
            return ''
        
        if html2text is None:
            logger.warning('html2text not available, using basic HTML stripping')
            # Basic HTML tag removal as fallback
            return re.sub(r'<[^>]+>', '', html_text)
        
        converter = self._get_html_converter()
        return converter.handle(html_text)
    
    def chunk_text(self, text: str) -> List[str]:
        '''Chunk text into sentences using Chonkie or fallback to simple sentence splitting.'''
        if not text:
            return []
        
        if SentenceChunker is not None:
            chunker = self._get_chunker()
            try:
                chunks = chunker.chunk(text)
                return [chunk.text for chunk in chunks]
            except Exception as e:
                logger.warning('Chonkie chunking failed, falling back to simple splitting', error=str(e))
        
        # Fallback: simple sentence splitting
        # Split on sentence endings followed by whitespace or end of string
        sentences = re.split(r'[.!?]+(?:\s+|$)', text)
        # Filter out empty sentences and strip whitespace
        filtered_sentences = [s.strip() for s in sentences if s.strip()]
        logger.debug('Fallback chunking', text=text, sentences=sentences, filtered_sentences=filtered_sentences, count=len(filtered_sentences))
        # If we get too many small chunks, return the original text as a single chunk
        if len(filtered_sentences) > 2:
            logger.debug('Too many chunks, returning original text')
            return [text]
        # If no sentences found, return the original text as a single chunk
        return filtered_sentences if filtered_sentences else [text]
    
    def prepare_entry_chunks(self, entries: List[Any]) -> List[SearchChunk]:
        '''Prepare chunks from RSS entries with HTML to Markdown conversion.'''
        chunks = []
        
        for entry_index, entry in enumerate(entries):
            # Extract and convert title and content (prefer content over summary for Reddit RSS)
            title = entry.get('title', '')
            content = entry.get('content', [{}])[0].get('value', '') if entry.get('content') else ''
            summary = entry.get('summary', '')
            
            # Use content if available, otherwise fall back to summary
            body_text = content if content else summary
            
            # Convert HTML to Markdown
            title_md = self.html_to_markdown(title)
            body_md = self.html_to_markdown(body_text)
            
            # Combine title and body
            full_text = f'{title_md}\n\n{body_md}'.strip()
            
            # Debug: Log the original and converted text
            logger.debug('Processing entry', 
                        entry_index=entry_index,
                        original_title=title[:100],
                        original_content=content[:100] if content else 'No content',
                        original_summary=summary[:100],
                        converted_title=title_md[:100],
                        converted_body=body_md[:100],
                        full_text=full_text[:200])
            
            # Chunk the combined text
            text_chunks = self.chunk_text(full_text)
            
            # Debug: Log the chunks
            logger.debug('Chunked text', 
                        entry_index=entry_index,
                        chunk_count=len(text_chunks),
                        chunks=[chunk[:100] for chunk in text_chunks])
            
            # Create SearchChunk objects
            for chunk_index, chunk_text in enumerate(text_chunks):
                if chunk_text.strip():  # Only add non-empty chunks
                    chunks.append(SearchChunk(
                        text=chunk_text,
                        entry_index=entry_index,
                        chunk_index=chunk_index,
                        entry=entry
                    ))
        
        logger.debug('Prepared chunks', total_chunks=len(chunks), total_entries=len(entries))
        return chunks
    
    def vector_search(self, 
                     entries: List[Any], 
                     query: str, 
                     cosine_threshold: float = 0.65,
                     top_k: int = 5) -> List[SearchResult]:
        '''Perform vector search on chunked entries.'''
        if SentenceTransformer is None:
            logger.warning('SentenceTransformer not available, cannot perform vector search')
            return []
        
        if not query or not entries:
            return []
        
        try:
            # Prepare chunks
            chunks = self.prepare_entry_chunks(entries)
            if not chunks:
                logger.warning('No chunks prepared for vector search')
                return []
            
            # Get embedding model
            model = self._get_embedding_model()
            if model is None:
                logger.error('Failed to load embedding model')
                return []
            
            # Encode query
            query_embedding = model.encode([query])[0]
            
            # Encode all chunks
            chunk_texts = [chunk.text for chunk in chunks]
            chunk_embeddings = model.encode(chunk_texts)
            
            # Compute cosine similarities
            similarities = np.dot(chunk_embeddings, query_embedding) / (
                np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
            )
            
            # Debug: Log all similarities
            logger.debug('All similarities', 
                        query=query,
                        similarities=[float(s) for s in similarities],
                        max_similarity=float(np.max(similarities)),
                        min_similarity=float(np.min(similarities)),
                        threshold=cosine_threshold)
            
            # Find chunks above threshold
            scored_chunks = []
            for i, (chunk, similarity) in enumerate(zip(chunks, similarities)):
                if similarity >= cosine_threshold:
                    scored_chunks.append((chunk, float(similarity)))
            
            # Sort by score (descending)
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            
            # Group by entry and find best score for each entry
            entry_scores: Dict[int, Tuple[Any, float, str, int]] = {}
            for chunk, score in scored_chunks:
                entry_index = chunk.entry_index
                if entry_index not in entry_scores or score > entry_scores[entry_index][1]:
                    entry_scores[entry_index] = (chunk.entry, score, chunk.text, chunk.chunk_index)
            
            # Convert to SearchResult objects and sort by score
            results = [
                SearchResult(
                    entry=entry,
                    score=score,
                    best_chunk_text=chunk_text,
                    best_chunk_index=chunk_index
                )
                for entry, score, chunk_text, chunk_index in entry_scores.values()
            ]
            
            # Sort by score and limit to top_k
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:top_k]
            
            logger.debug('Vector search completed', 
                        query=query, 
                        total_chunks=len(chunks),
                        chunks_above_threshold=len(scored_chunks),
                        unique_entries_found=len(results),
                        top_scores=[r.score for r in results[:3]])
            
            return results
            
        except Exception as e:
            logger.exception('Vector search failed', error=str(e))
            return []
    
    def keyword_search(self, entries: List[Any], query: str) -> List[SearchResult]:
        '''Perform simple keyword search as fallback.'''
        if not query or not entries:
            return []
        
        query_lower = query.lower()
        results = []
        
        for entry in entries:
            title = entry.get('title', '')
            content = entry.get('content', [{}])[0].get('value', '') if entry.get('content') else ''
            summary = entry.get('summary', '')
            
            # Use content if available, otherwise fall back to summary
            body_text = content if content else summary
            
            # Convert HTML to Markdown for better text matching
            title_md = self.html_to_markdown(title)
            body_md = self.html_to_markdown(body_text)
            
            combined_text = f'{title_md} {body_md}'.lower()
            
            if query_lower in combined_text:
                # Find the chunk containing the keyword
                chunks = self.chunk_text(f'{title_md}\n\n{body_md}')
                best_chunk = ''
                best_chunk_index = 0
                
                for i, chunk in enumerate(chunks):
                    if query_lower in chunk.lower():
                        best_chunk = chunk
                        best_chunk_index = i
                        break
                
                if not best_chunk and chunks:
                    best_chunk = chunks[0]
                
                results.append(SearchResult(
                    entry=entry,
                    score=1.0,  # Fixed score for keyword matches
                    best_chunk_text=best_chunk,
                    best_chunk_index=best_chunk_index
                ))
        
        logger.debug('Keyword search completed', query=query, matches=len(results))
        return results

# Global instance for reuse
_vector_search_engine = None

def get_vector_search_engine() -> VectorSearchEngine:
    '''Get or create the global vector search engine instance.'''
    global _vector_search_engine
    if _vector_search_engine is None:
        _vector_search_engine = VectorSearchEngine()
    return _vector_search_engine 