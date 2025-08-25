#!/usr/bin/env python3
# protocol chunker for intelligent splitting of OT-2 protocols
#   - builds on protocol_analyzer.py to create digestible chunks for LLM training
#   - implements multiple chunking strategies based on protocol type and complexity

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from protocol_analyzer import ProtocolAnalyzer

@dataclass
class Chunk:
    content: str
    protocol_name: str
    chunk_id: int
    chunk_type: str  # 'setup', 'main', 'loop', 'cleanup'
    start_line: int
    end_line: int
    context_lines: List[str]  # surrounding context for coherence
    operations: List[str]  # key operations in this chunk
    complexity: int
    protocol_type: str
    metadata: Dict[str, Any]

class ProtocolChunker:
    def __init__(self, analyzer_results_path: Optional[str] = None):
        self.chunks = []
        self.chunking_stats = {
            'total_protocols': 0,
            'total_chunks': 0,
            'chunks_by_type': {},
            'chunks_by_strategy': {},
            'avg_chunk_size': 0
        }
        
        # load analyzer results if provided
        if analyzer_results_path:
            self.load_analyzer_results(analyzer_results_path)
        else:
            self.analyzer_results = None

    def load_analyzer_results(self, results_path: str):
        results_path = Path(results_path)
        
        # load the main analysis report
        with open(results_path / "protocol_analysis_report.json", 'r') as f:
            self.analysis_report = json.load(f)
        
        # load detailed protocol structures
        with open(results_path / "protocol_structures.json", 'r') as f:
            self.protocol_structures = json.load(f)
        
        print(f"Loaded analysis results for {len(self.protocol_structures)} protocols")

    def extract_context_lines(self, lines: List[str], start: int, end: int, context_size: int = 5) -> List[str]:
        context_start = max(0, start - context_size)
        context_end = min(len(lines), end + context_size)
        return lines[context_start:context_end]

    def chunk_by_sections(self, code: str, protocol_structure: Dict, chunk_size: int = 150, overlap: int = 30) -> List[Chunk]:
        lines = code.split('\n')
        chunks = []
        protocol_name = protocol_structure['name']
        protocol_type = protocol_structure['protocol_type']
        
        # use identified sections from analyzer
        sections = protocol_structure.get('sections', {})
        
        if sections:
            # section-based chunking
            for section_name, section_info in sections.items():
                start = section_info['start']
                end = section_info['end']
                section_lines = end - start + 1
                
                if section_lines <= chunk_size:
                    # section fits in one chunk
                    chunk_content = '\n'.join(lines[start:end+1])
                    context = self.extract_context_lines(lines, start, end)
                    
                    chunk = Chunk(
                        content=chunk_content,
                        protocol_name=protocol_name,
                        chunk_id=len(chunks),
                        chunk_type=self.classify_chunk_type(section_name, chunk_content),
                        start_line=start,
                        end_line=end,
                        context_lines=context,
                        operations=self.extract_operations_from_chunk(chunk_content),
                        complexity=self.calculate_chunk_complexity(chunk_content),
                        protocol_type=protocol_type,
                        metadata={'section_name': section_name, 'strategy': 'section_based'}
                    )
                    chunks.append(chunk)
                else:
                    # section too large, use sliding window
                    chunks.extend(self.sliding_window_chunk_section(
                        lines, start, end, chunk_size, overlap, protocol_name, protocol_type, section_name
                    ))
        else:
            # fallback to simple sliding window if no sections identified
            chunks = self.sliding_window_chunk(lines, chunk_size, overlap, protocol_name, protocol_type)
        
        return chunks

    def sliding_window_chunk_section(self, lines: List[str], start: int, end: int, 
                                   chunk_size: int, overlap: int, protocol_name: str, 
                                   protocol_type: str, section_name: str) -> List[Chunk]:
        chunks = []
        current_pos = start
        chunk_id = 0
        
        while current_pos < end:
            chunk_end = min(current_pos + chunk_size, end + 1)
            chunk_content = '\n'.join(lines[current_pos:chunk_end])
            context = self.extract_context_lines(lines, current_pos, chunk_end - 1)
            
            chunk = Chunk(
                content=chunk_content,
                protocol_name=protocol_name,
                chunk_id=chunk_id,
                chunk_type=self.classify_chunk_type(section_name, chunk_content),
                start_line=current_pos,
                end_line=chunk_end - 1,
                context_lines=context,
                operations=self.extract_operations_from_chunk(chunk_content),
                complexity=self.calculate_chunk_complexity(chunk_content),
                protocol_type=protocol_type,
                metadata={'section_name': section_name, 'strategy': 'sliding_window_section'}
            )
            chunks.append(chunk)
            
            current_pos += chunk_size - overlap
            chunk_id += 1
        
        return chunks

    def sliding_window_chunk(self, lines: List[str], chunk_size: int, overlap: int, 
                           protocol_name: str, protocol_type: str) -> List[Chunk]:
        chunks = []
        current_pos = 0
        chunk_id = 0
        
        while current_pos < len(lines):
            chunk_end = min(current_pos + chunk_size, len(lines))
            chunk_content = '\n'.join(lines[current_pos:chunk_end])
            context = self.extract_context_lines(lines, current_pos, chunk_end - 1)
            
            chunk = Chunk(
                content=chunk_content,
                protocol_name=protocol_name,
                chunk_id=chunk_id,
                chunk_type=self.classify_chunk_type('unknown', chunk_content),
                start_line=current_pos,
                end_line=chunk_end - 1,
                context_lines=context,
                operations=self.extract_operations_from_chunk(chunk_content),
                complexity=self.calculate_chunk_complexity(chunk_content),
                protocol_type=protocol_type,
                metadata={'strategy': 'sliding_window'}
            )
            chunks.append(chunk)
            
            current_pos += chunk_size - overlap
            chunk_id += 1
        
        return chunks

    def loop_aware_chunk(self, code: str, protocol_structure: Dict) -> List[Chunk]:
        lines = code.split('\n')
        chunks = []
        protocol_name = protocol_structure['name']
        protocol_type = protocol_structure['protocol_type']
        
        # identify loop structures
        loop_starts = []
        loop_ends = []
        indent_stack = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r'^\s*for\s+', line) or re.match(r'^\s*while\s+', line):
                loop_starts.append(i)
                indent_stack.append(len(line) - len(line.lstrip()))
            elif stripped and indent_stack:
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_stack[-1] and loop_starts:
                    loop_ends.append(i - 1)
                    loop_starts.pop()
                    indent_stack.pop()
        
        # handle unclosed loops
        while loop_starts:
            loop_ends.append(len(lines) - 1)
            loop_starts.pop()
            indent_stack.pop()
        
        # create chunks around loops
        current_pos = 0
        chunk_id = 0
        
        for loop_start, loop_end in zip(loop_starts, loop_ends):
            # chunk before loop
            if current_pos < loop_start:
                pre_loop_content = '\n'.join(lines[current_pos:loop_start])
                if pre_loop_content.strip():
                    chunk = self.create_chunk(
                        pre_loop_content, protocol_name, chunk_id, 'setup',
                        current_pos, loop_start - 1, lines, protocol_type, 'loop_aware'
                    )
                    chunks.append(chunk)
                    chunk_id += 1
            
            # loop chunk
            loop_content = '\n'.join(lines[loop_start:loop_end + 1])
            chunk = self.create_chunk(
                loop_content, protocol_name, chunk_id, 'loop',
                loop_start, loop_end, lines, protocol_type, 'loop_aware'
            )
            chunks.append(chunk)
            chunk_id += 1
            current_pos = loop_end + 1
        
        # remaining content after last loop
        if current_pos < len(lines):
            remaining_content = '\n'.join(lines[current_pos:])
            if remaining_content.strip():
                chunk = self.create_chunk(
                    remaining_content, protocol_name, chunk_id, 'cleanup',
                    current_pos, len(lines) - 1, lines, protocol_type, 'loop_aware'
                )
                chunks.append(chunk)
        
        return chunks

    def create_chunk(self, content: str, protocol_name: str, chunk_id: int, chunk_type: str,
                    start_line: int, end_line: int, lines: List[str], protocol_type: str, strategy: str) -> Chunk:
        return Chunk(
            content=content,
            protocol_name=protocol_name,
            chunk_id=chunk_id,
            chunk_type=chunk_type,
            start_line=start_line,
            end_line=end_line,
            context_lines=self.extract_context_lines(lines, start_line, end_line),
            operations=self.extract_operations_from_chunk(content),
            complexity=self.calculate_chunk_complexity(content),
            protocol_type=protocol_type,
            metadata={'strategy': strategy}
        )

    def classify_chunk_type(self, section_name: str, content: str) -> str:
        content_lower = content.lower()
        
        if 'run' in section_name.lower() or 'main' in section_name.lower():
            if 'for ' in content_lower or 'while ' in content_lower:
                return 'loop'
            return 'main'
        elif any(keyword in content_lower for keyword in ['import', 'load_labware', 'load_instrument']):
            return 'setup'
        elif any(keyword in content_lower for keyword in ['home', 'drop_tip', 'return']):
            return 'cleanup'
        else:
            return 'main'

    def extract_operations_from_chunk(self, content: str) -> List[str]:
        operations = []
        operation_patterns = {
            'transfer': re.compile(r'\.(transfer|distribute|consolidate)\s*\('),
            'aspirate': re.compile(r'\.aspirate\s*\('),
            'dispense': re.compile(r'\.dispense\s*\('),
            'mix': re.compile(r'\.mix\s*\('),
            'pick_up_tip': re.compile(r'\.pick_up_tip\s*\('),
            'drop_tip': re.compile(r'\.drop_tip\s*\('),
            'delay': re.compile(r'\.delay\s*\('),
            'move_to': re.compile(r'\.move_to\s*\('),
            'touch_tip': re.compile(r'\.touch_tip\s*\('),
            'blow_out': re.compile(r'\.blow_out\s*\('),
            'home': re.compile(r'\.home\s*\('),
            'pause': re.compile(r'\.pause\s*\('),
            'for_loop': re.compile(r'^\s*for\s+'),
            'while_loop': re.compile(r'^\s*while\s+'),
            'if_statement': re.compile(r'^\s*if\s+')
        }
        
        for op_name, pattern in operation_patterns.items():
            if pattern.search(content):
                operations.append(op_name)
        
        return list(set(operations))

    def calculate_chunk_complexity(self, content: str) -> int:
        lines = content.split('\n')
        complexity = 0
        
        complexity += len([l for l in lines if l.strip()]) // 10  # non-empty lines
        complexity += content.count('for ') * 2  # loops
        complexity += content.count('while ') * 2
        complexity += content.count('if ') * 1  # conditionals
        complexity += content.count('.') // 5  # method calls
        
        return complexity

    def chunk_protocol(self, code: str, protocol_structure: Dict) -> List[Chunk]:
        protocol_type = protocol_structure['protocol_type']
        complexity = protocol_structure['complexity']
        
        # get chunking recommendations from analysis
        if self.analysis_report and 'chunking_recommendations' in self.analysis_report:
            recommendations = self.analysis_report['chunking_recommendations']
            type_recommendations = recommendations.get('chunk_sizes', {}).get(protocol_type, {})
            strategy = type_recommendations.get('strategy', 'simple_chunking')
            
            if strategy == 'section_based':
                chunk_size = type_recommendations.get('setup_chunk', 150)
                return self.chunk_by_sections(code, protocol_structure, chunk_size)
            elif strategy == 'loop_aware':
                return self.loop_aware_chunk(code, protocol_structure)
            elif strategy == 'sliding_window':
                chunk_size = type_recommendations.get('base_chunk', 200)
                overlap = type_recommendations.get('overlap', 50)
                return self.chunk_by_sections(code, protocol_structure, chunk_size, overlap)
        
        # fallback to adaptive chunking based on complexity
        if complexity > 20:
            return self.chunk_by_sections(code, protocol_structure, 120, 40)  # smaller chunks for complex protocols
        elif protocol_type in ['pcr', 'serial_dilution']:
            return self.loop_aware_chunk(code, protocol_structure)
        else:
            return self.chunk_by_sections(code, protocol_structure, 150, 30)

    def chunk_all_protocols(self, protocols_dir: str) -> List[Chunk]:
        protocols_path = Path(protocols_dir)
        all_chunks = []
        
        if not self.protocol_structures:
            print("No protocol structures loaded. Run analyzer first.")
            return []
        
        print(f"Chunking {len(self.protocol_structures)} protocols...")
        
        for i, structure in enumerate(self.protocol_structures):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(self.protocol_structures)}")
            
            protocol_name = structure['name']
            protocol_folder = protocols_path / protocol_name
            
            # extract code from protocol folder
            analyzer = ProtocolAnalyzer(str(protocols_path))
            code = analyzer.extract_protocol_code(protocol_folder)
            
            if code:
                protocol_chunks = self.chunk_protocol(code, structure)
                all_chunks.extend(protocol_chunks)
                
                # update stats
                self.chunking_stats['chunks_by_type'][structure['protocol_type']] = \
                    self.chunking_stats['chunks_by_type'].get(structure['protocol_type'], 0) + len(protocol_chunks)
        
        self.chunks = all_chunks
        self.update_chunking_stats()
        return all_chunks

    def update_chunking_stats(self):
        if not self.chunks:
            return
        
        self.chunking_stats['total_protocols'] = len(set(chunk.protocol_name for chunk in self.chunks))
        self.chunking_stats['total_chunks'] = len(self.chunks)
        self.chunking_stats['avg_chunk_size'] = sum(len(chunk.content.split('\n')) for chunk in self.chunks) / len(self.chunks)
        
        # count by strategy
        for chunk in self.chunks:
            strategy = chunk.metadata.get('strategy', 'unknown')
            self.chunking_stats['chunks_by_strategy'][strategy] = \
                self.chunking_stats['chunks_by_strategy'].get(strategy, 0) + 1

    def save_chunks(self, output_dir: str = "chunked_data"):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # save chunks as JSON for training pipeline
        chunks_data = []
        for chunk in self.chunks:
            chunks_data.append({
                'content': chunk.content,
                'protocol_name': chunk.protocol_name,
                'chunk_id': chunk.chunk_id,
                'chunk_type': chunk.chunk_type,
                'start_line': chunk.start_line,
                'end_line': chunk.end_line,
                'context_lines': chunk.context_lines,
                'operations': chunk.operations,
                'complexity': chunk.complexity,
                'protocol_type': chunk.protocol_type,
                'metadata': chunk.metadata
            })
        
        with open(output_path / "protocol_chunks.json", 'w') as f:
            json.dump(chunks_data, f, indent=2)
        
        # save chunking statistics
        with open(output_path / "chunking_stats.json", 'w') as f:
            json.dump(self.chunking_stats, f, indent=2)
        
        print(f"\nChunking complete! Results saved to {output_path}")
        print(f"Generated {self.chunking_stats['total_chunks']} chunks from {self.chunking_stats['total_protocols']} protocols")
        print(f"Average chunk size: {self.chunking_stats['avg_chunk_size']:.1f} lines")
        print(f"Chunks by protocol type: {self.chunking_stats['chunks_by_type']}")
        print(f"Chunks by strategy: {self.chunking_stats['chunks_by_strategy']}")

def main():
    # initialize chunker with existing analysis results
    chunker = ProtocolChunker("analysis_results")
    
    # chunk all protocols
    base_dir = "/Users/bellap/Desktop/Protocols-develop/protoBuilds"
    chunks = chunker.chunk_all_protocols(base_dir)
    
    # save results
    chunker.save_chunks()

if __name__ == "__main__":
    main()