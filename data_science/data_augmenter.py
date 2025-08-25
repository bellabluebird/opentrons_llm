#!/usr/bin/env python3
# data augmentation pipeline for OT-2 protocol training data
#   - transforms chunks into diverse training examples with realistic user prompts
#   - expands ~3000 chunks into 15000+ training examples through strategic variations

import json
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import itertools

@dataclass
class TrainingExample:
    prompt: str
    code: str
    protocol_type: str
    chunk_type: str
    operations: List[str]
    difficulty: str  # 'beginner', 'intermediate', 'advanced'
    style: str  # 'formal', 'casual', 'question', 'conversational'
    metadata: Dict[str, Any]

class DataAugmenter:
    def __init__(self, chunks_path: str):
        self.chunks = self.load_chunks(chunks_path)
        self.training_examples = []
        self.augmentation_stats = {
            'original_chunks': 0,
            'generated_examples': 0,
            'examples_by_style': defaultdict(int),
            'examples_by_type': defaultdict(int),
            'examples_by_difficulty': defaultdict(int)
        }
        
        # realistic user query templates
        self.prompt_templates = self.setup_prompt_templates()
        
        # parameter variation rules
        self.parameter_variations = self.setup_parameter_variations()

    def load_chunks(self, chunks_path: str) -> List[Dict]:
        with open(chunks_path, 'r') as f:
            chunks_data = json.load(f)
        print(f"Loaded {len(chunks_data)} chunks for augmentation")
        return chunks_data

    def setup_prompt_templates(self) -> Dict[str, List[str]]:
        return {
            'formal': [
                "Generate an OpenTrons protocol that performs {operation}.",
                "Create a protocol to {operation} using the OT-2 robot.",
                "Write an OpenTrons script for {operation}.",
                "Implement a protocol that will {operation}.",
                "Design an automated procedure to {operation}."
            ],
            'casual': [
                "I need to {operation} on the OT-2, can you help?",
                "How do I {operation} using OpenTrons?",
                "Can you write code to {operation}?",
                "I want to {operation} - what would the script look like?",
                "Help me automate {operation} on my robot."
            ],
            'question': [
                "What's the OpenTrons code for {operation}?",
                "How would you {operation} in an OT-2 protocol?",
                "What does the script look like to {operation}?",
                "Can you show me how to {operation}?",
                "What's the best way to {operation} with OpenTrons?"
            ],
            'conversational': [
                "I'm trying to {operation} but not sure how to code it.",
                "My lab needs to {operation} - can you walk me through it?",
                "I've been struggling with {operation} automation.",
                "Could you help me figure out how to {operation}?",
                "I'm new to OpenTrons - how do I {operation}?"
            ],
            'specific_context': [
                "For a {protocol_type} protocol, {operation}.",
                "In my {protocol_type} workflow, I need to {operation}.",
                "This is part of a {protocol_type} procedure: {operation}.",
                "During {protocol_type}, how do I {operation}?",
                "My {protocol_type} protocol requires {operation}."
            ]
        }

    def setup_parameter_variations(self) -> Dict[str, Dict]:
        return {
            'volumes': {
                'small': ['5', '10', '15', '20'],
                'medium': ['50', '100', '150', '200'],
                'large': ['500', '1000', '1500']
            },
            'tips': [
                'p10_single', 'p20_single', 'p300_single', 'p1000_single',
                'p10_multi', 'p20_multi', 'p300_multi'
            ],
            'plates': [
                '96-well', '384-well', '24-well', '6-well',
                'corning_96_wellplate_360ul_flat',
                'nest_96_wellplate_200ul_flat',
                'biorad_96_wellplate_200ul_pcr'
            ],
            'modules': [
                'temperature module', 'magnetic module', 'thermocycler',
                'heater-shaker', 'plate reader'
            ],
            'positions': ['A1', 'B2', 'C3', 'H12', 'A12', 'H1'],
            'temperatures': ['4', '25', '37', '65', '95'],
            'times': ['30 seconds', '1 minute', '2 minutes', '5 minutes', '10 minutes']
        }

    def extract_operation_description(self, chunk: Dict) -> str:
        operations = chunk.get('operations', [])
        protocol_type = chunk.get('protocol_type', 'unknown')
        chunk_type = chunk.get('chunk_type', 'main')
        
        # create natural language descriptions from operations
        descriptions = []
        
        if 'transfer' in operations:
            descriptions.append('transfer liquids between wells')
        if 'aspirate' in operations and 'dispense' in operations:
            descriptions.append('pipette samples')
        if 'mix' in operations:
            descriptions.append('mix solutions')
        if 'pick_up_tip' in operations:
            descriptions.append('handle pipette tips')
        if 'for_loop' in operations:
            descriptions.append('repeat operations across multiple wells')
        if 'delay' in operations:
            descriptions.append('include timing delays')
        if 'move_to' in operations:
            descriptions.append('position the pipette')
        
        # protocol-specific descriptions
        if protocol_type == 'pcr':
            if chunk_type == 'setup':
                descriptions.append('set up PCR thermocycling')
            elif chunk_type == 'loop':
                descriptions.append('run PCR thermal cycles')
        elif protocol_type == 'serial_dilution':
            descriptions.append('perform serial dilutions')
        elif protocol_type == 'magnetic_bead':
            descriptions.append('process magnetic beads')
        elif protocol_type == 'plate_mapping':
            descriptions.append('distribute samples across plates')
        
        if not descriptions:
            descriptions = ['perform laboratory automation']
        
        return random.choice(descriptions)

    def apply_parameter_variations(self, code: str, chunk: Dict) -> List[str]:
        variations = [code]  # include original
        
        # volume variations
        volume_patterns = [
            (r'(\d+)\s*[μu]?[lL]', lambda m: f"{random.choice(self.parameter_variations['volumes']['small'])}uL"),
            (r'(\d+)\s*ml', lambda m: f"{random.choice(self.parameter_variations['volumes']['medium'])}uL"),
        ]
        
        for pattern, replacement in volume_patterns:
            if re.search(pattern, code):
                varied_code = re.sub(pattern, replacement, code)
                if varied_code != code:
                    variations.append(varied_code)
        
        # plate variations
        plate_pattern = r"load_labware\s*\(\s*['\"]([^'\"]*plate[^'\"]*)['\"]"
        if re.search(plate_pattern, code):
            new_plate = random.choice(self.parameter_variations['plates'])
            varied_code = re.sub(plate_pattern, f'load_labware("{new_plate}"', code)
            variations.append(varied_code)
        
        # tip variations
        tip_pattern = r"load_instrument\s*\(\s*['\"]([^'\"]*)['\"]"
        if re.search(tip_pattern, code):
            new_tip = random.choice(self.parameter_variations['tips'])
            varied_code = re.sub(tip_pattern, f'load_instrument("{new_tip}"', code)
            variations.append(varied_code)
        
        # position variations
        position_pattern = r"['\"]([A-H]\d+)['\"]"
        positions_in_code = re.findall(position_pattern, code)
        if positions_in_code:
            varied_code = code
            for pos in positions_in_code:
                new_pos = random.choice(self.parameter_variations['positions'])
                varied_code = varied_code.replace(f'"{pos}"', f'"{new_pos}"', 1)
            variations.append(varied_code)
        
        return list(set(variations))  # remove duplicates

    def add_code_style_variations(self, code: str) -> List[str]:
        variations = [code]
        
        # add comments variation
        lines = code.split('\n')
        commented_lines = []
        for line in lines:
            commented_lines.append(line)
            if any(op in line.lower() for op in ['transfer', 'aspirate', 'dispense', 'mix']):
                if random.random() < 0.3:  # 30% chance to add comment
                    commented_lines.append(f"    # {random.choice(['Step completed', 'Processing sample', 'Moving to next step'])}")
        
        variations.append('\n'.join(commented_lines))
        
        # variable name variations
        var_variations = {
            'plate': ['source_plate', 'dest_plate', 'sample_plate'],
            'pipette': ['p300', 'multichannel', 'single_channel'],
            'tip_rack': ['tips', 'tip_box', 'pipette_tips']
        }
        
        varied_code = code
        for original, alternatives in var_variations.items():
            if original in code.lower():
                new_var = random.choice(alternatives)
                # Simple replacement - could be more sophisticated
                varied_code = varied_code.replace(original, new_var)
        
        variations.append(varied_code)
        
        return list(set(variations))

    def generate_realistic_errors(self, code: str) -> List[Tuple[str, str]]:
        """Generate common user errors for training robustness"""
        error_examples = []
        
        # missing import error
        if 'from opentrons import protocol_api' in code:
            error_code = code.replace('from opentrons import protocol_api', '')
            error_examples.append((error_code, "Missing import statement"))
        
        # incorrect volume units
        if 'uL' in code:
            error_code = code.replace('uL', 'ul')  # common typo
            error_examples.append((error_code, "Incorrect volume units"))
        
        # missing tip pickup
        if 'pipette.aspirate' in code and 'pick_up_tip' in code:
            lines = code.split('\n')
            filtered_lines = [line for line in lines if 'pick_up_tip' not in line]
            error_code = '\n'.join(filtered_lines)
            error_examples.append((error_code, "Missing tip pickup"))
        
        return error_examples

    def classify_difficulty(self, chunk: Dict, prompt_style: str) -> str:
        complexity = chunk.get('complexity', 0)
        operations = chunk.get('operations', [])
        protocol_type = chunk.get('protocol_type', 'basic')
        
        advanced_ops = ['for_loop', 'while_loop', 'if_statement', 'try_except']
        intermediate_ops = ['mix', 'delay', 'move_to', 'touch_tip']
        
        if complexity > 15 or any(op in operations for op in advanced_ops):
            return 'advanced'
        elif complexity > 5 or any(op in operations for op in intermediate_ops):
            return 'intermediate'
        else:
            return 'beginner'

    def augment_chunk(self, chunk: Dict) -> List[TrainingExample]:
        examples = []
        operation_desc = self.extract_operation_description(chunk)
        protocol_type = chunk.get('protocol_type', 'unknown')
        chunk_type = chunk.get('chunk_type', 'main')
        operations = chunk.get('operations', [])
        
        # generate parameter variations of the code
        code_variations = self.apply_parameter_variations(chunk['content'], chunk)
        
        # add style variations
        all_code_variations = []
        for code_var in code_variations:
            all_code_variations.extend(self.add_code_style_variations(code_var))
        
        # generate prompts in different styles
        for style, templates in self.prompt_templates.items():
            for template in templates[:2]:  # limit to 2 templates per style
                for code_var in all_code_variations[:3]:  # limit to 3 code variations
                    if style == 'specific_context':
                        prompt = template.format(operation=operation_desc, protocol_type=protocol_type)
                    else:
                        prompt = template.format(operation=operation_desc)
                    
                    difficulty = self.classify_difficulty(chunk, style)
                    
                    example = TrainingExample(
                        prompt=prompt,
                        code=code_var,
                        protocol_type=protocol_type,
                        chunk_type=chunk_type,
                        operations=operations,
                        difficulty=difficulty,
                        style=style,
                        metadata={
                            'original_chunk_id': chunk.get('chunk_id'),
                            'protocol_name': chunk.get('protocol_name'),
                            'augmentation_method': 'template_variation'
                        }
                    )
                    examples.append(example)
        
        # add error examples for robustness (5% of examples)
        if random.random() < 0.05:
            error_examples = self.generate_realistic_errors(chunk['content'])
            for error_code, error_desc in error_examples:
                prompt = f"Fix this OpenTrons code that has {error_desc}: {operation_desc}"
                example = TrainingExample(
                    prompt=prompt,
                    code=chunk['content'],  # correct code as target
                    protocol_type=protocol_type,
                    chunk_type=chunk_type,
                    operations=operations,
                    difficulty='advanced',
                    style='error_correction',
                    metadata={
                        'original_chunk_id': chunk.get('chunk_id'),
                        'protocol_name': chunk.get('protocol_name'),
                        'augmentation_method': 'error_correction',
                        'error_type': error_desc
                    }
                )
                examples.append(example)
        
        return examples

    def augment_all_chunks(self) -> List[TrainingExample]:
        print(f"Augmenting {len(self.chunks)} chunks...")
        all_examples = []
        
        for i, chunk in enumerate(self.chunks):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(self.chunks)}")
            
            chunk_examples = self.augment_chunk(chunk)
            all_examples.extend(chunk_examples)
        
        self.training_examples = all_examples
        self.update_augmentation_stats()
        return all_examples

    def update_augmentation_stats(self):
        self.augmentation_stats['original_chunks'] = len(self.chunks)
        self.augmentation_stats['generated_examples'] = len(self.training_examples)
        
        for example in self.training_examples:
            self.augmentation_stats['examples_by_style'][example.style] += 1
            self.augmentation_stats['examples_by_type'][example.protocol_type] += 1
            self.augmentation_stats['examples_by_difficulty'][example.difficulty] += 1

    def balance_dataset(self):
        """Balance the dataset across protocol types and difficulty levels"""
        # group examples by protocol type
        by_type = defaultdict(list)
        for example in self.training_examples:
            by_type[example.protocol_type].append(example)
        
        # find minimum count to balance to
        min_count = min(len(examples) for examples in by_type.values()) if by_type else 0
        target_count = max(min_count, 200)  # at least 200 examples per type
        
        balanced_examples = []
        for protocol_type, examples in by_type.items():
            if len(examples) >= target_count:
                # randomly sample if we have too many
                balanced_examples.extend(random.sample(examples, target_count))
            else:
                # use all examples if we have too few
                balanced_examples.extend(examples)
                print(f"Warning: Only {len(examples)} examples for {protocol_type} (target: {target_count})")
        
        self.training_examples = balanced_examples
        print(f"Balanced dataset to {len(balanced_examples)} examples")

    def save_training_data(self, output_dir: str = "training_data"):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # save in multiple formats for different training frameworks
        
        # JSONL format for Hugging Face
        with open(output_path / "training_data.jsonl", 'w') as f:
            for example in self.training_examples:
                training_record = {
                    'input': example.prompt,
                    'output': example.code,
                    'protocol_type': example.protocol_type,
                    'difficulty': example.difficulty,
                    'style': example.style,
                    'metadata': example.metadata
                }
                f.write(json.dumps(training_record) + '\n')
        
        # JSON format for custom training loops
        training_data = []
        for example in self.training_examples:
            training_data.append({
                'prompt': example.prompt,
                'code': example.code,
                'protocol_type': example.protocol_type,
                'chunk_type': example.chunk_type,
                'operations': example.operations,
                'difficulty': example.difficulty,
                'style': example.style,
                'metadata': example.metadata
            })
        
        with open(output_path / "training_data.json", 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # save augmentation statistics
        with open(output_path / "augmentation_stats.json", 'w') as f:
            json.dump(self.augmentation_stats, f, indent=2)
        
        # create train/validation split
        random.shuffle(self.training_examples)
        split_idx = int(0.9 * len(self.training_examples))
        
        train_examples = self.training_examples[:split_idx]
        val_examples = self.training_examples[split_idx:]
        
        # save splits
        with open(output_path / "train_data.jsonl", 'w') as f:
            for example in train_examples:
                record = {'input': example.prompt, 'output': example.code}
                f.write(json.dumps(record) + '\n')
        
        with open(output_path / "val_data.jsonl", 'w') as f:
            for example in val_examples:
                record = {'input': example.prompt, 'output': example.code}
                f.write(json.dumps(record) + '\n')
        
        print(f"\nAugmentation complete! Results saved to {output_path}")
        print(f"Generated {self.augmentation_stats['generated_examples']} examples from {self.augmentation_stats['original_chunks']} chunks")
        print(f"Expansion ratio: {self.augmentation_stats['generated_examples'] / self.augmentation_stats['original_chunks']:.1f}x")
        print(f"Examples by style: {dict(self.augmentation_stats['examples_by_style'])}")
        print(f"Examples by difficulty: {dict(self.augmentation_stats['examples_by_difficulty'])}")
        print(f"Training set: {len(train_examples)} examples")
        print(f"Validation set: {len(val_examples)} examples")

def main():
    # initialize augmenter with chunked data
    augmenter = DataAugmenter("chunked_data/protocol_chunks.json")
    
    # generate all augmented examples
    examples = augmenter.augment_all_chunks()
    
    # balance the dataset
    augmenter.balance_dataset()
    
    # save training data
    augmenter.save_training_data()

if __name__ == "__main__":
    main()