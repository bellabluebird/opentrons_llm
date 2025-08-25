#!/usr/bin/env python3
# training dataset generator with biological context and validation
#  - analyzes protocol chunks to understand biological purpose
#  - creates training examples with proper context and validation
#  - builds on protocol_analyzer.py and protocol_chunker.py results

import json                                     # read protocol data and save results
import re                                       # pattern matching for biological terms
import ast                                      # python syntax validation
import random                                   # generate varied training examples
from pathlib import Path                        # file handling  
from typing import Dict, List, Tuple, Any, Optional, Set  # type hints for readability
from dataclasses import dataclass              # structured data storage
from collections import defaultdict            # automatic dictionary creation
import pandas as pd

# data structure to hold biological information about protocols
@dataclass
class BiologicalContext:
    protocol_type: str              # pcr, serial_dilution, magnetic_bead, etc
    scientific_goal: str            # what the protocol is trying to accomplish
    reagents: List[str]             # chemicals and biological materials used
    equipment: List[str]            # labware and modules needed
    typical_volumes: Dict[str, str] # common volume ranges used
    temperature_conditions: List[str] # temperatures mentioned in protocol
    timing_requirements: List[str]  # time delays and incubation periods
    biological_process: str         # main biological process type
    expected_outcome: str           # what should happen after protocol runs

# complete training example with code and context
@dataclass 
class TrainingExample:
    prompt: str                     # natural language request from user
    code: str                       # python code that fulfills the request
    biological_context: BiologicalContext  # scientific context
    dependencies: Dict[str, List[str]]  # variables that need to be defined
    validation_status: Dict[str, bool]  # which validation checks passed
    difficulty: str                 # beginner, intermediate, advanced
    style: str                      # formal, casual, question, etc
    operations: List[str]           # opentrons operations used in code
    metadata: Dict[str, Any]        # additional information

class BiologicalContextExtractor:
    # extracts biological meaning from protocol descriptions and code
    def __init__(self):
        # set up keyword patterns and biological knowledge
        self.biological_patterns = self.setup_biological_patterns()
        self.reagent_database = self.setup_reagent_database() 
        self.process_knowledge = self.setup_process_knowledge()

    def setup_biological_patterns(self) -> Dict[str, List[str]]:
        # define keywords that help identify different biological processes
        # used to classify protocols based on description text
        return {
            'pcr_keywords': [
                'amplification', 'primer', 'polymerase', 'denaturation', 'annealing', 
                'extension', 'thermocycler', 'dna', 'template', 'cycles'
            ],
            'serial_dilution_keywords': [
                'dilution', 'concentration', 'serial', 'gradient', 'dose-response',
                'standards', 'calibration', 'titration'
            ],
            'magnetic_bead_keywords': [
                'magnetic', 'beads', 'capture', 'purification', 'isolation',
                'binding', 'washing', 'elution', 'cleanup'
            ],
            'cell_culture_keywords': [
                'cells', 'media', 'culture', 'growth', 'viability', 'passage',
                'seeding', 'confluency', 'incubation'
            ],
            'elisa_keywords': [
                'elisa', 'antibody', 'antigen', 'binding', 'detection', 'substrate',
                'colorimetric', 'absorbance', 'immunoassay'
            ]
        }

    def setup_reagent_database(self) -> Dict[str, Dict]:
        # basic information about common lab reagents
        # helps validate protocol parameters and understand biological context
        return {
            'dna': {
                'typical_concentrations': ['1 ng/uL', '10 ng/uL', '100 ng/uL'],
                'storage_conditions': ['4C', '-20C', '-80C'],
                'stability': 'stable at -20C for years',
                'common_volumes': ['1 uL', '2 uL', '5 uL']
            },
            'primer': {
                'typical_concentrations': ['10 uM', '100 uM'],
                'storage_conditions': ['-20C'],
                'stability': 'stable at -20C for 2+ years',
                'common_volumes': ['0.5 uL', '1 uL', '2 uL']
            },
            'polymerase': {
                'typical_concentrations': ['5 U/uL', '2.5 U/uL'],
                'storage_conditions': ['-20C'],
                'stability': 'stable at -20C for 6 months',
                'common_volumes': ['0.2 uL', '0.5 uL', '1 uL']
            },
            'buffer': {
                'typical_concentrations': ['1X', '2X', '5X', '10X'],
                'storage_conditions': ['4C', 'room temperature'],
                'stability': 'stable for months',
                'common_volumes': ['2.5 uL', '5 uL', '10 uL']
            }
        }

    def setup_process_knowledge(self) -> Dict[str, Dict]:
        return {
            'pcr': {
                'standard_conditions': {
                    'denaturation': '95C for 30 seconds',
                    'annealing': '55-65C for 30 seconds', 
                    'extension': '72C for 1-2 minutes'
                },
                'typical_cycles': '25-40 cycles',
                'reaction_volume': '25-50 uL',
                'expected_outcome': 'amplified DNA product'
            },
            'serial_dilution': {
                'dilution_factors': ['1:2', '1:5', '1:10'],
                'typical_steps': '6-12 dilution points',
                'mixing_requirements': '5-10 pipette cycles',
                'expected_outcome': 'concentration gradient'
            },
            'magnetic_bead': {
                'binding_time': '5-15 minutes',
                'washing_cycles': '3-5 washes',
                'elution_volume': '10-50% of binding volume',
                'expected_outcome': 'purified target molecules'
            }
        }

    def extract_from_readme(self, readme_path: Path) -> Dict[str, Any]:
        context = {
            'description': '',
            'reagents': [],
            'equipment': [],
            'process_type': 'unknown',
            'scientific_goal': ''
        }
        
        try:
            if readme_path.suffix == '.json':
                with open(readme_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # extract description from various possible fields
                description_text = ''
                for field in ['description', 'summary', 'notes', 'protocol_description']:
                    if field in data and data[field]:
                        description_text += str(data[field]) + ' '
                
                context['description'] = description_text.strip()
                
                # extract other relevant fields
                if 'reagents' in data:
                    context['reagents'] = data['reagents']
                if 'equipment' in data:
                    context['equipment'] = data['equipment']
                if 'labware' in data:
                    context['equipment'].extend(data['labware'])
                    
            elif readme_path.suffix == '.md':
                with open(readme_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    context['description'] = content
                    
        except Exception as e:
            print(f"Error reading {readme_path}: {e}")
            
        return context

    def classify_biological_process(self, description: str, code: str = '') -> str:
        # figure out what biological process this protocol performs
        # combines description text and code to find matching keywords
        text = (description + ' ' + code).lower()
        
        # count how many keywords match each process type
        scores = {}
        for process, keywords in self.biological_patterns.items():
            process_name = process.replace('_keywords', '')
            scores[process_name] = sum(1 for keyword in keywords if keyword in text)
        
        # return the process type with the most keyword matches
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                return max(scores, key=scores.get)
        
        # if no keywords match, we don't know what process this is
        return 'unknown'

    def extract_reagents_from_text(self, text: str) -> List[str]:
        reagents = []
        text_lower = text.lower()
        
        # common reagent patterns
        reagent_patterns = [
            r'\b(dna|rna)\b',
            r'\b(primer|primers)\b', 
            r'\b(polymerase|taq|phusion)\b',
            r'\b(buffer|pbs|tris)\b',
            r'\b(antibody|antibodies)\b',
            r'\b(enzyme|enzymes)\b',
            r'\b(substrate|substrates)\b',
            r'\b(media|medium)\b',
            r'\b(cells|cell line)\b'
        ]
        
        for pattern in reagent_patterns:
            matches = re.findall(pattern, text_lower)
            reagents.extend(matches)
            
        return list(set(reagents))  # remove duplicates

    def extract_biological_context(self, readme_data: Dict, code: str, protocol_structure: Dict) -> BiologicalContext:
        description = readme_data.get('description', '')
        
        # classify the biological process
        biological_process = self.classify_biological_process(description, code)
        protocol_type = protocol_structure.get('protocol_type', 'unknown')
        
        # extract reagents
        reagents = readme_data.get('reagents', [])
        if not reagents:
            reagents = self.extract_reagents_from_text(description + ' ' + code)
        
        # extract equipment
        equipment = readme_data.get('equipment', [])
        if not equipment:
            equipment = protocol_structure.get('labware_types', [])
            equipment.extend(protocol_structure.get('pipette_types', []))
        
        # get process-specific knowledge
        process_info = self.process_knowledge.get(biological_process, {})
        
        return BiologicalContext(
            protocol_type=protocol_type,
            scientific_goal=self.extract_scientific_goal(description, biological_process),
            reagents=reagents,
            equipment=equipment,
            typical_volumes=self.extract_volumes_from_code(code),
            temperature_conditions=self.extract_temperatures(code, description),
            timing_requirements=self.extract_timing(code, description),
            biological_process=biological_process,
            expected_outcome=process_info.get('expected_outcome', 'processed samples')
        )

    def extract_scientific_goal(self, description: str, process: str) -> str:
        # simple goal extraction based on process type and description keywords
        if process == 'pcr':
            if 'quantitative' in description.lower() or 'qpcr' in description.lower():
                return 'quantitative DNA amplification'
            return 'DNA amplification'
        elif process == 'serial_dilution':
            if 'standard' in description.lower():
                return 'create calibration standards'
            return 'create concentration gradient'
        elif process == 'magnetic_bead':
            return 'purify target molecules'
        else:
            return 'process biological samples'

    def extract_volumes_from_code(self, code: str) -> Dict[str, str]:
        volumes = {}
        
        # extract volume patterns from code
        volume_patterns = [
            r'(\d+\.?\d*)\s*[μu]?[lL]',  # microliters
            r'(\d+\.?\d*)\s*ml',         # milliliters
        ]
        
        found_volumes = []
        for pattern in volume_patterns:
            found_volumes.extend(re.findall(pattern, code))
        
        if found_volumes:
            volumes['typical_volume'] = f"{found_volumes[0]} uL"
            if len(found_volumes) > 1:
                volumes['max_volume'] = f"{max(map(float, found_volumes))} uL"
                volumes['min_volume'] = f"{min(map(float, found_volumes))} uL"
        
        return volumes

    def extract_temperatures(self, code: str, description: str) -> List[str]:
        text = code + ' ' + description
        temp_pattern = r'(\d+\.?\d*)\s*[°]?[cC]'
        temperatures = re.findall(temp_pattern, text)
        return [f"{temp}C" for temp in set(temperatures)]

    def extract_timing(self, code: str, description: str) -> List[str]:
        text = code + ' ' + description
        time_patterns = [
            r'(\d+\.?\d*)\s*seconds?',
            r'(\d+\.?\d*)\s*minutes?',
            r'(\d+\.?\d*)\s*hours?'
        ]
        
        timings = []
        for pattern in time_patterns:
            matches = re.findall(pattern, text)
            timings.extend(matches)
            
        return list(set(timings))

class DependencyTracker:
    # tracks what variables are defined and used in code chunks
    # helps make sure chunks have all the variables they need to run
    def __init__(self):
        self.variable_patterns = self.setup_variable_patterns()

    def setup_variable_patterns(self) -> Dict[str, str]:
        # regex patterns to find where variables are defined in opentrons code
        return {
            'labware': r'(\w+)\s*=\s*protocol\.load_labware',
            'pipette': r'(\w+)\s*=\s*protocol\.load_instrument', 
            'module': r'(\w+)\s*=\s*protocol\.load_module',
            'tips': r'(\w+)\s*=.*tip.*rack'
        }

    def extract_variable_dependencies(self, code: str) -> Dict[str, List[str]]:
        # analyze code to find what variables are defined and used
        # this helps us add missing definitions when creating training examples
        lines = code.split('\n')
        dependencies = {
            'defined': [],              # variables that get created in this code
            'used': [],                 # variables that the code expects to exist
            'required_imports': [],     # import statements needed
            'required_setup': []        # setup lines that should be included
        }
        
        # track variable definitions
        for line in lines:
            for var_type, pattern in self.variable_patterns.items():
                matches = re.findall(pattern, line)
                for match in matches:
                    dependencies['defined'].append(match)
                    if var_type in ['labware', 'pipette', 'module']:
                        dependencies['required_setup'].append(line.strip())
        
        # track variable usage
        for line in lines:
            for var in dependencies['defined']:
                if var in line and f'{var} =' not in line:
                    dependencies['used'].append(var)
        
        # check for required imports
        if 'protocol_api' in code:
            dependencies['required_imports'].append('from opentrons import protocol_api')
        if 'math.' in code:
            dependencies['required_imports'].append('import math')
        
        return dependencies

    def make_chunk_executable(self, chunk_code: str, all_dependencies: Dict) -> str:
        """Add necessary imports and variable definitions to make chunk executable"""
        
        # normalize indentation first
        lines = chunk_code.split('\n')
        normalized_lines = []
        
        for line in lines:
            if line.strip():  # non-empty line
                # remove existing indentation and normalize to 4 spaces per level
                stripped = line.lstrip()
                # estimate indentation level (rough heuristic)
                if line.startswith('    '):
                    indent_level = (len(line) - len(stripped)) // 4
                else:
                    indent_level = 0
                normalized_lines.append('    ' * indent_level + stripped)
            else:
                normalized_lines.append('')  # preserve empty lines
        
        normalized_chunk = '\n'.join(normalized_lines)
        
        # start building executable code
        executable_code = []
        
        # add required imports (always at the top)
        required_imports = all_dependencies.get('required_imports', [])
        if 'from opentrons import protocol_api' not in required_imports:
            if any(op in chunk_code for op in ['protocol.', 'ProtocolContext', 'protocol_api']):
                required_imports.insert(0, 'from opentrons import protocol_api')
        
        for import_stmt in required_imports:
            if import_stmt not in normalized_chunk:
                executable_code.append(import_stmt)
        
        # check if we need a function wrapper
        needs_function_wrapper = any(op in normalized_chunk for op in [
            'protocol.', 'pipette.', '.transfer', '.aspirate', '.dispense',
            'load_labware', 'load_instrument', 'load_module'
        ])
        
        # add function definition if needed
        if needs_function_wrapper and 'def run(' not in normalized_chunk:
            executable_code.append('')  # blank line
            executable_code.append('def run(protocol: protocol_api.ProtocolContext):')
            
            # add required setup code inside function
            for setup_line in all_dependencies.get('required_setup', []):
                if setup_line not in normalized_chunk:
                    executable_code.append(f'    {setup_line}')
            
            # add blank line before main code
            if all_dependencies.get('required_setup'):
                executable_code.append('')
            
            # indent the entire chunk for function body
            for line in normalized_lines:
                if line.strip():
                    executable_code.append(f'    {line}')
                else:
                    executable_code.append('')
        else:
            # no function wrapper needed, add setup at module level
            for setup_line in all_dependencies.get('required_setup', []):
                if setup_line not in normalized_chunk:
                    executable_code.append(setup_line)
            
            if all_dependencies.get('required_setup'):
                executable_code.append('')
                
            executable_code.extend(normalized_lines)
        
        return '\n'.join(executable_code)

class ProtocolValidator:
    # checks if generated code is valid and safe to run
    def __init__(self):
        self.valid_api_calls = self.setup_valid_api_calls()

    def setup_valid_api_calls(self) -> Set[str]:
        # list of legitimate opentrons API method calls
        # used to catch invalid or dangerous method calls in generated code
        return {
            # protocol methods
            'load_labware', 'load_instrument', 'load_module', 'comment', 'pause',
            'delay', 'home', 'set_rail_lights', 'max_speeds',
            
            # pipette methods  
            'aspirate', 'dispense', 'transfer', 'distribute', 'consolidate',
            'pick_up_tip', 'drop_tip', 'mix', 'blow_out', 'touch_tip',
            'air_gap', 'move_to', 'return_tip', 'reset_tipracks',
            
            # labware methods
            'wells', 'rows', 'cols', 'top', 'bottom', 'center', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6',
            'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
            'B9', 'B10', 'B11', 'B12', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10',
            'C11', 'C12', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12',
            
            # module methods
            'set_temperature', 'deactivate', 'engage', 'disengage',
            'open_lid', 'close_lid', 'set_block_temperature', 'set_lid_temperature',
            'set_and_wait_for_temperature', 'deactivate_heater', 'open_labware_latch', 'close_labware_latch',
            
            # mathematical methods (commonly used in protocols)
            'ceil', 'floor', 'sqrt', 'pow', 'round',
            
            # common python methods used in protocols
            'format', 'append', 'split', 'join', 'strip', 'replace', 'upper', 'lower',
            'range', 'len', 'enumerate', 'zip', 'min', 'max', 'sum', 'abs',
            
            # protocol-specific patterns
            'get_values', 'get', 'set', 'update', 'clear', 'copy', 'keys', 'values', 'items'
        }

    def validate_syntax(self, code: str) -> Tuple[bool, str]:
        # check if the generated code has valid python syntax
        try:
            ast.parse(code)
            return True, "Valid syntax"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

    def validate_opentrons_api(self, code: str) -> Tuple[bool, str]:
        # check if code only uses safe and valid opentrons API calls
        invalid_methods = []
        
        # find all method calls in the code
        method_pattern = r'\.(\w+)\s*\('
        methods_used = re.findall(method_pattern, code)
        
        # only flag obviously dangerous method calls
        # most opentrons methods are safe so we allow unknown methods
        dangerous_patterns = [
            'delete', 'remove', 'destroy', 'kill', 'terminate',
            'format_disk', 'rm', 'rmdir', 'unlink', 'system', 'exec'
        ]
        
        for method in methods_used:
            if method not in self.valid_api_calls:
                # check if this method could be dangerous
                if method in dangerous_patterns:
                    invalid_methods.append(method)
        
        if invalid_methods:
            return False, f"Potentially dangerous methods: {invalid_methods}"
        return True, "Valid API usage"

    def validate_volume_conservation(self, code: str) -> Tuple[bool, str]:
        # check if the volumes used in the protocol make sense
        # catch obviously wrong volumes that would cause problems
        volumes = re.findall(r'(\d+\.?\d*)', code)
        
        if volumes:
            volumes_float = [float(v) for v in volumes]
            max_volume = max(volumes_float)
            
            # check for unrealistic volumes
            if max_volume > 10000:  # more than 10mL is unusual for most protocols
                return False, f"Suspiciously large volume: {max_volume}"
            if any(v < 0 for v in volumes_float):
                return False, "Negative volume detected"
        
        return True, "Volume conservation passed"

    def validate_biological_feasibility(self, code: str, biological_context: BiologicalContext) -> Tuple[bool, str]:
        # check if the protocol makes sense from a biological perspective
        # catch protocols that would never work in a real lab
        issues = []
        
        # check PCR temperature requirements
        if biological_context.biological_process == 'pcr':
            temps = [float(t.replace('C', '')) for t in biological_context.temperature_conditions if t.endswith('C')]
            if temps:
                # PCR needs high temp to denature DNA
                if not any(90 <= t <= 100 for t in temps):
                    issues.append("Missing denaturation temperature (90-100C)")
                # PCR needs medium temp for primers to bind
                if not any(50 <= t <= 70 for t in temps):
                    issues.append("Missing annealing temperature (50-70C)")
        
        # check if reagent volumes make sense
        if 'primer' in biological_context.reagents:
            if not any('0.5' in v or '1' in v or '2' in v for v in biological_context.typical_volumes.values()):
                issues.append("Unusual primer volume (typically 0.5-2 uL)")
        
        if issues:
            return False, f"Biological feasibility issues: {'; '.join(issues)}"
        return True, "Biologically feasible"

    def comprehensive_validate(self, code: str, biological_context: BiologicalContext) -> Dict[str, Tuple[bool, str]]:
        # run all validation checks and return results
        return {
            'syntax': self.validate_syntax(code),
            'api': self.validate_opentrons_api(code),
            'volume': self.validate_volume_conservation(code),
            'biology': self.validate_biological_feasibility(code, biological_context)
        }

class DatasetGenerator:
    # main class for creating training data from protocol chunks
    # combines biological context extraction with validation
    def __init__(self, chunks_path: str, protocols_base_dir: str):
        # store paths and initialize helper classes
        self.chunks_path = chunks_path
        self.protocols_base_dir = Path(protocols_base_dir)
        self.context_extractor = BiologicalContextExtractor()
        self.dependency_tracker = DependencyTracker()
        self.validator = ProtocolValidator()
        
        # load the chunked protocols and set up storage for results
        self.chunks = self.load_chunks()
        self.training_examples = []
        self.generation_stats = defaultdict(int)

    def load_chunks(self) -> List[Dict]:
        with open(self.chunks_path, 'r') as f:
            return json.load(f)

    def is_valid_training_chunk(self, chunk: Dict) -> bool:
        # filter out code chunks that would make bad training examples
        # we want chunks that represent complete, meaningful protocol operations
        code = chunk['content']
        
        # skip helper functions that need class context to work
        if 'def ' in code and 'self.' in code and 'class ' not in code:
            return False  # this is a method but there's no class definition
        
        # skip chunks that are too short unless they do something important
        if code.strip().count('\n') < 3 and not any(op in code for op in [
            'transfer', 'aspirate', 'dispense', 'load_labware', 'load_instrument'
        ]):
            return False  # too short and doesn't contain meaningful protocol steps
        
        # skip chunks that just extract parameters
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        if len(lines) <= 2 and all('=' in line and 'get_values' in line for line in lines):
            return False  # this is just parameter setup, not useful protocol code
        
        return True

    def enhance_chunk_with_context(self, chunk: Dict) -> Optional[TrainingExample]:
        # convert a raw code chunk into a complete training example
        # adds biological context, dependencies, and validation
        
        # skip chunks that wouldn't make good training examples
        if not self.is_valid_training_chunk(chunk):
            return None
            
        protocol_name = chunk['protocol_name']
        protocol_folder = self.protocols_base_dir / protocol_name
        
        # get biological information from the protocol's README file
        readme_files = list(protocol_folder.glob('README.*'))
        readme_data = {}
        if readme_files:
            readme_data = self.context_extractor.extract_from_readme(readme_files[0])
        
        # extract biological context from the README and code
        biological_context = self.context_extractor.extract_biological_context(
            readme_data, chunk['content'], chunk
        )
        
        # figure out what variables this code needs to run
        dependencies = self.dependency_tracker.extract_variable_dependencies(chunk['content'])
        
        # add missing imports and variable definitions to make code executable
        executable_code = self.dependency_tracker.make_chunk_executable(
            chunk['content'], dependencies
        )
        
        # check if the code is valid and safe
        validation_results = self.validator.comprehensive_validate(
            executable_code, biological_context
        )
        validation_status = {check: result[0] for check, result in validation_results.items()}
        
        # create a natural language prompt that matches the biological context
        prompt = self.generate_contextual_prompt(chunk, biological_context)
        
        return TrainingExample(
            prompt=prompt,
            code=executable_code,
            biological_context=biological_context,
            dependencies=dependencies,
            validation_status=validation_status,
            difficulty=self.classify_difficulty(chunk, biological_context),
            style='contextual',
            operations=chunk.get('operations', []),
            metadata={
                'original_chunk_id': chunk.get('chunk_id'),
                'protocol_name': protocol_name,
                'validation_details': {check: result[1] for check, result in validation_results.items()}
            }
        )

    def generate_contextual_prompt(self, chunk: Dict, bio_context: BiologicalContext) -> str:
        # create natural language prompts that match the biological context
        # simpler code gets simpler prompts, complex code gets detailed requests
        
        process = bio_context.biological_process
        goal = bio_context.scientific_goal
        reagents = ', '.join(bio_context.reagents[:3]) if bio_context.reagents else 'samples'
        
        # match prompt complexity to code complexity
        complexity = chunk.get('complexity', 0)
        
        if complexity > 15:  # complex protocols need detailed requests
            prompts = [
                f"Design a comprehensive {process} protocol to {goal} using {reagents}, including proper controls and optimization.",
                f"Create an automated {process} workflow for {goal} with {reagents}, ensuring reproducibility and efficiency.",
                f"Implement a robust {process} protocol for {goal} that handles {reagents} with appropriate quality controls."
            ]
        elif complexity > 5:  # intermediate protocols get standard requests
            prompts = [
                f"Set up a {process} protocol to {goal} using {reagents}.",
                f"Create an automated procedure for {process} with {reagents} to achieve {goal}.",
                f"Design a {process} workflow for {goal} using {reagents}."
            ]
        else:  # simple protocols get casual requests
            prompts = [
                f"Help me {goal.replace('create', 'make').replace('amplify', 'amplify')} using {reagents}.",
                f"I need to {goal} with {reagents}, can you write the protocol?",
                f"Show me how to {goal} using {reagents} on the OT-2."
            ]
        
        return random.choice(prompts)

    def classify_difficulty(self, chunk: Dict, bio_context: BiologicalContext) -> str:
        # figure out if this protocol is beginner, intermediate, or advanced
        # considers both code complexity and biological complexity
        
        base_complexity = chunk.get('complexity', 0)
        operations = chunk.get('operations', [])
        
        # add points for biological complexity factors
        bio_complexity_factors = 0
        if len(bio_context.reagents) > 5:
            bio_complexity_factors += 2  # lots of reagents makes it harder
        if len(bio_context.temperature_conditions) > 2:
            bio_complexity_factors += 2  # temperature control adds difficulty
        if bio_context.biological_process in ['pcr', 'magnetic_bead']:
            bio_complexity_factors += 1  # these processes are inherently complex
            
        total_complexity = base_complexity + bio_complexity_factors
        
        # classify based on total complexity and presence of loops
        if total_complexity > 20 or 'for_loop' in operations:
            return 'advanced'
        elif total_complexity > 8 or len(operations) > 5:
            return 'intermediate'
        else:
            return 'beginner'

    def generate_dataset(self) -> List[TrainingExample]:
        # process all chunks and create training examples with biological context
        print(f"Processing {len(self.chunks)} chunks to create training examples...")
        
        for i, chunk in enumerate(self.chunks):
            # print progress updates every 50 chunks
            if i % 50 == 0:
                print(f"Progress: {i}/{len(self.chunks)}")
            
            try:
                # try to convert this chunk into a training example
                training_example = self.enhance_chunk_with_context(chunk)
                if training_example is not None:  # only keep good examples
                    self.training_examples.append(training_example)
                    
                    # keep track of what we're generating
                    self.generation_stats['total_generated'] += 1
                    self.generation_stats[f'difficulty_{training_example.difficulty}'] += 1
                    self.generation_stats[f'process_{training_example.biological_context.biological_process}'] += 1
                    
                    # track validation results to see how we're doing
                    for check, passed in training_example.validation_status.items():
                        self.generation_stats[f'validation_{check}_{"pass" if passed else "fail"}'] += 1
                else:
                    self.generation_stats['filtered_out'] += 1
                    
            except Exception as e:
                # basic error catching, continue processing other chunks
                print(f"Error processing chunk {chunk.get('chunk_id', 'unknown')}: {e}")
                self.generation_stats['errors'] += 1
        
        return self.training_examples

    def save_dataset(self, output_dir: str = "training_data"):
        # save all the training examples in different formats
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # prepare data for export in multiple formats
        training_data = []
        validation_report = []
        
        for example in self.training_examples:
            # main training data
            training_record = {
                'prompt': example.prompt,
                'code': example.code,
                'biological_context': {
                    'protocol_type': example.biological_context.protocol_type,
                    'scientific_goal': example.biological_context.scientific_goal,
                    'biological_process': example.biological_context.biological_process,
                    'reagents': example.biological_context.reagents,
                    'expected_outcome': example.biological_context.expected_outcome
                },
                'difficulty': example.difficulty,
                'operations': example.operations,
                'validation_passed': all(example.validation_status.values())
            }
            training_data.append(training_record)
            
            # validation report
            validation_record = {
                'protocol_name': example.metadata['protocol_name'],
                'validation_status': example.validation_status,
                'validation_details': example.metadata['validation_details'],
                'dependencies': example.dependencies
            }
            validation_report.append(validation_record)
        
        # save training data in multiple formats
        with open(output_path / "training_data.json", 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # JSONL for Hugging Face
        with open(output_path / "training_data.jsonl", 'w') as f:
            for record in training_data:
                f.write(json.dumps({
                    'input': record['prompt'],
                    'output': record['code']
                }) + '\n')
        
        # validation report
        with open(output_path / "validation_report.json", 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        # generation statistics
        with open(output_path / "generation_stats.json", 'w') as f:
            json.dump(dict(self.generation_stats), f, indent=2)
        
        # create filtered datasets
        valid_examples = [ex for ex in training_data if ex['validation_passed']]
        with open(output_path / "validated_training_data.jsonl", 'w') as f:
            for record in valid_examples:
                f.write(json.dumps({
                    'input': record['prompt'],
                    'output': record['code']
                }) + '\n')
        
        print(f"\nDataset generation complete!")
        print(f"Total examples generated: {len(training_data)}")
        print(f"Validation passed: {len(valid_examples)} ({len(valid_examples)/len(training_data)*100:.1f}%)")
        print(f"Results saved to {output_path}")
        print(f"\nGeneration statistics: {dict(self.generation_stats)}")

def main():
    # initialize dataset generator
    generator = DatasetGenerator(
        "chunked_data/protocol_chunks.json",
        "/Users/bellap/Desktop/Protocols-develop/protoBuilds"
    )
    
    # generate dataset
    examples = generator.generate_dataset()
    
    # save results
    generator.save_dataset()

if __name__ == "__main__":
    main()