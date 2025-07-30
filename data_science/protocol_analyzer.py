#!/user/bin/env python3
# protocol structure analyzer for intelligent chunking
#   - identify patterns/structures to evaluate dataset and develop chunking strategies
#   - heavily annotated for bella's learning + future implementation purposes 

import json                                     # read and write json files  
import re                                       # regex for parsing code + pattern matching
from pathlib import Path                        # file pathing
from collections import defaultdict, Counter    # auto-dicting, better counting
import pandas as pd                             # csv import and export; data science library
from typing import Dict, List, Tuple, Any       # readability and type hinting

class ProtocolAnalyzer:
    # initializing project directory
    def _init_(self, base_dir: str):        # base_dir = directory
        self.analysis_results = {           # sets up new storage directory for analysis outputs
            'protocol_structures': [],
            'function_patterns': defaultdict(int),
            'section_lengths': defaultdict(list),
            'common_patterns': defaultdict(int),
            'protocol_types': defaultdict(list),
            'complexity_metrics': []
        }

    # extracting python code from .py.json files
    # identifies code from keywords and converts it to a string for future processing
    def extract_protocol_code(self, folder_path: Path) -> str: 
        # check for files that have .py.json, then store in list
        py_json_files = list(folder_path.glob("*.py.json"))    
        if not py_json_files:
            # if nothing comes back with that ending, return nothing
            return None                                        
        
        try:
            with open(py_json_files[0], 'r', encoding='utf-8') as f:
                # parses file contents into a python dictionary; this makes our work way easier
                data = json.load(f)                            
                # json files are structures as key-value pairs
                if 'content' in data:   
                    # if the key 'content' exists, we can assume it contains the python code                    
                    return data['content']
                elif 'code' in data:      
                    # used by opentrons API/some manual templates                     
                    return data['code']
                elif 'python_code' in data:
                    # common in many intermediate formats                    
                    return data['python_code']
                else:                                          
                    # try to find anything that could be python code
                    for key, value in data.items():            
                        # iterate over any key:value pairs in our new dictionary
                        if isinstance(value, str) and 'def run' in value:  
                            # if a value is a string and contains "def run", we can assume it's python code due to syntax
                            return value                       
        # basic error catching                
        except Exception as e:                                 
            print(f"Error reading {py_json_files[0]}: {e}")
            return None

    # parsing + feature extraction from raw code  
    # analyzes the structure of the protocol code to identify sections, functions, and key operations
    # returns a dictionary with the protocol structure and key features
    # lots of regex patterns used to identify specific structures; i used chatgpt to help with this
    # this is the main function that does the heavy lifting of analyzing the protocol code      
    def analyze_protocol_structure(self, code: str, protocol_name: str) -> Dict:
        # split the code into lines for easier processing
        lines = code.split('\n')
        
        # builds output dictionary to hold information about the protocol structure
        structure = {
            'name': protocol_name,
            'total_lines': len(lines),      # script length
            'sections': {},                 # line ranges for each section
            'functions': [],                # names of all functions
            'imports': [],                  # all imports
            'protocol_type': 'unknown',     # type of protocol (ie PCR, serial dilution)
            'complexity': 0,                # score of complexity based on structure
            'has_custom_labware': False,    # true if labware not from opentrons
            'pipette_types': [],            # list of pipette types used
            'labware_types': [],            # list of labware types used
            'key_operations': []            # all matched operations 
        }
        
        # set up for logical section tracking
        # always starts with the header section
        # entry/exit to store line ranges
        current_section = 'header'
        section_start = 0
        in_function = False
        function_name = None
        function_start = 0
        
        # pattern matching regex for well-known structures
        import_pattern = re.compile(r'^(from|import)\s+')
        function_pattern = re.compile(r'^def\s+(\w+)\s*\(')
        run_function_pattern = re.compile(r'^def\s+run\s*\(')
        comment_pattern = re.compile(r'^\s*#')
        labware_pattern = re.compile(r'load_labware\s*\(\s*[\'"]([^\'"]+)[\'"]')
        pipette_pattern = re.compile(r'load_instrument\s*\(\s*[\'"]([^\'"]+)[\'"]')
        
        # regex to identify common, well-known operations in protocols
        operation_patterns = {
            'transfer': re.compile(r'\.(transfer|distribute|consolidate)\s*\('),
            'pick_up_tip': re.compile(r'\.pick_up_tip\s*\('),
            'aspirate': re.compile(r'\.aspirate\s*\('),
            'dispense': re.compile(r'\.dispense\s*\('),
            'mix': re.compile(r'\.mix\s*\('),
            'blow_out': re.compile(r'\.blow_out\s*\('),
            'touch_tip': re.compile(r'\.touch_tip\s*\('),
            'air_gap': re.compile(r'\.air_gap\s*\('),
            'move_to': re.compile(r'\.move_to\s*\('),
            'delay': re.compile(r'\.delay\s*\('),
            'thermocycler': re.compile(r'thermocycler|set_block_temperature'),
            'temperature': re.compile(r'temperature_module|temp_mod'),
            'magnetic': re.compile(r'magnetic_module|mag_mod'),
            'drop_tip': re.compile(r'\.drop_tip\s*\('),
            'pause': re.compile(r'\.pause\s*\('),
            'comment': re.compile(r'\.comment\s*\('),
            'home': re.compile(r'\.home\s*\('),
            'set_flow_rate': re.compile(r'\.flow_rate\.'),
            'load_module': re.compile(r'load_module\s*\('),
            'engage': re.compile(r'\.engage\s*\('),
            'disengage': re.compile(r'\.disengage\s*\('),
            'open_lid': re.compile(r'\.open_lid\s*\('),
            'close_lid': re.compile(r'\.close_lid\s*\('),
            'set_lid_temperature': re.compile(r'\.set_lid_temperature\s*\('),
            'deactivate': re.compile(r'\.deactivate\s*\('),
            'try_except': re.compile(r'^\s*try\s*:|^\s*except\s+'),
            'with_statement': re.compile(r'^\s*with\s+'),
            'for_loop': re.compile(r'^\s*for\s+'),
            'while_loop': re.compile(r'^\s*while\s+'),
            'if_statement': re.compile(r'^\s*if\s+')
        }

        # groups of identifiable operations for easier analysis; not used in this version
        # but could be useful for future analysis or categorization
        operation_groups = {
            'pipetting': [
                'transfer', 'aspirate', 'dispense', 'mix', 'touch_tip', 'blow_out', 'air_gap'
            ],
            'tip_handling': [
                'pick_up_tip', 'drop_tip'
            ],
            'labware_control': [
                'move_to', 'home', 'pause', 'comment'
            ],
            'flow_control': [
                'set_flow_rate'
            ],
            'delay': ['delay'],
            'module_loading': ['load_module'],
            'temperature_module': [
                'temperature', 'deactivate', 'set_lid_temperature'
            ],
            'magnetic_module': [
                'magnetic', 'engage', 'disengage'
            ],
            'thermocycler': [
                'thermocycler', 'set_lid_temperature'
            ],
            'control_flow': [
                'if_statement', 'for_loop', 'while_loop', 'try_except', 'with_statement'
            ]
        }
        
        for i, line in enumerate(lines):
            stripped = line.strip()

            # if line is an import statement, record it
            if import_pattern.match(stripped):
                structure['imports'].append(stripped)
                continue  # imports don't need more analysis

            # if this line is a function definition
            func_match = function_pattern.match(stripped)
            if func_match:
                # if we were already in a function, close that section
                if in_function:
                    structure['sections'][function_name] = {
                        'start': function_start,
                        'end': i - 1,
                        'lines': i - function_start
                    }

                # update function tracking
                function_name = func_match.group(1)
                function_start = i
                in_function = True
                structure['functions'].append(function_name)

                # if it's the run() function, mark this as the main operational section
                if run_function_pattern.match(stripped):
                    current_section = 'run_function'
                    section_start = i

            # if we’re inside the run() function, track pipettes/labware/operations
            if current_section == 'run_function':
                labware_match = labware_pattern.search(line)
                if labware_match:
                    structure['labware_types'].append(labware_match.group(1))

                pipette_match = pipette_pattern.search(line)
                if pipette_match:
                    structure['pipette_types'].append(pipette_match.group(1))

                # for each operation pattern, see if this line matches
                matched_ops = set()
                for op_name, op_pattern in operation_patterns.items():
                    if op_pattern.search(line):
                        matched_ops.add(op_name)

                # update key operations and global counters
                for op_name in matched_ops:
                    structure['key_operations'].append(op_name)
                    self.analysis_results['common_patterns'][op_name] += 1

            # if this is a blank line following a non-blank one, treat it as a section break
            if i > 0 and not stripped and lines[i - 1].strip():
                if current_section and current_section not in structure['sections']:
                    structure['sections'][current_section] = {
                        'start': section_start,
                        'end': i - 1,
                        'lines': i - section_start
                    }
                    self.analysis_results['section_lengths'][current_section].append(i - section_start)
                    section_start = i + 1  # start new section after the blank line

        # close out the final function section if the script ends inside one
        if in_function:
            structure['sections'][function_name] = {
                'start': function_start,
                'end': len(lines) - 1,
                'lines': len(lines) - function_start
            }

        # deduplicate labware/pipettes just in case they were referenced multiple times
        structure['labware_types'] = list(set(structure['labware_types']))
        structure['pipette_types'] = list(set(structure['pipette_types']))

        # classify what kind of protocol this is based on its operations
        structure['protocol_type'] = self.classify_protocol_type(structure)

        # calculate a rough complexity score based on structure
        structure['complexity'] = self.calculate_complexity(structure)

        # if any labware type is not from Opentrons, we mark it as custom
        if any('opentrons' not in lw.lower() for lw in structure['labware_types']):
            structure['has_custom_labware'] = True

        return structure

    # classify the protocol type based on key operations
    # this is a simple heuristic-based classification system
    # could be expanded with more complex ML models in the future    
    def classify_protocol_type(self, structure: Dict) -> str:
        ops = set(structure['key_operations'])
        
        if 'thermocycler' in ops:
            return 'pcr'
        elif 'magnetic' in ops:
            return 'magnetic_bead'
        elif 'temperature' in ops and 'mix' in ops:
            return 'incubation'
        elif ops.intersection({'distribute', 'consolidate'}):
            return 'plate_mapping'
        elif 'for_loop' in ops and 'transfer' in ops:
            if len(structure['labware_types']) > 2:
                return 'complex_transfer'
            return 'serial_dilution'
        elif 'mix' in ops:
            return 'mixing'
        else:
            return 'basic_transfer'
    
    # calculate a complexity score based on the protocol structure
    # currently calculates based on number of lines, functions, key operations, and labware types
    # could be expanded with more complex metrics in the future
    def calculate_complexity(self, structure: Dict) -> int:
        complexity = 0
        
        complexity += structure['total_lines'] // 100
        complexity += len(structure['functions']) * 2
        complexity += len(set(structure['key_operations']))
        complexity += len(set(structure['labware_types'])) * 2

        for op in ['for_loop', 'while_loop', 'if_statement']:
            complexity += structure['key_operations'].count(op) * 3
        
        return complexity
    
    # analyze all protocols in the directory
    def analyze_all_protocols(self):
        # create list of all subdirectories in the base directory to go through
        protocol_folders = [f for f in self.base_dir.iterdir() if f.is_dir()]
        
        print(f"Analyzing {len(protocol_folders)} protocols...")
        
        # loop through each protocol folder
        for i, folder in enumerate(protocol_folders):
            # print updates 
            if i % 50 == 0:
                print(f"Progress: {i}/{len(protocol_folders)}")
            
            # extract code as string
            code = self.extract_protocol_code(folder)
            if code:
                # folder.name gets passed as the name of the protocol; analysis run
                structure = self.analyze_protocol_structure(code, folder.name)

                # appending index to store protocol names that fall under each type
                self.analysis_results['protocol_structures'].append(structure)
                self.analysis_results['protocol_types'][structure['protocol_type']].append(folder.name)

                # store summary stats 
                self.analysis_results['complexity_metrics'].append({
                    'name': folder.name,
                    'complexity': structure['complexity'],
                    'lines': structure['total_lines'],
                    'type': structure['protocol_type']
                })

    # create simple analysis report 
    def generate_analysis_report(self):
        # building the structure of the report 
        report = {
            'summary': { 
                # count how many protocols were analyzed
                'total_protocols': len(self.analysis_results['protocol_structures']),
                # break down protocols by type
                'protocol_types': dict(Counter(s['protocol_type'] for s in self.analysis_results['protocol_structures'])),
                # find average length of protocols
                'avg_lines': sum(s['total_lines'] for s in self.analysis_results['protocol_structures']) / len(self.analysis_results['protocol_structures']),
                # initialize complexity variables
                'complexity_distribution': {}
            },
            # creating placeholders for various analysis results
            'chunking_recommendations': {},
            'common_patterns': dict(self.analysis_results['common_patterns']),
            'section_analysis': {},
            'protocol_type_details': {}
        }
        
        # analyze complexity score distribution
        complexities = [m['complexity'] for m in self.analysis_results['complexity_metrics']]
        report['summary']['complexity_distribution'] = {
            'min': min(complexities),
            'max': max(complexities),
            'mean': sum(complexities) / len(complexities),
            'quartiles': {
                '25%': sorted(complexities)[len(complexities)//4],
                '50%': sorted(complexities)[len(complexities)//2],
                '75%': sorted(complexities)[3*len(complexities)//4]
            }
        }
        
        # analyze section length distribution
        for section, lengths in self.analysis_results['section_lengths'].items():
            if lengths:
                report['section_analysis'][section] = {
                    'avg_lines': sum(lengths) / len(lengths),
                    'min_lines': min(lengths),
                    'max_lines': max(lengths),
                    'optimal_chunk_size': self.recommend_chunk_size(lengths)
                }
        
        # group protocols by type and analyze their structures
        for ptype, protocols in self.analysis_results['protocol_types'].items():
            # iterate through each type and find all that match that type
            type_structures = [s for s in self.analysis_results['protocol_structures'] if s['protocol_type'] == ptype]
            
            # count how many of each type and their average length
            report['protocol_type_details'][ptype] = {
                'count': len(protocols),
                'avg_lines': sum(s['total_lines'] for s in type_structures) / len(type_structures) if type_structures else 0,
                # find most common operations for this type
                'common_operations': Counter([op for s in type_structures for op in s['key_operations']]).most_common(5),
                # return example protocols of this type
                'example_protocols': protocols[:3]
            }
        
        # run chunking recommendations function
        report['chunking_recommendations'] = self.generate_chunking_recommendations()
        
        return report

    # recommend chunk size + specific recommendations based on analysis
    def chunking_recommendations(self):
        return None
    
    # save results in previous analysis_results directory
    def save_results(self, outputdir: str = "analysis_results"):
        return None

    # main function to run the analysis yippee
    if __name__ == "__main__":
        main()
