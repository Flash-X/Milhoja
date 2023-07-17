import json_sections as sects
import packet_generation_utility as mutil
import argparse
import json
import os
import cpp2c_cgkit
from collections import defaultdict

def generate_cpp2c_outer(data: dict):
    with open('cg-tpl.cpp2c_outer.cpp', 'w') as outer:
        outer.writelines([
            '/* _connector:cpp2c_outer */\n',
            f'/* _param:class_name = {data["name"]} */\n\n',
            f'/* {"_param:release = 0 */" if data.get(sects.EXTRA_STREAMS, 0) == 0 else "/* _param:release = packet_h->releaseExtraQueue(id) */"} \n\n',
            '/* _link:cpp2c */'
        ])

def insert_host_arguments(data: dict, connectors: dict):
    connectors['get_host_members'] = ['const int queue1_h = packet_h->asynchronousQueue();\n',
                                    'const int _nTiles_h = packet_h->_nTiles_h;\n']
    n_streams = data.get(sects.EXTRA_STREAMS, 0)
    connectors['get_host_members'].extend([
        f'const int queue{i}_h = packet_h->extraAsynchronousQueue({i});\n'
        for i in range(2, n_streams+2)
    ])
    connectors['c2f_argument_list'].append( ('queue1_h', 'const int') )
    connectors['c2f_argument_list'].extend([
        (f'queue{i}_h', 'const int') for i in range(2, n_streams+2)
    ])
    connectors['c2f_argument_list'].append( ('_nTiles_h', 'const int'))
    connectors['c2f_argument_list'].append( ('_nTiles_d', 'const void*') )
    connectors['c2f_argument_list'].extend([
        (f'_{item}_d', 'const void*') for item in data[sects.ORDER]
    ])
    
def generate_cpp2c_helper(data: dict):
    connectors = defaultdict(list)
    connectors['c2f_argument_list'] = [ ('packet_h', 'void*') ]
    insert_host_arguments(data, connectors)
    # data.get(sects.GENERAL, {}).pop("nTiles", None)
    connectors['instance_args'] = [ (key, f'const {dtype}') for key,dtype in data.get(sects.GENERAL, {}).items() if key != "nTiles" ] 

    connectors['c2f_arguments'] = [ item[0] for item in connectors['c2f_argument_list'] ]
    with open('cg-tpl.cpp2c_helper.cpp', 'w') as helper:
        helper.writelines(['/* _connector:get_host_members */\n' ] + connectors['get_host_members'])
        helper.writelines(
            [ '\n/* _connector:c2f_argument_list */\n' ] +
            [ ',\n'.join([ f"{item[1]} {item[0]}" for item in connectors["c2f_argument_list"] ]) ] +
            
            [ '\n\n/* _connector:c2f_arguments */\n'] + 
            [ ',\n'.join([ f"{item[0]}" for item in connectors['c2f_argument_list']]) ] + 
            
            [ '\n\n/* _connector:get_device_members */\n' ] + 
            [ 'void* _nTiles_d = static_cast<void*>( packet_h->_nTiles_d );\n'] +
            [ ''.join([ f'void* _{item}_d = static_cast<void*>( packet_h->_{item}_d );\n' for item in data[sects.ORDER] ]) ] +
            
            ['\n/* _connector:instance_args */\n'] +
            [ ','.join( [ f'{item[1]} {item[0]}' for item in connectors['instance_args'] ] ) ] +

            ['\n\n/* _connector:get_host_members */\n'] + 
            [ ','.join([ item[0] for item in connectors['instance_args'] ])]
        )
        
def main(data: dict):
    generate_cpp2c_outer(data)
    generate_cpp2c_helper(data)
    cpp2c_cgkit.main(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generates the C to Fortran interoperability layer.")
    parser.add_argument("JSON", help="The JSON file used to generated the data packet.")
    args = parser.parse_args()

    with open(args.JSON, 'r') as fp:
        data = json.load(fp)
        data['name'] = os.path.basename(args.JSON).replace('.json', '')
        main(data)
