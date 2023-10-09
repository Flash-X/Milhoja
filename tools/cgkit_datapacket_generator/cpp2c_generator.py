import json_sections as sects
import argparse
import json
import os
import cpp2c_cgkit
from packet_generation_utility import Language
from collections import defaultdict
from DataPacketMemberVars import DataPacketMemberVars

_ARG_LIST_KEY = "c2f_argument_list"
_INST_ARGS_KEY = "instance_args"
_HOST_MEMBERS_KEY = "get_host_members"


def _generate_cpp2c_outer(data: dict):
    """
    Generates the outer template for the cpp2c layer with the name
    `cg-tpl.cpp2c_outer.cpp`.

    :param dict data: The dictionary containing the data packet JSON data.
    """
    file_name = 'cg-tpl.cpp2c_outer.cpp'
    if os.path.isfile(file_name):
        print(f"Warning: {file_name} already exists. Overwriting.")

    with open(file_name, 'w') as outer:
        outer.writelines([
            '/* _connector:cpp2c_outer */\n',
            f'/* _param:class_name = {data["name"]} */\n\n',
            f'/* _param:taskfunctionname = {data[sects.TASK_FUNCTION_NAME]} */\n',
            f'/* _param:taskfunctionnametf = {data[sects.TASK_FUNCTION_NAME]}_tf */\n',
            f'/* _param:taskfunctionnamec2f = {data[sects.TASK_FUNCTION_NAME]}_c2f */\n',
            f'/* _param:instantiate = instantiate_{data["name"]}_C */\n',
            f'/* _param:deletion = delete_{data["name"]}_C */\n',
            f'/* _param:release = release_{data["name"]}_extra_queue_C */\n\n',
            '/* _link:cpp2c */'
        ])


def _insert_connector_arguments(data: dict, connectors: dict, dpinfo_order: list):
    """
    Inserts various connector arguments into the connectors dictionary.
    
    :param dict data: The dictionary containing the data packet JSON data.
    :param dict connectors: The connectors dictionary to write to containing all cgkit connectors.
    :param list dpinfo_order: A list of DataPacketMemberVars objects that use all of the items in the task_function_argument_list.
    """

    nTiles_language = 'int' if data[sects.LANG] == Language.fortran else 'std::size_t'

    # initialize the boilerplate values for host members.
    # one queue always exists in the data packet
    # need to insert nTiles host manually
    connectors[_HOST_MEMBERS_KEY] = [
        'const int queue1_h = packet_h->asynchronousQueue();\n',
        f'const {nTiles_language} _nTiles_h = packet_h->_nTiles_h;\n'
    ]
    # insert the number of extra streams as a connector
    n_streams = data.get(sects.EXTRA_STREAMS, 0)
    connectors[_HOST_MEMBERS_KEY].extend([
        f'const int queue{i}_h = packet_h->extraAsynchronousQueue({i});\n' + 
        f'if (queue{i}_h < 0)\n' + 
        f'\tthrow std::overflow_error("[cgkit.cpp2c.cxx] Potential overflow error when accessing async queue id.");\n'
        for i in range(2, n_streams+2)
    ])

    # extend the argument list connector using dpinfo_order
    connectors[_ARG_LIST_KEY].extend(
        [('packet_h', 'void*')] +
        [('queue1_h', 'const int')] +
        [(f'queue{i}_h', 'const int') for i in range(2, n_streams+2)] +
        [('_nTiles_h', f'const {nTiles_language}')] +
        [(item.device, 'const void*') for item in dpinfo_order]
    )


def _generate_cpp2c_helper(data: dict):
    """
    Generates the helper template for the cpp2c layer.
    
    :param data: The dict containing the data packet JSON data.
    """
    connectors = defaultdict(list)
    # generate DataPacketMemberVars instance for each item in TFAL.
    dpinfo_order = ([DataPacketMemberVars(item, '', '', False) for item in data[sects.ORDER]])
    # insert connectors into dictionary
    _insert_connector_arguments(data, connectors, dpinfo_order)
    # instance args only found in general section
    connectors[_INST_ARGS_KEY] = [
        (key, f'{dtype}') for key, dtype in data.get(sects.GENERAL, {}).items() if key != "nTiles"
    ] + [('packet', 'void**')]

    # print warning when overwriting
    file_name = 'cg-tpl.cpp2c_helper.cpp'
    if os.path.isfile(file_name):
        print(f'Warning: {file_name} already exists. Overwriting.')
        
    # insert all connectors into helper template file
    with open('cg-tpl.cpp2c_helper.cpp', 'w') as helper:
        helper.writelines(
            ['/* _connector:get_host_members */\n'] +
            connectors[_HOST_MEMBERS_KEY]
        )
        helper.writelines(
            ['\n/* _connector:c2f_argument_list */\n'] +
            [',\n'.join([f"{item[1]} {item[0]}" for item in connectors[_ARG_LIST_KEY]])] +

            ['\n\n/* _connector:c2f_arguments */\n'] +
            [',\n'.join([f"{item[0]}" for item in connectors[_ARG_LIST_KEY]])] +

            ['\n\n/* _connector:get_device_members */\n'] +
            [''.join([f'void* {item.device} = static_cast<void*>( packet_h->{item.device} );\n' 
                      for item in dpinfo_order])] +

            ['\n/* _connector:instance_args */\n'] +
            [','.join([f'{item[1]} {item[0]}' for item in connectors[_INST_ARGS_KEY]])] +

            ['\n\n/* _connector:host_members */\n'] +
            [','.join([item for item in data.get(sects.GENERAL, {}) if item != 'nTiles'])]
        )


def generate_cpp2c(data: dict):
    """
    Driver for cpp2c generator.
    
    :param dict data: The dict containing the data packet JSON.
    """
    _generate_cpp2c_outer(data)
    _generate_cpp2c_helper(data)
    cpp2c_cgkit.generate_datapacket_cpp2c_layer(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generates the C to Fortran interoperability layer.")
    parser.add_argument("JSON", help="The JSON file used to generated the data packet.")
    args = parser.parse_args()

    with open(args.JSON, 'r') as fp:
        data = json.load(fp)
        data['name'] = os.path.basename(args.JSON).replace('.json', '')
        generate_cpp2c(data)