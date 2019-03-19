import re
from subprocess import check_output

def extract_features(design_file, yosys_binary='yosys', abc_binary='abc'):
    '''
    Returns features of a given circuit as a tuple.
    Features are listed below
    '''
    number_of_wires = 0
    number_of_public_wires = 0
    number_of_cells = 0
    ands = 0
    ors = 0
    nots = 0

    input_pins = 0
    output_pins = 0
    edges = 0
    levels = 0
    latches = 0

    yosys_command = "read_verilog " + design_file + "; stat"
    try:
        proc = check_output([yosys_binary, '-QT', '-p', yosys_command])
        lines = proc.decode("utf-8").split('\n')
        for line in lines:
            if 'Number of wires' in line:
                number_of_wires = int(line.strip().split()[-1])
            if 'Number of public wires' in line:
                number_of_public_wires = int(line.strip().split()[-1])
            if 'Number of cells' in line:
                number_of_cells = int(line.strip().split()[-1])
            if '$and' in line:
                ands = int(line.strip().split()[-1])
            if '$or' in line:
                ors = int(line.strip().split()[-1])
            if '$not' in line:
                nots = int(line.strip().split()[-1])
        yosys_stats = (number_of_wires, number_of_public_wires, number_of_cells, ands, ors, nots)
    except Exception as e:
        print(e)
        return None, None
    
    abc_command = "read_verilog " + design_file + "; print_stats"
    try:
        proc = check_output([abc_binary, '-c', abc_command])
        lines = proc.decode("utf-8").split('\n')
        for line in lines:
            if 'i/o' in line:
                ob = re.search(r'i/o *= *[1-9]+ */ *[1-9]+', line)
                input_pins = int(ob.group().split('=')[1].strip().split('/')[0].strip())
                output_pins = int(ob.group().split('=')[1].strip().split('/')[1].strip())
        
                ob = re.search(r'edge *= *[1-9]+', line)
                edges = int(ob.group().split('=')[1].strip())

                ob = re.search(r'lev *= *[1-9]+', line)
                levels = int(ob.group().split('=')[1].strip())

                ob = re.search(r'lat *= *[1-9]+', line)
                latches = int(ob.group().split('=')[1].strip())
        abc_stats = (input_pins, output_pins, edges, levels, latches)
    except Exception as e:
        print(e)
        return None, None
    
    return yosys_stats, abc_stats