import re
from multiprocessing import Process, Manager
from subprocess import check_output
from collections import defaultdict

def yosys_stats(design_file, yosys_binary, stats):
    yosys_command = "read_verilog " + design_file + "; stat"
    try:
        proc = check_output([yosys_binary, '-QT', '-p', yosys_command])
        lines = proc.decode("utf-8").split('\n')
        for line in lines:
            if 'Number of wires' in line:
                stats['number_of_wires'] = int(line.strip().split()[-1])
            if 'Number of public wires' in line:
                stats['number_of_public_wires'] = int(line.strip().split()[-1])
            if 'Number of cells' in line:
                stats['number_of_cells'] = int(line.strip().split()[-1])
            if '$and' in line:
                stats['ands'] = int(line.strip().split()[-1])
            if '$or' in line:
                stats['ors'] = int(line.strip().split()[-1])
            if '$not' in line:
                stats['nots'] = int(line.strip().split()[-1])
    except Exception as e:
        print(e)
        return None
    return stats

def abc_stats(design_file, abc_binary, stats):    
    abc_command = "read_verilog " + design_file + "; print_stats"
    try:
        proc = check_output([abc_binary, '-c', abc_command])
        lines = proc.decode("utf-8").split('\n')
        for line in lines:
            if 'i/o' in line:
                ob = re.search(r'i/o *= *[0-9]+ */ *[0-9]+', line)
                stats['input_pins'] = int(ob.group().split('=')[1].strip().split('/')[0].strip())
                stats['output_pins'] = int(ob.group().split('=')[1].strip().split('/')[1].strip())
        
                ob = re.search(r'edge *= *[0-9]+', line)
                stats['edges'] = int(ob.group().split('=')[1].strip())

                ob = re.search(r'lev *= *[0-9]+', line)
                stats['levels'] = int(ob.group().split('=')[1].strip())

                ob = re.search(r'lat *= *[0-9]+', line)
                stats['latches'] = int(ob.group().split('=')[1].strip())
    except Exception as e:
        print(e)
        return None
    
    return stats

def extract_features(design_file, yosys_binary='yosys', abc_binary='abc'):
    '''
    Returns features of a given circuit as a tuple.
    Features are listed below
    '''
    manager = Manager()
    features = manager.dict()
    p1 = Process(target=yosys_stats, args=(design_file, yosys_binary, features))
    p2 = Process(target=abc_stats, args=(design_file, abc_binary, features))
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    return dict(features)