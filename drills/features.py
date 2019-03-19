from subprocess import check_output

def extract_features(design_file, yosys_binary='yosys'):
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

    yosys_command = "read_verilog " + design_file + "; stat"
    try:
        proc = check_output([yosys_binary, '-QT', '-p', yosys_command])
        lines = proc.decode("utf-8").split('\n')
        for line in lines:
            if 'Number of wires' in line:
                number_of_wires = int(line.strip().split()[-1])
            if 'Number of public wires' in line:
                number_of_public_wires = int(line.strip().split()[-1])
            if 'Number of public cells' in line:
                number_of_cells = int(line.strip().split()[-1])
            if '$and' in line:
                ands = int(line.strip().split()[-1])
            if '$or' in line:
                ors = int(line.strip().split()[-1])
            if '$not' in line:
                nots = int(line.strip().split()[-1])
        return (number_of_wires, number_of_public_wires, number_of_cells, ands, ors, nots)
    except Exception as e:
        print(e)
        return None