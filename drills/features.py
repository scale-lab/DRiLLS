import re
from subprocess import check_output

def extract_features(design_file, yosys_binary='yosys'):
    yosys_command = "read_verilog " + design_file + "; stat"
    try:
        proc = check_output([yosys_binary, '-QT', '-p', yosys_command])
        lines = proc.decode("utf-8").split('\n')
        for line in lines:
            print(line.strip().split('\t'))
    except Exception as e:
        print(e)
        return None