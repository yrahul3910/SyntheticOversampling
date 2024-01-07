import sys

from raise_utils.interpret import ScottKnott


if __name__ == '__main__':
    values = {}
    now = []
    is_value_line = False
    cur = ""
    for line in sys.stdin:
        if line.strip() == "":
            continue
        if not is_value_line:
            cur = line.strip()
            values[cur] = []
            is_value_line = True
        else:
            for val in line.split():
                values[cur].append(float(val))
            is_value_line = False
    
    sk = ScottKnott(values)
    sk.pprint()


##  type test_sk.txt| python2 sk_stats.py --text 30 --latex True
