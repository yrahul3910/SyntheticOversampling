import sys
from pprint import pprint
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

    rxs = sk.get_results()
    rxs = [(rx.rank, rx.rx, float(rx.tiles().split('),')[1].split(',')[2])) for rx in rxs]
    rxs = [rx for rx in rxs if rx[1] not in ['Howso1', 'Howso2', 'SDV_FASTML', 'DAZZLE1', 'DAZZLE2', 'WFO2']]
    rxs = [(rx[0], 'datasynthesizer' if rx[1] == 'DS' else rx[1].lower(), f'{rx[2]}*' if rx[0] == max([rx[0] for rx in rxs]) else rx[2]) for rx in rxs]
    sorted_rxs = sorted(rxs, key=lambda p: p[1] if p[1] != 'no' else 'AAAAA')  # hack so that "No" is the first one
    print('Rxs below are:', [x[1] for x in sorted_rxs])
    sorted_rxs = [str(x[2]) for x in sorted_rxs]
    print('\n'.join(sorted_rxs))


##  type test_sk.txt| python2 sk_stats.py --text 30 --latex True
