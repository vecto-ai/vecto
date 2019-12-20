def get_pairs(fname):
    pairs = []
    with open(fname) as file_in:
        id_line = 0
        for line in file_in:
            if line.strip() == '':
                continue
            try:
                id_line += 1
                if "\t" in line:
                    parts = line.lower().split("\t")
                else:
                    parts = line.lower().split()
                left = parts[0]
                right = parts[1]
                right = right.strip()
                if "/" in right:
                    right = [i.strip() for i in right.split("/")]
                else:
                    right = [i.strip() for i in right.split(",")]
                pairs.append([left, right])
            except:
                print("error reading pairs")
                print("in file", fname)
                print("in line", id_line, line)
                exit(-1)
    return pairs
