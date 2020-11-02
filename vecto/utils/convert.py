import sys
def main():
    path = sys.argv[1]
    print(path)
    with open(path, encoding='utf-8', errors='ignore') as f_in:
        with open(path + ".out", "w", encoding='utf-8') as f_out:
            for l in f_in:
                label, text = l.rstrip().split(None, 1)
                f_out.write(f"{label}\t{text}\n")


if __name__ == '__main__':
    main()