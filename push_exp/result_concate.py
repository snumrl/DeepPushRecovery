import glob

result_csv_path = glob.glob("results/main_CrouchSimulation/**/_result.csv")

a = None

with open('result.csv', 'w') as fout:
    for filename in result_csv_path:
        with open(filename, 'r') as f:
            if a is None:
                a = f.readline()
                fout.write(a)
            else:
                f.readline()
            fout.write(f.read())

