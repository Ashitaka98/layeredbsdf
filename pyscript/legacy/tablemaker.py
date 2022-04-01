import os

materials = []

def getMaterialNames(path):
    filesList = os.listdir(path)
    for fileName in filesList:
        fileAbsPath = os.path.join(path, fileName)
        if os.path.isdir(fileAbsPath):
            pass
        else:
            tmp = os.path.splitext(fileName)
            rawName = tmp[0]
            extName = tmp[1]
            print(rawName)
            if extName == ".spd":
                tmp = os.path.splitext(rawName)
                ior = tmp[0]
                par = tmp[1]
                if par == '.eta':
                    materials.append(ior)


def readFiles(path, outfile):
    for ior in materials:
        eta_list = []
        k_list = []
        with open(os.path.join(path, ior + '.eta.spd'), 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if '0' <= line[0] and line[0] <= '9':
                    eta_list.append(line.split()[1])
        
        with open(os.path.join(path, ior + '.k.spd'), 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if '0' <= line[0] and line[0] <= '9':
                    k_list.append(line.split()[1])
        
        if len(eta_list) != len(k_list):
            print(f"ERROR! List lengths didn't match at {ior}!")

        for i in range(len(eta_list)):
            outfile.write(eta_list[i] + ' ' + k_list[i] + '\n')


if __name__ == '__main__':
    getMaterialNames('../data/ior/')
    print(len(materials))

    outfile = open('eta-k_table.txt', 'w')
    readFiles('../data/ior/', outfile)
    outfile.close()
