import numpy as np

def eva1(As:list,Cs:dict,GG:np.ndarray,nodes:list):
    layer = len(As)
    count = 0 
    mindensity = np.Inf
    geodensity = 1
    avgdensity = 0
    within = 0
    cross = 0

    print("Size of each layer: ", [len(node) for node in nodes])
    
    for i in range(layer):
        density = 0
        if len(nodes[i])>=2:
            density = As[i][nodes[i],:][:,nodes[i]].sum() / (len(nodes[i])*(len(nodes[i])-1))
        mindensity = min(mindensity,density)
        geodensity *= density
        avgdensity += density
        within += density
        count += 1

        print(f"Within-layer {i+1}, the edge density is {density}")

        for j in range(layer):
            if j>i and GG[i,j]!=0:
                density = 0
                if len(nodes[i])>=2 and len(nodes[j])>=2:
                    density = Cs[(i,j)][nodes[i],:][:,nodes[j]].sum() / (len(nodes[i])*len(nodes[j]))
                mindensity = min(mindensity,density)
                geodensity *= density
                avgdensity += density
                cross += density
                count += 1
                print(f"Cross-layer {i+1,j+1}, the edge density is {density}")

    geodensity = np.power(geodensity,1/count)
    avgdensity /= count
    crossdensity = within*cross
    

    print(f"Minimum edge density {mindensity}")
    print(f"Geometric mean edge density {geodensity}")
    print(f"Average mean edge density {avgdensity}")
    print(f"Cross edge density {crossdensity}")


    return mindensity,geodensity,avgdensity,crossdensity


def eva2(As: list, Cs: dict, GG: np.ndarray, nodes: list):
    layer = len(As)
    count = 0
    mindensity = np.Inf
    geodensity = 1
    avgdensity = 0
    within = 0
    cross = 0

    for i in range(layer):
        density = 0
        if len(nodes[i]) >= 2:
            density = As[i][nodes[i], :][:, nodes[i]].nnz / (len(nodes[i])*2)
        mindensity = min(mindensity, density)
        geodensity *= density
        avgdensity += density
        within += density
        count += 1

        print(f"Within-layer {i+1}, the average density is {density}")

        for j in range(layer):
            if j > i and GG[i, j] != 0:
                density = 0
                if len(nodes[i]) >= 2 and len(nodes[j]) >= 2:
                    density = Cs[(i, j)][nodes[i], :][:, nodes[j]
                                                      ].nnz / np.sqrt((len(nodes[i])*len(nodes[j])))
                mindensity = min(mindensity, density)
                geodensity *= density
                avgdensity += density
                cross += density
                count += 1
                print(f"Cross-layer {i+1,j+1}, the average density is {density}")
                
    geodensity = np.power(geodensity,1/count)
    avgdensity /= count
    crossdensity = within*cross

    print(f"Minimum average density {mindensity}")
    print(f"Geometric mean average density {geodensity}")
    print(f"Average mean average density {avgdensity}")
    print(f"Cross average density {crossdensity}")

    return mindensity, geodensity,avgdensity,crossdensity
