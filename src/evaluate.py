import numpy as np

def evaluation(As:list,Cs:dict,GG:np.ndarray,nodes:list):
    layer = len(As)
    count = 0 
    min_density = np.Inf
    geo_density = 1
    avg_density = 0
    within = 0
    cross = 0

    print("Size of each layer: ", [len(node) for node in nodes])
    print("Edge density of each network: ")
    
    for i in range(layer):
        edge_density = 0
        if len(nodes[i])>=2:
            edge_density = As[i][nodes[i],:][:,nodes[i]].sum() / (len(nodes[i])*(len(nodes[i])-1))
        min_density = min(min_density,edge_density)
        geo_density *= edge_density
        avg_density += edge_density
        within += edge_density
        count += 1

        print("Within-layer {}, the edge density is {:.4g}".format(i+1,edge_density))

        for j in range(layer):
            if j>i and GG[i,j]!=0:
                edge_density = 0
                if len(nodes[i])>=2 and len(nodes[j])>=2:
                    edge_density = Cs[(i,j)][nodes[i],:][:,nodes[j]].sum() / (len(nodes[i])*len(nodes[j]))
                min_density = min(min_density,edge_density)
                geo_density *= edge_density
                avg_density += edge_density
                cross += edge_density
                count += 1
                print("Cross-layer {}, the edge density is {:.4g}".format((i+1,j+1),edge_density))

    geo_density = np.power(geo_density,1/count)
    avg_density /= count
    cross_density = within*cross

    print("\n")
    print("Our defined metric to measure the quality of the mined result:")
    print("Minimum edge density is {:.4g}".format(min_density))
    print("Geometric mean edge density is {:.4g}".format(geo_density))
    print("Average mean edge density is {:.4g}".format(avg_density))
    print("Cross edge density is {:.4g}".format(cross_density))


    return min_density,geo_density,avg_density,cross_density
