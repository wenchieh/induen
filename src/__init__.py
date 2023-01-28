from src import nncf
from src import fraudar
from src import corduen
from src import baselines
from src import corduen
from src import greedy
from src import boosting

CONMF   = nncf.CONMF
CNMF    = nncf.CNMF
FRAUDAR = fraudar.Fraudar
AGGR    = corduen.aggregation
CORDUEN = corduen.Corduen
BOOST   = boosting.fastNeighborBoosting
greedyCharikar = baselines.greedyCharikar
greedyoqc      = baselines.greedyOqc
greedyBipartite = fraudar.greedyBipartite
# CorduenWithoutMF = corduen.CorduenWithoutMF



