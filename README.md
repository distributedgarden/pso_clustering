# PSO for Clustering
Implements the particle swarm optimization algorithm for clustering proposed in "A particle swarm optimization approach to clustering" by Tunchan Cura [1].

### Datasets:
- iris
- wine
- thyroid
- cmc
- glass

# Usage
Images will save to the `data/` directory.

```
# navigate to the repository root directory
> python3 -m venv env
> source env/bin/activate
> python3 -m pip install -r requirements.txt

# run all datasets
> python3 scripts/simulation.py --all

# run custom set of datasets
> python3 scripts/simulation.py --datasets iris wine
```


# References
[1]: Cura, T. (2012). A particle swarm optimization approach to clustering. Expert Systems With Applications, 39(1), 1582–1588. https://doi.org/10.1016/j.eswa.2011.07.123