def generate_edge_types():
    # capture all edge types
    edges = ['UF', 'UM', 'TF', 'TM']

    # edge_types = ['UF_TM','UF_TM', ...] 
    edge_types = []

    for i in edges:
        for j in edges:
            edge_types.append(f"{i}_{j}")

    edge_types.append('total_te')

    return edge_types

# list of datasets used in the project
dataset_list = ['Skripal', 'Navalny', 'Ukraine']

# mapping dictionary to the datasets names
dict_list = {'Skripal': 'Skripal', 'Navalny': 'Navalny', 'Ukraine': 'Ukraine'}