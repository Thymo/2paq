def sort_dict_by_val(x):
    return {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}

def unique(collection):
    return list(dict.fromkeys(collection))
