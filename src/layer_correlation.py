def layer_correlation(df, source, target):
    sum_of_numerator = 0
    sum_of_denominator = 0
    for index, row in df.iterrows():
        if row[source] > 0 and row[target] > 0:
            sum_of_numerator += 1
        else:
            sum_of_numerator += 0
    for index, row in df.iterrows():
        if row[target] > 0:
            sum_of_denominator += 1
        else:
            sum_of_denominator += 0
    return sum_of_numerator / sum_of_denominator

def weighted_layer_correlation(df, source, target):
    sum_of_numerator = 0
    sum_of_denominator = 0
    for index, row in df.iterrows():
        if row[source] > 0 and row[target] > 0:
            sum_of_numerator += row[target]
        else:
            sum_of_numerator += 0
    for index, row in df.iterrows():
        if row[target] > 0:
            sum_of_denominator += row[target]
        else:
            sum_of_denominator += 0
    return sum_of_numerator / sum_of_denominator
