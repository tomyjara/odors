import random

import pandas as pd
import json


def filter(df, contains=[], not_contains=[]):
    for column in contains:
        df = df[df[column] == 1]
    for column in not_contains:
        df = df[df[column] == 0]
    return df


def get_tags(string_of_tags):
    # print(str(string_of_tags))
    # print(type(string_of_tags))
    if str(string_of_tags) != 'nan':
        result = string_of_tags.split('@')
    else:
        result = []
    return result


def build_balanced_multilabel_dataset(dataset_df, target_tags=[], target_positives=None):
    tag_frequences = []

    for tag in target_tags:
        tag_frequences.append((tag, len(dataset_df[dataset_df[tag] == 1])))

    # ordenamos los tags de menor a mayor en funcion de cuanto aparecen en el dataset
    tag_frequences = sorted(tag_frequences, key=lambda tup: tup[1], reverse=False)

    min_tag, tag_freq = tag_frequences.pop(0)
    target_len = positives_target(tag_freq, target_positives)
    tag_df = dataset_df[dataset_df[min_tag] == 1].sample(n=target_len)
    balanced_dataset_subset = tag_df

    processed_tags = [min_tag]

    while len(tag_frequences) > 0:
        # obtenemos el siguiente tag
        curr_tag, curr_tag_freq = tag_frequences.pop(0)
        # nos quedamos con aquellas filas que no contengan tags ya procesados (por invariante ya cumplen con el el numero de apariciones objetivo)
        curr_tag_df = filter(dataset_df, contains=[curr_tag], not_contains=processed_tags)
        processed_tags.append(curr_tag)

        # calculamos cuantas filas se deben agregar para llegar al objetivo actual en base a los que ya se tienen
        curr_tag_train_freq = len(balanced_dataset_subset[balanced_dataset_subset[curr_tag] == 1])
        missing_tag = target_len - curr_tag_train_freq
        # filtramos usando ese numero
        curr_tag_df = curr_tag_df.sample(n=missing_tag)
        balanced_dataset_subset = pd.concat([balanced_dataset_subset, curr_tag_df])

    balanced_dataset_subset_ids = balanced_dataset_subset.pubchem_id
    balanced_dataset_subset_complement = dataset_df[~dataset_df.pubchem_id.isin(balanced_dataset_subset_ids)]
    print(f'Balanced dataset: {len(balanced_dataset_subset)}')
    print(f'Balanced dataset complement: {len(balanced_dataset_subset_complement)}')
    return balanced_dataset_subset, balanced_dataset_subset_complement


def build_ubalanced_multilabel_dataset(dataset_df, target_tags=[], target_positives_ratio=None):
    tag_frequences = []

    for tag in target_tags:
        tag_frequences.append((tag, len(dataset_df[dataset_df[tag] == 1])))

    # ordenamos los tags de menor a mayor en funcion de cuanto aparecen en el dataset
    tag_frequences = sorted(tag_frequences, key=lambda tup: tup[1], reverse=False)

    min_tag, tag_freq = tag_frequences.pop(0)
    tag_df = dataset_df[dataset_df[min_tag] == 1]
    tag_df = tag_df.sample(frac=target_positives_ratio)
    shape = tag_df.shape
    unbalanced_dataset_subset = tag_df

    # saco las filas que contienen la primer etiqueta
    dataset_df = dataset_df[~dataset_df.pubchem_id.isin(unbalanced_dataset_subset.pubchem_id)]

    processed_tags = [min_tag]
    while len(tag_frequences) > 0:
        # obtenemos el siguiente tag
        curr_tag, curr_tag_freq = tag_frequences.pop(0)

        number_of_examples_already_in_ds = filter(unbalanced_dataset_subset, contains=[curr_tag]).shape[0]

        # sacando los processed tags evitamos desbalancear clases ya computadas, pero no garantiza que haya nuevos ejemplos de la clase actual
        # podria suceder dado que reducimoms el espacio de busqueda
        curr_tag_df = filter(dataset_df, contains=[curr_tag], not_contains=processed_tags)
        processed_tags.append(curr_tag)
        curr_shape = curr_tag_df.shape[0]
        target_number_of_examples = (curr_tag_freq*target_positives_ratio)

        #recalculamos el target positives ratio de modo que de igual sabiendo que algunos ejemplos ya entraron arrastrados por las categorias anteriores
        target_positives_recalculated = (target_number_of_examples - number_of_examples_already_in_ds) / curr_shape
        # filtramos usando ese numero
        curr_tag_df = curr_tag_df.sample(frac=target_positives_recalculated)
        unbalanced_dataset_subset = pd.concat([unbalanced_dataset_subset, curr_tag_df])
        dataset_df = dataset_df[~dataset_df.pubchem_id.isin(unbalanced_dataset_subset.pubchem_id)]

        final_count_of_tag_in_ds = filter(unbalanced_dataset_subset, contains=[curr_tag]).shape[0]

        print(curr_tag)

    unbalanced_dataset_subset_ids = unbalanced_dataset_subset.pubchem_id
    unbalanced_dataset_subset_complement = dataset_df[~dataset_df.pubchem_id.isin(unbalanced_dataset_subset_ids)]
    print(f'Unbalanced dataset: {len(unbalanced_dataset_subset)}')
    print(f'Unbalanced dataset complement: {len(unbalanced_dataset_subset_complement)}')
    return unbalanced_dataset_subset, unbalanced_dataset_subset_complement


def positives_target(tag_freq, target_positives):
    if target_positives is not None:
        if target_positives <= tag_freq:
            target_len = target_positives
        else:
            raise Exception(f'Target positives {target_positives} < min target frequence {tag_freq}')
    else:
        target_len = tag_freq
    return target_len


def amount_of_molecules_by_tag(molecules_df):
    molecules_by_tag = {}
    for tags in molecules_df['flavor_profile']:
        for tag in tags:
            if tag in molecules_by_tag:
                molecules_by_tag[tag] += 1
            else:
                molecules_by_tag[tag] = 1
    molecules_by_tag = dict(sorted(molecules_by_tag.items(), key=lambda item: item[1], reverse=True))

    return molecules_by_tag


def randomize_tags(dataset, target_positives_dataset):
    for tag in target_tags:
        dataset = dataset.rename(index=dict(zip(dataset.index, range(len(dataset)))))
        dataset_randoms = random.sample(range(len(dataset)), target_positives_dataset)
        dataset[tag] = 0
        for position in dataset_randoms:
            dataset.at[position, tag] = 1

    return dataset


def train_test_validation_balanced(common_tags, target_tags, target_positives_train, target_positives_validation_test):
    train, test = build_balanced_multilabel_dataset(common_tags, target_tags=target_tags,
                                                    target_positives=target_positives_train)
    test, validation = build_balanced_multilabel_dataset(test, target_tags=target_tags,
                                                         target_positives=target_positives_validation_test)
    validation, garbage = build_balanced_multilabel_dataset(validation, target_tags=target_tags,
                              target_positives = target_positives_validation_test)

    return train, test, validation


def train_test_validation_unbalanced(common_tags, target_tags, target_positives_train, target_positives_validation_test):
    #los ratios estan cambiados porque si primero saco un 60% digamos, luego necesito el 50 de lo que queda para test y el 100 de lo nuevo que queda para validation
    utrain, utest = build_ubalanced_multilabel_dataset(common_tags, target_tags=target_tags, target_positives_ratio=target_positives_train)
    utest, uvalidation = build_ubalanced_multilabel_dataset(utest, target_tags=target_tags, target_positives_ratio=target_positives_validation_test)
    uvalidation, ugarbage = build_ubalanced_multilabel_dataset(uvalidation, target_tags=target_tags, target_positives_ratio=1)

    return utrain, utest, uvalidation

if __name__ == '__main__':
    # read molecules df
    molecules_df = pd.read_csv('molescules.csv')

    # append positive/negative flags
    molecules_df['flavor_profile'] = molecules_df['flavor_profile'].apply(get_tags)
    molecules_df['has_odor'] = molecules_df['odor'].apply(
        lambda tag: 1 if (not (tag != tag)) and tag != 'odorless' else (0 if (not (tag != tag)) else -1))
    molecules_df['one_tag'] = molecules_df['flavor_profile'].apply(lambda tags: 1 if len(tags) == 1 else 0)
    molecules_df['has_bitter'] = molecules_df['flavor_profile'].apply(
        lambda tags: 1 if 'bitter' in tags and ('sweet' not in tags) and ('sweet-like' not in tags) else 0)
    molecules_df['no_sweet'] = molecules_df['flavor_profile'].apply(
        lambda tags: 1 if ('sweet' not in tags and 'sweet-like' not in tags) else 0)
    molecules_df['has_sweet'] = molecules_df['flavor_profile'].apply(lambda tags: 1 if 'sweet' in tags else 0)

    sulfur_not = ['floral', 'bitter']
    bitter_not = ['floral', 'sulfurous', 'sulfur', 'sulfury']
    floral_not = ['bitter', 'sulfurous', 'sulfur', 'sulfury']
    # herbal_not = ['bitter', 'fruity', 'herbal', 'floral', 'green', 'woody']
    # green_not = ['bitter', 'fruity', 'herbal', 'floral', 'green', 'woody']
    # herbal_not = ['bitter', 'fruity', 'herbal', 'floral', 'green', 'woody']
    # woody_not = ['bitter', 'fruity', 'herbal', 'floral', 'green', 'woody']

    molecules_df['sulfur'] = molecules_df['flavor_profile'].apply(lambda tags: 1 if (('sulfurous' in tags or 'sulfury' in tags or 'sulfur' in tags) and len(set(sulfur_not) & set(tags)) == 0) else 0)
    molecules_df['fruity'] = molecules_df['flavor_profile'].apply(lambda tags: 1 if ('fruity' in tags or 'fruit' in tags) else 0)  # and ('wood' not in tags and'green' not in tags and 'woody' not in tags and 'herbal' not in tags and 'bitter' not in tags) else 0)
    molecules_df['green'] = molecules_df['flavor_profile'].apply(lambda tags: 1 if 'green' in tags else 0)  # and ('wood' not in tags and'green' not in tags and 'woody' not in tags and 'herbal' not in tags and 'bitter' not in tags) else 0) and ('wood' not in tags and 'floral' not in tags and 'woody' not in tags and 'herbal' not in tags and 'bitter' not in tags and 'fruit' not in tags and 'fruity' not in tags) else 0)
    molecules_df['floral'] = molecules_df['flavor_profile'].apply(lambda tags: 1 if 'floral' in tags else 0)  # and ('wood' not in tags and'green' not in tags and 'woody' not in tags and 'herbal' not in tags and 'bitter' not in tags) else 0) and ('wood' not in tags and 'green' not in tags and 'woody' not in tags and 'herbal' not in tags and 'bitter' not in tags and 'fruit' not in tags and 'fruity' not in tags) else 0)
    molecules_df['woody'] = molecules_df['flavor_profile'].apply(lambda tags: 1 if ('woody' in tags or 'wood' in tags) else 0)  # and ('wood' not in tags and'green' not in tags and 'woody' not in tags and 'herbal' not in tags and 'bitter' not in tags) else 0) and ('green' not in tags and 'floral' not in tags and 'herbal' not in tags and 'bitter' not in tags and 'fruit' not in tags and 'fruity' not in tags) else 0)
    molecules_df['herbal'] = molecules_df['flavor_profile'].apply(lambda tags: 1 if 'herbal' in tags else 0)  # and ('wood' not in tags and'green' not in tags and 'woody' not in tags and 'herbal' not in tags and 'bitter' not in tags) else 0) and ('wood' not in tags and 'green' not in tags and 'woody' not in tags and 'floral' not in tags and 'bitter' not in tags and 'fruit' not in tags and 'fruity' not in tags) else 0)
    molecules_df['bitter'] = molecules_df['flavor_profile'].apply(lambda tags: 1 if 'bitter' in tags else 0)  # and ('wood' not in tags and'green' not in tags and 'woody' not in tags and 'herbal' not in tags and 'bitter' not in tags) else 0) and ('wood' not in tags and 'green' not in tags and 'woody' not in tags and 'herbal' not in tags and 'floral' not in tags and 'fruit' not in tags and 'fruity' not in tags) else 0)
    molecules_df['spicy'] = molecules_df['flavor_profile'].apply(lambda tags: 1 if 'spicy' in tags else 0)

    SUBDIR = 'common_tags/unbalanced'
    molecules_df = molecules_df.filter(
        items=['pubchem_id', 'common_name', 'inchi', 'smile', 'sulfur', 'bitter', 'fruity', 'herbal', 'floral', 'green',
               'woody', 'flavor_profile'])

    sulfur = molecules_df[(molecules_df['sulfur'] == 1)]
    sulfur.to_csv(f'{SUBDIR}/sulfur.csv', index=False)

    fruity = molecules_df[(molecules_df['fruity'] == 1)]
    fruity.to_csv(f'{SUBDIR}/fruity.csv', index=False)

    green = molecules_df[(molecules_df['green'] == 1)]
    green.to_csv(f'{SUBDIR}/green.csv', index=False)

    floral = molecules_df[(molecules_df['floral'] == 1)]
    floral.to_csv(f'{SUBDIR}/floral.csv', index=False)

    woody = molecules_df[(molecules_df['woody'] == 1)]
    woody.to_csv(f'{SUBDIR}/woody.csv', index=False)

    herbal = molecules_df[(molecules_df['herbal'] == 1)]
    herbal.to_csv(f'{SUBDIR}/herbal.csv', index=False)

    bitter = molecules_df[(molecules_df['bitter'] == 1)]
    bitter.to_csv(f'{SUBDIR}/bitter.csv', index=False)

    #spicy = molecules_df[(molecules_df['spicy'] == 1)]
    #spicy.to_csv(f'{SUBDIR}/spicy.csv', index=False)

    common_tags = pd.concat([fruity, bitter, green, floral, woody]).sample(frac=1)
    common_tags = common_tags.drop_duplicates(subset=['pubchem_id'])
    common_tags.to_csv(f'{SUBDIR}/common_tags.csv', index=False)

    target_tags = ['green', 'bitter', 'fruity', 'floral', 'woody']

    target_positives_train = 143
    target_positives_validation_test = 50

    #train, test, validation = train_test_validation_balanced(common_tags, target_tags, target_positives_train, target_positives_validation_test)
    utrain, utest, uvalidation = train_test_validation_unbalanced(common_tags, target_tags, 0.6, 0.5)

    #if randomize:
        #train = randomize_tags(train, target_positives_train)
        #test = randomize_tags(test, target_positives_validation_test)
        #validation = randomize_tags(validation, target_positives_validation_test)

    #train.to_csv(f'{SUBDIR}/common_tags_train.csv', index=False)
    #test.to_csv(f'{SUBDIR}/common_tags_test.csv', index=False)
    #validation.to_csv(f'{SUBDIR}/common_tags_validation.csv', index=False)

    utrain.to_csv(f'{SUBDIR}/unbalanced_common_tags_train.csv', index=False)
    utest.to_csv(f'{SUBDIR}/unbalanced_common_tags_test.csv', index=False)
    uvalidation.to_csv(f'{SUBDIR}/unbalanced_common_tags_validation.csv', index=False)

    exit(0)

    # consideramos solo sweet o también sweet-like?
    molecules_df['only_sweet'] = molecules_df['flavor_profile'].apply(
        lambda tags: 1 if len(tags) == 1 and tags[0] == 'sweet' else 0)

    # consideramos que contenga sweet y lo que sea o solo sweet y una cosa mas?
    molecules_df['sweet_and_something_else'] = molecules_df['flavor_profile'].apply(
        lambda tags: 1 if len(tags) > 1 and 'sweet' in tags else 0)

    molecules_by_tag = amount_of_molecules_by_tag(molecules_df)
    one_tag_molecules = (molecules_df[molecules_df['one_tag'] == 1])['common_name']
    only_sweet = (molecules_df[molecules_df['only_sweet'] == 1])['common_name']
    no_sweet = (molecules_df[molecules_df['no_sweet'] == 1])

    bitter = (molecules_df[molecules_df['has_bitter'] == 1])
    odorless = molecules_df[molecules_df['has_odor'] == 0]
    not_odorless = molecules_df[molecules_df['has_odor'] == 1]
    sweet_and_something_else = (molecules_df[molecules_df['sweet_and_something_else'] == 1])

    odorless.to_csv('odor/odorless.csv', index=False)
    not_odorless.to_csv('odor/not_odorless.csv', index=False)

    one_tag_molecules.to_csv('one_tag_molecules.csv', index=False)
    only_sweet.to_csv('only_sweet.csv', index=False)
    sweet_and_something_else.to_csv('sweet_and_something_else.csv', index=False)
    bitter.to_csv('bitter.csv', index=False)
    no_sweet.to_csv('no_sweet.csv', index=False)
    molecules_df.to_csv('molecules_df.csv', index=False)

    with open('molecules_by_tag.json', 'w') as f:
        json.dump(molecules_by_tag, f, indent=4)

    print('ONE TAG')
    print(one_tag_molecules.shape)
    print(one_tag_molecules)
    print()

    print('ONLY SWEET')
    print(only_sweet.shape)
    print(only_sweet)
    print()

    print('HAS BITTER')
    print(bitter.shape)
    print(bitter)
    print()

    print('SWEET AND SOMETHING ELSE')
    print(sweet_and_something_else.shape)
    print(sweet_and_something_else)
    print()
