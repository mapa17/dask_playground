import time
from glob import glob
from dask.distributed import Client
from dask import delayed, compute
import dask.bag as db

import pandas as pd

data_files = glob('../data/top-things/*/*.csv')

def get_most_mentioned(df):
    try:
        # Make sure we have no nans
        df['total_mentions'] = df['total_mentions'].fillna(0).astype(int)
        most_mentioned_prod_per_cat = df.groupby('category', as_index=False).apply(lambda frame: frame.loc[frame['total_mentions'].idxmax()])
    except Exception as e:
        most_mentioned_prod_per_cat = pd.DataFrame()
    return most_mentioned_prod_per_cat

def get_most_mentioned_product_from_single_file(fn):
    df = pd.read_csv(fn)
    return get_most_mentioned(df)

def sequential_execution(files):
    partial_best = []
    for file in files:
        partial_best.append(get_most_mentioned_product_from_single_file(file))
    all_bests = pd.concat(partial_best).reset_index(drop=True)
    most_mentioned = get_most_mentioned(all_bests)
    return most_mentioned

def parallel_execution(files, client=None, nWorkers=4):
    if client is None:
        client = Client(n_workers=nWorkers)

    partial_best = []
    for file in files:
        partial_best.append(delayed(get_most_mentioned_product_from_single_file)(file))
    all_bests = delayed(pd.concat)(partial_best).reset_index(drop=True)
    execution_plan = delayed(get_most_mentioned)(all_bests)

    most_mentioned = execution_plan.compute()

    return most_mentioned, execution_plan, client


def batch_parallel_execution(files, batchSize, client=None, nWorkers=4):
    if client is None:
        client = Client(n_workers=nWorkers)

    batches = db.from_sequence(files, partition_size=batchSize)
    partial_best = batches.map(get_most_mentioned_product_from_single_file)

    all_bests = delayed(pd.concat)(partial_best).reset_index(drop=True)
    # all_bests = pd.concat(partial_best).reset_index(drop=True)
    execution_plan = delayed(get_most_mentioned)(all_bests)

    most_mentioned = execution_plan.compute()

    return most_mentioned, execution_plan, client

def batch_parallel_execution2(files, batchSize, client=None, nWorkers=4):
    if client is None:
        client = Client(n_workers=nWorkers)

    batches = db.from_sequence(files, partition_size=batchSize)
    partial_best = batches.map(get_most_mentioned_product_from_single_file)
    all_bests = pd.concat(partial_best).reset_index(drop=True)
    most_mentioned = get_most_mentioned(all_bests)
    most_mentioned = compute(most_mentioned)

    return most_mentioned


#filenames = sorted(glob(os.path.join('data', 'nycflights', '*.csv')))

if __name__ == '__main__':
    print('Sequential execution ...')
    click = time.time()
    most_mentioned = sequential_execution(data_files)
    print(most_mentioned[['category', 'name', 'total_mentions']])
    clack = time.time()
    seq_runtime = clack-click

    # Starting DASK cluster with 4 workers
    client = Client(n_workers=4)

    print('Parallel execution ...')
    click = time.time()
    most_mentioned, plan, _ = parallel_execution(data_files, client=client)
    clack = time.time()
    par_runtime=clack-click
    print(most_mentioned[['category', 'name', 'total_mentions']])

    print('Batch execution ...')
    click = time.time()
    most_mentioned, plan, _ = batch_parallel_execution(data_files, 100, client=client)
    clack = time.time()
    batch_runtime = clack - click
    print(most_mentioned[['category', 'name', 'total_mentions']])

    client.close()

    print(f'Sequential runtime: {seq_runtime}')
    print(f'Parallel runtime: {par_runtime}')
    print(f'Batch runtime: {batch_runtime}')

