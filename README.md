# YZBoost

YZBoost is a yet another implementation of gradient boosting over
Oblivious Decision Trees.  This is my graduation project for Large
Scale Machine Learning class of [Yandex School of Data
Analysis](https://yandexdataschool.com/).

## Performance

Oblivious Decision Trees are very memory efficient -- just $2^{depth}$ leaf
values and $depth$ pairs $(feature, threshold)$.  But unfortunately they
are very difficult to build and are much less accurate.

It takes $O(BF2^L + FN)$ time build $L$-th layer of the decision tree,
where $L$ is the layer (depth) the layer, $B$ is the number of bins,
$N$ is the number of samples, and $F$ is the number of features.

This makes them uncomparable to traditional decision trees boosters
like `LightGBM` and `XGBoost`.  However there is another reference
implementation `CatBoost`.

See [Benchmarks](notes/Bench_xgb_lgbm_catboost.ipynb) for benchmarks of
other algorithms.  See [My run](notes/myrun.txt) for highlight of my run.

<table>
  <tr>
    <td></td>
    <td>yzboost</td>
    <td>CatBoost</td>
    <td>XGBoost</td>
    <td>LightGBM</td>
  </tr>
  <tr>
    <td>Accuracy</td>
    <td>0.749722</td>
    <td>0.743498</td>
    <td>0.70721</td>
    <td>0.693924</td>
  </tr>
  <tr>
    <td>Train time</td>
    <td>155.53</td>
    <td>220.42</td>
    <td>193.75</td>
    <td>121.58</td>
  </tr>
</table>

## Building

You will need any `gcc` compiler with `OpenMP 4.0` support.  Then simply run `make` to build.

    make

## Running

There are two primary modes supported: `train` and `predict`.  You can run `yzboost` without any arguments to see usage.  As of now:

```
$ ./yzboost
Expected at least 1 command line argument
Usage: ./yzboost <action> [<options>]
Where <action> is one of:
  train
  predict

To train
 $ ./yzboost train --input=data.csv --model=higgs-model.txt --tree-depth=10 --num-trees=100 --validation-fraction=0.1

To predict
 $ ./yzboost predict --input=data.csv --model=higgs-model.txt --output=higgs-predictions.txt

All options:
  --input=somefile.csv            input file with data
  --output=predictions            where will the predictions go
  --model=higgs-model.txt         where will the model be stored or loaded from
  --num-trees=100                 number of trees to use
  --tree-depth=10                 how big those trees should be
  --validation-fraction=0.1       which fraction of samples to use as validation split
  --seed=123                      random seed
  --logging-enabled=0             to print to screen
  --silent=1                      to shut up
  --show-file=1                   to see where the error is coming from
  --discretization-bins=100       how many bins to use for discretization
  --discretization-samples=50000  how many samples to use to do feature binarization

```
