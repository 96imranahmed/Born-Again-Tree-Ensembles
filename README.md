# Born-Again-Tree-Ensembles
A reimplementation of Vidal et al.'s [paper](https://arxiv.org/pdf/2003.11132.pdf) on Born-Again Tree Ensembles. Currently implemented only for the `Depth` optimization objective, as outlined in the body of the main paper.

(To be added: a fun tree picture to show this works)

## Key takeaways
- Implemented approach is **relatively inefficient** as coded, which currently makes the model impractical for general use. For example, it takes ~30 seconds to merge a forest with three trees on my computer. Plenty of opportunities to optimize the code.
- There are a couple of minor typos in the algorithms in the original paper which I attempt to resolve below:
  * On page 5, in the algorithm, there is an `else` missing between `MEMORIZE((Zl, Zr), 0) return  0` and `MEMORIZE((Zl, Zr), 0) return  1`
  * On page 13, in `extract-optimal-solutions`, `EXPORT a split on feature j with level z_j^L` should be `EXPORT a split on feature j with level z_j^l`

## Instructions

The model should be relatively simple to run. The `BATDepth` object accepts a trained random forest classifier, and runs the optimization with the `.fit()` call. This returns a `DecisionTree` which has a `predict` method to help validate that it outputs the same predictions as the original forest.

```python
from model import BATDepth

c_tree = BATDepth(classifier, log = False)
output_decision_tree = c_tree.fit()
test_predict = output_decision_tree.predict(test_data[test_data.columns.difference(['target'])].values)
``` 

I have created a set of tests with various UCI datasets in the `tests` directory. `Basic_test.py` is a variant of the example detailed in the Appendix of the paper. `HTRU_test.py` is the same model as applied to the [HTRU2 dataset](http://archive.ics.uci.edu/ml/datasets/HTRU2).

Running `BATDepth` on an RF (>2 estimators) trained on HTRU2 gives the following output: 
```sh
RF ROC score: 0.9532293936092957
RF Accuracy: 0.9784916201117319
Hyperplane z-bounds:  (array([1, 1, 1, 1, 1, 1, 1, 1]), array([1, 9, 3, 2, 2, 1, 1, 3]))
BATDepth accuracy: 0.9784916201117319
```
Equivalent accuracies between the BATDepth and RF models suggest that **decision trees have been perfectly reconstructed**.

## Todos

PRs to address the below are welcomed :smile:

- Add tree visualisation code **(in progress)**
- Add tree pruning code as specified in the paper
- Improve efficiency of `BATDepth` algorithm (possibly re-implement in C++, to validate timings provided in the paper)
- Add implementations for `DL` and `L` optimisation objectives
- If reimplementation proves efficient for practical use, 'package-ify' this code

## License
Generic MIT license detailed [here](https://github.com/96imranahmed/Born-Again-Tree-Ensembles/blob/master/LICENSE)

## Contact
Feel free to reach-out if you have any questions (contact details on GitHub or my [website](https://imranahmed.io)). I am not affiliated in any way with the authors of the original paper.
