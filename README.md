# SIVIR

This package aims at using the SIVI in the logistic regression setup. This algorithm is brought up to solve the degenarcy phenomenon in the posterior in the mean-field case, which introduce some extra terms to modify the lower and upper bound, making the new surrogate converging to the ELBO on both side. This alorithm comes from (Mingzhang Y., Mingyuan Z. 2018)(http://proceedings.mlr.press/v80/yin18b/yin18b.pdf).

## Example of the SIVI
Giving an example using the attached data.

```{r}
x = waveform$X.train
y = waveform$y.train
result = sivi_lr(x, y)
pos = result$sample_pos
dim(pos)
```

The output of the function "sivi_lr" will automatically print the objective values and the standerd deviation of this part.
