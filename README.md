# SIVIR

This package aims at using the SIVI in the logistic regression setup. This algorithm is brought up to solve the degenarcy phenomenon in the posterior in the mean-field case, which introduce some extra terms to modify the lower and upper bound, making the new surrogate converging to the ELBO on both side. This alorithm comes from (Mingzhang Y., Mingyuan Z. 2018)(http://proceedings.mlr.press/v80/yin18b/yin18b.pdf).

## Installation 

For installing this package
```{r}
install.packages("devtools") # if you have not install this package yet.
devtools::install_github("wanghg1996/SIVIR")
```

## Example of the SIVI
Giving an example using the attached data.

```{r}
library(SIVIR)
x = waveform$X.train
y = waveform$y.train
result = sivi_lr(x, y)
pos = result$sample_pos
dim(pos)
```

The output of the function "sivi_lr" will automatically print the objective values and the standerd deviation of this part.

```{r}
>iter: 51 loss = 120.9965 , std = 49.9286 
>iter: 101 loss = 98.06868 , std = 1.933512 
>iter: 151 loss = 93.56685 , std = 1.204331 
>iter: 201 loss = 90.4817 , std = 0.8567058 
>iter: 251 loss = 89.46041 , std = 0.6002979 
>iter: 301 loss = 89.35255 , std = 0.5086475 
>iter: 351 loss = 89.36044 , std = 0.622364 
>iter: 401 loss = 89.06226 , std = 0.4958853 
>iter: 451 loss = 89.26981 , std = 0.8869689 
>iter: 501 loss = 89.14639 , std = 0.473536
```

When we tuning the corresponding parameters like K, J and iter_nums, the output may get better or worse correspondingly, for which I have a more detailed explain in the vignette.
