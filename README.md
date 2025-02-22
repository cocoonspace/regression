regression
=======
[![Go Reference](https://pkg.go.dev/badge/github.com/cocoonspace/regression.svg)](https://pkg.go.dev/github.com/cocoonspace/regression)
[![Go Report Card](https://goreportcard.com/badge/github.com/cocoonspace/regression)](https://goreportcard.com/report/github.com/cocoonspace/regression)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/cocoonspace/regression/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/cocoonspace/regression/tree/master)
[![License][license-image]][license-url]

[license-image]: http://img.shields.io/badge/license-MIT-green.svg?style=flat-square
[license-url]: LICENSE.txt

Multivariable Linear Regression in Go (golang)

This is a fork of github.com/sajari/regression.

Differences are:
- no output code, if you want to dump anything to stdout, it's your job,
- better performance & lower memory usage,
- ability to retrain with new datapoints.

installation
------------

    $ go get github.com/cocoonspace/regression

Supports Go 1.18+

example usage
-------------

Import the package, create a regression and add data to it. You can use as many variables as you like, in the below example there are 3 variables for each observation.

```go
package main

import (
	"fmt"

	"github.com/cocoonspace/regression"
)

func main() {
	r := &regression.Regression{}
	r.Train(
		regression.DataPoint{Observed:11.2, Variables:[]float64{587000, 16.5, 6.2}},
		regression.DataPoint{Observed:13.4, Variables:[]float64{643000, 20.5, 6.4}},
		regression.DataPoint{Observed:40.7, Variables:[]float64{635000, 26.3, 9.3}},
		regression.DataPoint{Observed:5.3, Variables:[]float64{692000, 16.5, 5.3}},
		regression.DataPoint{Observed:24.8, Variables:[]float64{1248000, 19.2, 7.3}},
		regression.DataPoint{Observed:12.7, Variables:[]float64{643000, 16.5, 5.9}},
		regression.DataPoint{Observed:20.9, Variables:[]float64{1964000, 20.2, 6.4}},
		regression.DataPoint{Observed:35.7, Variables:[]float64{1531000, 21.3, 7.6}},
		regression.DataPoint{Observed:8.7, Variables:[]float64{713000, 17.2, 4.9}},
		regression.DataPoint{Observed:9.6, Variables:[]float64{749000, 14.3, 6.4}},
		regression.DataPoint{Observed:14.5, Variables:[]float64{7895000, 18.1, 6}},
		regression.DataPoint{Observed:26.9, Variables:[]float64{762000, 23.1, 7.4}},
		regression.DataPoint{Observed:15.7, Variables:[]float64{2793000, 19.1, 5.8}},
		regression.DataPoint{Observed:36.2, Variables:[]float64{741000, 24.7, 8.6}},
		regression.DataPoint{Observed:18.1, Variables:[]float64{625000, 18.6, 6.5}},
		regression.DataPoint{Observed:28.9, Variables:[]float64{854000, 24.9, 8.3}},
		regression.DataPoint{Observed:14.9, Variables:[]float64{716000, 17.9, 6.7}},
		regression.DataPoint{Observed:25.8, Variables:[]float64{921000, 22.4, 8.6}},
		regression.DataPoint{Observed:21.7, Variables:[]float64{595000, 20.2, 8.4}},
		regression.DataPoint{Observed:25.7, Variables:[]float64{3353000, 16.9, 6.7}},
	)
	r.Run()
}
```

Note: You can also add data points one by one.

Once calculated you can print the data, look at the R^2, Variance, residuals, etc. You can also access the coefficients directly to use elsewhere, e.g.

```go
// Get the coefficient for the "Inhabitants" variable 0:
c := r.Coeff(0)
```

You can also use the model to predict new data points

```go
prediction, err := r.Predict([]float64{587000, 16.5, 6.2})
```

Feature crosses are supported so your model can capture fixed non-linear relationships

```go

r.Train(
  regression.DataPoint{Observed:11.2, Variables:[]float64{587000, 16.5, 6.2}},
)
//Add a new feature which is the first variable (index 0) to the power of 2
r.AddCross(PowCross(0, 2))
r.Run()

```
